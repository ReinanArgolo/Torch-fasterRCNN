from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

from modules import get_model
from transforms import ResizeShortSide


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Det:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    label: int


def _ensure_chw_float01_3ch(img: torch.Tensor) -> torch.Tensor:
    x = img
    if x.dtype != torch.float32:
        x = x.float()

    if x.ndim == 3 and x.shape[0] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1).contiguous()

    if x.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape={tuple(x.shape)}")

    if x.max().item() > 1.0:
        x = x / 255.0

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] == 4:
        x = x[:3, ...]

    return x


def _load_image_for_infer(path: str, resize_min: Optional[int], resize_max: Optional[int]) -> Tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB")

    if resize_min is not None:
        sample_transform = ResizeShortSide(min_size=int(resize_min), max_size=int(resize_max or 1333))
        img_t, _ = sample_transform(pil, {})
        if not isinstance(img_t, torch.Tensor):
            img_t = torchvision.transforms.functional.to_tensor(img_t)
        img_t = _ensure_chw_float01_3ch(img_t)
        pil = torchvision.transforms.functional.to_pil_image(img_t)
        return pil, img_t

    img_t = torchvision.transforms.functional.to_tensor(pil)
    img_t = _ensure_chw_float01_3ch(img_t)
    return pil, img_t


def _find_fold_checkpoints(run_dir: str) -> List[Tuple[str, str]]:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # Prefer CV layout: run_dir/fold_*/best.pth
    fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    out: List[Tuple[str, str]] = []
    for fd in fold_dirs:
        ckpt = fd / "best.pth"
        if ckpt.exists():
            out.append((fd.name, str(ckpt)))

    if out:
        return out

    # Fallback: single run best.pth
    ckpt = base / "best.pth"
    if ckpt.exists():
        return [(base.name, str(ckpt))]

    raise FileNotFoundError(f"No best.pth found in {run_dir} (expected fold_*/best.pth or best.pth)")


def _predict(model, img_t: torch.Tensor, score_thresh: float) -> List[Det]:
    model.eval()
    with torch.no_grad():
        out = model([img_t.to(DEVICE)])[0]

    boxes = out.get("boxes")
    scores = out.get("scores")
    labels = out.get("labels")
    if boxes is None or scores is None or labels is None:
        return []

    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    dets: List[Det] = []
    for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        dets.append(Det(bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), score=float(s), label=int(l)))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _wbf(
    per_model: Sequence[Sequence[Det]],
    image_size_hw: Tuple[int, int],
    weights: Optional[Sequence[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
) -> List[Det]:
    """Lightweight Weighted Boxes Fusion.

    This is a small, dependency-free implementation meant for quick ensemble visualization.

    - Boxes are fused per-class.
    - Coordinates are normalized to [0,1] internally.
    """

    h, w = int(image_size_hw[0]), int(image_size_hw[1])
    if h <= 0 or w <= 0:
        return []

    n_models = len(per_model)
    if weights is None:
        weights = [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"wbf weights length ({len(weights)}) must equal number of models ({n_models})")

    by_label: Dict[int, List[Tuple[int, Det]]] = defaultdict(list)
    for m_idx, dets in enumerate(per_model):
        for d in dets:
            if d.score < float(skip_box_thr):
                continue
            by_label[int(d.label)].append((m_idx, d))

    fused: List[Det] = []

    for label, items in by_label.items():
        # sort by score desc
        items = sorted(items, key=lambda t: t[1].score, reverse=True)

        clusters: List[List[Tuple[int, Det]]] = []
        cluster_boxes: List[List[float]] = []  # representative box (normalized)

        def _norm_box(d: Det) -> List[float]:
            x1, y1, x2, y2 = d.bbox_xyxy
            return [x1 / w, y1 / h, x2 / w, y2 / h]

        for m_idx, det in items:
            b = _norm_box(det)

            best_i = -1
            best_iou = 0.0
            for i, rb in enumerate(cluster_boxes):
                iou = _iou_xyxy(b, rb)
                if iou >= float(iou_thr) and iou > best_iou:
                    best_iou = iou
                    best_i = i

            if best_i < 0:
                clusters.append([(m_idx, det)])
                cluster_boxes.append(b)
            else:
                clusters[best_i].append((m_idx, det))
                # update representative box as current fused estimate
                cluster_boxes[best_i] = _fuse_cluster_box(clusters[best_i], w, h, weights)

        # finalize clusters -> Det
        for cluster in clusters:
            fused_box = _fuse_cluster_box(cluster, w, h, weights)
            fused_score = _fuse_cluster_score(cluster, weights, conf_type=conf_type)
            x1n, y1n, x2n, y2n = fused_box
            fused.append(
                Det(
                    bbox_xyxy=(x1n * w, y1n * h, x2n * w, y2n * h),
                    score=float(fused_score),
                    label=int(label),
                )
            )

    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


def _fuse_cluster_box(cluster: Sequence[Tuple[int, Det]], w: int, h: int, weights: Sequence[float]) -> List[float]:
    # Weighted average of coords with weight = model_weight * score
    sum_w = 0.0
    acc = [0.0, 0.0, 0.0, 0.0]
    for m_idx, d in cluster:
        mw = float(weights[m_idx])
        ww = mw * float(d.score)
        x1, y1, x2, y2 = d.bbox_xyxy
        acc[0] += (x1 / w) * ww
        acc[1] += (y1 / h) * ww
        acc[2] += (x2 / w) * ww
        acc[3] += (y2 / h) * ww
        sum_w += ww
    if sum_w <= 0.0:
        # fallback: simple avg
        n = max(1, len(cluster))
        x1 = sum(d.bbox_xyxy[0] for _, d in cluster) / n
        y1 = sum(d.bbox_xyxy[1] for _, d in cluster) / n
        x2 = sum(d.bbox_xyxy[2] for _, d in cluster) / n
        y2 = sum(d.bbox_xyxy[3] for _, d in cluster) / n
        return [x1 / w, y1 / h, x2 / w, y2 / h]
    return [v / sum_w for v in acc]


def _fuse_cluster_score(cluster: Sequence[Tuple[int, Det]], weights: Sequence[float], conf_type: str = "avg") -> float:
    if not cluster:
        return 0.0

    # Use only model weights here (classic WBF behavior variants differ)
    wsum = 0.0
    ssum = 0.0
    smax = 0.0
    for m_idx, d in cluster:
        mw = float(weights[m_idx])
        wsum += mw
        ssum += float(d.score) * mw
        smax = max(smax, float(d.score))

    if conf_type == "max":
        return float(smax)
    # default: avg
    if wsum <= 0.0:
        return float(sum(float(d.score) for _, d in cluster) / len(cluster))
    return float(ssum / wsum)


def _label_color(label: int) -> Tuple[int, int, int]:
    # Simple deterministic palette
    palette = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 120, 255),
        (255, 200, 50),
        (200, 50, 255),
        (50, 255, 220),
    ]
    return palette[int(label) % len(palette)]


def _draw_dets(
    pil: Image.Image,
    dets: Sequence[Det],
    title: str,
    score_thresh: float,
    max_boxes: int,
) -> Image.Image:
    img = pil.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Title bar
    pad = 6
    bar_h = 22
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    draw.text((pad, 3), title, fill=(255, 255, 255), font=font)

    kept = 0
    for d in dets:
        if kept >= int(max_boxes):
            break
        if d.score < float(score_thresh):
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        color = _label_color(d.label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = f"{d.label}:{d.score:.2f}"
        draw.text((x1 + 2, max(0, y1 + 2)), txt, fill=color, font=font)
        kept += 1

    return img


def _fit_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    # Keep aspect ratio and pad
    canvas = Image.new("RGB", (cell_w, cell_h), (20, 20, 20))
    fitted = ImageOps.contain(img, (cell_w, cell_h))
    x = (cell_w - fitted.width) // 2
    y = (cell_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _make_grid(rows: List[List[Image.Image]], cell_w: int, cell_h: int) -> Image.Image:
    if not rows:
        raise ValueError("No rows to make grid")
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)

    grid = Image.new("RGB", (n_cols * cell_w, n_rows * cell_h), (0, 0, 0))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            cell = _fit_cell(img, cell_w, cell_h)
            grid.paste(cell, (c * cell_w, r * cell_h))
    return grid


def _write_html(out_dir: str, grid_png_name: str, per_image_rows: List[Tuple[str, List[str]]], col_names: List[str]) -> str:
    # Simple local-friendly report
    html_path = os.path.join(out_dir, "report.html")
    lines: List[str] = []
    lines.append("<html><head><meta charset='utf-8'><title>Ensemble Grid</title></head><body>")
    lines.append("<h2>Grid</h2>")
    lines.append(f"<p><img src='{grid_png_name}' style='max-width: 100%; height: auto;'/></p>")
    lines.append("<h2>Por imagem</h2>")
    lines.append("<table border='1' cellspacing='0' cellpadding='4'>")

    # header
    lines.append("<tr>")
    lines.append("<th>Arquivo</th>")
    for cn in col_names:
        lines.append(f"<th>{cn}</th>")
    lines.append("</tr>")

    for file_name, img_paths in per_image_rows:
        lines.append("<tr>")
        lines.append(f"<td>{file_name}</td>")
        for p in img_paths:
            lines.append(f"<td><img src='{p}' style='max-width: 360px; height: auto;'/></td>")
        lines.append("</tr>")

    lines.append("</table>")
    lines.append("</body></html>")

    os.makedirs(out_dir, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return html_path


def _load_yaml_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping/dict, got: {type(data)}")
    return data


def _cfg_get(cfg: dict, keys: Sequence[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    p = argparse.ArgumentParser(
        description=(
            "Ensemble test: carrega best.pth de cada fold de um run, roda inferência, "
            "e gera um grid (cada fold + ensemble WBF). Output é PNG/HTML (headless/SSH friendly)."
        )
    )
    p.add_argument("--config", default=None, help="Opcional: config.yaml do projeto (para ler model.name/num_classes/params)")
    p.add_argument("--run-dir", required=True, help="Diretório do run (ex.: outputs/.../run_YYYYMMDD_HHMMSS)")
    p.add_argument("--images", required=True, help="Diretório com imagens para inferência")
    p.add_argument("--out-dir", required=True, help="Diretório de saída (vai criar grid.png e report.html)")

    p.add_argument("--model", default=None, help="Sobrescreve model.name do config.yaml")
    p.add_argument("--num-classes", type=int, default=None, help="Sobrescreve model.num_classes do config.yaml")

    p.add_argument("--score-thresh", type=float, default=0.4, help="Score mínimo para desenhar/usar no WBF")
    p.add_argument("--max-boxes", type=int, default=50, help="Máximo de boxes desenhadas por imagem")

    p.add_argument("--resize-min", type=int, default=None)
    p.add_argument("--resize-max", type=int, default=None)

    p.add_argument("--limit", type=int, default=6, help="Número de imagens para visualizar")
    p.add_argument("--shuffle", action="store_true", help="Embaralha imagens antes de selecionar --limit")
    p.add_argument("--seed", type=int, default=42)

    # WBF
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--wbf-skip-box-thr", type=float, default=0.0)
    p.add_argument("--wbf-conf", choices=["avg", "max"], default="avg")
    p.add_argument(
        "--wbf-weights",
        type=float,
        nargs="*",
        default=None,
        help="Lista de pesos por fold/modelo (mesma ordem do fold_1..fold_k). Se omitido, pesos iguais.",
    )

    # Grid
    p.add_argument("--cell", type=int, default=560, help="Tamanho base da célula do grid (px)")

    args = p.parse_args()

    cfg = _load_yaml_config(args.config)
    model_name = args.model or _cfg_get(cfg, ["model", "name"], "fasterrcnn_resnet50_fpn_v2")
    num_classes = args.num_classes or int(_cfg_get(cfg, ["model", "num_classes"], 2))
    model_params = _cfg_get(cfg, ["model", "params"], {})
    if model_params is None:
        model_params = {}
    if not isinstance(model_params, dict):
        raise SystemExit("config.yaml: model.params deve ser um dict")

    ckpts = _find_fold_checkpoints(args.run_dir)
    if args.wbf_weights is not None and len(args.wbf_weights) not in (0, len(ckpts)):
        raise SystemExit(f"--wbf-weights deve ter {len(ckpts)} valores (ou ser omitido).")

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        raise SystemExit(f"--images must be a directory: {args.images}")

    files = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.name.lower().endswith(IMG_EXTS)]
    if not files:
        raise SystemExit(f"No images found in {args.images}")

    if args.shuffle:
        random.seed(int(args.seed))
        random.shuffle(files)

    files = files[: int(args.limit)]

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "per_image")
    os.makedirs(vis_dir, exist_ok=True)

    # Load all models (one-by-one on GPU to avoid OOM, but keep state on CPU to reload quickly)
    # Given typical fold count is small (5), we can keep them instantiated; if you hit OOM, switch to sequential.
    models: List[Tuple[str, torch.nn.Module]] = []
    for fold_name, ckpt_path in ckpts:
        model = get_model(model_name, num_classes=int(num_classes), pretrained=False, **model_params)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()
        models.append((fold_name, model))

    weights = args.wbf_weights if args.wbf_weights is not None and len(args.wbf_weights) else None

    grid_rows: List[List[Image.Image]] = []
    per_image_rows: List[Tuple[str, List[str]]] = []

    col_names = [name for name, _ in models] + ["ensemble_wbf"]

    for img_path in files:
        pil, img_t = _load_image_for_infer(str(img_path), args.resize_min, args.resize_max)
        h, w = pil.height, pil.width

        per_model_dets: List[List[Det]] = []
        per_model_imgs: List[Image.Image] = []

        for fold_name, model in models:
            dets = _predict(model, img_t, score_thresh=float(args.score_thresh))
            per_model_dets.append(dets)
            per_model_imgs.append(
                _draw_dets(
                    pil,
                    dets,
                    title=fold_name,
                    score_thresh=float(args.score_thresh),
                    max_boxes=int(args.max_boxes),
                )
            )

        dets_wbf = _wbf(
            per_model=per_model_dets,
            image_size_hw=(h, w),
            weights=weights,
            iou_thr=float(args.wbf_iou),
            skip_box_thr=float(args.wbf_skip_box_thr),
            conf_type=str(args.wbf_conf),
        )
        img_wbf = _draw_dets(
            pil,
            dets_wbf,
            title="ensemble_wbf",
            score_thresh=float(args.score_thresh),
            max_boxes=int(args.max_boxes),
        )

        row_imgs = per_model_imgs + [img_wbf]
        grid_rows.append(row_imgs)

        # Save per-image panel for easier inspection
        safe_name = img_path.name
        out_paths: List[str] = []
        for col_img, col_name in zip(row_imgs, col_names):
            out_name = f"{Path(safe_name).stem}__{col_name}.jpg"
            out_rel = os.path.join("per_image", out_name)
            out_abs = os.path.join(args.out_dir, out_rel)
            col_img.save(out_abs, quality=92)
            out_paths.append(out_rel)
        per_image_rows.append((safe_name, out_paths))

    grid = _make_grid(grid_rows, cell_w=int(args.cell), cell_h=int(args.cell))
    grid_png = os.path.join(args.out_dir, "grid.png")
    grid.save(grid_png)

    html_path = _write_html(args.out_dir, "grid.png", per_image_rows, col_names)

    print(
        {
            "run_dir": args.run_dir,
            "images": len(files),
            "models": len(models),
            "out_dir": args.out_dir,
            "grid": grid_png,
            "report": html_path,
            "device": str(DEVICE),
            "model": model_name,
            "num_classes": int(num_classes),
        }
    )


if __name__ == "__main__":
    main()
