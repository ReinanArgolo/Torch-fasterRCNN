from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

from modules import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Det:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    label: int


def _find_fold_checkpoints(run_dir: str) -> List[Tuple[str, str]]:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    out: List[Tuple[str, str]] = []
    for fd in fold_dirs:
        ckpt = fd / "best.pth"
        if ckpt.exists():
            out.append((fd.name, str(ckpt)))

    if out:
        return out

    ckpt = base / "best.pth"
    if ckpt.exists():
        return [(base.name, str(ckpt))]

    raise FileNotFoundError(f"No best.pth found in {run_dir} (expected fold_*/best.pth or best.pth)")


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


def _cv2_bgr_to_pil_rgb(frame_bgr) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_rgb_to_cv2_bgr(img: Image.Image):
    rgb = torchvision.transforms.functional.pil_to_tensor(img).permute(1, 2, 0).contiguous().numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


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


def _fuse_cluster_box(cluster: Sequence[Tuple[int, Det]], w: int, h: int, weights: Sequence[float]) -> List[float]:
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

    if wsum <= 0.0:
        return float(sum(float(d.score) for _, d in cluster) / len(cluster))
    return float(ssum / wsum)


def _wbf(
    per_model: Sequence[Sequence[Det]],
    image_size_hw: Tuple[int, int],
    weights: Optional[Sequence[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
) -> List[Det]:
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
        items = sorted(items, key=lambda t: t[1].score, reverse=True)

        clusters: List[List[Tuple[int, Det]]] = []
        cluster_boxes: List[List[float]] = []

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
                cluster_boxes[best_i] = _fuse_cluster_box(clusters[best_i], w, h, weights)

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


def _label_color(label: int) -> Tuple[int, int, int]:
    palette = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 120, 255),
        (255, 200, 50),
        (200, 50, 255),
        (50, 255, 220),
    ]
    return palette[int(label) % len(palette)]


def _draw_dets(pil: Image.Image, dets: Sequence[Det], title: str, score_thresh: float, max_boxes: int) -> Image.Image:
    img = pil.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

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
        draw.text((x1 + 2, max(0, y1 + 2)), f"{d.label}:{d.score:.2f}", fill=color, font=font)
        kept += 1

    return img


def _fit_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    canvas = Image.new("RGB", (cell_w, cell_h), (20, 20, 20))
    fitted = ImageOps.contain(img, (cell_w, cell_h))
    x = (cell_w - fitted.width) // 2
    y = (cell_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _make_single_row_grid(cells: List[Image.Image], cell_w: int, cell_h: int) -> Image.Image:
    n_cols = len(cells)
    grid = Image.new("RGB", (n_cols * cell_w, cell_h), (0, 0, 0))
    for c, img in enumerate(cells):
        cell = _fit_cell(img, cell_w, cell_h)
        grid.paste(cell, (c * cell_w, 0))
    return grid


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
            "Gera um MP4 com grid por frame: cada fold + ensemble WBF. "
            "Ideal para rodar em servidor via SSH e assistir localmente (baixando o MP4)."
        )
    )
    p.add_argument("--config", default=None, help="Opcional: config.yaml do projeto (para ler model.name/num_classes/params)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--video", required=True, help="Caminho para o vídeo de entrada (mp4/avi/etc)")
    p.add_argument("--out", default=None, help="Caminho para o mp4 de saída (obrigatório a menos que --no-mp4)")

    p.add_argument("--gui", action="store_true", help="Mostra janela com o grid (para rodar localmente com desktop)")
    p.add_argument("--no-mp4", action="store_true", help="Não gera MP4; útil quando usar --gui")

    p.add_argument("--model", default=None, help="Sobrescreve model.name do config.yaml")
    p.add_argument("--num-classes", type=int, default=None, help="Sobrescreve model.num_classes do config.yaml")

    p.add_argument("--score-thresh", type=float, default=0.4)
    p.add_argument("--max-boxes", type=int, default=50)

    p.add_argument("--every", type=int, default=5, help="Processa 1 frame a cada N frames (stride)")
    p.add_argument("--max-frames", type=int, default=300, help="Limite de frames processados")
    p.add_argument("--out-fps", type=float, default=10.0)

    # WBF
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--wbf-skip-box-thr", type=float, default=0.0)
    p.add_argument("--wbf-conf", choices=["avg", "max"], default="avg")
    p.add_argument("--wbf-weights", type=float, nargs="*", default=None)

    # Grid
    p.add_argument("--cell", type=int, default=540)

    args = p.parse_args()

    cfg = _load_yaml_config(args.config)

    model_name = args.model or _cfg_get(cfg, ["model", "name"], "fasterrcnn_resnet50_fpn_v2")
    num_classes = args.num_classes or int(_cfg_get(cfg, ["model", "num_classes"], 2))
    model_params = _cfg_get(cfg, ["model", "params"], {})
    if model_params is None:
        model_params = {}
    if not isinstance(model_params, dict):
        raise SystemExit("config.yaml: model.params deve ser um dict")

    if args.no_mp4 and args.out is not None:
        # allow passing --out but ignore it
        args.out = None
    if not args.no_mp4 and not args.out:
        raise SystemExit("Você precisa passar --out (ou então usar --no-mp4)")

    if args.gui:
        # Basic headless guard: imshow will fail without a display.
        if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
            raise SystemExit("--gui requer um ambiente com display (rodar localmente). No servidor headless, gere MP4.")

    ckpts = _find_fold_checkpoints(args.run_dir)
    if args.wbf_weights is not None and len(args.wbf_weights) not in (0, len(ckpts)):
        raise SystemExit(f"--wbf-weights deve ter {len(ckpts)} valores (ou ser omitido).")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {args.video}")

    # Load models
    models: List[Tuple[str, torch.nn.Module]] = []
    for fold_name, ckpt_path in ckpts:
        model = get_model(model_name, num_classes=int(num_classes), pretrained=False, **model_params)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()
        models.append((fold_name, model))

    weights = args.wbf_weights if args.wbf_weights is not None and len(args.wbf_weights) else None
    col_names = [n for n, _ in models] + ["ensemble_wbf"]

    # Determine output video size
    cell = int(args.cell)
    out_w = len(col_names) * cell
    out_h = cell

    writer = None
    if not args.no_mp4:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, float(args.out_fps), (out_w, out_h))
        if not writer.isOpened():
            raise SystemExit(f"Não consegui criar VideoWriter em: {args.out}")

    frame_idx = 0
    written = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % int(args.every) != 0:
            frame_idx += 1
            continue

        pil = _cv2_bgr_to_pil_rgb(frame_bgr)
        img_t = torchvision.transforms.functional.to_tensor(pil)
        img_t = _ensure_chw_float01_3ch(img_t)

        per_model_dets: List[List[Det]] = []
        per_model_imgs: List[Image.Image] = []

        for fold_name, model in models:
            dets = _predict(model, img_t, score_thresh=float(args.score_thresh))
            per_model_dets.append(dets)
            per_model_imgs.append(_draw_dets(pil, dets, fold_name, float(args.score_thresh), int(args.max_boxes)))

        dets_wbf = _wbf(
            per_model=per_model_dets,
            image_size_hw=(pil.height, pil.width),
            weights=weights,
            iou_thr=float(args.wbf_iou),
            skip_box_thr=float(args.wbf_skip_box_thr),
            conf_type=str(args.wbf_conf),
        )
        img_wbf = _draw_dets(pil, dets_wbf, "ensemble_wbf", float(args.score_thresh), int(args.max_boxes))

        grid = _make_single_row_grid(per_model_imgs + [img_wbf], cell_w=cell, cell_h=cell)
        out_frame_bgr = _pil_rgb_to_cv2_bgr(grid)

        if writer is not None:
            writer.write(out_frame_bgr)

        if args.gui:
            cv2.imshow("ensemble_grid", out_frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        written += 1
        if written >= int(args.max_frames):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.gui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(
        {
            "run_dir": args.run_dir,
            "video": args.video,
            "out": args.out,
            "models": len(models),
            "frames_written": written,
            "every": int(args.every),
            "out_fps": float(args.out_fps),
            "device": str(DEVICE),
            "model": model_name,
            "num_classes": int(num_classes),
            "gui": bool(args.gui),
            "no_mp4": bool(args.no_mp4),
        }
    )


if __name__ == "__main__":
    main()
