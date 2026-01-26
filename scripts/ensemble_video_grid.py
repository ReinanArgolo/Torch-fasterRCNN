from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

# Allow running via: python scripts/ensemble_video_grid.py
# (so sibling package `modules/` is importable)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _pil_resample(name: str):
    # Pillow compatibility across versions
    try:
        resampling = Image.Resampling  # type: ignore[attr-defined]
    except Exception:
        resampling = Image

    mapping = {
        "nearest": getattr(resampling, "NEAREST", 0),
        "bilinear": getattr(resampling, "BILINEAR", 2),
        "bicubic": getattr(resampling, "BICUBIC", 3),
        "lanczos": getattr(resampling, "LANCZOS", 1),
    }
    return mapping.get(str(name).lower(), mapping["bicubic"])


def _contain_resize(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    if max_w <= 0 or max_h <= 0:
        return img
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1e9)
    if scale >= 0.999 and w <= max_w and h <= max_h:
        return img

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resample = _pil_resample("lanczos" if scale < 1.0 else "bicubic")
    return img.resize((nw, nh), resample=resample)


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


def _draw_dets(pil: Image.Image, dets: Sequence[Det], score_thresh: float, max_boxes: int) -> Image.Image:
    img = pil.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

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


def _fit_cell(img: Image.Image, cell_w: int, cell_h: int, title: str, subtitle: Optional[str] = None) -> Image.Image:
    canvas = Image.new("RGB", (cell_w, cell_h), (20, 20, 20))

    header_h = max(26, int(cell_h * 0.06))
    inner_w = cell_w
    inner_h = max(1, cell_h - header_h)

    fitted = _contain_resize(img, inner_w, inner_h)
    x = (cell_w - fitted.width) // 2
    y = header_h + (inner_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, cell_w, header_h], fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 6
    draw.text((pad, 3), str(title), fill=(255, 255, 255), font=font)
    if subtitle:
        draw.text((pad, max(3, header_h - 14)), str(subtitle), fill=(200, 200, 200), font=font)
    return canvas


def _make_single_row_grid(
    cells: List[Image.Image],
    cell_w: int,
    cell_h: int,
    titles: Sequence[str],
    subtitles: Optional[Sequence[Optional[str]]] = None,
) -> Image.Image:
    n_cols = len(cells)
    grid = Image.new("RGB", (n_cols * cell_w, cell_h), (0, 0, 0))
    for c, img in enumerate(cells):
        st = None
        if subtitles is not None and c < len(subtitles):
            st = subtitles[c]
        t = titles[c] if c < len(titles) else f"col_{c}"
        cell = _fit_cell(img, cell_w, cell_h, title=t, subtitle=st)
        grid.paste(cell, (c * cell_w, 0))
    return grid


def _make_grid(
    cells: List[Image.Image],
    cell_w: int,
    cell_h: int,
    grid_cols: int,
    grid_rows: int,
    titles: Sequence[str],
    subtitles: Optional[Sequence[Optional[str]]] = None,
) -> Image.Image:
    cols = int(grid_cols)
    rows = int(grid_rows)
    if cols <= 0 or rows <= 0:
        raise ValueError(f"grid_cols/grid_rows must be > 0, got cols={cols} rows={rows}")

    n_slots = cols * rows
    if len(cells) > n_slots:
        raise ValueError(f"Too many cells ({len(cells)}) for grid {rows}x{cols} ({n_slots} slots)")

    # Pad with empty images so layout is stable.
    if len(cells) < n_slots:
        empty = Image.new("RGB", (1, 1), (20, 20, 20))
        cells = list(cells) + [empty] * (n_slots - len(cells))

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    for i, img in enumerate(cells):
        r = i // cols
        c = i % cols

        t = titles[i] if i < len(titles) else f"cell_{i}"
        st = None
        if subtitles is not None and i < len(subtitles):
            st = subtitles[i]

        cell = _fit_cell(img, cell_w, cell_h, title=t, subtitle=st)
        grid.paste(cell, (c * cell_w, r * cell_h))
    return grid


def _resize_bgr_to_fit(frame_bgr, max_w: Optional[int], max_h: Optional[int]):
    if frame_bgr is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return frame_bgr
    if not max_w or not max_h or int(max_w) <= 0 or int(max_h) <= 0:
        return frame_bgr

    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return frame_bgr
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


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


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def main():
    p = argparse.ArgumentParser(
        description=(
            "Gera um MP4 com grid por frame: cada fold + ensemble WBF. "
            "Ideal para rodar em servidor via SSH e assistir localmente (baixando o MP4)."
        )
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Opcional: YAML. Lê model.* e também (se presentes) video_grid.* "
            "(run_dir/video/out/gui/no_mp4/thresholds/WBF/grid)."
        ),
    )
    p.add_argument("--run-dir", required=False, default=None, help="Pasta do run (best.pth ou fold_*/best.pth)")
    p.add_argument("--video", required=False, default=None, help="Caminho para o vídeo de entrada (mp4/avi/etc)")
    p.add_argument("--out", default=None, help="Caminho para o mp4 de saída (obrigatório a menos que --no-mp4)")

    p.add_argument("--gui", action="store_true", help="Mostra janela com o grid (para rodar localmente com desktop)")
    p.add_argument("--no-mp4", action="store_true", help="Não gera MP4; útil quando usar --gui")

    p.add_argument("--model", default=None, help="Sobrescreve model.name do config.yaml")
    p.add_argument("--num-classes", type=int, default=None, help="Sobrescreve model.num_classes do config.yaml")

    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--max-boxes", type=int, default=None)

    p.add_argument("--every", type=int, default=None, help="Processa 1 frame a cada N frames (stride)")
    p.add_argument("--max-frames", type=int, default=None, help="Limite de frames processados")
    p.add_argument("--out-fps", type=float, default=None)

    # WBF
    p.add_argument("--wbf-iou", type=float, default=None)
    p.add_argument("--wbf-skip-box-thr", type=float, default=None)
    p.add_argument("--wbf-conf", choices=["avg", "max"], default=None)
    p.add_argument("--wbf-weights", type=float, nargs="*", default=None)

    # Grid
    p.add_argument("--cell", type=int, default=None)
    p.add_argument("--grid-cols", type=int, default=None, help="Número de colunas do grid")
    p.add_argument("--grid-rows", type=int, default=None, help="Número de linhas do grid")
    p.add_argument("--out-w", type=int, default=None, help="Largura (px) do canvas final/MP4 (ex: 1920)")
    p.add_argument("--out-h", type=int, default=None, help="Altura (px) do canvas final/MP4 (ex: 1080)")
    p.add_argument("--gui-max-w", type=int, default=None, help="Largura máxima (px) para o grid no --gui (só display)")
    p.add_argument("--gui-max-h", type=int, default=None, help="Altura máxima (px) para o grid no --gui (só display)")

    args = p.parse_args()

    cfg = _load_yaml_config(args.config)

    # Optional script-specific section in YAML
    vcfg = _cfg_get(cfg, ["video_grid"], {})
    if vcfg is None:
        vcfg = {}
    if not isinstance(vcfg, dict):
        raise SystemExit("config.yaml: video_grid deve ser um dict")

    # Allow run/video/out/gui/no_mp4 to come from YAML when omitted.
    args.run_dir = _coalesce(args.run_dir, _cfg_get(vcfg, ["run_dir"], None))
    args.video = _coalesce(args.video, _cfg_get(vcfg, ["video"], None))
    args.out = _coalesce(args.out, _cfg_get(vcfg, ["out"], None))

    if not args.gui:
        args.gui = bool(_cfg_get(vcfg, ["gui"], False))
    if not args.no_mp4:
        args.no_mp4 = bool(_cfg_get(vcfg, ["no_mp4"], False))

    # Numeric defaults (CLI overrides YAML; YAML overrides hardcoded default)
    args.score_thresh = float(_coalesce(args.score_thresh, _cfg_get(vcfg, ["score_thresh"], 0.4)))
    args.max_boxes = int(_coalesce(args.max_boxes, _cfg_get(vcfg, ["max_boxes"], 50)))
    args.every = int(_coalesce(args.every, _cfg_get(vcfg, ["every"], 5)))
    args.max_frames = int(_coalesce(args.max_frames, _cfg_get(vcfg, ["max_frames"], 300)))
    args.out_fps = float(_coalesce(args.out_fps, _cfg_get(vcfg, ["out_fps"], 10.0)))

    args.wbf_iou = float(_coalesce(args.wbf_iou, _cfg_get(vcfg, ["wbf"], {}).get("iou", 0.55)))
    args.wbf_skip_box_thr = float(
        _coalesce(args.wbf_skip_box_thr, _cfg_get(vcfg, ["wbf"], {}).get("skip_box_thr", 0.0))
    )
    args.wbf_conf = str(_coalesce(args.wbf_conf, _cfg_get(vcfg, ["wbf"], {}).get("conf_type", "avg")))
    if args.wbf_weights is None:
        w = _cfg_get(vcfg, ["wbf"], {}).get("weights", None)
        if w is not None:
            if not isinstance(w, list):
                raise SystemExit("config.yaml: video_grid.wbf.weights deve ser uma lista de floats")
            args.wbf_weights = [float(x) for x in w]

    args.cell = int(_coalesce(args.cell, _cfg_get(vcfg, ["grid"], {}).get("cell", 540)))

    grid_cfg = _cfg_get(vcfg, ["grid"], {})
    if grid_cfg is None:
        grid_cfg = {}
    if not isinstance(grid_cfg, dict):
        raise SystemExit("config.yaml: video_grid.grid deve ser um dict")

    args.grid_cols = int(_coalesce(args.grid_cols, grid_cfg.get("cols", 0)))
    args.grid_rows = int(_coalesce(args.grid_rows, grid_cfg.get("rows", 0)))

    args.out_w = _coalesce(args.out_w, _cfg_get(vcfg, ["out_w"], None), _cfg_get(vcfg, ["output"], {}).get("w", None))
    args.out_h = _coalesce(args.out_h, _cfg_get(vcfg, ["out_h"], None), _cfg_get(vcfg, ["output"], {}).get("h", None))
    if args.out_w is not None:
        args.out_w = int(args.out_w)
    if args.out_h is not None:
        args.out_h = int(args.out_h)

    args.gui_max_w = int(_coalesce(args.gui_max_w, _cfg_get(vcfg, ["gui_max_w"], 1600)))
    args.gui_max_h = int(_coalesce(args.gui_max_h, _cfg_get(vcfg, ["gui_max_h"], 900)))

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
    if not args.run_dir:
        raise SystemExit("Você precisa passar --run-dir (ou definir video_grid.run_dir no YAML)")
    if not args.video:
        raise SystemExit("Você precisa passar --video (ou definir video_grid.video no YAML)")
    if not args.no_mp4 and not args.out:
        raise SystemExit("Você precisa passar --out (ou então usar --no-mp4)")

    if args.gui:
        # Basic headless guard: imshow will fail without a display.
        if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
            raise SystemExit("--gui requer um ambiente com display (rodar localmente). No servidor headless, gere MP4.")

        try:
            cv2.namedWindow("ensemble_grid", cv2.WINDOW_NORMAL)
        except Exception:
            pass

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

    subtitles: List[Optional[str]] = []
    if weights is None:
        for _ in models:
            subtitles.append(None)
    else:
        for i in range(len(models)):
            subtitles.append(f"w={float(weights[i]):.2f}")
    subtitles.append(f"iou={float(args.wbf_iou):.2f} conf={str(args.wbf_conf)}")

    # Determine output video size + cell size
    cell = int(args.cell)
    n_views = len(col_names)

    if int(args.grid_cols) > 0 and int(args.grid_rows) > 0:
        grid_cols = int(args.grid_cols)
        grid_rows = int(args.grid_rows)
    elif int(args.grid_cols) > 0:
        grid_cols = int(args.grid_cols)
        grid_rows = (n_views + grid_cols - 1) // grid_cols
    else:
        # Backward compatible default: single row
        grid_cols = n_views
        grid_rows = 1

    if grid_cols <= 0 or grid_rows <= 0:
        raise SystemExit("grid inválido: defina video_grid.grid.cols/rows (ou use --grid-cols/--grid-rows)")

    if args.out_w is not None and args.out_h is not None and int(args.out_w) > 0 and int(args.out_h) > 0:
        out_w = int(args.out_w)
        out_h = int(args.out_h)
        cell_w = max(1, out_w // grid_cols)
        cell_h = max(1, out_h // grid_rows)
    else:
        cell_w = cell
        cell_h = cell
        out_w = grid_cols * cell_w
        out_h = grid_rows * cell_h

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
            per_model_imgs.append(_draw_dets(pil, dets, float(args.score_thresh), int(args.max_boxes)))

        dets_wbf = _wbf(
            per_model=per_model_dets,
            image_size_hw=(pil.height, pil.width),
            weights=weights,
            iou_thr=float(args.wbf_iou),
            skip_box_thr=float(args.wbf_skip_box_thr),
            conf_type=str(args.wbf_conf),
        )
        img_wbf = _draw_dets(pil, dets_wbf, float(args.score_thresh), int(args.max_boxes))

        grid = _make_grid(
            per_model_imgs + [img_wbf],
            cell_w=cell_w,
            cell_h=cell_h,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            titles=col_names,
            subtitles=subtitles,
        )

        if grid.size != (out_w, out_h):
            canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
            gx, gy = grid.size
            ox = max(0, (out_w - gx) // 2)
            oy = max(0, (out_h - gy) // 2)
            canvas.paste(grid, (ox, oy))
            grid = canvas
        out_frame_bgr = _pil_rgb_to_cv2_bgr(grid)

        if writer is not None:
            writer.write(out_frame_bgr)

        if args.gui:
            show_bgr = _resize_bgr_to_fit(out_frame_bgr, args.gui_max_w, args.gui_max_h)
            cv2.imshow("ensemble_grid", show_bgr)
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
            "gui_max_w": int(args.gui_max_w),
            "gui_max_h": int(args.gui_max_h),
            "out_w": int(out_w),
            "out_h": int(out_h),
            "grid_cols": int(grid_cols),
            "grid_rows": int(grid_rows),
        }
    )


if __name__ == "__main__":
    main()
