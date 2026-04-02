#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
import torch
from PIL import Image, ImageDraw, ImageFont

# Permite executar via `python scripts/inspect_augmentations.py` sem precisar exportar PYTHONPATH
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import COCODetectionDataset
from transforms import build_sample_transform


def _rp(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _to_pil(img: torch.Tensor) -> Image.Image:
    # img: CHW float32 0..1 (expected)
    x = img.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32:
        x = x.float()
    x = x.clamp(0.0, 1.0)
    # manual conversion to avoid torchvision dependency differences
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c > 3:
        x = x[:3]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _draw_boxes(
    pil_img: Image.Image,
    boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    color: str = "red",
    width: int = 3,
    title: Optional[str] = None,
) -> Image.Image:
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if title:
        draw.rectangle([0, 0, out.width, 18], fill="black")
        draw.text((4, 2), title, fill="white", font=font)

    if boxes is None:
        return out

    b = boxes.detach().cpu()
    if b.numel() == 0:
        return out

    b = b.reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(b.tolist()):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels is not None and labels.numel() > i:
            lab = int(labels.detach().cpu().reshape(-1)[i].item())
            draw.text((x1 + 2, y1 + 2), str(lab), fill="yellow", font=font)

    return out


def _concat_h(images: List[Image.Image], pad: int = 6, bg=(30, 30, 30)) -> Image.Image:
    if not images:
        raise ValueError("No images to concat")
    heights = [im.height for im in images]
    max_h = max(heights)
    widths = [im.width for im in images]
    total_w = sum(widths) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, max_h), color=bg)
    x = 0
    for im in images:
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad
    return canvas


def describe_augmentations(cfg: Dict[str, Any]) -> str:
    tcfg = cfg.get("training", {}).get("transforms", {}) if isinstance(cfg, dict) else {}

    short_sizes = tcfg.get("short_sizes", [640, 672, 704, 736, 768, 800])
    max_size = tcfg.get("max_size", 1333)
    hflip_p = tcfg.get("hflip_p", 0.5)

    cj = tcfg.get(
        "color_jitter",
        {"p": 0.5, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
    )
    use_iou_crop = bool(tcfg.get("use_iou_crop", False))
    min_iou = float(tcfg.get("min_iou", 0.3))

    geom_cfg = tcfg.get("geom", tcfg.get("geometry", {})) or {}
    rot_deg = float(geom_cfg.get("rotation_deg", 0.0))
    rot_p = float(geom_cfg.get("rotation_p", 0.0))

    persp_cfg = geom_cfg.get("perspective", {}) or {}
    persp_p = float(persp_cfg.get("p", 0.0))
    persp_dist = float(persp_cfg.get("distortion_scale", 0.05))

    zoom_cfg = geom_cfg.get("zoom_out", {}) or {}
    zoom_p = float(zoom_cfg.get("p", 0.0))
    zoom_side_range = zoom_cfg.get("side_range", [1.0, 1.4])

    photo_cfg = tcfg.get("photometric", {}) or {}
    pd_cfg = photo_cfg.get("photometric_distort", {}) or {}
    pd_p = float(pd_cfg.get("p", 0.0))

    autocontrast_p = float(photo_cfg.get("autocontrast_p", 0.0))
    equalize_p = float(photo_cfg.get("equalize_p", 0.0))

    blur_cfg = photo_cfg.get("gaussian_blur", {}) or {}
    blur_p = float(blur_cfg.get("p", 0.0))
    blur_kernel = blur_cfg.get("kernel_size", 3)
    blur_sigma = blur_cfg.get("sigma", (0.1, 2.0))

    noise_cfg = photo_cfg.get("gaussian_noise", {}) or {}
    noise_p = float(noise_cfg.get("p", 0.0))
    noise_std = float(noise_cfg.get("std", 0.01))

    sanitize_cfg = tcfg.get("sanitize_boxes", {}) or {}
    sanitize_enabled = bool(sanitize_cfg.get("enabled", True))
    sanitize_min_size = float(sanitize_cfg.get("min_size", 1.0))

    lines = []
    lines.append("Data Augmentation (train/sample_transforms):")
    lines.append("- ToImage()")
    lines.append("- ConvertImageDtype(float32) -> escala [0,1]")
    if use_iou_crop:
        lines.append(f"- RandomIoUCrop(sampler_options=[{min_iou}])")
    lines.append(f"- RandomHorizontalFlip(p={hflip_p})")

    if zoom_p > 0:
        lines.append(f"- RandomZoomOut(p={zoom_p}, side_range={zoom_side_range})")

    if rot_deg > 0 and rot_p > 0:
        lines.append(f"- RandomRotation(degrees=±{rot_deg}, p={rot_p})")

    if persp_p > 0:
        lines.append(f"- RandomPerspective(distortion_scale={persp_dist}, p={persp_p})")

    lines.append(
        "- RandomApply(ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h}), p={p})".format(
            b=cj.get("brightness", 0.2),
            c=cj.get("contrast", 0.2),
            s=cj.get("saturation", 0.2),
            h=cj.get("hue", 0.02),
            p=cj.get("p", 0.5),
        )
    )

    if pd_p > 0:
        lines.append(f"- RandomPhotometricDistort(p={pd_p})")

    if autocontrast_p > 0:
        lines.append(f"- RandomAutocontrast(p={autocontrast_p})")

    if equalize_p > 0:
        lines.append(f"- RandomEqualize(p={equalize_p})")

    if blur_p > 0:
        lines.append(f"- GaussianBlur(kernel_size={blur_kernel}, sigma={blur_sigma}, p={blur_p})")

    if noise_p > 0 and noise_std > 0:
        lines.append(f"- GaussianNoise(std={noise_std}, p={noise_p})")

    lines.append(f"- RandomChoice(Resize short_side in {list(short_sizes)}, max_size={max_size})")

    if sanitize_enabled:
        lines.append(f"- SanitizeBoundingBoxes(min_size={sanitize_min_size})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Visualiza o Data Augmentation aplicado no dataset")
    ap.add_argument("--config", required=True, help="YAML de config (ex.: config_new_whales_rcnn.yaml)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Qual split usar")
    ap.add_argument("--k", type=int, default=6, help="Quantas imagens (indices) amostrar")
    ap.add_argument("--repeats", type=int, default=3, help="Quantas variações augmentadas por imagem")
    ap.add_argument("--seed", type=int, default=42, help="Seed para escolher indices")
    ap.add_argument("--out-dir", default=None, help="Diretório de saída (default: outputs/aug_inspect/<run_stamp>)")
    ap.add_argument("--indices", default=None, help="Lista de indices separados por vírgula (override de --k)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    project_root = cfg.get("project_root", os.getcwd())
    data_cfg = cfg.get("data", {})

    if args.split == "train":
        images_dir = _rp(project_root, data_cfg["images"]["train_dir"])
        ann_path = _rp(project_root, data_cfg["annotations"]["train_json"])
        sample_transform = build_sample_transform(cfg.get("training", {}).get("transforms", {}))
    else:
        images_dir = _rp(project_root, data_cfg["images"].get("val_dir", ""))
        ann_path = _rp(project_root, data_cfg["annotations"].get("val_json", ""))
        # validação no treino está configurada para sem resize/aug no Dataset
        sample_transform = None

    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(project_root, "outputs", "aug_inspect", run_stamp)
    os.makedirs(out_dir, exist_ok=True)

    print(describe_augmentations(cfg))
    print(f"\n[IO] images_dir={images_dir}")
    print(f"[IO] ann_path={ann_path}")
    print(f"[IO] out_dir={out_dir}\n")

    # Dataset raw (sem sample_transforms) e dataset aug (com sample_transforms)
    ds_raw = COCODetectionDataset(images_dir, ann_path, sample_transforms=None)
    ds_aug = COCODetectionDataset(images_dir, ann_path, sample_transforms=sample_transform)

    n = len(ds_raw)
    if n == 0:
        raise SystemExit("Dataset vazio (0 imagens resolvidas). Verifique paths do images_dir e annotation_json.")

    # Seleção de indices
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        rng = random.Random(args.seed)
        k = min(args.k, n)
        indices = rng.sample(range(n), k)

    # Sanity: garantir que os image_ids batem no idx
    mismatches = 0
    for idx in indices:
        raw_id = int(ds_raw.image_ids[idx])
        aug_id = int(ds_aug.image_ids[idx])
        if raw_id != aug_id:
            mismatches += 1
    if mismatches:
        print(f"[WARN] {mismatches} indices com image_id diferente entre ds_raw e ds_aug. Ainda assim vou salvar outputs.")

    for idx in indices:
        raw_img, raw_tgt = ds_raw[idx]
        raw_pil = _to_pil(raw_img)
        raw_boxes = raw_tgt.get("boxes")
        raw_labels = raw_tgt.get("labels")
        raw_drawn = _draw_boxes(raw_pil, raw_boxes, raw_labels, color="lime", title=f"raw idx={idx}")

        aug_variants: List[Image.Image] = []
        for r in range(args.repeats):
            aug_img, aug_tgt = ds_aug[idx]
            aug_pil = _to_pil(aug_img)
            aug_boxes = aug_tgt.get("boxes")
            aug_labels = aug_tgt.get("labels")
            aug_drawn = _draw_boxes(aug_pil, aug_boxes, aug_labels, color="red", title=f"aug r={r+1}")
            aug_variants.append(aug_drawn)

        grid = _concat_h([raw_drawn] + aug_variants)
        out_path = os.path.join(out_dir, f"idx_{idx:05d}.jpg")
        grid.save(out_path, quality=95)
        print(f"[OK] saved {out_path}")

    print("\nPronto. Abra os .jpg gerados para ver o antes/depois + variações.")


if __name__ == "__main__":
    main()
