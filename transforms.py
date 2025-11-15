from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
from PIL import Image


def _resize_pil(img: Image.Image, new_w: int, new_h: int) -> Image.Image:
    return img.resize((new_w, new_h), resample=Image.BILINEAR)


class ResizeShortSide:
    """
    Resize keeping aspect ratio so that the shorter side == min_size, and the longer side <= max_size.
    Scales bounding boxes and area accordingly.
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = int(min_size)
        self.max_size = int(max_size)

    def __call__(self, img: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        w, h = img.size
        if w <= 0 or h <= 0:
            return img, target
        short, long = (h, w) if h < w else (w, h)
        scale = self.min_size / float(short)
        if round(scale * long) > self.max_size:
            scale = self.max_size / float(long)

        if abs(scale - 1.0) < 1e-6:
            return img, target

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img2 = _resize_pil(img, new_w, new_h)

        t2 = dict(target)
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes = boxes * scale
            # clip
            boxes[:, 0::2].clamp_(min=0, max=new_w - 1)
            boxes[:, 1::2].clamp_(min=0, max=new_h - 1)

            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > 0) & (h > 0)

            t2["boxes"] = boxes[keep]
            if "labels" in t2:
                t2["labels"] = t2["labels"][keep]
            if "iscrowd" in t2:
                t2["iscrowd"] = t2["iscrowd"][keep]
            if "area" in t2:
                t2["area"] = t2["area"][keep] * (scale**2)
        elif "area" in target:
            # no boxes, but has area
            t2["area"] = target["area"] * (scale**2)

        return img2, t2


def build_sample_transform(cfg: Optional[dict]) -> Optional[ResizeShortSide]:
    """Build a paired sample transform (image+target). Currently supports ResizeShortSide.
    Expected cfg example:
    {
      "resize": {"min_size": 800, "max_size": 1333}
    }
    Returns a callable (img, target) -> (img, target) or None.
    """
    if not cfg:
        return None
    resize = cfg.get("resize") if isinstance(cfg, dict) else None
    if resize:
        min_size = int(resize.get("min_size", 800))
        max_size = int(resize.get("max_size", 1333))
        return ResizeShortSide(min_size=min_size, max_size=max_size)
    return None
