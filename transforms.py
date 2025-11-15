from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image as PILImage

from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image as TvImage, BoundingBoxes as TvBoxes


def _infer_hw(img) -> Tuple[int, int] | Tuple[None, None]:
    if isinstance(img, torch.Tensor):
        return int(img.shape[-2]), int(img.shape[-1])
    if isinstance(img, PILImage.Image):
        return int(img.height), int(img.width)
    if isinstance(img, np.ndarray) and img.ndim >= 2:
        return int(img.shape[0]), int(img.shape[1])
    if isinstance(img, TvImage):
        return int(img.shape[-2]), int(img.shape[-1])
    return None, None


def _wrap_tv_tensors(img, target: Dict[str, Any]):
    boxes = target.get("boxes", None)
    h, w = _infer_hw(img)
    target = dict(target)
    if boxes is not None and not isinstance(boxes, TvBoxes):
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        canvas_size = (h, w) if (h is not None and w is not None) else None
        target["boxes"] = TvBoxes(boxes_t, format="XYXY", canvas_size=canvas_size)
    elif isinstance(boxes, TvBoxes) and boxes.canvas_size is None and h is not None and w is not None:
        target["boxes"] = TvBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(h, w))
    return img, target


def _ensure_chw_float(img: torch.Tensor) -> torch.Tensor:
    # Converte para float32 e reordena para CHW se vier HWC
    x = img.detach().to(torch.float32)
    if x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(2, 0, 1).contiguous()
    return x


def _unwrap_tv_tensors(img, target: Dict[str, Any]):
    # Apenas garante imagem como Tensor CHW float32; não converte boxes (TvBoxes funciona no modelo)
    if isinstance(img, (TvImage, torch.Tensor, np.ndarray)):
        img_t = torch.as_tensor(img, dtype=torch.float32) if not isinstance(img, torch.Tensor) else img
    elif isinstance(img, PILImage.Image):
        img_t = torch.from_numpy(np.array(img))
    else:
        img_t = torch.as_tensor(img, dtype=torch.float32)
    img_t = _ensure_chw_float(img_t)
    return img_t, dict(target)


def _make_multiscale_resize(short_sizes, max_size):
    # Escolhe aleatoriamente um tamanho curto e aplica Resize mantendo aspecto
    choices = []
    for s in short_sizes:
        s = int(s)
        try:
            op = T.Resize(size=(s,), max_size=max_size, antialias=True)
        except TypeError:
            try:
                op = T.Resize(size=(s,), max_size=max_size)
            except TypeError:
                op = T.Resize(size=s)
        choices.append(op)
    return T.RandomChoice(choices)


def build_sample_transform(cfg: Dict[str, Any]):
    short_sizes = cfg.get("short_sizes", [640, 672, 704, 736, 768, 800])
    max_size = int(cfg.get("max_size", 1333))
    hflip_p = float(cfg.get("hflip_p", 0.5))

    cj_cfg = cfg.get("color_jitter", {"p": 0.5, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02})
    cj_p = float(cj_cfg.get("p", 0.5))

    use_iou_crop = bool(cfg.get("use_iou_crop", False))
    min_iou = float(cfg.get("min_iou", 0.3))

    resize_op = _make_multiscale_resize(short_sizes, max_size)

    aug = [
        T.ToImage(),                         # PIL/ndarray -> Tensor (não garante CHW se já for Tensor)
        T.ConvertImageDtype(torch.float32),
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomApply([T.ColorJitter(
            brightness=cj_cfg.get("brightness", 0.2),
            contrast=cj_cfg.get("contrast", 0.2),
            saturation=cj_cfg.get("saturation", 0.2),
            hue=cj_cfg.get("hue", 0.02),
        )], p=cj_p),
        resize_op,
    ]
    if use_iou_crop:
        aug.insert(3, T.RandomIoUCrop(sampler_options=[min_iou]))

    pipeline = T.Compose(aug)

    def transform(img, target):
        img, target = _wrap_tv_tensors(img, target)
        img, target = pipeline(img, target)
        img, target = _unwrap_tv_tensors(img, target)  # garante CHW float32
        return img, target

    return transform
