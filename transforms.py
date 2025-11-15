from __future__ import annotations

from typing import Dict, Any, Tuple

import torch

from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image as TvImage, BoundingBoxes as TvBoxes


def _wrap_tv_tensors(img: torch.Tensor, target: Dict[str, Any]) -> Tuple[TvImage, Dict[str, Any]]:
    # img -> tv_tensors.Image, boxes -> tv_tensors.BoundingBoxes (XYXY)
    if not isinstance(img, torch.Tensor):
        # Se vier como PIL, ToImage no pipeline cuidará depois
        pass
    boxes = target.get("boxes", None)
    h = img.shape[-2] if isinstance(img, torch.Tensor) else target.get("height", None)
    w = img.shape[-1] if isinstance(img, torch.Tensor) else target.get("width", None)

    # O pipeline v2 aceita tanto PIL quanto Tensor; garantimos tipos dos boxes
    target = dict(target)  # evitar mutação
    if boxes is not None and not isinstance(boxes, TvBoxes):
        canvas_size = (h, w) if (h is not None and w is not None) else None
        target["boxes"] = TvBoxes(boxes, format="XYXY", canvas_size=canvas_size)
    return img, target


def _unwrap_tv_tensors(img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Converte tv_tensors de volta para tensores "puros" onde necessário
    target = dict(target)
    if isinstance(target.get("boxes", None), TvBoxes):
        target["boxes"] = target["boxes"].as_tensor()
    return img, target


def build_sample_transform(cfg: Dict[str, Any]):

    # Config com defaults sensatos para detecção
    # Pode ser controlado via YAML em training.transforms
    short_sizes = cfg.get("short_sizes", [640, 672, 704, 736, 768, 800])
    max_size = int(cfg.get("max_size", 1333))
    hflip_p = float(cfg.get("hflip_p", 0.5))

    cj_cfg = cfg.get("color_jitter", {"p": 0.5, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02})
    cj_p = float(cj_cfg.get("p", 0.5))

    use_iou_crop = bool(cfg.get("use_iou_crop", False))
    min_iou = float(cfg.get("min_iou", 0.3))  # se habilitar IoU crop

    aug = [
        # Garante tipos corretos
        T.ToImage(),                         # PIL/numpy -> Tensor [C,H,W]
        T.ConvertImageDtype(torch.float32),  # float32 em [0,1]
        # Augs
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomApply([T.ColorJitter(
            brightness=cj_cfg.get("brightness", 0.2),
            contrast=cj_cfg.get("contrast", 0.2),
            saturation=cj_cfg.get("saturation", 0.2),
            hue=cj_cfg.get("hue", 0.02),
        )], p=cj_p),
        # Multi-scale training
        T.RandomResize(sizes=short_sizes, max_size=max_size, antialias=True),
    ]

    if use_iou_crop:
        # Recorte que preserva objetos com certo IoU com a janela
        # sampler_options recebe uma lista de IoUs alvo; usamos um único valor
        aug.insert(3, T.RandomIoUCrop(sampler_options=[min_iou]))

    pipeline = T.Compose(aug)

    def transform(img, target):
        img, target = _wrap_tv_tensors(img, target)
        img, target = pipeline(img, target)
        img, target = _unwrap_tv_tensors(img, target)
        return img, target

    return transform
