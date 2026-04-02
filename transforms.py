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
    # Boxes -> TvBoxes (XYXY) com canvas_size
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


def _ensure_chw3(img: torch.Tensor) -> torch.Tensor:
    # Garante [C,H,W] e 3 canais
    x = img.detach().to(torch.float32)
    if x.ndim == 2:
        # [H,W] -> [1,H,W]
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        C, H, W = x.shape[0], x.shape[1], x.shape[2]
        # Se primeira dim não é canal e última parece canal, permuta HWC->CHW
        if C not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(2, 0, 1).contiguous()
    # Agora ajusta canais para 3
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] == 4:
        x = x[:3, ...]
    return x


def _unwrap_tv_tensors(img, target: Dict[str, Any]):
    # Imagem -> Tensor CHW float32 3ch; Boxes -> Tensor Nx4 float32
    if isinstance(img, torch.Tensor):
        img_t = img
    elif isinstance(img, PILImage.Image):
        img_t = torch.from_numpy(np.array(img))
    else:
        img_t = torch.as_tensor(img)
    img_t = _ensure_chw3(img_t)

    tgt = dict(target)
    b = tgt.get("boxes", None)
    if isinstance(b, TvBoxes):
        tgt["boxes"] = b.detach().to(torch.float32).clone()
    elif torch.is_tensor(b):
        tgt["boxes"] = b.detach().to(torch.float32).clone()
    return img_t, tgt


class RandomGaussianNoise:
    """Adds light Gaussian noise to the image only (keeps target intact)."""

    def __init__(self, p: float = 0.0, std: float = 0.01):
        self.p = float(p)
        self.std = float(std)

    def __call__(self, img, target: Dict[str, Any]):
        if self.p <= 0.0 or self.std <= 0.0:
            return img, target
        if torch.rand(()) >= self.p:
            return img, target

        noise = torch.randn_like(img) * self.std
        img_noisy = img + noise
        # Clamp to valid range; torchvision v2 works with float images in [0,1].
        img_noisy = img_noisy.clamp(0.0, 1.0)
        return img_noisy, target


def _make_multiscale_resize(short_sizes, max_size):
    # RandomChoice de Resize para compatibilidade de versões
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

    geom_cfg = cfg.get("geom", cfg.get("geometry", {})) or {}
    rot_deg = float(geom_cfg.get("rotation_deg", 0.0))
    rot_p = float(geom_cfg.get("rotation_p", 0.0))

    persp_cfg = geom_cfg.get("perspective", {}) or {}
    persp_p = float(persp_cfg.get("p", 0.0))
    persp_dist = float(persp_cfg.get("distortion_scale", 0.05))

    zoom_cfg = geom_cfg.get("zoom_out", {}) or {}
    zoom_p = float(zoom_cfg.get("p", 0.0))
    zoom_side_range = zoom_cfg.get("side_range", [1.0, 1.4])
    if not isinstance(zoom_side_range, (list, tuple)) or len(zoom_side_range) != 2:
        zoom_side_range = [1.0, 1.4]
    zoom_fill = zoom_cfg.get("fill", 0)

    photo_cfg = cfg.get("photometric", {}) or {}
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

    sanitize_cfg = cfg.get("sanitize_boxes", {}) or {}
    sanitize_enabled = bool(sanitize_cfg.get("enabled", True))
    sanitize_min_size = float(sanitize_cfg.get("min_size", 1.0))

    resize_op = _make_multiscale_resize(short_sizes, max_size)

    aug = [
        T.ToImage(),                        # -> tv_tensors.Image (C,H,W) na maioria dos casos
        T.ConvertImageDtype(torch.float32), # -> float32 [0,1]
        T.RandomHorizontalFlip(p=hflip_p),
    ]
    if use_iou_crop:
        aug.insert(3, T.RandomIoUCrop(sampler_options=[min_iou]))

    if zoom_p > 0:
        try:
            aug.append(T.RandomApply([T.RandomZoomOut(side_range=tuple(zoom_side_range), fill=zoom_fill)], p=zoom_p))
        except AttributeError:
            pass

    if rot_deg > 0 and rot_p > 0:
        aug.append(T.RandomApply([T.RandomRotation(degrees=(-rot_deg, rot_deg))], p=rot_p))

    if persp_p > 0:
        aug.append(T.RandomApply([T.RandomPerspective(distortion_scale=persp_dist, p=1.0)], p=persp_p))

    aug.append(T.RandomApply([T.ColorJitter(
        brightness=cj_cfg.get("brightness", 0.2),
        contrast=cj_cfg.get("contrast", 0.2),
        saturation=cj_cfg.get("saturation", 0.2),
        hue=cj_cfg.get("hue", 0.02),
    )], p=cj_p))

    if pd_p > 0:
        try:
            aug.append(T.RandomApply([T.RandomPhotometricDistort()], p=pd_p))
        except AttributeError:
            pass

    if autocontrast_p > 0:
        aug.append(T.RandomApply([T.RandomAutocontrast()], p=autocontrast_p))

    if equalize_p > 0:
        aug.append(T.RandomApply([T.RandomEqualize()], p=equalize_p))

    if blur_p > 0:
        aug.append(T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)], p=blur_p))

    if noise_p > 0 and noise_std > 0:
        aug.append(RandomGaussianNoise(p=noise_p, std=noise_std))

    aug.append(resize_op)

    if sanitize_enabled:
        try:
            aug.append(T.SanitizeBoundingBoxes(min_size=sanitize_min_size))
        except TypeError:
            aug.append(T.SanitizeBoundingBoxes())

    pipeline = T.Compose(aug)

    def transform(img, target):
        img, target = _wrap_tv_tensors(img, target)
        img, target = pipeline(img, target)
        img, target = _unwrap_tv_tensors(img, target)  # garante CHW float32 3ch e boxes Tensor
        return img, target

    return transform


class ResizeShortSide:
    """Deterministic resize keeping aspect ratio.

    Primarily used for inference utilities.
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = int(min_size)
        self.max_size = int(max_size)

    def __call__(self, img, target: Dict[str, Any]):
        # Use torchvision v2 Resize when possible; keep it simple and deterministic.
        try:
            op = T.Resize(size=(self.min_size,), max_size=self.max_size, antialias=True)
        except TypeError:
            try:
                op = T.Resize(size=(self.min_size,), max_size=self.max_size)
            except TypeError:
                op = T.Resize(size=self.min_size)

        # For inference, target may not have boxes; still support if present.
        img, target = _wrap_tv_tensors(img, target)
        img, target = op(img, target)

        # Prefer returning PIL if input was PIL, to keep older utilities working.
        if isinstance(img, TvImage):
            # Convert tv_tensors.Image -> Tensor, then caller can decide.
            img = img.as_subclass(torch.Tensor)
        return img, target


def build_val_sample_transform(cfg: Dict[str, Any]):
    """Deterministic validation transform: resize only (no flip/jitter/crop)."""
    resize_cfg = cfg.get("resize", {}) if isinstance(cfg, dict) else {}
    min_size = int(resize_cfg.get("min_size", 800))
    max_size = int(resize_cfg.get("max_size", cfg.get("max_size", 1333) if isinstance(cfg, dict) else 1333))

    try:
        resize_op = T.Resize(size=(min_size,), max_size=max_size, antialias=True)
    except TypeError:
        try:
            resize_op = T.Resize(size=(min_size,), max_size=max_size)
        except TypeError:
            resize_op = T.Resize(size=min_size)

    pipeline = T.Compose(
        [
            T.ToImage(),
            T.ConvertImageDtype(torch.float32),
            resize_op,
        ]
    )

    def transform(img, target):
        img, target = _wrap_tv_tensors(img, target)
        img, target = pipeline(img, target)
        img, target = _unwrap_tv_tensors(img, target)
        return img, target

    return transform
