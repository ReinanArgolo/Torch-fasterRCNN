from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


class COCODetectionDataset(Dataset):
    """
    Minimal COCO-style dataset for object detection compatible with torchvision detection models.

    Returns (image_tensor, target_dict) where target has keys:
    - boxes (FloatTensor[N, 4]) in [x1, y1, x2, y2]
    - labels (Int64Tensor[N]) starting at 1
    - image_id (Int64Tensor[1])
    - area (Tensor[N])
    - iscrowd (UInt8Tensor[N])
    """

    def __init__(
        self,
        images_dir: str,
        annotation_json: str,
        transforms: Any | None = None,
        sample_transforms: Optional[Callable[[Image.Image, Dict[str, torch.Tensor]], Tuple[Image.Image, Dict[str, torch.Tensor]]]] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.transforms = transforms
        self.sample_transforms = sample_transforms
        self.coco = COCO(annotation_json)
        all_img_ids = list(sorted(self.coco.getImgIds()))

        # Index files recursively for robust filename resolution (supports nested folders)
        self._file_index_rel_lower: Dict[str, str] = {}
        self._file_index_basename_lower: Dict[str, List[str]] = {}
        if os.path.isdir(self.images_dir):
            for root, _dirs, files in os.walk(self.images_dir):
                for f in files:
                    full = os.path.join(root, f)
                    if not os.path.isfile(full):
                        continue
                    rel = os.path.relpath(full, self.images_dir)
                    rel_norm = rel.replace(os.sep, "/")
                    self._file_index_rel_lower[rel_norm.lower()] = full
                    rel_key = self._normalize_rel_key(rel_norm)
                    if rel_key:
                        self._file_index_rel_lower[rel_key.lower()] = full
                    base_lower = os.path.basename(f).lower()
                    self._file_index_basename_lower.setdefault(base_lower, []).append(full)

        self._resolved_paths: Dict[int, str] = {}
        resolved_ids: List[int] = []

        missing: List[int] = []
        ambiguous: List[int] = []

        for img_id in all_img_ids:
            info = self.coco.loadImgs([img_id])[0]
            fname = info["file_name"]
            path = self._resolve_filename(fname)
            if path is not None:
                self._resolved_paths[img_id] = path
                resolved_ids.append(img_id)
            else:
                # mark missing/ambiguous for diagnostics
                if isinstance(getattr(self, "_last_resolve_status", None), str) and self._last_resolve_status == "ambiguous":
                    ambiguous.append(img_id)
                else:
                    missing.append(img_id)

        # Diagnostics summary (helps catch dataset path/JSON mismatches)
        total = len(all_img_ids)
        if total > 0:
            msg = (
                f"[Dataset] {os.path.basename(annotation_json)} | images_dir={self.images_dir} | "
                f"total={total} resolved={len(resolved_ids)} missing={len(missing)} ambiguous={len(ambiguous)}"
            )
            print(msg)

        self.image_ids = resolved_ids

    def _resolve_filename(self, file_name: str) -> Optional[str]:
        # Reset last status
        self._last_resolve_status = "missing"

        # 1) Try direct path joining images_dir + file_name (supports subfolders from COCO JSON).
        rel_norm = str(file_name).replace("\\", "/").strip()
        rel_candidates = [rel_norm]
        rel_no_leading_slash = rel_norm.lstrip("/")
        if rel_no_leading_slash and rel_no_leading_slash != rel_norm:
            rel_candidates.append(rel_no_leading_slash)
        rel_no_dot_slash = rel_no_leading_slash
        while rel_no_dot_slash.startswith("./"):
            rel_no_dot_slash = rel_no_dot_slash[2:]
        if rel_no_dot_slash and rel_no_dot_slash not in rel_candidates:
            rel_candidates.append(rel_no_dot_slash)

        images_root_abs = os.path.abspath(self.images_dir)
        for rel_candidate in rel_candidates:
            joined = os.path.normpath(os.path.join(self.images_dir, rel_candidate))
            joined_abs = os.path.abspath(joined)
            if os.path.commonpath([images_root_abs, joined_abs]) != images_root_abs:
                continue
            if os.path.isfile(joined_abs):
                self._last_resolve_status = "ok"
                return joined_abs

        # 2) Try indexed relative path (case-insensitive)
        rel_key = self._normalize_rel_key(rel_norm)
        full = self._file_index_rel_lower.get(rel_key.lower()) if rel_key else None
        if full is not None:
            self._last_resolve_status = "ok"
            return full

        # 3) Try basename exact (case-insensitive) anywhere under images_dir
        rel_norm = rel_no_dot_slash if rel_no_dot_slash else rel_norm
        base_name = os.path.basename(rel_norm)
        candidates = self._file_index_basename_lower.get(base_name.lower(), [])
        if len(candidates) == 1:
            self._last_resolve_status = "ok"
            return candidates[0]
        if len(candidates) > 1:
            # Never guess: ambiguous basename across subfolders.
            self._last_resolve_status = "ambiguous"
            return None

        # 4) No prefix matching here on purpose: variants like quadro_0000_*.jpg should be explicit in the COCO JSON.
        return None

    @staticmethod
    def _normalize_rel_key(file_name: str) -> str:
        rel = str(file_name).replace("\\", "/").strip()
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if not rel:
            return ""
        norm = os.path.normpath(rel).replace("\\", "/")
        return "" if norm in ("", ".") else norm

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        # Use resolved path built at init
        path = self._resolved_paths.get(img_id)
        if path is None:
            # This should not happen if init filtered correctly; raise clear error
            raise FileNotFoundError(f"Image file for id {img_id} not found under {self.images_dir}")

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image '{path}' (image_id={img_id}): {e}") from e

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for a in anns:
            x, y, w, h = a["bbox"]
            # skip invalid/empty boxes
            if w <= 0 or h <= 0:
                continue
            # Convert to [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(a.get("category_id", 1))
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        # Ensure tensors have correct shapes even when there are no annotations
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.uint8)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        # Apply paired image+target transforms (e.g., resize that also scales bboxes)
        if self.sample_transforms is not None:
            img, target = self.sample_transforms(img, target)

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            # Se o pipeline de sample_transforms já retornou Tensor, não reconverta via PIL/NumPy.
            # Garanta apenas tipo/escala corretos e formato CHW.
            if isinstance(img, torch.Tensor):
                # dtype e escala
                if img.dtype != torch.float32:
                    img = img.float()
                # Se veio em 0..255, normalize para 0..1
                if torch.is_floating_point(img) and img.max().item() > 1.0:
                    img = img / 255.0
                # Formato: se vier HWC, permutar para CHW
                if img.ndim == 3 and img.shape[0] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
                    img = img.permute(2, 0, 1).contiguous()
            else:
                # Default to ToTensor without normalization; torchvision models expect [0,1]
                img = self.default_to_tensor(img)

        return img, target

    @staticmethod
    def default_to_tensor(pil_img: Image.Image) -> torch.Tensor:
        # Convert PIL to tensor in [0,1]
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(2, 0, 1)
        return arr


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))
