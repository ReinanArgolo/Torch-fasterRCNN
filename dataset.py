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

        # Index files in directory for robust filename resolution
        try:
            self._dir_files = sorted([f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))])
        except FileNotFoundError:
            self._dir_files = []

        self._dir_set_lower = {f.lower() for f in self._dir_files}
        self._resolved_paths: Dict[int, str] = {}
        resolved_ids: List[int] = []

        for img_id in all_img_ids:
            info = self.coco.loadImgs([img_id])[0]
            fname = info["file_name"]
            path = self._resolve_filename(fname)
            if path is not None:
                self._resolved_paths[img_id] = path
                resolved_ids.append(img_id)
            # If not found, skip this image_id silently (or could log)

        self.image_ids = resolved_ids

    def _resolve_filename(self, file_name: str) -> Optional[str]:
        # Exact match
        exact_path = os.path.join(self.images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        # Case-insensitive exact match
        lower = file_name.lower()
        if lower in self._dir_set_lower:
            # find the original cased filename
            for f in self._dir_files:
                if f.lower() == lower:
                    return os.path.join(self.images_dir, f)

        # Prefix match: handle cases like 'quadro_0000.jpg' vs 'quadro_0000_abcdef.jpg'
        base, ext = os.path.splitext(file_name)
        candidates = [f for f in self._dir_files if f.startswith(base + "_") and f.lower().endswith(ext.lower())]
        if len(candidates) == 1:
            return os.path.join(self.images_dir, candidates[0])
        if len(candidates) > 1:
            # pick the first deterministically
            return os.path.join(self.images_dir, sorted(candidates)[0])

        # Fallback: any startswith base (without enforcing underscore)
        candidates = [f for f in self._dir_files if f.startswith(base) and f.lower().endswith(ext.lower())]
        if candidates:
            return os.path.join(self.images_dir, sorted(candidates)[0])

        return None

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

        img = Image.open(path).convert("RGB")

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
