from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

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
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.transforms = transforms
        self.coco = COCO(annotation_json)
        self.image_ids = list(sorted(self.coco.getImgIds()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        path = os.path.join(self.images_dir, file_name)

        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for a in anns:
            x, y, w, h = a["bbox"]
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(a.get("category_id", 1))
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
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
