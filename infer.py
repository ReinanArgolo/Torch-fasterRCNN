from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from dataset import COCODetectionDataset, collate_fn
from train import get_model, DEVICE


class ImageFolderDataset(Dataset):
    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self.files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.images_dir, file_name)
        img = Image.open(path).convert("RGB")
        # To tensor [0,1]
        tensor = torchvision.transforms.functional.to_tensor(img)
        # Use index as image_id for consistency in outputs; file mapping saved separately
        target = {"image_id": torch.tensor([idx], dtype=torch.int64)}
        return tensor, target


def main():
    parser = argparse.ArgumentParser(description="Inference for Faster R-CNN")
    parser.add_argument("--images", type=str, required=True, help="Directory with images for inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes including background")
    parser.add_argument("--model", type=str, default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    ds = ImageFolderDataset(args.images)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = get_model(args.model, num_classes=args.num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    results = []
    file_map = []

    with torch.no_grad():
        for (images, targets) in tqdm(loader, desc="Infer"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            out = outputs[0]
            img_idx = int(targets[0]["image_id"].item())
            file_map.append(ds.files[img_idx])

            boxes = out["boxes"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            labels = out["labels"].detach().cpu().tolist()

            for box, score, label in zip(boxes, scores, labels):
                if score < args.score_thresh:
                    continue
                x1, y1, x2, y2 = box
                results.append({
                    "file_name": ds.files[img_idx],
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "label": int(label),
                })

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Saved results to {args.out}")
    else:
        print(json.dumps({"results": results[:50]}, indent=2))  # print sample if no path provided


if __name__ == "__main__":
    main()
