from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
import yaml

from dataset import COCODetectionDataset, collate_fn
from modules import get_model
from coco_eval import compute_coco_map


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO mAP on validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--images-dir", type=str, default=None, help="Optional: override validation images dir")
    parser.add_argument("--ann-json", type=str, default=None, help="Optional: override validation COCO annotation json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    project_root = args.project_root or cfg.get("project_root", os.getcwd())
    def rp(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    data_cfg = cfg["data"]
    val_images = (
        args.images_dir
        or (rp(data_cfg["images"]["val_dir"]) if isinstance(data_cfg.get("images"), dict) else rp(data_cfg.get("val_images", "")))
    )
    val_ann = (
        args.ann_json
        or (rp(data_cfg["annotations"]["val_json"]) if isinstance(data_cfg.get("annotations"), dict) else rp(data_cfg.get("val_annotations", "")))
    )

    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = str(model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"))
    model_params = model_cfg.get("params", {}) if isinstance(model_cfg.get("params", {}), dict) else {}

    # IMPORTANT: do not resize externally for COCO evaluation.
    # COCOeval uses GT boxes from the original JSON; resizing in the Dataset would mismatch coords.
    ds = COCODetectionDataset(val_images, val_ann, sample_transforms=None)
    num_workers = int(cfg.get("training", {}).get("num_workers", 4))
    batch_size = int(cfg.get("training", {}).get("batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = get_model(model_name, num_classes=num_classes, pretrained=False, **model_params)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)

    ap, ap50, ap75 = compute_coco_map(model, loader, ds, device=DEVICE, score_thresh=args.score_thresh)
    print({"AP": ap, "AP50": ap50, "AP75": ap75})


if __name__ == "__main__":
    main()
