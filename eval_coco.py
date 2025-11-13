from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
import yaml

from dataset import COCODetectionDataset, collate_fn
from train import get_model, DEVICE
from coco_eval import compute_coco_map


def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO mAP on validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    project_root = args.project_root or cfg.get("project_root", os.getcwd())
    def rp(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    data_cfg = cfg["data"]
    val_images = rp(data_cfg["images"]["val_dir"]) if isinstance(data_cfg["images"], dict) else rp(data_cfg.get("val_images", ""))
    val_ann = rp(data_cfg["annotations"]["val_json"]) if isinstance(data_cfg["annotations"], dict) else rp(data_cfg.get("val_annotations", ""))

    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = str(model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"))

    ds = COCODetectionDataset(val_images, val_ann)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=int(cfg.get("training",{}).get("num_workers", 4)), collate_fn=collate_fn)

    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)

    ap, ap50, ap75 = compute_coco_map(model, loader, ds, device=DEVICE, score_thresh=args.score_thresh)
    print({"AP": ap, "AP50": ap50, "AP75": ap75})


if __name__ == "__main__":
    main()
