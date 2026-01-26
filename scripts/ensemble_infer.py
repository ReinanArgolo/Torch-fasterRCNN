from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision

from dataset import collate_fn
from infer import ImageFolderDataset
from modules import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_model(model, loader, score_thresh: float) -> Dict[str, List[dict]]:
    model.eval()
    per_file: Dict[str, List[dict]] = defaultdict(list)
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                file_name = tgt.get("file_name")
                if file_name is None:
                    # fallback: ImageFolderDataset uses image_id index
                    file_name = str(int(tgt["image_id"].item()))

                boxes = out["boxes"].detach().cpu()
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu()

                keep = scores >= score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    per_file[file_name].append({
                        "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(s),
                        "label": int(l),
                    })
    return per_file


def _nms_fuse(dets: List[dict], iou_thresh: float) -> List[dict]:
    if not dets:
        return []

    out: List[dict] = []
    by_label: Dict[int, List[dict]] = defaultdict(list)
    for d in dets:
        by_label[int(d["label"])].append(d)

    for label, ds in by_label.items():
        boxes = torch.tensor([d["bbox_xyxy"] for d in ds], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in ds], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes, scores, float(iou_thresh)).tolist()
        for k in keep:
            out.append(ds[k])

    # stable ordering by score desc
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def main():
    p = argparse.ArgumentParser(description="Simple ensemble inference by NMS-fusing predictions from multiple checkpoints")
    p.add_argument("--images", required=True, help="Directory with images for inference")
    p.add_argument("--checkpoints", required=True, nargs="+", help="List of checkpoints (.pth) from multiple folds/models")
    p.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for NMS fusion")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    ds = ImageFolderDataset(args.images, sample_transform=None)

    # Add file_name to target so downstream aggregation is stable
    def _collate(batch):
        images, targets = collate_fn(batch)
        targets = list(targets)
        for t, f in zip(targets, [ds.files[int(tt["image_id"].item())] for tt in targets]):
            t["file_name"] = f
        return images, targets

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate, num_workers=2)

    aggregated: Dict[str, List[dict]] = defaultdict(list)

    for ckpt_path in args.checkpoints:
        model = get_model(args.model, num_classes=args.num_classes, pretrained=False).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

        per_file = _run_model(model, loader, score_thresh=float(args.score_thresh))
        for fn, dets in per_file.items():
            aggregated[fn].extend(dets)

        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    fused = {fn: _nms_fuse(dets, iou_thresh=float(args.iou_thresh)) for fn, dets in aggregated.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": fused}, f)

    print({"images": len(fused), "checkpoints": len(args.checkpoints), "out": args.out})


if __name__ == "__main__":
    main()
