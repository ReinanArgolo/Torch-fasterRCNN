from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def predict_results(
    model,
    data_loader,
    device: torch.device,
    score_thresh: float = 0.05,
) -> List[Dict]:
    """Run inference and return results in COCO json format.

    Returns a list of dicts with keys: image_id, category_id, bbox[x,y,w,h], score
    """
    model.eval()
    results: List[Dict] = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item())
            boxes = out["boxes"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            labels = out["labels"].detach().cpu().tolist()
            for box, score, label in zip(boxes, scores, labels):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )
    return results


def compute_coco_map(
    model,
    data_loader,
    dataset,
    device: torch.device,
    iou_type: str = "bbox",
    score_thresh: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute COCO mAP (AP@[.5:.95]) and return (AP, AP50, AP75).

    Requires dataset.coco to be available (pycocotools COCO) and that the data_loader
    yields targets with 'image_id'.
    """
    if not hasattr(dataset, "coco"):
        raise ValueError("Dataset must have a 'coco' attribute for evaluation.")

    coco_gt: COCO = dataset.coco
    results = predict_results(model, data_loader, device=device, score_thresh=score_thresh)

    if len(results) == 0:
        return 0.0, 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    ap = float(stats[0]) if stats is not None else 0.0
    ap50 = float(stats[1]) if stats is not None else 0.0
    ap75 = float(stats[2]) if stats is not None else 0.0
    return ap, ap50, ap75
