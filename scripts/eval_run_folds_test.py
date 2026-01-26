from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Ensure repository root is on sys.path when running as `python scripts/<file>.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
import yaml

from coco_eval import compute_coco_map
from dataset import COCODetectionDataset, collate_fn
from modules import get_model
from transforms import build_val_sample_transform


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_fold_checkpoints(run_dir: str) -> List[Tuple[str, str]]:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    out: List[Tuple[str, str]] = []
    for fd in fold_dirs:
        ckpt = fd / "best.pth"
        if ckpt.exists():
            out.append((fd.name, str(ckpt)))

    if out:
        return out

    ckpt = base / "best.pth"
    if ckpt.exists():
        return [(base.name, str(ckpt))]

    raise FileNotFoundError(f"No best.pth found in {run_dir} (expected fold_*/best.pth or best.pth)")


@dataclass(frozen=True)
class _FoldMetric:
    name: str
    checkpoint: str
    ap: float
    ap50: float
    ap75: float


def _nms_fuse_single(
    dets: List[Dict],
    iou_thresh: float,
    max_dets: int,
) -> List[Dict]:
    if not dets:
        return []

    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for d in dets:
        by_label[int(d["label"])].append(d)

    fused: List[Dict] = []
    for label, ds in by_label.items():
        boxes = torch.tensor([d["bbox_xyxy"] for d in ds], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in ds], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes, scores, float(iou_thresh)).tolist()
        keep = keep[: max(0, int(max_dets))]
        for k in keep:
            fused.append(ds[k])

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


@torch.no_grad()
def _predict_results_ensemble_nms(
    models: Sequence[torch.nn.Module],
    data_loader,
    device: torch.device,
    score_thresh: float,
    iou_thresh: float,
    max_dets: int,
) -> List[Dict]:
    for m in models:
        m.eval()

    results: List[Dict] = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        per_model_outputs = []
        for m in models:
            per_model_outputs.append(m(images))

        # Fuse per-image
        for img_idx_in_batch, tgt in enumerate(targets):
            image_id = int(tgt["image_id"].item())

            dets_xyxy: List[Dict] = []
            for outs in per_model_outputs:
                out = outs[img_idx_in_batch]
                boxes = out["boxes"].detach().cpu()
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu()

                keep = scores >= float(score_thresh)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    dets_xyxy.append({
                        "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(s),
                        "label": int(l),
                    })

            fused = _nms_fuse_single(dets_xyxy, iou_thresh=float(iou_thresh), max_dets=int(max_dets))

            for d in fused:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                results.append({
                    "image_id": image_id,
                    "category_id": int(d["label"]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(d["score"]),
                })

    return results


def _compute_coco_map_from_results(
    coco_gt: COCO,
    results: List[Dict],
    image_ids: Optional[List[int]] = None,
    iou_type: str = "bbox",
) -> Tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if image_ids is not None:
        coco_eval.params.imgIds = list(image_ids)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    ap = float(stats[0]) if stats is not None else 0.0
    ap50 = float(stats[1]) if stats is not None else 0.0
    ap75 = float(stats[2]) if stats is not None else 0.0
    return ap, ap50, ap75


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate each fold checkpoint on TEST set and optionally ensemble them (COCO mAP)")
    p.add_argument("--run-dir", required=True, help="Run directory containing fold_*/best.pth")

    # Dataset (explicit, so it works even if config.yaml isn't aligned)
    p.add_argument("--images-dir", required=True, help="Directory containing test images")
    p.add_argument("--ann-json", required=True, help="COCO annotation JSON for test split")

    # Optional config just to reuse model params/transforms
    p.add_argument("--config", default=None, help="Optional YAML config to read model/transforms")
    p.add_argument("--project-root", default=None, help="Optional base dir for relative paths in --config")

    p.add_argument("--model", default=None, help="Override model name (else read from --config, else fasterrcnn_resnet50_fpn_v2)")
    p.add_argument("--num-classes", type=int, default=None, help="Override num classes (else read from --config, else 2)")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--score-thresh", type=float, default=0.05)

    p.add_argument(
        "--sample-transform",
        choices=["none", "config-val"],
        default="none",
        help=(
            "Whether to apply an external sample transform before the model. "
            "For COCO eval, default is 'none' to keep predictions in original COCO image coordinates. "
            "Use 'config-val' only if you also adapt GT to the resized space (not recommended)."
        ),
    )

    p.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Optional subset of fold directory names to evaluate (e.g. fold_1 fold_3)",
    )

    # Ensemble
    p.add_argument("--ensemble", choices=["none", "nms"], default="nms")
    p.add_argument("--ensemble-iou", type=float, default=0.5)
    p.add_argument("--ensemble-max-dets", type=int, default=300)

    p.add_argument("--out-dir", required=True, help="Where to write metrics_test.json/csv")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")

    args = p.parse_args()

    device = DEVICE_DEFAULT
    if args.device is not None:
        device = torch.device(args.device)

    cfg = None
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    model_name = str(
        args.model
        or (cfg.get("model", {}).get("name") if isinstance(cfg, dict) else None)
        or "fasterrcnn_resnet50_fpn_v2"
    )
    num_classes = int(
        args.num_classes
        or (cfg.get("model", {}).get("num_classes") if isinstance(cfg, dict) else None)
        or 2
    )
    model_params = {}
    if isinstance(cfg, dict):
        mp = cfg.get("model", {}).get("params", {})
        if isinstance(mp, dict):
            model_params = mp

    # NOTE: For COCO mAP evaluation, do NOT resize externally by default.
    # Torchvision's detection models postprocess boxes back to the *input* image size.
    # If we resize in the Dataset, predictions become relative to the resized image, but COCO GT stays original -> AP collapses to ~0.
    sample_transform = None
    if args.sample_transform == "config-val" and isinstance(cfg, dict):
        sample_transform = build_val_sample_transform(cfg.get("training", {}).get("transforms", {}))

    ds = COCODetectionDataset(args.images_dir, args.ann_json, sample_transforms=sample_transform)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
    )

    folds = _find_fold_checkpoints(args.run_dir)
    if args.folds:
        wanted = set(args.folds)
        folds = [(n, pth) for (n, pth) in folds if n in wanted]
        if not folds:
            raise ValueError(f"No folds matched --folds={args.folds}")

    os.makedirs(args.out_dir, exist_ok=True)

    metrics: Dict[str, object] = {
        "run_dir": args.run_dir,
        "images_dir": args.images_dir,
        "ann_json": args.ann_json,
        "model": {"name": model_name, "num_classes": num_classes, "params": model_params},
        "score_thresh": float(args.score_thresh),
        "sample_transform": str(args.sample_transform),
        "folds": [],
    }

    fold_metrics: List[_FoldMetric] = []

    json_path = os.path.join(args.out_dir, "metrics_test.json")
    csv_path = os.path.join(args.out_dir, "metrics_test.csv")

    for fold_name, ckpt_path in folds:
        model = get_model(model_name, num_classes=num_classes, pretrained=False, **model_params).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ap, ap50, ap75 = compute_coco_map(model, loader, ds, device=device, score_thresh=float(args.score_thresh))
        fm = _FoldMetric(name=fold_name, checkpoint=ckpt_path, ap=ap, ap50=ap50, ap75=ap75)
        fold_metrics.append(fm)

        metrics["folds"].append({
            "fold": fold_name,
            "checkpoint": ckpt_path,
            "AP": ap,
            "AP50": ap50,
            "AP75": ap75,
        })

        # Persist after each fold so partial results survive interrupts
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "checkpoint", "AP", "AP50", "AP75"])
            for fm in fold_metrics:
                w.writerow([fm.name, fm.checkpoint, f"{fm.ap:.6f}", f"{fm.ap50:.6f}", f"{fm.ap75:.6f}"])

        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Ensemble evaluation (optional)
    if args.ensemble != "none":
        # Load all fold models at once for correctness (slower, but simple).
        models = []
        for fold_name, ckpt_path in folds:
            m = get_model(model_name, num_classes=num_classes, pretrained=False, **model_params).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(ckpt["model_state"])
            models.append(m)

        if args.ensemble == "nms":
            results = _predict_results_ensemble_nms(
                models=models,
                data_loader=loader,
                device=device,
                score_thresh=float(args.score_thresh),
                iou_thresh=float(args.ensemble_iou),
                max_dets=int(args.ensemble_max_dets),
            )
            ap, ap50, ap75 = _compute_coco_map_from_results(
                coco_gt=ds.coco,
                results=results,
                image_ids=list(ds.image_ids) if hasattr(ds, "image_ids") else None,
            )
            metrics["ensemble"] = {
                "type": "nms",
                "iou": float(args.ensemble_iou),
                "max_dets": int(args.ensemble_max_dets),
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
            }

        # Cleanup
        try:
            for m in models:
                del m
        except Exception:
            pass
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Write final outputs (append ensemble if present)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "checkpoint", "AP", "AP50", "AP75"])
        for fm in fold_metrics:
            w.writerow([fm.name, fm.checkpoint, f"{fm.ap:.6f}", f"{fm.ap50:.6f}", f"{fm.ap75:.6f}"])
        if isinstance(metrics.get("ensemble"), dict):
            e = metrics["ensemble"]
            w.writerow(["ensemble_nms", "-", f"{e['AP']:.6f}", f"{e['AP50']:.6f}", f"{e['AP75']:.6f}"])

    print({"out_dir": args.out_dir, "folds": len(folds), "ensemble": args.ensemble})


if __name__ == "__main__":
    main()
