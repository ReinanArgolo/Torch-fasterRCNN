from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, Optional

import torch
import torchvision
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset import COCODetectionDataset, collate_fn
from utils import AverageMeter, build_optimizer, build_scheduler, save_checkpoint, set_seed, format_time
from coco_eval import compute_coco_map
from transforms import build_sample_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "fasterrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None,
            num_classes=num_classes,
        )
    elif name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model


def train_one_epoch(model, loader, optimizer, epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    loss_meter = AverageMeter("loss")
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for images, targets in pbar:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                losses: Dict[str, torch.Tensor] = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses: Dict[str, torch.Tensor] = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), n=len(images))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    return loss_meter.avg

def evaluate_loss(model, loader):
    # For torchvision detection models, validation loss can be obtained by keeping model in train mode
    # but disabling gradients.
    model.train()
    loss_meter = AverageMeter("val_loss")
    pbar = tqdm(loader, desc="[val_loss]", leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            losses: Dict[str, torch.Tensor] = model(images, targets)
            loss = sum(losses.values())
            loss_meter.update(loss.item(), n=len(images))
            pbar.set_postfix({"val_loss": f"{loss_meter.avg:.4f}"})
    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on COCO-style dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--project-root", type=str, default=None, help="Override project root for relative paths")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--eval-map-every", type=int, default=None, help="Evaluate COCO mAP every N epochs (0 to disable)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve root
    project_root = args.project_root or cfg.get("project_root", os.getcwd())

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    data_cfg = cfg["data"]
    train_images = rp(data_cfg["images"]["train_dir"]) if isinstance(data_cfg["images"], dict) else rp(data_cfg["train_images"])  # backward
    val_images = rp(data_cfg["images"].get("val_dir", "")) if isinstance(data_cfg["images"], dict) else rp(data_cfg.get("val_images", ""))

    train_ann = rp(data_cfg["annotations"]["train_json"]) if isinstance(data_cfg["annotations"], dict) else rp(data_cfg["train_annotations"])  # backward
    val_ann = rp(data_cfg["annotations"].get("val_json", "")) if isinstance(data_cfg["annotations"], dict) else rp(data_cfg.get("val_annotations", ""))

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})

    epochs = args.epochs or int(train_cfg.get("epochs", 10))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    lr = args.lr or float(train_cfg.get("lr", 0.005))
    num_workers = args.num_workers or int(train_cfg.get("num_workers", 4))
    seed = args.seed or int(train_cfg.get("seed", 42))
    amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    eval_map_every = args.eval_map_every if args.eval_map_every is not None else int(train_cfg.get("eval_map_every", 0))

    exp_dir = os.path.join(project_root, out_cfg.get("dir", "outputs"), out_cfg.get("experiment", "exp"))
    os.makedirs(exp_dir, exist_ok=True)

    set_seed(seed)

    # Datasets and loaders
    # Build paired transforms (e.g., resize with bbox scaling)
    sample_tf_cfg = train_cfg.get("transforms", {})
    sample_transform = build_sample_transform(sample_tf_cfg)

    train_ds = COCODetectionDataset(train_images, train_ann, sample_transforms=sample_transform)
    val_ds = COCODetectionDataset(val_images, val_ann, sample_transforms=sample_transform) if val_images and val_ann and os.path.exists(val_ann) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) if val_ds else None

    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = str(model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"))
    pretrained = bool(model_cfg.get("pretrained", True))
    model = get_model(model_name, num_classes, pretrained=pretrained)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)), weight_decay=float(train_cfg.get("weight_decay", 5e-4)))
    scheduler = build_scheduler(optimizer, step_size=int(train_cfg.get("lr_step_size", 3)), gamma=float(train_cfg.get("lr_gamma", 0.1)))

    scaler = torch.amp.GradScaler("cuda") if amp else None

    start_epoch = 1
    best_val = float("inf")
    best_map = -float("inf")
    resume_path = args.resume or out_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        best_map = ckpt.get("best_map", best_map)
        print(f"Resumed from {resume_path} at epoch {start_epoch-1}")

    save_every = int(out_cfg.get("save_every", 1))

    # Early stopping configuration
    es_cfg = cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_monitor = str(es_cfg.get("monitor", "val_loss"))  # "val_loss" or "map"
    es_mode = str(es_cfg.get("mode", "min"))  # "min" for loss, "max" for map
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_patience = int(es_cfg.get("patience", 5))
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, scaler)
        val_loss = None
        val_map = None
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader)
            if val_loss < best_val - es_min_delta:
                best_val = val_loss
            # Compute mAP optionally
            if eval_map_every and (epoch % eval_map_every == 0 or epoch == start_epoch):
                ap, ap50, ap75 = compute_coco_map(model, val_loader, val_ds, device=DEVICE)
                val_map = ap
                if val_map > best_map + es_min_delta:
                    best_map = val_map

        scheduler.step()

        # Save checkpoint
        if (epoch % save_every) == 0 or epoch == epochs:
            ckpt_path = os.path.join(exp_dir, f"epoch_{epoch}.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "best_val": best_val,
                    "best_map": best_map,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        # Early stopping check
        if es_enabled and val_loader is not None:
            current = None
            if es_monitor == "val_loss":
                current = val_loss if val_loss is not None else float("inf")
                improved = current <= (best_val + es_min_delta)
                # improvement already updated above; count no-improve based on mode
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            elif es_monitor == "map":
                # ensure we have a map value; if not computed this epoch, compute now quickly
                if val_map is None:
                    ap, ap50, ap75 = compute_coco_map(model, val_loader, val_ds, device=DEVICE)
                    val_map = ap
                current = val_map
                improved = current >= (best_map - es_min_delta)
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if epochs_no_improve >= es_patience:
                print(f"Early stopping triggered after {epoch} epochs (monitor={es_monitor}, patience={es_patience}).")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
