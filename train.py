from __future__ import annotations
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import argparse
import yaml
import torch
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import COCODetectionDataset, collate_fn
from transforms import build_sample_transform
from utils import build_optimizer, build_scheduler, set_seed
from modules import get_model
from modules.trainer import Trainer

# ADDITIONS
import json
import numpy as np
# END ADDITIONS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Train Faster R-CNN (refatorado)")
    p.add_argument("--config", required=True)
    p.add_argument("--project-root", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-map-every", type=int, default=None)
    return p.parse_args()


def rp(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


# ADDITIONS: utilitários para CV em COCO
def _load_coco(ann_path: str) -> dict:
    with open(ann_path, "r") as f:
        data = json.load(f)
    return data


def _write_coco_subset(base: dict, image_ids_set: set[int], out_path: str):
    images = [im for im in base.get("images", []) if im["id"] in image_ids_set]
    anns = [an for an in base.get("annotations", []) if an["image_id"] in image_ids_set]
    subset = {
        "info": base.get("info", {}),
        "licenses": base.get("licenses", []),
        "categories": base.get("categories", []),
        "images": images,
        "annotations": anns,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subset, f)


def _build_kfold_splits(ann_path: str, n_folds: int, seed: int):
    base = _load_coco(ann_path)
    images = base.get("images", [])
    img_ids = np.array([im["id"] for im in images], dtype=np.int64)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(img_ids))
    folds_idx = np.array_split(perm, n_folds)

    splits = []
    for i in range(n_folds):
        val_idx = folds_idx[i]
        train_idx = np.concatenate([folds_idx[j] for j in range(n_folds) if j != i], axis=0) if n_folds > 1 else val_idx
        train_ids = set(img_ids[train_idx].tolist())
        val_ids = set(img_ids[val_idx].tolist())
        splits.append((train_ids, val_ids))
    return base, splits
# END ADDITIONS


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    project_root = args.project_root or cfg.get("project_root", os.getcwd())
    data_cfg = cfg["data"]

    train_images = rp(project_root, data_cfg["images"]["train_dir"])
    val_images = rp(project_root, data_cfg["images"].get("val_dir", ""))
    train_ann = rp(project_root, data_cfg["annotations"]["train_json"])
    val_ann = rp(project_root, data_cfg["annotations"].get("val_json", ""))

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    es_cfg = cfg.get("early_stopping", {})
    cv_cfg = cfg.get("cross_validation", {})  # ADDITION

    epochs = args.epochs or int(train_cfg.get("epochs", 10))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    lr = args.lr or float(train_cfg.get("lr", 0.005))
    num_workers = args.num_workers or int(train_cfg.get("num_workers", 4))
    seed = args.seed or int(train_cfg.get("seed", 42))
    amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    eval_map_every = args.eval_map_every if args.eval_map_every is not None else int(train_cfg.get("eval_map_every", 0))
    sample_vis_count = int(train_cfg.get("sample_vis_count", 3))
    sample_vis_thresh = float(train_cfg.get("sample_vis_thresh", 0.1))
    save_every = int(out_cfg.get("save_every", 1))

    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    exp_dir = os.path.join(project_root, out_cfg.get("dir", "outputs"), out_cfg.get("experiment", "exp"), run_stamp)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Execução em: {exp_dir}")

    set_seed(seed)

    # Transforms
    train_sample_transform = build_sample_transform(train_cfg.get("transforms", {}))
    val_sample_transform = None

    # CV toggle
    cv_enabled = bool(cv_cfg.get("enabled", False))
    n_folds = int(cv_cfg.get("folds", 5))

    if cv_enabled:
        print(f"Validação cruzada K={n_folds} habilitada. Gerando folds a partir de {train_ann}")
        base_json, splits = _build_kfold_splits(train_ann, n_folds=n_folds, seed=seed)

        # Em CV, usamos o diretório de treino para treino e validação (subconjuntos)
        cv_images_root = train_images

        fold_summaries = []
        for fold_idx, (train_ids, val_ids) in enumerate(splits, start=1):
            print(f"Fold {fold_idx}/{n_folds}: treino={len(train_ids)} imagens, val={len(val_ids)} imagens")
            fold_dir = os.path.join(exp_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            fold_train_ann = os.path.join(fold_dir, "instances_train.json")
            fold_val_ann = os.path.join(fold_dir, "instances_val.json")
            _write_coco_subset(base_json, train_ids, fold_train_ann)
            _write_coco_subset(base_json, val_ids, fold_val_ann)

            # Datasets e loaders para o fold
            train_ds = COCODetectionDataset(cv_images_root, fold_train_ann, sample_transforms=train_sample_transform)
            val_ds = COCODetectionDataset(cv_images_root, fold_val_ann, sample_transforms=val_sample_transform)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

            # Modelo/otimizador por fold
            num_classes = int(model_cfg.get("num_classes", 2))
            model_name = str(model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"))
            pretrained = bool(model_cfg.get("pretrained", True))
            model = get_model(model_name, num_classes, pretrained=pretrained).to(DEVICE)

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = build_optimizer(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)), weight_decay=float(train_cfg.get("weight_decay", 5e-4)))
            scheduler = build_scheduler(optimizer, step_size=int(train_cfg.get("lr_step_size", 3)), gamma=float(train_cfg.get("lr_gamma", 0.1)))
            scaler = torch.amp.GradScaler("cuda") if amp else None

            # Resume desabilitado por padrão em CV (ajuste se necessário)
            start_epoch = 1

            # Trainer e treino
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=DEVICE,
                scaler=scaler,
                eval_map_every=eval_map_every,
                save_every=save_every,
                early_stopping_cfg=es_cfg,
                sample_vis_count=sample_vis_count,
                exp_dir=fold_dir,
                sample_vis_thresh=sample_vis_thresh,
            )

            trainer.fit(train_loader, val_loader, train_ds, val_ds, start_epoch, epochs)

            # Opcional: aqui você pode coletar métricas do fold a partir do trainer, se ele expuser
            # ex.: fold_summaries.append({"fold": fold_idx, "val_mAP": trainer.best_map, "val_loss": trainer.best_val_loss})

        # Opcional: agregue métricas entre folds aqui e salve em exp_dir
        return

    # Caminho padrão (sem CV): usa treino/val configurados
    train_ds = COCODetectionDataset(train_images, train_ann, sample_transforms=train_sample_transform)
    val_ds = COCODetectionDataset(val_images, val_ann, sample_transforms=val_sample_transform) if val_images and val_ann and os.path.exists(val_ann) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) if val_ds else None

    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = str(model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"))
    pretrained = bool(model_cfg.get("pretrained", True))
    model = get_model(model_name, num_classes, pretrained=pretrained).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)), weight_decay=float(train_cfg.get("weight_decay", 5e-4)))
    scheduler = build_scheduler(optimizer, step_size=int(train_cfg.get("lr_step_size", 3)), gamma=float(train_cfg.get("lr_gamma", 0.1)))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    start_epoch = 1
    resume_path = args.resume or out_cfg.get("resume")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumido de {resume_path} epoch={start_epoch-1}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        scaler=scaler,
        eval_map_every=eval_map_every,
        save_every=save_every,
        early_stopping_cfg=es_cfg,
        sample_vis_count=sample_vis_count,
        exp_dir=exp_dir,
        sample_vis_thresh=sample_vis_thresh,
    )

    trainer.fit(train_loader, val_loader, train_ds, val_ds, start_epoch, epochs)


if __name__ == "__main__":
    main()
