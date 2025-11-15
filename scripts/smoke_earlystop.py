"""Smoke test for Trainer early-stopping behavior using a tiny dummy dataset.

This script uses your existing `modules.get_model` and `modules.trainer.Trainer` to
run a few epochs on synthetic data. It avoids COCO annotations and the heavy mAP
computation by setting eval_map_every=0.

Run with: python scripts/smoke_earlystop.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Dataset, DataLoader

from modules import get_model
from utils import build_optimizer, build_scheduler
from modules.trainer import Trainer


class DummyDetectionDataset(Dataset):
    def __init__(self, n=20, img_size=(3, 128, 128)):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # random image
        img = torch.rand(self.img_size, dtype=torch.float32)
        # single box in format x1,y1,x2,y2
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return img, target


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Smoke test device: {device}")
    model = get_model("fasterrcnn_resnet50_fpn_v2", num_classes=2, pretrained=False).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, lr=0.005)
    scheduler = build_scheduler(optimizer, step_size=2, gamma=0.1)

    # smaller dataset for a fast smoke test; use GPU if available for speed
    train_ds = DummyDetectionDataset(n=20, img_size=(3, 128, 128))
    val_ds = DummyDetectionDataset(n=8, img_size=(3, 128, 128))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    es_cfg = {
        "enabled": True,
        "monitor": "val_loss",
        "mode": None,
        "min_delta": 1e-4,
        "patience": 2,
        "restore_best_weights": True,
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=None,
        eval_map_every=0,
        save_every=1,
        early_stopping_cfg=es_cfg,
        sample_vis_count=0,
        exp_dir=str(Path("outputs") / "smoke_test"),
    )

    try:
        out = trainer.fit(train_loader, val_loader, train_ds, val_ds, start_epoch=1, epochs=6)
        print("Smoke test finished on device:", device)
        print("History keys:", out.keys())
    except RuntimeError as e:
        msg = str(e).lower()
        if "outofmemory" in msg.replace(" ", "") or "out of memory" in msg:
            print("GPU out of memory detected. Clearing cache and retrying on CPU...")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Retry on CPU with smaller batch/image size to ensure the smoke test completes
            device = torch.device("cpu")
            model = get_model("fasterrcnn_resnet50_fpn_v2", num_classes=2, pretrained=False).to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = build_optimizer(params, lr=0.005)
            scheduler = build_scheduler(optimizer, step_size=2, gamma=0.1)

            train_ds = DummyDetectionDataset(n=10, img_size=(3, 64, 64))
            val_ds = DummyDetectionDataset(n=4, img_size=(3, 64, 64))
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=None,
                eval_map_every=0,
                save_every=1,
                early_stopping_cfg=es_cfg,
                sample_vis_count=0,
                exp_dir=str(Path("outputs") / "smoke_test"),
            )
            out = trainer.fit(train_loader, val_loader, train_ds, val_ds, start_epoch=1, epochs=4)
            print("Smoke test finished on CPU.")
            print("History keys:", out.keys())
        else:
            raise


if __name__ == "__main__":
    main()
