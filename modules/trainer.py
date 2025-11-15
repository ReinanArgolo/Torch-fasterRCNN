import os
import math
import random
import time
import torch
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, format_time
from coco_eval import compute_coco_map
from .visualization import visualize_samples
from .metrics import export_metrics

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        scaler=None,
        eval_map_every=0,
        save_every=1,
        early_stopping_cfg=None,
        sample_vis_count=3,
        exp_dir=".",
        sample_vis_thresh=0.1,  # novo
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.eval_map_every = eval_map_every
        self.save_every = save_every
        self.es_cfg = early_stopping_cfg or {}
        self.sample_vis_count = sample_vis_count
        self.exp_dir = exp_dir
        self.sample_vis_thresh = sample_vis_thresh
        self.best_val = float("inf")
        self.best_map = -float("inf")
        self.epochs_no_improve = 0
        # path where we'll keep the best checkpoint for early stopping restoration
        self.best_ckpt_path = os.path.join(self.exp_dir, "best.pth")

    def _train_one_epoch(self, loader, epoch: int):
        self.model.train()
        meter = AverageMeter("loss")
        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                # API atual de AMP
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    losses = self.model(images, targets)
                    loss = sum(losses.values())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(images, targets)
                loss = sum(losses.values())
                loss.backward()
                self.optimizer.step()
            meter.update(loss.item(), n=len(images))
            pbar.set_postfix({"loss": f"{meter.avg:.4f}"})
        return meter.avg

    def _evaluate_loss(self, loader):
        self.model.train()
        meter = AverageMeter("val_loss")
        pbar = tqdm(loader, desc="[val_loss]", leave=False)
        with torch.no_grad():
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                losses = self.model(images, targets)
                loss = sum(losses.values())
                meter.update(loss.item(), n=len(images))
                pbar.set_postfix({"val_loss": f"{meter.avg:.4f}"})
        return meter.avg

    def _early_stopping_check(self, epoch_metric, monitor_key: str):
        if not self.es_cfg.get("enabled", False):
            return False
        # reconcile defaults with requested specs
        mode = self.es_cfg.get("mode")
        min_delta = float(self.es_cfg.get("min_delta", 1e-4))
        patience = int(self.es_cfg.get("patience", 10))
        restore_best = bool(self.es_cfg.get("restore_best_weights", True))

        # if mode is not set, auto-detect from monitor_key
        if mode is None:
            if "map" in monitor_key.lower():
                mode = "max"
            else:
                mode = "min"
        improved = False
        # choose comparison based on mode
        if mode == "min":
            if epoch_metric < (self.best_val - min_delta):
                self.best_val = epoch_metric
                improved = True
        else:
            if epoch_metric > (self.best_map + min_delta):
                self.best_map = epoch_metric
                improved = True

        # Save best checkpoint when improved
        if improved:
            # always save model state and optimizer/scheduler state to best_ckpt_path
            save_checkpoint(
                {
                    "epoch": None,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "sched_state": self.scheduler.state_dict(),
                    "best_val": self.best_val,
                    "best_map": self.best_map,
                },
                self.best_ckpt_path,
            )
        if improved:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        # if patience reached, optionally restore best and signal stop
        if self.epochs_no_improve >= patience:
            if restore_best and os.path.exists(self.best_ckpt_path):
                try:
                    ckpt = torch.load(self.best_ckpt_path, map_location="cpu")
                    if "model_state" in ckpt:
                        self.model.load_state_dict(ckpt["model_state"], strict=False)
                    if "optim_state" in ckpt and self.optimizer is not None:
                        try:
                            self.optimizer.load_state_dict(ckpt["optim_state"])
                        except Exception:
                            # optimizer state may be incompatible; ignore if so
                            pass
                    if "sched_state" in ckpt and self.scheduler is not None:
                        try:
                            self.scheduler.load_state_dict(ckpt["sched_state"])
                        except Exception:
                            pass
                    print(f"[Trainer] Restored best checkpoint from {self.best_ckpt_path} before stopping.")
                except Exception as e:
                    print(f"[Trainer] Failed to restore best checkpoint: {e}")
            return True
        return False

    def fit(self, train_loader, val_loader, train_ds, val_ds, start_epoch, epochs):
        hist_epochs, hist_train, hist_val, hist_map = [], [], [], []
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            val_loss = None
            val_map = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                if self.eval_map_every > 0 and (epoch % self.eval_map_every) == 0:
                    # Compute COCO mAP on validation set (AP@[.5:.95])
                    self.model.eval()
                    with torch.no_grad():
                        # compute_coco_map returns a tuple (AP, AP50, AP75)
                        try:
                            ap, ap50, ap75 = compute_coco_map(self.model, val_loader, val_ds, device=self.device)
                        except TypeError:
                            # Backward-compat: older signature may not require dataset, try without it
                            ap, ap50, ap75 = compute_coco_map(self.model, val_loader, device=self.device)  # type: ignore
                    val_map = float(ap)
                    print(f"[Epoch {epoch}] mAP: {val_map:.4f} | AP50: {ap50:.4f} | AP75: {ap75:.4f}")

            hist_epochs.append(epoch)
            hist_train.append(train_loss)
            hist_val.append(val_loss if val_loss is not None else math.nan)
            hist_map.append(val_map if val_map is not None else math.nan)

            # Visualizações de amostra (fallback para train_ds se val_ds estiver vazio)
            sample_dir = os.path.join(self.exp_dir, f"epoch_{epoch:03d}", "samples")
            os.makedirs(sample_dir, exist_ok=True)
            ds_ref = None
            src_name = ""
            if val_ds is not None and len(val_ds) > 0:
                ds_ref = val_ds
                src_name = "val"
            elif train_ds is not None and len(train_ds) > 0:
                ds_ref = train_ds
                src_name = "train"
            else:
                print(f"[Epoch {epoch}] Nenhum dataset disponível para visualizar amostras.")
            if self.sample_vis_count > 0 and ds_ref is not None:
                n = len(ds_ref)
                k = min(self.sample_vis_count, n)
                if k > 0:
                    import random
                    idxs = random.sample(range(n), k)
                    counts = visualize_samples(self.model, ds_ref, idxs, sample_dir, self.device, score_thresh=self.sample_vis_thresh)
                    total_drawn = sum(counts.values())
                    print(f"[Epoch {epoch}] {k} amostras salvas de '{src_name}' em {sample_dir} (boxes desenhadas: {total_drawn}; thresh={self.sample_vis_thresh}).")
                else:
                    print(f"[Epoch {epoch}] Dataset '{src_name}' vazio; nenhuma amostra salva.")

            # Salvar checkpoint
            if (epoch % self.save_every) == 0 or epoch == epochs:
                ckpt_path = os.path.join(self.exp_dir, f"epoch_{epoch:03d}", f"checkpoint_epoch_{epoch:03d}.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "sched_state": self.scheduler.state_dict(),
                        "best_val": self.best_val,
                        "best_map": self.best_map,
                    },
                    ckpt_path,
                )
                print(f"Checkpoint salvo: {ckpt_path}")

            stop = False
            # Decide which metric to monitor
            monitor = self.es_cfg.get("monitor", "val_loss")
            # If configured, prefer val_mAP automatically when available
            if self.es_cfg.get("prefer_val_mAP", False) and self.eval_map_every > 0:
                monitor = "val_mAP"
            # accept multiple monitor aliases; prefer val_mAP if configured
            monitor_alias = monitor
            if monitor_alias in ("map", "mAP"):
                monitor_alias = "val_mAP"
            if monitor_alias == "val_mAP":
                metric_current = val_map
            else:
                metric_current = val_loss
            if val_loader is not None and metric_current is not None:
                stop = self._early_stopping_check(metric_current, monitor)
                if stop:
                    print("Early stopping acionado.")
            dur = time.time() - t0
            print(f"Epoch {epoch} fim em {format_time(dur)} | train={train_loss:.4f} val={val_loss} map={val_map}")
            if stop:
                break

        export_metrics(self.exp_dir, hist_epochs, hist_train, hist_val, hist_map)
        print("Treinamento completo.")
        # After training finishes normally, if configured, restore best weights
        if self.es_cfg.get("enabled", False) and bool(self.es_cfg.get("restore_best_weights", True)):
            if os.path.exists(self.best_ckpt_path):
                try:
                    ckpt = torch.load(self.best_ckpt_path, map_location="cpu")
                    if "model_state" in ckpt:
                        self.model.load_state_dict(ckpt["model_state"], strict=False)
                    print(f"[Trainer] Final best checkpoint restored from {self.best_ckpt_path}.")
                except Exception as e:
                    print(f"[Trainer] Failed to restore final best checkpoint: {e}")
        return {
            "epochs": hist_epochs,
            "train_loss": hist_train,
            "val_loss": hist_val,
            "map": hist_map,
        }