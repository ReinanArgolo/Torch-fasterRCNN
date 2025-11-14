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
        mode = self.es_cfg.get("mode", "min")
        min_delta = float(self.es_cfg.get("min_delta", 0.0))
        patience = int(self.es_cfg.get("patience", 5))
        improved = False
        if monitor_key == "val_loss":
            if epoch_metric < (self.best_val - min_delta):
                self.best_val = epoch_metric
                improved = True
        else:
            if epoch_metric > (self.best_map + min_delta):
                self.best_map = epoch_metric
                improved = True
        if improved:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= patience

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
                    self.model.eval()
                    with torch.no_grad():
                        map_stats = compute_coco_map(self.model, val_loader, device=self.device)
                    val_map = map_stats.get("bbox_mAP") or map_stats.get("mAP") or map_stats.get("map") or None
                    print(f"[Epoch {epoch}] mAP: {val_map}")

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
            monitor = self.es_cfg.get("monitor", "val_loss")
            metric_current = val_map if (monitor == "map") else val_loss
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
        return {
            "epochs": hist_epochs,
            "train_loss": hist_train,
            "val_loss": hist_val,
            "map": hist_map,
        }