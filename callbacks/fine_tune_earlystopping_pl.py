"""Fine-tuning aware EarlyStopping for PyTorch Lightning.

This callback monitors a validation metric (default 'val_loss') and triggers early
stopping when no improvement is observed for `patience` consecutive validation
epochs. If `prefer_val_mAP=True` and 'val_mAP' is found in the logged metrics,
it will switch to monitoring 'val_mAP'. The best checkpoint is saved to disk and
restored onto the LightningModule at training end.

Notes:
 - Ensure your LightningModule logs validation metrics using `self.log('val_loss', ...)`
   and/or `self.log('val_mAP', ..., on_epoch=True, sync_dist=True)`.
 - This file assumes `pytorch_lightning` is available in the environment when used.
"""
from typing import Optional
import os
import torch

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - import error will surface when used
    pl = None


class FineTuneEarlyStoppingPL(pl.Callback if pl is not None else object):
    """EarlyStopping callback optimized for fine-tuning with PyTorch Lightning.

    Parameters
    ----------
    monitor : Optional[str]
        Metric to monitor (default 'val_loss'). If None, default is 'val_loss'.
    prefer_val_mAP : bool
        If True and 'val_mAP' is present in `trainer.callback_metrics`, prefer it.
    patience : int
        Number of validation epochs with no improvement to wait before stopping.
    min_delta : float
        Minimum change to count as improvement.
    ckpt_dir : str
        Directory to save the best checkpoint.
    verbose : bool
        If True, prints progress messages.
    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        prefer_val_mAP: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        ckpt_dir: str = "checkpoints",
        verbose: bool = True,
    ):
        if pl is None:
            raise RuntimeError("pytorch_lightning is required to use FineTuneEarlyStoppingPL. Install 'pytorch-lightning'.")
        super().__init__()
        self.user_monitor = monitor
        self.prefer_val_mAP = bool(prefer_val_mAP)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = True
        self.ckpt_dir = ckpt_dir
        self.verbose = bool(verbose)

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # runtime state
        self.monitor = monitor or "val_loss"
        self.mode = None
        self.best = None
        self.wait = 0
        self.best_ckpt_path = None

    def _set_mode(self):
        if "map" in self.monitor.lower():
            self.mode = "max"
            self.best = -float("inf")
        else:
            self.mode = "min"
            self.best = float("inf")

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics = trainer.callback_metrics

        # allow switching to val_mAP if preferred and present
        if self.user_monitor is None and self.prefer_val_mAP and "val_mAP" in metrics:
            if self.monitor != "val_mAP":
                if self.verbose:
                    print("[FineTuneEarlyStoppingPL] Using 'val_mAP' as monitor (found in callback_metrics).")
                self.monitor = "val_mAP"
                self._set_mode()

        if self.mode is None:
            self._set_mode()

        if self.monitor not in metrics:
            raise ValueError(
                f"Monitored metric '{self.monitor}' not found in trainer.callback_metrics at validation end. "
                f"Ensure your validation_step logs it via self.log('val_loss', ... , on_epoch=True) or self.log('val_mAP', ..., on_epoch=True). "
                f"Available keys: {list(metrics.keys())}"
            )

        val = metrics[self.monitor]
        if isinstance(val, torch.Tensor):
            current = float(val.detach().cpu().item())
        else:
            current = float(val)

        improved = False
        if self.mode == "min":
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if self.best is None or improved:
            self.best = current
            self.wait = 0
            best_path = os.path.join(self.ckpt_dir, "best.ckpt")
            trainer.save_checkpoint(best_path)
            self.best_ckpt_path = best_path
            if self.verbose:
                print(f"[FineTuneEarlyStoppingPL] Improvement detected ({self.monitor}={self.best:.6f}). Saved checkpoint: {best_path}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"[FineTuneEarlyStoppingPL] No improvement ({self.monitor}={current:.6f}). wait={self.wait}/{self.patience}")
            if self.wait >= self.patience:
                if self.verbose:
                    print("[FineTuneEarlyStoppingPL] Patience exhausted. Requesting early stop.")
                # Request stop in a trainer-compatible way
                trainer.should_stop = True

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.restore_best_weights and self.best_ckpt_path is not None and os.path.exists(self.best_ckpt_path):
            if self.verbose:
                print(f"[FineTuneEarlyStoppingPL] Restoring best checkpoint from {self.best_ckpt_path}")
            ckpt = torch.load(self.best_ckpt_path, map_location=pl_module.device)
            state_dict = ckpt.get("state_dict", ckpt)
            # load state dict (allow partial/strict=False to be robust)
            pl_module.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print("[FineTuneEarlyStoppingPL] Best weights restored onto LightningModule.")
        else:
            if self.verbose:
                print("[FineTuneEarlyStoppingPL] No best checkpoint found to restore.")
