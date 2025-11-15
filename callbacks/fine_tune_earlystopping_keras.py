"""Fine-tuning aware EarlyStopping for tf.keras.

Usage notes:
 - Default monitor: 'val_loss'
 - If prefer_val_mAP=True and 'val_mAP' appears in logs, the callback will switch to 'val_mAP'
 - mode is set automatically: 'min' for losses, 'max' for mAP-like metrics
 - patience default: 10
 - min_delta default: 1e-4
 - restore_best_weights enforced (True)
 - This callback raises a clear error if validation is not enabled or the monitored metric is missing

This file assumes `tensorflow` is available in the environment when imported/used.
"""
from typing import Optional
import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - import error will surface at runtime when used
    tf = None


class FineTuneEarlyStopping(tf.keras.callbacks.Callback if tf is not None else object):
    """EarlyStopping callback optimized for fine-tuning pretrained models.

    Parameters
    ----------
    monitor : Optional[str]
        Metric key to monitor (e.g. 'val_loss' or 'val_mAP'). If None, default is 'val_loss'.
    prefer_val_mAP : bool
        If True and 'val_mAP' exists in validation logs, switch to monitoring 'val_mAP'.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    verbose : int
        Verbosity (0: silent, 1: print messages).

    Behavior
    --------
    - Raises ValueError if validation isn't configured in `model.fit` (no validation pass).
    - Raises ValueError if the monitored metric isn't present in `logs`.
    - Always attempts to restore the best weights at the end of training.
    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        prefer_val_mAP: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: int = 1,
    ):
        if tf is None:
            raise RuntimeError("TensorFlow is required to use FineTuneEarlyStopping. Install 'tensorflow'.")
        super().__init__()
        self.user_monitor = monitor
        self.prefer_val_mAP = bool(prefer_val_mAP)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = int(verbose)

        # enforced behavior
        self.restore_best_weights = True

        # runtime state
        self.monitor = monitor or "val_loss"
        self.mode = None
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self._initialized = False

    def _set_mode_for_monitor(self):
        if "map" in self.monitor.lower():
            self.mode = "max"
            self.best = -np.Inf
        else:
            self.mode = "min"
            self.best = np.Inf

    def on_train_begin(self, logs=None):
        # Keras sets self.params before callbacks are run
        params = getattr(self, "params", {})
        do_val = params.get("do_validation", False)
        if not do_val:
            raise ValueError(
                "FineTuneEarlyStopping requires validation to be run each epoch. "
                "Call model.fit(..., validation_data=..., validation_split>0 or validation_steps=...)."
            )

        self.monitor = self.user_monitor or "val_loss"
        self._set_mode_for_monitor()
        self.wait = 0
        self.best_weights = None
        self._initialized = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # If prefer_val_mAP and val_mAP present, switch monitor automatically
        if self.prefer_val_mAP and "val_mAP" in logs:
            if self.monitor != "val_mAP":
                if self.verbose:
                    print("[FineTuneEarlyStopping] Switching monitor to 'val_mAP' (found in logs).")
                self.monitor = "val_mAP"
                self._set_mode_for_monitor()

        if self.monitor not in logs:
            raise ValueError(
                f"Monitored metric '{self.monitor}' not found in logs for epoch {epoch}. "
                f"Available keys: {list(logs.keys())}. Ensure your validation step computes and logs this metric."
            )

        current = float(logs.get(self.monitor))

        if self.mode == "min":
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if self.best is None or improved:
            self.best = current
            self.wait = 0
            # save best weights (in-memory)
            self.best_weights = self.model.get_weights()
            if self.verbose:
                print(f"[FineTuneEarlyStopping] Epoch {epoch}: {self.monitor} improved to {self.best:.6f}. Saved best weights.")
        else:
            self.wait += 1
            if self.verbose:
                print(f"[FineTuneEarlyStopping] Epoch {epoch}: no improvement ({self.monitor}={current:.6f}). wait={self.wait}/{self.patience}")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print(f"[FineTuneEarlyStopping] Stopping training. Restoring best weights from epoch with {self.monitor}={self.best:.6f}.")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        # If training ended normally, restore best weights as well (helpful for fine-tuning)
        if self.restore_best_weights and getattr(self, "best_weights", None) is not None and getattr(self, "model", None) is not None:
            if self.verbose:
                print("[FineTuneEarlyStopping] Restoring best weights at training end.")
            self.model.set_weights(self.best_weights)
