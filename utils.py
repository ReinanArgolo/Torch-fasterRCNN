from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str | torch.device] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_optimizer(
    params,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> Optimizer:
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(optimizer: Optimizer, step_size: int = 3, gamma: float = 0.1) -> _LRScheduler:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
