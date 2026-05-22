"""Gestor de checkpoints: guarda el mejor por una métrica + el último."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Maneja `best.pt` (mejor por métrica) y `last.pt` (último epoch)."""

    def __init__(
        self,
        ckpt_dir: Path | str,
        metric: str = "val_auc",
        mode: str = "max",
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        if mode not in ("max", "min"):
            raise ValueError(f"mode='{mode}' inválido")
        self.mode = mode
        self.best_value: float = -math.inf if mode == "max" else math.inf
        self.best_epoch: int = -1

    def _is_better(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def maybe_save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if not self._is_better(float(value)):
            return False
        self.best_value = float(value)
        self.best_epoch = epoch
        self._save(self.ckpt_dir / "best.pt", model, optimizer, epoch, metrics)
        return True

    def save_last(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        self._save(self.ckpt_dir / "last.pt", model, optimizer, epoch, metrics)

    @staticmethod
    def _save(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
