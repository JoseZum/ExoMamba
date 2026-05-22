"""Early stopping por métrica de validación con paciencia configurable."""

from __future__ import annotations

import math


class EarlyStopping:
    """Detiene el entrenamiento si la métrica no mejora durante `patience` epochs.

    Args:
        metric: nombre de la métrica a vigilar (solo para logs).
        patience: cuántos epochs sin mejora se toleran.
        mode: "max" (mayor es mejor: AUC, F1, ...) o "min" (menor es mejor: loss).
        min_delta: mejora mínima para contar como mejora.
    """

    def __init__(
        self,
        metric: str = "val_auc",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode='{mode}' inválido (max o min)")
        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0
        self.stopped = False

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Registra la métrica del epoch actual. Devuelve True si mejoró."""
        if math.isnan(value):
            self.bad_epochs += 1
        elif self._is_better(value):
            self.best = value
            self.bad_epochs = 0
            return True
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            self.stopped = True
        return False
