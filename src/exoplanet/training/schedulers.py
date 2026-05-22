"""Construcción del scheduler de learning rate."""

from __future__ import annotations

from typing import Any

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any] | None
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Crea un scheduler (cosine o ninguno).

    cfg ejemplos:
      {"type": "none"}                   → None
      {"type": "cosine", "t_max": 50}    → CosineAnnealingLR
    """
    if cfg is None:
        return None
    sch_type = cfg.get("type", "none").lower()
    if sch_type == "none":
        return None
    if sch_type == "cosine":
        t_max = int(cfg.get("t_max", 50))
        eta_min = float(cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    raise ValueError(f"scheduler.type='{sch_type}' no soportado (none, cosine)")
