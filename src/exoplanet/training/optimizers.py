"""Construcción del optimizador desde la configuración."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Crea un optimizador (adam/adamw) con los hiperparámetros del config."""
    opt_type = cfg.get("type", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer.type='{opt_type}' no soportado (adam, adamw)")
