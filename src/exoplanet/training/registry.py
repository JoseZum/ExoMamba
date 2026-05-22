"""Registro de modelos: mapea string del YAML a clase de modelo."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from exoplanet.models.cnn_baseline import CNNBaseline
from exoplanet.models.mamba_baseline import MambaBaseline
from exoplanet.models.random_baseline import RandomBaseline

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn_baseline": CNNBaseline,
    "random_baseline": RandomBaseline,
    "mamba_baseline": MambaBaseline,   # requiere mamba-ssm (WSL2)
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Construye un modelo desde el config `{type, params}`."""
    model_type = cfg.get("type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model.type='{model_type}' no registrado. "
            f"Disponibles: {sorted(MODEL_REGISTRY.keys())}"
        )
    params = cfg.get("params", {}) or {}
    return MODEL_REGISTRY[model_type](**params)
