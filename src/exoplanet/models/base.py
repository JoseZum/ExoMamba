"""Interfaz común para todos los modelos del proyecto.

Contrato: cada modelo recibe el batch como dict (lo que devuelve el collate)
y produce logits de forma (B,) para clasificación binaria con BCEWithLogitsLoss.

Tier 1 (CNN, Mamba puro) lee solo `global_view`.
Tier 2 (ExoMamba V1/V2) leerá además `local_view` y `scalar_features`.
La firma no cambia entre tiers - cambia qué llaves usa cada implementación.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """ABC para todos los modelos de exoplaneta."""

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Recibe un batch dict y devuelve logits de forma (B,)."""
        ...
