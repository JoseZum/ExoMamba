"""Baseline aleatorio estratificado (Fase 5.a).

Predice una probabilidad fija basada en el prior de clase del train:
  prob(positivo) = n_pos / (n_pos + n_neg) ≈ 0.383 con nuestro dataset

Es el piso mínimo absoluto contra el que comparar todos los demás modelos.
Si una arquitectura "compleja" no supera consistentemente este baseline,
entonces algo está roto.

Como necesita conocer el prior del train ANTES del primer forward, se le
pasa explícitamente vía constructor (no se calcula dentro del modelo -
eso requeriría romper la abstracción del DataLoader).

Output: el modelo NO entrena (no tiene parámetros entrenables - un Linear
fantasma sin uso para que el optimizer no se queje); siempre devuelve el
mismo logit (logit(prior)).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from exoplanet.models.base import BaseModel


class RandomBaseline(BaseModel):
    """Devuelve siempre logit(prior). AUC esperado: 0.5."""

    def __init__(self, prior_positive: float = 0.383) -> None:
        super().__init__()
        if not 0.0 < prior_positive < 1.0:
            raise ValueError(f"prior_positive fuera de (0, 1): {prior_positive}")
        # Calculamos el logit que dará sigmoid(logit) = prior.
        logit = math.log(prior_positive / (1 - prior_positive))
        # Parámetro "fantasma" sin uso real - Adam exige al menos 1 param entrenable.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
        # El logit base se guarda como buffer (no se actualiza con gradient).
        self.register_buffer("base_logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size = batch["global_view"].shape[0]
        # Multiplicamos por 0 a `dummy` para que aparezca en el grafo de cómputo
        # sin afectar el valor (necesario para que loss.backward() no falle).
        out = self.base_logit + 0.0 * self.dummy.sum()
        return out.expand(batch_size).clone()
