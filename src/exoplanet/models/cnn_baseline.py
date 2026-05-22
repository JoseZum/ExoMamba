"""CNN 1D inspirada en AstroNet (Shallue & Vanderburg, 2018).

Arquitectura simple sobre la vista global (1, 18000):

  Conv1d(1 → 16, k=5) + Norm + ReLU + AvgPool(2)   →  (B, 16, 9000)
  Conv1d(16 → 32, k=5) + Norm + ReLU + AvgPool(2)  →  (B, 32, 4500)
  Conv1d(32 → 64, k=5) + Norm + ReLU + AvgPool(2)  →  (B, 64, 2250)
  Conv1d(64 → 128, k=5) + Norm + ReLU + AvgPool(2) →  (B, 128, 1125)
  AdaptiveAvgPool1d(1)                              →  (B, 128)
  Linear(128 → 64) + ReLU + Dropout                 →  (B, 64)
  Linear(64 → 1)                                    →  (B, 1)
  squeeze                                           →  (B,)

Salida: logits para BCEWithLogitsLoss. No aplica sigmoide.

Este NO es una reproducción de AstroNet (que usa dos ramas: global + local).
Es la versión simplificada que vive en Tier 1 (solo vista global).

Decisiones tras debug:

  - Pooling: AvgPool en vez de MaxPool. La señal de interés es una BAJADA del
    flujo, y MaxPool elige el valor más alto de cada ventana — literalmente
    descarta los puntos del tránsito a favor de los puntos sin tránsito.
    AvgPool preserva la bajada porque el promedio de la región baja cuando
    hay tránsito.

  - Centrado: se resta `input_offset` al input antes de la primera Conv1d.
    Las curvas viven alrededor de 1.0 (mediana normalizada). Restar 1.0
    convierte la señal en desviaciones desde 0 — convención estándar para
    redes neuronales, da gradientes iniciales más sanos.

  - Normalización: GroupNorm por default en vez de BatchNorm.
    BatchNorm con batch_size=16 sobre dataset de ~1.100 ejemplos es
    inestable: cada batch ve pocas curvas y las running stats de eval
    divergen de las stats de train. GroupNorm normaliza por grupo de
    canales DENTRO de cada muestra, sin depender del batch.
    Configurable vía `norm: "batch" | "group"` para ablations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from exoplanet.models.base import BaseModel


def _make_norm(norm: str, channels: int, num_groups: int) -> nn.Module:
    """Selector de capa de normalización."""
    if norm == "batch":
        return nn.BatchNorm1d(channels)
    if norm == "group":
        g = min(num_groups, channels)
        while channels % g != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=channels)
    raise ValueError(f"norm='{norm}' no soportado (batch, group)")


def _block(
    in_ch: int,
    out_ch: int,
    kernel: int = 5,
    pool: int = 2,
    norm: str = "group",
    num_groups: int = 8,
) -> nn.Sequential:
    """Bloque Conv1d + Norm + ReLU + AvgPool con padding 'same' para kernel impar."""
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
        _make_norm(norm, out_ch, num_groups),
        nn.ReLU(inplace=True),
        nn.AvgPool1d(pool),
    )


class CNNBaseline(BaseModel):
    """CNN 1D baseline para clasificación binaria de curvas de luz."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        kernel_size: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        input_offset: float = 1.0,
        standardize: bool = False,
        norm: str = "batch",
        num_groups: int = 8,
    ) -> None:
        """
        Args:
            in_channels, channels, kernel_size, hidden_dim, dropout: arquitectura.
            input_offset: valor que se RESTA al input antes de la primera Conv1d.
                Default 1.0 (curvas normalizadas por mediana). Ignorado si
                `standardize=True` porque la estandarización ya recentra.
            standardize: si True, estandariza cada curva del batch a mean=0,
                std=1 ANTES de la primera Conv1d. Esto amplifica la señal
                relativa (un dip del 0.5% queda como una desviación de
                varios sigmas, mucho más detectable que -0.005). Default False
                para mantener compatibilidad con v0/v1.
            norm: tipo de normalización entre conv y ReLU.
                "batch" → BatchNorm1d (default; v1 con GroupNorm fue peor
                porque GN normaliza por muestra y diluye la señal).
                "group" → GroupNorm (disponible para ablation).
            num_groups: número de grupos para GroupNorm. Se ajusta hacia abajo
                si no divide a `channels`. Ignorado si norm="batch".
        """
        super().__init__()
        self.input_offset = input_offset
        self.standardize = standardize
        blocks: list[nn.Module] = []
        prev = in_channels
        for c in channels:
            blocks.append(
                _block(prev, c, kernel=kernel_size, pool=2, norm=norm, num_groups=num_groups)
            )
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = batch["global_view"]                 # (B, 1, L)
        if self.standardize:
            # Per-sample z-score: cada curva queda mean=0, std=1.
            # Amplifica la señal relativa (transit dip del 0.5% pasa de ser
            # una desviación de 0.005 a una de varios sigmas).
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std
        else:
            x = x - self.input_offset            # centrar alrededor de 0
        x = self.features(x)                     # (B, C, L')
        x = self.pool(x)                         # (B, C, 1)
        x = self.head(x)                         # (B, 1)
        return x.squeeze(-1)                     # (B,)
