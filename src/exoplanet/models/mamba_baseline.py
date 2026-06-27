"""Mamba puro para clasificación binaria de curvas de luz (Fase 8 - Tier 1).

Arquitectura:

  Input (B, 1, L)
    │
    transpose          →  (B, L, 1)
    Linear(1 → D)      →  (B, L, D)         embedding por timestep
    │
    [Mamba block + residual + norm] × N_LAYERS
    │
    mean over L        →  (B, D)            pool global
    Linear(D → 1)      →  (B, 1)
    squeeze            →  (B,)              logits para BCEWithLogitsLoss

Notas operativas:

  - mamba-ssm SOLO funciona en Linux/WSL2 con CUDA + nvcc. Por eso el
    import se hace dentro de __init__ y NO al cargar el módulo. Esto
    permite que el resto del paquete (Dataset, training loop, CNN) se
    pueda importar en Windows nativo sin que mamba-ssm aún esté
    instalado.

  - El input se estandariza por muestra (mismo fix que CNN v2): cada
    curva queda con mean=0, std=1 antes de la primera Linear. Esto
    amplifica la señal relativa del tránsito.

  - Diseñado para batch_size=16 + L=18000 en RTX 3050 (4 GB VRAM).
    Si OOM: bajar `d_model` o activar `gradient_checkpointing` desde
    el YAML.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from exoplanet.models.base import BaseModel


class MambaBaseline(BaseModel):
    """Mamba puro sobre vista global. Para Fase 8."""

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        standardize: bool = True,
        input_offset: float = 1.0,
    ) -> None:
        """
        Args:
            in_channels: canales del input (1 para vista global).
            d_model: dimensión interna del Mamba. Chico para caber en 4 GB.
            d_state: dimensión del state space de cada bloque Mamba.
            d_conv: kernel de la convolución 1D interna de Mamba.
            expand: factor de expansión del MLP interno.
            n_layers: número de bloques Mamba apilados con residual.
            dropout: dropout entre bloques y antes de la cabeza.
            standardize: si True, z-score por muestra antes del embedding.
            input_offset: si standardize=False, se resta este valor (default 1.0).
        """
        super().__init__()

        # Import LAZY: solo cuando se instancia. Permite que el paquete cargue
        # en Windows nativo sin tener mamba-ssm.
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "mamba-ssm no está instalado. Mamba solo funciona en Linux/WSL2.\n"
                "Instalación en WSL2 Ubuntu 24.04:\n"
                "  pip install causal-conv1d\n"
                "  pip install mamba-ssm\n"
                f"Error original: {e}"
            ) from e

        self.standardize = standardize
        self.input_offset = input_offset

        self.embed = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = batch["global_view"]                  # (B, 1, L)

        # Centrar / estandarizar (igual que CNN v4)
        if self.standardize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std
        else:
            x = x - self.input_offset

        # (B, 1, L) -> (B, L, 1) -> (B, L, D)
        x = x.transpose(1, 2)
        h = self.embed(x)

        # Stack de bloques Mamba con residual + LayerNorm
        for mamba, ln in zip(self.layers, self.layer_norms, strict=True):
            h = ln(h + mamba(h))

        # Pool global por mean sobre la secuencia
        h = self.final_norm(h.mean(dim=1))        # (B, D)
        h = self.dropout(h)
        logits = self.head(h)                     # (B, 1)
        return logits.squeeze(-1)                 # (B,)
