"""ExoMamba V1 - Mamba global + CNN local (Tier 2, Fase 10).

Arquitectura híbrida:

  global_view (B, 1, 18000)        local_view (B, 1, 201)
        │                                  │
   [Mamba encoder]                  [CNN local head]
   (reusa mamba_baseline.py)        3 bloques Conv1d + BN + ReLU + MaxPool
        │                                  │
   mean pool → (B, d_model)         AdaptiveAvgPool1d → (B, 64)
        └──────────── concat ──────────────┘
                       │
                  (B, d_model + 64)
                       │
              Linear(→ head_hidden) + ReLU + Dropout
                       │
              Linear(→ 1) → logit (B,)

Notas:

  - El encoder global reusa la lógica de `mamba_baseline.py` (embedding,
    stack de bloques Mamba con residual + LayerNorm, pool por media,
    estandarización por muestra). Se reimplementa internamente (no se
    importa MambaBaseline) para devolver el VECTOR pooled en vez del
    logit; la lógica es idéntica a la de `mamba_baseline.py` hasta antes
    de la cabeza.

  - La rama local es una CNN 1D simple inspirada en la rama local de
    AstroNet (Shallue & Vanderburg, 2018), pero más pequeña: 3 bloques
    Conv1d (16→32→64) con BatchNorm + ReLU + MaxPool(2) cada uno, y un
    AdaptiveAvgPool1d global al final. Suficiente para una secuencia de
    201 puntos.

  - Fusión: late concat (default ATAT-style). FiLM o cross-attention
    quedan como mejora futura si concat satura.

  - mamba-ssm SOLO funciona en Linux/WSL2 con CUDA + nvcc. El import
    está LAZY (dentro de __init__) para que el resto del paquete pueda
    importarse en Windows nativo aunque mamba-ssm no esté instalado.

  - Si el batch no trae `local_view` (es None), el forward levanta
    ValueError. ExoMambaV1 NO es retro-compatible con splits Tier 1.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from exoplanet.models.base import BaseModel


def _local_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    """Bloque Conv1d + BatchNorm + ReLU + MaxPool(2) para la rama local."""
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class ExoMambaV1(BaseModel):
    """Mamba global + CNN local con fusión late concat. Tier 2 - Fase 10."""

    def __init__(
        self,
        # --- Encoder global (Mamba) ---
        in_channels: int = 1,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        mamba_dropout: float = 0.1,
        standardize: bool = True,
        input_offset: float = 1.0,
        # --- Rama local (CNN) ---
        local_channels: tuple[int, ...] = (16, 32, 64),
        local_kernel: int = 5,
        local_dropout: float = 0.0,
        # --- Cabeza de fusión ---
        head_hidden: int = 64,
        head_dropout: float = 0.3,
    ) -> None:
        """
        Args:
            in_channels: canales del input global (1).
            d_model: dimensión interna del Mamba. Default 64 para 4 GB VRAM.
            d_state, d_conv, expand: hiperparámetros internos de cada bloque Mamba.
            n_layers: número de bloques Mamba apilados con residual.
            mamba_dropout: dropout entre bloques y antes del pool del encoder global.
            standardize: si True, z-score por muestra antes del embedding global.
            input_offset: si standardize=False, valor que se resta al input global.
            local_channels: canales de los 3 bloques Conv1d locales (16, 32, 64 por default).
            local_kernel: kernel de las convs locales.
            local_dropout: dropout después del pool global de la rama local.
            head_hidden: dimensión oculta del MLP de fusión.
            head_dropout: dropout dentro del MLP de fusión (default 0.3).
        """
        super().__init__()

        # --- Import LAZY de mamba-ssm ---
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "mamba-ssm no está instalado. ExoMambaV1 (Mamba+local) solo funciona "
                "en Linux/WSL2. Instalación en WSL2 Ubuntu 24.04:\n"
                "  pip install causal-conv1d\n"
                "  pip install mamba-ssm\n"
                f"Error original: {e}"
            ) from e

        # --- Encoder global (reusa lógica de mamba_baseline.py) ---
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
        self.mamba_dropout = nn.Dropout(mamba_dropout)

        # --- Rama local (CNN inspirada en AstroNet-local-branch, S&V 2018) ---
        local_blocks: list[nn.Module] = []
        prev = 1
        for c in local_channels:
            local_blocks.append(_local_block(prev, c, kernel=local_kernel))
            prev = c
        self.local_features = nn.Sequential(*local_blocks)
        self.local_pool = nn.AdaptiveAvgPool1d(1)
        self.local_dropout = nn.Dropout(local_dropout)
        local_out_dim = local_channels[-1]

        # --- Cabeza de fusión (late concat → MLP → logit) ---
        fused_dim = d_model + local_out_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encode_global(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder global (Mamba). Devuelve (B, d_model)."""
        # Centrar / estandarizar (igual que MambaBaseline)
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
        h = self.final_norm(h.mean(dim=1))   # (B, D)
        h = self.mamba_dropout(h)
        return h

    def _encode_local(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder local (CNN). Devuelve (B, local_channels[-1])."""
        h = self.local_features(x)           # (B, C, L')
        h = self.local_pool(h).squeeze(-1)   # (B, C)
        h = self.local_dropout(h)
        return h

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        global_view = batch["global_view"]
        local_view = batch.get("local_view")
        if local_view is None:
            raise ValueError(
                "ExoMambaV1 requires local_view; train with Tier 2 splits "
                "(data/splits/tier2_*.csv) and configure data.local_dir."
            )

        g = self._encode_global(global_view)   # (B, d_model)
        l = self._encode_local(local_view)     # (B, local_channels[-1])
        fused = torch.cat([g, l], dim=-1)      # (B, d_model + local_dim)
        logits = self.head(fused)              # (B, 1)
        return logits.squeeze(-1)              # (B,)
