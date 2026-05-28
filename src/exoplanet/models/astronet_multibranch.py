"""AstroNet multibranch — reproducción fiel de Shallue & Vanderburg (2018).

Referencia: Shallue, C. J. & Vanderburg, A. (2018).
  "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain
  around Kepler-80 and an Eighth Planet around Kepler-90."
  AJ, 155(2), 94. https://arxiv.org/pdf/1712.05044
  (ver Tabla 2: "Final architecture of the convolutional neural network")

Arquitectura (dos ramas + cabeza fully-connected):

  global_view (B, 1, 18000)             local_view (B, 1, 201)
       │                                        │
  [5 bloques globales]                    [2 bloques locales]
   cada bloque = Conv1d ×2                cada bloque = Conv1d ×2
                + MaxPool(5, s=2)                     + MaxPool(7, s=2)
   canales: 16→32→64→128→256              canales: 16→32
       │                                        │
   AdaptiveAvgPool1d(1)                   AdaptiveAvgPool1d(1)
   → (B, 256)                             → (B, 32)
       └───────────── concat ────────────────────┘
                       │
                  (B, 288)
                       │
              Linear(288 → 512) + ReLU + Dropout
              Linear(512 → 512) + ReLU + Dropout
              Linear(512 → 512) + ReLU + Dropout
              Linear(512 → 1)
                       │
                  squeeze → (B,)  logits

Adaptaciones respecto al paper original:

  - Longitud global: 18000 (TESS, ~27 días a 2 min) vs 2001 (Kepler, paper).
  - Longitud local: 201 (igual al paper, ventana phase-folded centrada).
  - AdaptiveAvgPool1d(1) al final de cada rama (en vez de Flatten directo):
    con secuencia global de 18000 → 256 canales, un Flatten daría un FC
    de entrada gigantesco. El paper original con L=2001 termina con una
    salida razonable; nosotros con L=18000 necesitamos colapsar
    explícitamente a (B, C). Es la única adaptación a la arquitectura
    fuera de la longitud de entrada.
  - Kernels y poolings: idénticos al paper (k=5 conv, MaxPool 5/2 global,
    MaxPool 7/2 local).
  - Cabeza FC: 4 capas (512→512→512→1) con ReLU + Dropout(0.3), como en
    el paper.

Notas operativas:

  - Modelo "pesado" en cómputo y VRAM por los 256 canales en las últimas
    capas globales y los FC de 512. Pensado para `batch_size=8` + FP16 +
    gradient_checkpointing en RTX 3050 (4 GB). Sin FP16 da OOM.
  - Pure-torch: corre en Windows nativo (no requiere mamba-ssm/WSL2).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from exoplanet.models.base import BaseModel


def _conv_pair(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    """Par de Conv1d + ReLU (sin BN, fiel al paper) con padding 'same'."""
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad),
        nn.ReLU(inplace=True),
    )


class AstroNetMultibranch(BaseModel):
    """Reproducción fiel de AstroNet (Shallue & Vanderburg, 2018) adaptada a TESS."""

    def __init__(
        self,
        global_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        local_channels: tuple[int, ...] = (16, 32),
        kernel: int = 5,
        pool_kernel_global: int = 5,
        pool_kernel_local: int = 7,
        pool_stride: int = 2,
        fc_hidden: tuple[int, ...] = (512, 512, 512),
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            global_channels: canales de cada bloque de la rama global (5 por default).
            local_channels: canales de cada bloque de la rama local (2 por default).
            kernel: tamaño del kernel de las Conv1d (5 en el paper).
            pool_kernel_global: kernel del MaxPool en la rama global (5).
            pool_kernel_local: kernel del MaxPool en la rama local (7).
            pool_stride: stride de los MaxPools (2 en el paper).
            fc_hidden: dimensiones de los FC ocultos en la cabeza (512, 512, 512).
            dropout: dropout aplicado entre cada FC oculto (0.3).
        """
        super().__init__()

        # --- Rama global: 5 bloques (Conv ×2 + MaxPool) ---
        global_blocks: list[nn.Module] = []
        prev = 1
        for c in global_channels:
            global_blocks.append(_conv_pair(prev, c, kernel=kernel))
            global_blocks.append(nn.MaxPool1d(kernel_size=pool_kernel_global, stride=pool_stride))
            prev = c
        self.global_branch = nn.Sequential(*global_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        global_out_dim = global_channels[-1]

        # --- Rama local: 2 bloques (Conv ×2 + MaxPool) ---
        local_blocks: list[nn.Module] = []
        prev = 1
        for c in local_channels:
            local_blocks.append(_conv_pair(prev, c, kernel=kernel))
            local_blocks.append(nn.MaxPool1d(kernel_size=pool_kernel_local, stride=pool_stride))
            prev = c
        self.local_branch = nn.Sequential(*local_blocks)
        self.local_pool = nn.AdaptiveAvgPool1d(1)
        local_out_dim = local_channels[-1]

        # --- Cabeza FC: 4 capas (3 ocultas con ReLU+Dropout + 1 final) ---
        fused_dim = global_out_dim + local_out_dim
        fc_layers: list[nn.Module] = []
        prev_dim = fused_dim
        for h in fc_hidden:
            fc_layers.append(nn.Linear(prev_dim, h))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = h
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        global_view = batch["global_view"]
        local_view = batch.get("local_view")
        if local_view is None:
            raise ValueError(
                "AstroNetMultibranch requires local_view; train with Tier 2 splits "
                "(data/splits/tier2_*.csv) and configure data.local_dir."
            )

        g = self.global_branch(global_view)      # (B, C_g, L_g')
        g = self.global_pool(g).squeeze(-1)      # (B, C_g)

        l = self.local_branch(local_view)        # (B, C_l, L_l')
        l = self.local_pool(l).squeeze(-1)       # (B, C_l)

        fused = torch.cat([g, l], dim=-1)        # (B, C_g + C_l)
        logits = self.head(fused)                # (B, 1)
        return logits.squeeze(-1)                # (B,)
