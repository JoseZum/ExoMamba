"""Función de collate propia para batches con valores `None` opcionales.

El `default_collate` de PyTorch falla si algún campo del sample es `None`.
Como nuestro Dataset devuelve `local_view=None` y `scalar_features=None` en
Tier 1, necesitamos un collate explícito que los maneje.

Si un campo es None en algún sample, debe ser None en todos los samples del batch, 
sino lanzamos un error para evitar batches mixtos que confunden al modelo.
"""

from __future__ import annotations

from typing import Any

import torch


def _stack_optional(samples: list[dict], key: str) -> torch.Tensor | None:
    presents = [s[key] is not None for s in samples]
    if not any(presents):
        return None
    if not all(presents):
        n_present = sum(presents)
        raise ValueError(
            f"Batch mixto en '{key}': {n_present}/{len(samples)} samples lo tienen. "
            f"Debe ser todos o ninguno."
        )
    return torch.stack([s[key] for s in samples])


def collate_lightcurves(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Combina samples de LightCurveDataset en un batch."""
    if not batch:
        raise ValueError("Batch vacío")
    return {
        "tic_id": torch.tensor([s["tic_id"] for s in batch], dtype=torch.long),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.float32),
        "global_view": torch.stack([s["global_view"] for s in batch]),
        "local_view": _stack_optional(batch, "local_view"),
        "scalar_features": _stack_optional(batch, "scalar_features"),
    }
