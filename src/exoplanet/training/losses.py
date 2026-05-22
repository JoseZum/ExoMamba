"""Construcción de la función de pérdida desde la configuración."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_loss(cfg: dict[str, Any], pos_count: int, neg_count: int) -> nn.Module:
    """Construye una BCEWithLogitsLoss con pos_weight opcional.

    Args:
        cfg: subconfig de loss. Espera `type` y `pos_weight`.
            - type: "bce" (única opción por ahora).
            - pos_weight:
                * "balanced": pos_weight = neg_count / pos_count (compensa desbalance).
                * float: usa ese valor directo.
                * None: sin pesos.
        pos_count: número de ejemplos positivos en train.
        neg_count: número de ejemplos negativos en train.
    """
    loss_type = cfg.get("type", "bce").lower()
    if loss_type != "bce":
        raise ValueError(f"loss.type='{loss_type}' no soportado (solo 'bce')")

    pw_raw = cfg.get("pos_weight", None)
    if pw_raw == "balanced":
        if pos_count == 0:
            raise ValueError("No hay ejemplos positivos para calcular pos_weight balanced")
        pw = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    elif pw_raw is None:
        pw = None
    else:
        pw = torch.tensor([float(pw_raw)], dtype=torch.float32)

    return nn.BCEWithLogitsLoss(pos_weight=pw)
