"""Reproducibilidad: fijar todas las semillas relevantes en un solo lugar."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (incluyendo CUDA).

    Args:
        seed: semilla entera.
        deterministic: si True, fuerza modo determinista en cuDNN. Más lento,
            pero garantiza reproducibilidad bit a bit. Default False (más rápido,
            con pequeñas variaciones numéricas entre corridas).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
