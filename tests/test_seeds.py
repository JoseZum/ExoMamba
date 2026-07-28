"""Pruebas de set_seed para verificar que sean la misma secuencia aleatoria."""

from __future__ import annotations

import os

import numpy as np
import torch

from exoplanet.utils import set_seed


def test_torch_reproducible() -> None:
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_numpy_reproducible() -> None:
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_seeds_distintas_dan_resultados_distintos() -> None:
    set_seed(1)
    a = torch.randn(10)
    set_seed(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_flag_deterministic_configura_cudnn() -> None:
    """`deterministic=True` tiene que apagar el autotuner de cuDNN, no sólo la semilla.

    Importa para el paper: sin esto, dos corridas con la misma semilla pueden diferir
    en los últimos decimales porque cuDNN elige algoritmos distintos entre corridas.
    """
    previo = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

        set_seed(42, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previo


def test_set_seed_fija_python_hashseed() -> None:
    """PYTHONHASHSEED afecta el orden de iteración de sets, que puede filtrarse a los splits."""
    set_seed(789)
    assert os.environ["PYTHONHASHSEED"] == "789"
