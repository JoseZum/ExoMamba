"""Pruebas de set_seed: misma semilla → misma secuencia aleatoria."""

from __future__ import annotations

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
