"""Pruebas del CNN baseline: forward, output shape, parámetros."""

from __future__ import annotations

import torch

from exoplanet.models import CNNBaseline


def _dummy_batch(batch_size: int = 4, length: int = 18000) -> dict:
    return {
        "global_view": torch.randn(batch_size, 1, length),
        "local_view": None,
        "scalar_features": None,
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "tic_id": torch.tensor([1, 2, 3, 4]),
    }


def test_forward_output_shape() -> None:
    model = CNNBaseline()
    batch = _dummy_batch(batch_size=4)
    logits = model(batch)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_backward_pasa() -> None:
    model = CNNBaseline()
    batch = _dummy_batch(batch_size=2)
    logits = model(batch)
    loss = logits.mean()
    loss.backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grads_ok


def test_params_count_razonable() -> None:
    model = CNNBaseline()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # ~30-300K params para esta arquitectura (cabe en 4GB VRAM)
    assert 10_000 < n < 1_000_000


def test_kwargs_arquitectura_funciona() -> None:
    model = CNNBaseline(channels=(8, 16), hidden_dim=16, dropout=0.0)
    batch = _dummy_batch(batch_size=2, length=1000)
    logits = model(batch)
    assert logits.shape == (2,)
