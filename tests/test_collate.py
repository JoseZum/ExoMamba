"""Pruebas de collate_lightcurves: batches con None, sin None y mixtos."""

from __future__ import annotations

import pytest
import torch

from exoplanet.training import collate_lightcurves


def _sample(tic: int, label: int, with_local: bool = False, with_scalars: bool = False) -> dict:
    return {
        "tic_id": tic,
        "label": label,
        "global_view": torch.zeros((1, 18000), dtype=torch.float32),
        "local_view": torch.zeros((1, 200), dtype=torch.float32) if with_local else None,
        "scalar_features": torch.zeros((5,), dtype=torch.float32) if with_scalars else None,
    }


def test_batch_con_todos_none() -> None:
    batch = [_sample(1, 1), _sample(2, 0), _sample(3, 1)]
    out = collate_lightcurves(batch)
    assert out["global_view"].shape == (3, 1, 18000)
    assert out["label"].tolist() == [1.0, 0.0, 1.0]
    assert out["tic_id"].tolist() == [1, 2, 3]
    assert out["local_view"] is None
    assert out["scalar_features"] is None


def test_batch_con_local_y_scalars() -> None:
    batch = [
        _sample(1, 1, with_local=True, with_scalars=True),
        _sample(2, 0, with_local=True, with_scalars=True),
    ]
    out = collate_lightcurves(batch)
    assert out["local_view"].shape == (2, 1, 200)
    assert out["scalar_features"].shape == (2, 5)


def test_batch_mixto_lanza_error() -> None:
    batch = [_sample(1, 1, with_local=True), _sample(2, 0, with_local=False)]
    with pytest.raises(ValueError, match="Batch mixto"):
        collate_lightcurves(batch)


def test_batch_vacio_lanza_error() -> None:
    with pytest.raises(ValueError, match="Batch vacío"):
        collate_lightcurves([])
