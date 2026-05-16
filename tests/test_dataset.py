"""Prueba básica para LightCurveDataset (Fase 4).

Si los splits o los .pt aún no existen en este entorno, el test se omite
(no falla). Esto permite que el repositorio clonado en limpio pase pytest sin
necesidad de regenerar todo el preprocesamiento.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from exoplanet.data import LightCurveDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO_ROOT / "data" / "splits" / "train_tics.csv"
PROCESSED = REPO_ROOT / "data" / "processed" / "global"


@pytest.fixture(scope="module")
def dataset() -> LightCurveDataset:
    if not TRAIN_CSV.exists():
        pytest.skip(f"No existe {TRAIN_CSV}; corré scripts/make_splits.py primero.")
    if not PROCESSED.exists() or not any(PROCESSED.glob("*.pt")):
        pytest.skip(f"No hay .pt en {PROCESSED}; corré scripts/preprocess_global.py primero.")
    return LightCurveDataset(TRAIN_CSV, processed_dir=PROCESSED)


def test_dataset_non_empty(dataset: LightCurveDataset) -> None:
    assert len(dataset) > 0


def test_sample_schema(dataset: LightCurveDataset) -> None:
    sample = dataset[0]
    assert set(sample.keys()) == {"tic_id", "global_view", "local_view", "scalar_features", "label"}
    assert isinstance(sample["tic_id"], int)
    assert isinstance(sample["label"], int) and sample["label"] in (0, 1)
    assert sample["local_view"] is None
    assert sample["scalar_features"] is None


def test_global_view_shape_and_dtype(dataset: LightCurveDataset) -> None:
    sample = dataset[0]
    gv = sample["global_view"]
    assert torch.is_tensor(gv)
    assert gv.dtype == torch.float32
    assert gv.shape == (1, 18000)
