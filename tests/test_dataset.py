"""Pruebas básicas para LightCurveDataset (Fase 4 — Tier 1 y Tier 2).

Si los splits o los .pt aún no existen en este entorno, los tests se omiten
(no fallan). Esto permite que el repositorio clonado en limpio pase pytest sin
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
LOCAL_PROCESSED = REPO_ROOT / "data" / "processed" / "local"
TIER2_TRAIN_CSV = REPO_ROOT / "data" / "splits" / "tier2_train_tics.csv"


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


@pytest.fixture(scope="module")
def dataset_with_local() -> LightCurveDataset:
    if not TRAIN_CSV.exists():
        pytest.skip(f"No existe {TRAIN_CSV}.")
    if not PROCESSED.exists() or not any(PROCESSED.glob("*.pt")):
        pytest.skip(f"No hay .pt globales en {PROCESSED}.")
    if not LOCAL_PROCESSED.exists() or not any(LOCAL_PROCESSED.glob("*.pt")):
        pytest.skip(f"No hay .pt locales en {LOCAL_PROCESSED}; corré scripts/preprocess_local.py.")
    return LightCurveDataset(
        TRAIN_CSV, processed_dir=PROCESSED, local_dir=LOCAL_PROCESSED
    )


def test_local_view_optional_load(dataset_with_local: LightCurveDataset) -> None:
    """Si local_dir está seteado y existe el .pt local, debe devolver tensor (1, 201).

    Si NO existe el .pt local para un TIC dado, local_view debe ser None
    (no debe levantar excepción).
    """
    found_loaded = False
    for i in range(min(len(dataset_with_local), 50)):
        sample = dataset_with_local[i]
        lv = sample["local_view"]
        if lv is not None:
            assert torch.is_tensor(lv)
            assert lv.dtype == torch.float32
            assert lv.shape == (1, 201)
            found_loaded = True
            break
    assert found_loaded, "Ningún TIC del split train tiene local_view procesado"


def test_tier2_splits_all_load_local(dataset_with_local: LightCurveDataset) -> None:
    """En splits Tier 2 todos los TICs deben tener local_view."""
    if not TIER2_TRAIN_CSV.exists():
        pytest.skip(f"No existe {TIER2_TRAIN_CSV}; corré scripts/make_tier2_splits.py.")
    ds = LightCurveDataset(
        TIER2_TRAIN_CSV, processed_dir=PROCESSED, local_dir=LOCAL_PROCESSED
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["local_view"] is not None
    assert sample["local_view"].shape == (1, 201)
