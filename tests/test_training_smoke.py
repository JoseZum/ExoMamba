"""Smoke test end-to-end del training loop con configs/smoke.yaml.

Si los datos preprocesados o splits no existen, el test se omite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from exoplanet.training import load_config, run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
TRAIN_CSV = REPO_ROOT / "data" / "splits" / "train_tics.csv"
PROCESSED = REPO_ROOT / "data" / "processed" / "global"


@pytest.fixture(scope="module")
def smoke_summary(tmp_path_factory) -> dict:
    if not SMOKE_CONFIG.exists():
        pytest.skip(f"No existe {SMOKE_CONFIG}")
    if not TRAIN_CSV.exists():
        pytest.skip(f"No existe {TRAIN_CSV}; corré scripts/make_splits.py primero.")
    if not PROCESSED.exists() or not any(PROCESSED.glob("*.pt")):
        pytest.skip(f"No hay .pt en {PROCESSED}; corré scripts/preprocess_global.py primero.")

    cfg = load_config(SMOKE_CONFIG)
    # Redirigimos output_dir a un tmp para no ensuciar experiments/
    cfg["experiment"]["output_dir"] = str(tmp_path_factory.mktemp("smoke_exp"))
    return run_training(cfg)


def test_smoke_devuelve_run_dir(smoke_summary: dict) -> None:
    assert "run_dir" in smoke_summary
    run_dir = Path(smoke_summary["run_dir"])
    assert run_dir.exists()


def test_smoke_artefactos_creados(smoke_summary: dict) -> None:
    run_dir = Path(smoke_summary["run_dir"])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env_info.txt").exists()
    assert (run_dir / "git_info.txt").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()


def test_smoke_metrics_csv_tiene_columnas_esperadas(smoke_summary: dict) -> None:
    run_dir = Path(smoke_summary["run_dir"])
    df = pd.read_csv(run_dir / "metrics.csv")
    expected = {"epoch", "train_loss", "val_loss", "val_auc_roc", "val_f1", "lr"}
    assert expected.issubset(df.columns)
    assert len(df) >= 1  # al menos 1 epoch
