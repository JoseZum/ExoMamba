"""Helpers de paths: nombrar y crear el directorio de cada corrida experimental."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_experiment_dir(base_dir: str | Path, run_name: str) -> Path:
    """Crea experiments/<YYYY-MM-DD_HH-MM-SS>_<run_name>/ y devuelve el path.

    El timestamp asegura que dos corridas con el mismo nombre no se pisen.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = run_name.replace(" ", "_").replace("/", "_")
    out = Path(base_dir) / f"{ts}_{safe_name}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    (out / "tensorboard").mkdir(parents=True, exist_ok=False)
    return out
