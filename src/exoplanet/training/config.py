"""Carga y validación de configs YAML del training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_KEYS = {"experiment", "data", "model", "training"}


def load_config(path: Path | str) -> dict[str, Any]:
    """
    Lee un YAML y verifica claves de primer nivel mínimas.

    No valida el contenido completo pq eso lo hacen las funciones constructoras
    como build_optimizer, build_loss, etc., al consumir cada sección del config.
    
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config no encontrado: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"El config {p} no es un dict YAML")
    missing = REQUIRED_TOP_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Faltan claves en el config: {sorted(missing)}")
    return cfg


def dump_config(cfg: dict[str, Any], path: Path | str) -> None:
    """Guarda el config como YAML (snapshot por corrida)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
