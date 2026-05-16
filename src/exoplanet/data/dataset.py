"""
Dataset PyTorch para curvas de luz preprocesadas (Fase 4 — Tier 1).

Lee un split CSV (train / val / test) generado por scripts/make_splits.py y, para
cada TIC, carga el .pt correspondiente desde data/processed/global/<tid>.pt.

Cada __getitem__ devuelve un dict con la interfaz acordada en CLAUDE.md, pensada
para que los modelos de Tier 1 (CNN, Mamba puro) y Tier 2 (ExoMamba V1/V2)
compartan la misma firma:

    {
        "tic_id":          int,
        "global_view":     tensor (1, L)     # L = 18000
        "local_view":      None              # se completará en Tier 2 (Fase 3.b)
        "scalar_features": None              # se completará en Tier 2 (Fase 3.b)
        "label":           int               # 0 = FP, 1 = CP
    }

Importante: los None NO los maneja el default_collate de PyTorch. La fase 6/7
(training loop) deberá usar un collate_fn propio o leer solo las claves
necesarias por modelo. Esa es decisión del training loop, no del Dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class LightCurveDataset(Dataset):
    """Curvas de luz TESS preprocesadas (vista global, longitud fija L=18000)."""

    def __init__(
        self,
        split_csv: str | Path,
        processed_dir: str | Path = "data/processed/global",
        check_files: bool = True,
    ) -> None:
        """Carga la tabla del split y verifica que cada .pt existe.

        Args:
            split_csv: ruta al CSV con columnas (tid, label).
            processed_dir: directorio con los .pt por TIC.
            check_files: si True (default), verifica al inicializar que todos
                los .pt existen y aborta con error si falta alguno. Esto evita
                fallar a mitad del entrenamiento.
        """
        self.processed_dir = Path(processed_dir)
        split_path = Path(split_csv)
        if not split_path.exists():
            raise FileNotFoundError(f"Split CSV no encontrado: {split_path}")

        df = pd.read_csv(split_path)
        if not {"tid", "label"}.issubset(df.columns):
            raise ValueError(
                f"Split CSV debe tener columnas (tid, label); tiene {list(df.columns)}"
            )
        self.tids: list[int] = df["tid"].astype(int).tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()

        if check_files:
            missing = [t for t in self.tids if not (self.processed_dir / f"{t}.pt").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} .pt faltantes en {self.processed_dir}. "
                    f"Ejemplos: {missing[:5]}"
                )

    def __len__(self) -> int:
        return len(self.tids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tid = self.tids[idx]
        label = self.labels[idx]
        payload = torch.load(self.processed_dir / f"{tid}.pt", weights_only=False)

        global_view = payload["global_view"]
        if not torch.is_tensor(global_view):
            raise TypeError(f"global_view de TIC {tid} no es tensor: {type(global_view)}")
        if global_view.dtype != torch.float32:
            global_view = global_view.float()

        return {
            "tic_id": tid,
            "global_view": global_view,
            "local_view": None,
            "scalar_features": None,
            "label": label,
        }
