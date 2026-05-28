"""
Dataset PyTorch para curvas de luz preprocesadas (Fase 4 — Tier 1 y Tier 2).

Lee un split CSV (train / val / test) generado por scripts/make_splits.py o
scripts/make_tier2_splits.py y, para cada TIC, carga el .pt correspondiente
desde `data/processed/global/<tid>.pt`. Si `local_dir` se pasa, también intenta
cargar `<local_dir>/<tid>.pt` con la vista local phase-folded (Fase 3.b).

Cada __getitem__ devuelve un dict con la interfaz acordada en CLAUDE.md, pensada
para que los modelos de Tier 1 (CNN, Mamba puro) y Tier 2 (ExoMamba V1/V2)
compartan la misma firma:

    {
        "tic_id":          int,
        "global_view":     tensor (1, L)     # L = 18000
        "local_view":      tensor (1, 201) | None    # Tier 2 — opcional
        "scalar_features": None                       # se completará en Tier 2 V2
        "label":           int                        # 0 = FP, 1 = CP
    }

Importante: los None NO los maneja el default_collate de PyTorch. La fase 6/7
(training loop) deberá usar un collate_fn propio o leer solo las claves
necesarias por modelo. Esa es decisión del training loop, no del Dataset.

Reglas:

  - Si `local_dir` es None (default), `local_view` SIEMPRE es None — comportamiento
    Tier 1 inalterado.
  - Si `local_dir` está seteado y existe `<local_dir>/<tid>.pt`, se carga.
  - Si `local_dir` está seteado y NO existe el archivo del TIC, `local_view` es
    None y el caller decide si filtra. Para garantizar consistencia, los splits
    Tier 2 (`data/splits/tier2_*_tics.csv`) ya filtran TICs sin local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from exoplanet.data.augment import Compose


class LightCurveDataset(Dataset):
    """Curvas de luz TESS preprocesadas (vista global L=18000, opcional vista local 201)."""

    def __init__(
        self,
        split_csv: str | Path,
        processed_dir: str | Path = "data/processed/global",
        check_files: bool = True,
        augment: Compose | None = None,
        local_dir: str | Path | None = None,
    ) -> None:
        """Carga la tabla del split y verifica que cada .pt existe.

        Args:
            split_csv: ruta al CSV con columnas (tid, label).
            processed_dir: directorio con los .pt globales por TIC.
            check_files: si True (default), verifica al inicializar que todos
                los .pt globales existen y aborta con error si falta alguno. Esto
                evita fallar a mitad del entrenamiento. Para `local_dir`, los
                archivos faltantes producen `local_view=None` en __getitem__,
                NO se verifican aquí.
            augment: Compose opcional de augmentations a aplicar SOLO al
                `global_view` en cada __getitem__. **Es responsabilidad del
                caller** pasar `augment=None` para val y test — el dataset no
                sabe en qué split está. Si `local_view` o `scalar_features`
                están poblados (Tier 2), el augmentation actual NO los toca.
            local_dir: directorio con los .pt locales por TIC (Fase 3.b /
                Tier 2). Default None — mantiene comportamiento Tier 1.
        """
        self.processed_dir = Path(processed_dir)
        self.local_dir = Path(local_dir) if local_dir is not None else None
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
        self.augment = augment

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

        if self.augment is not None:
            global_view = self.augment(global_view)

        local_view: torch.Tensor | None = None
        if self.local_dir is not None:
            local_path = self.local_dir / f"{tid}.pt"
            if local_path.exists():
                local_payload = torch.load(local_path, weights_only=False)
                lv = local_payload["local_view"]
                if not torch.is_tensor(lv):
                    raise TypeError(f"local_view de TIC {tid} no es tensor: {type(lv)}")
                if lv.dtype != torch.float32:
                    lv = lv.float()
                local_view = lv

        return {
            "tic_id": tid,
            "global_view": global_view,
            "local_view": local_view,
            "scalar_features": None,
            "label": label,
        }
