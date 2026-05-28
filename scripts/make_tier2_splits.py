"""
Genera splits Tier 2 a partir de los splits Tier 1 + manifiesto local.

Tier 2 es un SUBSET ESTRICTO de Tier 1: ningún TIC cambia de split (eso sería
data leakage al re-asignar). Lo único que cambia es que se filtran los TICs
sin local_view válido (`status != ok` en processed_local_manifest.csv).

Lee:
    data/splits/{train,val,test}_tics.csv
    data/splits/processed_local_manifest.csv

Escribe:
    data/splits/tier2_{train,val,test}_tics.csv (mismas cols que Tier 1: tid, label)
    Stats por split y por clase a stdout.

Uso:
    python scripts/make_tier2_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SPLITS_DIR = Path("data/splits")
LOCAL_MANIFEST = SPLITS_DIR / "processed_local_manifest.csv"
SPLITS_IN = {
    "train": SPLITS_DIR / "train_tics.csv",
    "val": SPLITS_DIR / "val_tics.csv",
    "test": SPLITS_DIR / "test_tics.csv",
}
SPLITS_OUT = {
    "train": SPLITS_DIR / "tier2_train_tics.csv",
    "val": SPLITS_DIR / "tier2_val_tics.csv",
    "test": SPLITS_DIR / "tier2_test_tics.csv",
}


def main() -> int:
    if not LOCAL_MANIFEST.exists():
        sys.exit(
            f"No existe {LOCAL_MANIFEST}. Corré scripts/preprocess_local.py primero."
        )
    manifest = pd.read_csv(LOCAL_MANIFEST)
    ok_tids = set(manifest.loc[manifest["status"].astype(str).str.lower() == "ok", "tid"].astype(int))
    print(f"TICs con local válido (status=ok): {len(ok_tids):,}")

    print(f"\n{'split':<6} {'N Tier1':>8} {'N Tier2':>8} "
          f"{'CP T1':>6} {'FP T1':>6} {'CP T2':>6} {'FP T2':>6} "
          f"{'CP loss':>8} {'FP loss':>8}")
    for name, src in SPLITS_IN.items():
        if not src.exists():
            sys.exit(f"No existe {src}.")
        df_t1 = pd.read_csv(src)
        df_t2 = df_t1[df_t1["tid"].astype(int).isin(ok_tids)].copy()
        # Preservar el orden y columnas originales (al menos tid, label).
        cols = [c for c in ("tid", "label") if c in df_t2.columns]
        df_t2 = df_t2[cols].reset_index(drop=True)
        SPLITS_OUT[name].parent.mkdir(parents=True, exist_ok=True)
        df_t2.to_csv(SPLITS_OUT[name], index=False)

        n_t1 = len(df_t1)
        n_t2 = len(df_t2)
        cp_t1 = int((df_t1["label"] == 1).sum())
        fp_t1 = int((df_t1["label"] == 0).sum())
        cp_t2 = int((df_t2["label"] == 1).sum())
        fp_t2 = int((df_t2["label"] == 0).sum())
        cp_loss = cp_t1 - cp_t2
        fp_loss = fp_t1 - fp_t2
        print(
            f"{name:<6} {n_t1:>8} {n_t2:>8} "
            f"{cp_t1:>6} {fp_t1:>6} {cp_t2:>6} {fp_t2:>6} "
            f"{cp_loss:>8} {fp_loss:>8}"
        )
        print(f"       → {SPLITS_OUT[name]}")

    print("\nNOTA: Tier 2 es subset estricto de Tier 1. Los TICs NO se reasignan "
          "de split (eso sería data leakage). Solo se filtran los TICs sin "
          "local_view procesado (status != ok).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
