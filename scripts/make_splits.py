"""
Splits train/val/test por TIC ID (Fase 4 — Tier 1).

Contrato:

  - El split es POR TIC ID, nunca por sector ni por archivo .pt. Cada estrella
    pertenece a un único fold (train, val o test). Esto previene leakage:
    si dos sectores de la misma estrella estuvieran en folds distintos, el
    modelo aprendería las características intrínsecas de la estrella en train
    y las reconocería trivialmente en test.

  - Estratificación por label (CP=1, FP=0). Dado el desbalance ~1:1.6 del
    dataset preprocesado, sin estratificación los folds podrían quedar
    sesgados. Estratificar preserva la proporción de clases en cada fold.

  - Proporciones 70 / 15 / 15 (train / val / test), per la propuesta original
    entregada en Etapa 1 del curso.

  - Seed = 42 (fijo y versionado en el archivo de salida vía nombre del run y
    en BITACORA.md). Para reproducibilidad estricta, no se cambia entre runs.

  - Solo se consideran TICs con un .pt válido en data/processed/global/.
    Si un tid figura en processed_manifest.csv como 'ok' pero el .pt no existe
    en disco (caso patológico), se reporta y se excluye.

Entradas:
  data/splits/processed_manifest.csv  (filtrado a status=ok)
  data/splits/tics_labeled.csv        (label por tid, 1=CP, 0=FP)
  data/processed/global/<tid>.pt      (verificación de existencia)

Salidas (versionadas):
  data/splits/train_tics.csv  (tid, label)
  data/splits/val_tics.csv    (tid, label)
  data/splits/test_tics.csv   (tid, label)

Uso:
  python scripts/make_splits.py
  python scripts/make_splits.py --seed 42 --train 0.70 --val 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_MANIFEST = Path("data/splits/processed_manifest.csv")
LABELS_PATH = Path("data/splits/tics_labeled.csv")
PROCESSED_DIR = Path("data/processed/global")

OUT_TRAIN = Path("data/splits/train_tics.csv")
OUT_VAL = Path("data/splits/val_tics.csv")
OUT_TEST = Path("data/splits/test_tics.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Splits train/val/test por TIC ID.")
    p.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad (default: 42).")
    p.add_argument("--train", type=float, default=0.70, help="Fracción train (default: 0.70).")
    p.add_argument("--val", type=float, default=0.15, help="Fracción val (default: 0.15).")
    return p.parse_args()


def load_eligible_tics() -> pd.DataFrame:
    """Devuelve dataframe con (tid, label) de TICs preprocesados con .pt en disco."""
    if not PROCESSED_MANIFEST.exists():
        sys.exit(f"[ERROR] No existe {PROCESSED_MANIFEST}. Corré preprocess_global.py primero.")
    if not LABELS_PATH.exists():
        sys.exit(f"[ERROR] No existe {LABELS_PATH}.")

    proc = pd.read_csv(PROCESSED_MANIFEST)
    proc["status"] = proc["status"].astype(str).str.strip().str.lower()
    proc_ok = proc[proc["status"] == "ok"][["tid"]].drop_duplicates(subset="tid")

    labels = pd.read_csv(LABELS_PATH)[["tid", "label"]].drop_duplicates(subset="tid")

    df = proc_ok.merge(labels, on="tid", how="inner")

    # Verificar existencia física del .pt — un manifest correcto pero sin archivo
    # en disco (p. ej. archivo borrado) indicaría inconsistencia.
    df["pt_exists"] = df["tid"].apply(lambda t: (PROCESSED_DIR / f"{t}.pt").exists())
    missing = df[~df["pt_exists"]]
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} TICs en manifest sin .pt en disco. Se excluyen.")
        for t in missing["tid"].tolist()[:10]:
            print(f"  - {t}")
    df = df[df["pt_exists"]].drop(columns=["pt_exists"]).reset_index(drop=True)

    if df["label"].isna().any():
        n_na = int(df["label"].isna().sum())
        sys.exit(f"[ERROR] {n_na} TICs sin label tras el merge. Revisar tics_labeled.csv.")

    df["label"] = df["label"].astype(int)
    return df


def stratified_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split en dos pasos:

      1) train (train_frac) vs temp (1 - train_frac), estratificado por label.
      2) val vs test del 'temp', estratificado por label.

    val_frac se interpreta sobre el total (no sobre temp), entonces dentro de
    temp la proporción de val es val_frac / (1 - train_frac).
    """
    assert 0 < train_frac < 1, "train_frac fuera de rango"
    assert 0 < val_frac < 1 - train_frac, "val_frac fuera de rango"

    temp_frac = 1 - train_frac
    val_within_temp = val_frac / temp_frac

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_frac,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_within_temp,
        stratify=temp_df["label"],
        random_state=seed,
        shuffle=True,
    )

    return (
        train_df.sort_values("tid").reset_index(drop=True),
        val_df.sort_values("tid").reset_index(drop=True),
        test_df.sort_values("tid").reset_index(drop=True),
    )


def report(name: str, df: pd.DataFrame, total: int) -> None:
    n = len(df)
    n_cp = int((df["label"] == 1).sum())
    n_fp = int((df["label"] == 0).sum())
    pct = 100 * n / total
    ratio = (n_fp / n_cp) if n_cp > 0 else float("inf")
    print(
        f"  {name:<6} | n={n:>4} ({pct:5.2f}%) | "
        f"CP={n_cp:>4} | FP={n_fp:>4} | ratio FP:CP = {ratio:.2f}:1"
    )


def assert_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    s_train = set(train["tid"])
    s_val = set(val["tid"])
    s_test = set(test["tid"])
    if s_train & s_val:
        sys.exit(f"[ERROR] {len(s_train & s_val)} TICs en train ∩ val")
    if s_train & s_test:
        sys.exit(f"[ERROR] {len(s_train & s_test)} TICs en train ∩ test")
    if s_val & s_test:
        sys.exit(f"[ERROR] {len(s_val & s_test)} TICs en val ∩ test")


def main() -> None:
    args = parse_args()

    df = load_eligible_tics()
    total = len(df)
    n_cp = int((df["label"] == 1).sum())
    n_fp = int((df["label"] == 0).sum())
    print(f"\n[INFO] TICs elegibles: {total} (CP={n_cp}, FP={n_fp})")
    print(f"[INFO] Split: train={args.train}, val={args.val}, test={1 - args.train - args.val:.2f}")
    print(f"[INFO] Seed: {args.seed}\n")

    train_df, val_df, test_df = stratified_split(df, args.train, args.val, args.seed)

    print("=== Distribución por fold ===")
    report("train", train_df, total)
    report("val", val_df, total)
    report("test", test_df, total)
    print()

    assert_no_overlap(train_df, val_df, test_df)
    print("[OK] Sin overlap de TICs entre folds.")

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_df[["tid", "label"]].to_csv(OUT_TRAIN, index=False)
    val_df[["tid", "label"]].to_csv(OUT_VAL, index=False)
    test_df[["tid", "label"]].to_csv(OUT_TEST, index=False)

    print(f"\n[OK] Escritos:")
    print(f"  - {OUT_TRAIN}")
    print(f"  - {OUT_VAL}")
    print(f"  - {OUT_TEST}")
    print("\n[POLÍTICA] test_tics.csv queda SELLADO hasta Fase 9. No tocar en tuning.")


if __name__ == "__main__":
    main()
