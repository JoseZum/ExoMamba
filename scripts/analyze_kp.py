"""Análisis de KP (Known Planets) y otras categorías del TOI catalog.

Pregunta 1: ¿Cuántos KP hay y son compatibles con los CP?
Pregunta 2: ¿De los samples del catálogo, cuántos ya tenemos descargados
            y procesados pero no estamos usando?

NO modifica nada. Solo lee y reporta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG_PATH = Path("data/raw/toi_catalog.csv")
LABELED_PATH = Path("data/splits/tics_labeled.csv")
MANIFEST_PATH = Path("data/splits/manifest.csv")
PROCESSED_MANIFEST_PATH = Path("data/splits/processed_manifest.csv")
PROCESSED_DIR = Path("data/processed/global")


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> int:
    if not CATALOG_PATH.exists():
        sys.exit(f"No existe {CATALOG_PATH}. Corré scripts/get_data.py.")

    # ----- 1) Inventario del catálogo TOI -----
    section("1) Inventario completo del catálogo TOI")
    cat = pd.read_csv(CATALOG_PATH, low_memory=False)
    print(f"Total filas: {len(cat):,}")
    print(f"TICs únicos: {cat['tid'].nunique():,}")
    print()
    print("Distribución por tfopwg_disp:")
    disp_counts = cat["tfopwg_disp"].value_counts(dropna=False)
    for d, n in disp_counts.items():
        print(f"  {str(d):8s}  {n:>6,}  ({100 * n / len(cat):4.1f}%)")

    # TICs únicos por disposición
    print()
    print("TICs únicos por disposición:")
    for d in disp_counts.index:
        if pd.isna(d):
            sub = cat[cat["tfopwg_disp"].isna()]
        else:
            sub = cat[cat["tfopwg_disp"] == d]
        print(f"  {str(d):8s}  {sub['tid'].nunique():>6,} TICs únicos")

    # ----- 2) Estado actual: qué estamos usando -----
    section("2) Estado actual de lo que estamos usando")
    if not LABELED_PATH.exists():
        sys.exit(f"No existe {LABELED_PATH}")
    labeled = pd.read_csv(LABELED_PATH)
    print(f"tics_labeled.csv: {len(labeled):,} filas, {labeled['tid'].nunique():,} TICs únicos")
    print()
    print("Distribución actual en tics_labeled.csv:")
    if "tfopwg_disp" in labeled.columns:
        for d, n in labeled["tfopwg_disp"].value_counts(dropna=False).items():
            print(f"  {str(d):8s}  {n:>6,}")
    print()
    print("Label distribution:")
    for lbl, n in labeled["label"].value_counts(dropna=False).items():
        print(f"  label={lbl}  {n:>6,}")

    # ----- 3) ¿Cuántos KP hay y cuáles ya tenemos descargados? -----
    section("3) KP (Known Planets) — cuántos y si ya los tenemos")
    kp_in_catalog = cat[cat["tfopwg_disp"] == "KP"]
    kp_tics = set(kp_in_catalog["tid"].astype(int).unique())
    print(f"KP únicos en catálogo: {len(kp_tics):,}")

    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        manifest["status"] = manifest["status"].astype(str).str.strip().str.lower()
        downloaded_tics = set(
            manifest[manifest["status"] == "ok"]["tid"].astype(int).unique()
        )
        kp_already_downloaded = kp_tics & downloaded_tics
        kp_not_downloaded = kp_tics - downloaded_tics
        print(f"  Descargados (en manifest, status=ok): {len(kp_already_downloaded):,}")
        print(f"  Por descargar: {len(kp_not_downloaded):,}")
    else:
        kp_already_downloaded = set()
        kp_not_downloaded = kp_tics
        print(f"  Manifest no encontrado, asumimos 0 descargados.")

    if PROCESSED_MANIFEST_PATH.exists():
        proc = pd.read_csv(PROCESSED_MANIFEST_PATH)
        proc["status"] = proc["status"].astype(str).str.strip().str.lower()
        processed_tics = set(
            proc[proc["status"] == "ok"]["tid"].astype(int).unique()
        )
        kp_processed = kp_tics & processed_tics
        print(f"  Procesados (en processed_manifest, status=ok): {len(kp_processed):,}")
    else:
        kp_processed = set()
        print(f"  processed_manifest no encontrado, asumimos 0 procesados.")

    if PROCESSED_DIR.exists():
        pt_files = {int(p.stem) for p in PROCESSED_DIR.glob("*.pt")}
        kp_with_pt = kp_tics & pt_files
        print(f"  Con .pt en disco: {len(kp_with_pt):,}")
    else:
        kp_with_pt = set()
        print(f"  processed/global/ no encontrado, asumimos 0 .pt.")

    # ----- 4) Otras categorías que ya tenemos pero no usamos -----
    section("4) Otras dispositivas — ¿hay muestras ya en disco no usadas?")
    if PROCESSED_DIR.exists():
        pt_files = {int(p.stem) for p in PROCESSED_DIR.glob("*.pt")}
        labeled_tics = set(labeled["tid"].astype(int).unique())
        # TICs con .pt pero NO en tics_labeled.csv
        pt_unused = pt_files - labeled_tics
        print(f"TICs con .pt en disco pero NO en tics_labeled.csv: {len(pt_unused):,}")

        if pt_unused:
            # ¿De qué disposición son?
            unused_disp = cat[cat["tid"].isin(pt_unused)]["tfopwg_disp"].value_counts(
                dropna=False
            )
            print("  Distribución por disposición:")
            for d, n in unused_disp.items():
                print(f"    {str(d):8s}  {n:>6,}")
    else:
        pt_unused = set()

    # ----- 5) Comparación de distribuciones CP vs KP -----
    section("5) ¿CP y KP tienen distribuciones parecidas?")
    cp = cat[cat["tfopwg_disp"] == "CP"]
    kp = cat[cat["tfopwg_disp"] == "KP"]
    print(f"n_CP = {len(cp):,}  n_KP = {len(kp):,}")

    features = ["st_tmag", "pl_orbper", "pl_trandep"]
    print()
    print(f"{'feature':<13} {'CP_n':>6} {'CP_med':>10} {'CP_q25':>10} {'CP_q75':>10}"
          f"   {'KP_n':>6} {'KP_med':>10} {'KP_q25':>10} {'KP_q75':>10}")
    print("-" * 100)
    for f in features:
        if f not in cat.columns:
            print(f"{f:<13}  (columna ausente)")
            continue
        cp_v = cp[f].dropna()
        kp_v = kp[f].dropna()
        if len(cp_v) == 0 or len(kp_v) == 0:
            print(f"{f:<13}  (sin datos)")
            continue

        def fmt(x: float) -> str:
            if abs(x) >= 1000:
                return f"{x:10.0f}"
            if abs(x) >= 1:
                return f"{x:10.3f}"
            return f"{x:10.4f}"

        print(
            f"{f:<13} {len(cp_v):>6,d} {fmt(cp_v.median())} {fmt(cp_v.quantile(0.25))} {fmt(cp_v.quantile(0.75))}"
            f"   {len(kp_v):>6,d} {fmt(kp_v.median())} {fmt(kp_v.quantile(0.25))} {fmt(kp_v.quantile(0.75))}"
        )

    # Test Kolmogorov-Smirnov para detectar diferencias estadísticas
    section("6) Test Kolmogorov-Smirnov (KS): ¿CP y KP vienen de la misma distribución?")
    try:
        from scipy import stats

        print("p-valor alto (>0.05) -> no podemos rechazar misma distribución (compatibles).")
        print("p-valor bajo (<0.05) -> CP y KP vienen de distribuciones distintas.")
        print()
        for f in features:
            if f not in cat.columns:
                continue
            cp_v = cp[f].dropna()
            kp_v = kp[f].dropna()
            if len(cp_v) < 10 or len(kp_v) < 10:
                continue
            ks_stat, p_value = stats.ks_2samp(cp_v, kp_v)
            verdict = "compatibles" if p_value > 0.05 else "DISTINTAS"
            print(f"  {f:<13}  KS={ks_stat:.4f}  p={p_value:.4e}  -> {verdict}")
    except ImportError:
        print("scipy no instalado, salto el test KS.")

    # ----- 7) Veredicto -----
    section("7) Veredicto operativo")
    print(f"KP en catálogo:             {len(kp_tics):,}")
    print(f"KP ya descargados:          {len(kp_already_downloaded):,}")
    print(f"KP ya procesados (.pt):     {len(kp_with_pt):,}")
    print(f"KP a descargar nuevos:      {len(kp_not_downloaded):,}")
    print()
    print(f"Samples .pt NO usados:      {len(pt_unused):,}")
    print()
    if len(kp_with_pt) > len(kp_tics) * 0.5:
        print(">>> Hay muchos KP ya en disco. Vale la pena agregarlos sin re-descargar.")
    elif len(kp_with_pt) > 0:
        print(f">>> Hay {len(kp_with_pt)} KP ya en disco, pero faltan {len(kp_not_downloaded)} por descargar.")
    else:
        print(">>> No hay KP en disco. Habría que descargarlos primero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
