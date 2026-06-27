"""
Análisis de factibilidad para la vista local phase-folded (Fase 3.b / Tier 2).

Verifica, sobre el catálogo TOI y los splits Tier 1 (train/val/test), cuántos
TICs tienen metadata utilizable para construir `local_view`:

    - period   (pl_orbper)
    - epoch    (pl_tranmid, BJD; se convertirá restando 2457000)
    - duration (pl_trandurh en horas, se convierte a días)

Filtros aplicados (deben coincidir con preprocess_local.py):

    a) los 3 campos válidos (no NaN, finitos, > 0)
    b) period <= 27.4 días (el sector TESS dura ~27 días; periodos mayores no
       garantizan un tránsito completo dentro de un solo sector)
    c) duration < period / 2 (descarta valores degenerados o erróneos del
       catálogo donde la duración reportada es absurdamente grande)

Output:
    - stdout: tabla resumen por split y por clase
    - data/splits/local_view_feasibility.md: mismo reporte en markdown

HARD STOP: si N_train tras filtros < 500, marcar Tier 2 como INVIABLE.

Uso:
    python scripts/analyze_local_view_feasibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path("data/splits")
TOI_SUMMARY = SPLITS_DIR / "toi_summary.csv"
TRAIN_CSV = SPLITS_DIR / "train_tics.csv"
VAL_CSV = SPLITS_DIR / "val_tics.csv"
TEST_CSV = SPLITS_DIR / "test_tics.csv"
OUT_MD = SPLITS_DIR / "local_view_feasibility.md"

PERIOD_MAX_DAYS = 27.4  # 1 sector TESS ~ 27.4 días
HARD_STOP_N_TRAIN = 500


def load_catalog_metadata() -> pd.DataFrame:
    """Lee el catálogo TOI y agrupa por TIC (un TIC puede tener varias TOIs).

    Para cada TIC se queda con la TOI cuya combinación (period, epoch, duration)
    sea válida y tenga la duración reportada más estrecha (criterio heurístico:
    valores más físicos y menos degenerados). Si ninguna TOI del TIC tiene los
    3 válidos, el TIC queda con NaN en los 3 campos.
    """
    df = pd.read_csv(TOI_SUMMARY)
    needed = {"tid", "pl_orbper", "pl_tranmid", "pl_trandurh"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise SystemExit(
            f"toi_summary.csv no tiene las columnas {missing}. "
            "Corré `python scripts/get_data.py` con la versión extendida."
        )

    # Sub-DataFrame con TOIs válidas: los 3 campos no-NaN y > 0.
    valid_mask = (
        df["pl_orbper"].notna()
        & df["pl_tranmid"].notna()
        & df["pl_trandurh"].notna()
        & (df["pl_orbper"] > 0)
        & (df["pl_trandurh"] > 0)
    )
    valid = df.loc[valid_mask].copy()

    # Si un TIC tiene varias TOIs válidas, nos quedamos con la más estrecha
    # en duración (suele corresponder al tránsito mejor caracterizado).
    valid = valid.sort_values(["tid", "pl_trandurh"], ascending=[True, True])
    valid_by_tic = valid.drop_duplicates(subset=["tid"], keep="first")

    # TICs sin ninguna TOI válida → NaN en los 3 campos.
    all_tics = df.drop_duplicates(subset=["tid"], keep="first")[["tid"]].copy()
    merged = all_tics.merge(
        valid_by_tic[["tid", "pl_orbper", "pl_tranmid", "pl_trandurh"]],
        on="tid",
        how="left",
    )
    return merged


def evaluate_split(split_df: pd.DataFrame, catalog: pd.DataFrame, name: str) -> dict:
    df = split_df.merge(catalog, on="tid", how="left")
    n_total = len(df)

    has_metadata = (
        df["pl_orbper"].notna()
        & df["pl_tranmid"].notna()
        & df["pl_trandurh"].notna()
    )
    period_ok = df["pl_orbper"] <= PERIOD_MAX_DAYS
    # duration ya está en horas; pasamos a días para comparar contra period.
    duration_days = df["pl_trandurh"] / 24.0
    duration_ok = duration_days < (df["pl_orbper"] / 2.0)

    full_ok = has_metadata & period_ok & duration_ok

    out = {
        "split": name,
        "n_total": n_total,
        "n_with_metadata": int(has_metadata.sum()),
        "n_period_too_long": int((has_metadata & ~period_ok).sum()),
        "n_duration_degenerate": int((has_metadata & period_ok & ~duration_ok).sum()),
        "n_ok": int(full_ok.sum()),
        "n_ok_cp": int(((df["label"] == 1) & full_ok).sum()),
        "n_ok_fp": int(((df["label"] == 0) & full_ok).sum()),
        "n_total_cp": int((df["label"] == 1).sum()),
        "n_total_fp": int((df["label"] == 0).sum()),
    }
    return out


def render_report(results: list[dict], catalog_summary: dict) -> str:
    lines = []
    lines.append("# Análisis de factibilidad - local_view (Fase 3.b / Tier 2)\n")
    lines.append("Generado por `scripts/analyze_local_view_feasibility.py`.\n")
    lines.append("## Catálogo TOI\n")
    lines.append(f"- Total TOIs (rows en `toi_summary.csv`): **{catalog_summary['n_rows']:,}**")
    lines.append(f"- TICs únicos: **{catalog_summary['n_tics']:,}**")
    lines.append(
        f"- TICs con al menos una TOI con (period, epoch, duration) válidos: "
        f"**{catalog_summary['n_tics_valid']:,}**"
    )
    lines.append("\nColumnas usadas: `pl_orbper`, `pl_tranmid` (BJD), `pl_trandurh` (horas).\n")

    lines.append("## Filtros aplicados\n")
    lines.append(f"1. Los 3 campos no-NaN y > 0.")
    lines.append(f"2. `pl_orbper` <= **{PERIOD_MAX_DAYS}** días (1 sector TESS).")
    lines.append(f"3. `pl_trandurh / 24 < pl_orbper / 2` (descarta duraciones degeneradas).\n")

    lines.append("## Conteos por split\n")
    lines.append("| Split | N total | Con metadata | Period > 27.4d | Dur degen | **N Tier 2 OK** | OK · CP | OK · FP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['split']} | {r['n_total']} | {r['n_with_metadata']} | "
            f"{r['n_period_too_long']} | {r['n_duration_degenerate']} | "
            f"**{r['n_ok']}** | {r['n_ok_cp']} | {r['n_ok_fp']} |"
        )

    lines.append("\n## Distribución de clases dentro de Tier 2 OK\n")
    lines.append("| Split | OK CP | OK FP | Ratio CP/(CP+FP) | Total Tier 1 CP | Total Tier 1 FP |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        denom = r["n_ok_cp"] + r["n_ok_fp"]
        ratio = (r["n_ok_cp"] / denom) if denom > 0 else 0.0
        lines.append(
            f"| {r['split']} | {r['n_ok_cp']} | {r['n_ok_fp']} | "
            f"{ratio:.3f} | {r['n_total_cp']} | {r['n_total_fp']} |"
        )

    train_n = next(r["n_ok"] for r in results if r["split"] == "train")
    lines.append("\n## Veredicto HARD STOP\n")
    if train_n >= HARD_STOP_N_TRAIN:
        lines.append(
            f"**N_train_tier2 = {train_n} >= {HARD_STOP_N_TRAIN}** → Tier 2 es VIABLE. "
            "Se puede continuar con `scripts/preprocess_local.py`."
        )
    else:
        lines.append(
            f"**N_train_tier2 = {train_n} < {HARD_STOP_N_TRAIN}** → Tier 2 INVIABLE. "
            "Pivot recomendado a solo Tier 1."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    for p in (TOI_SUMMARY, TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise SystemExit(f"No existe {p}.")

    raw_summary = pd.read_csv(TOI_SUMMARY)
    catalog = load_catalog_metadata()
    n_tics_valid = int(catalog[["pl_orbper", "pl_tranmid", "pl_trandurh"]].notna().all(axis=1).sum())
    catalog_summary = {
        "n_rows": len(raw_summary),
        "n_tics": int(raw_summary["tid"].nunique()),
        "n_tics_valid": n_tics_valid,
    }

    splits = [
        ("train", pd.read_csv(TRAIN_CSV)),
        ("val", pd.read_csv(VAL_CSV)),
        ("test", pd.read_csv(TEST_CSV)),
    ]
    results = [evaluate_split(df, catalog, name) for name, df in splits]

    # Imprimir a stdout
    print("\n=== Catálogo TOI ===")
    print(f"Filas: {catalog_summary['n_rows']:,}")
    print(f"TICs únicos: {catalog_summary['n_tics']:,}")
    print(f"TICs con al menos una TOI válida (period+epoch+duration): {catalog_summary['n_tics_valid']:,}")

    print("\n=== Tier 2 OK por split ===")
    print(f"{'split':<6} {'N':>5} {'meta':>5} {'P>27.4':>7} {'dur_deg':>8} {'OK':>5} {'OK CP':>6} {'OK FP':>6}")
    for r in results:
        print(
            f"{r['split']:<6} {r['n_total']:>5} {r['n_with_metadata']:>5} "
            f"{r['n_period_too_long']:>7} {r['n_duration_degenerate']:>8} "
            f"{r['n_ok']:>5} {r['n_ok_cp']:>6} {r['n_ok_fp']:>6}"
        )

    train_n = next(r["n_ok"] for r in results if r["split"] == "train")
    print(f"\nHARD STOP: N_train_tier2 = {train_n} (umbral {HARD_STOP_N_TRAIN})")
    if train_n < HARD_STOP_N_TRAIN:
        print(">>> Tier 2 INVIABLE - abortar pipeline local_view.")
        verdict = 1
    else:
        print(">>> Tier 2 viable, continuar con preprocess_local.py.")
        verdict = 0

    OUT_MD.write_text(render_report(results, catalog_summary), encoding="utf-8")
    print(f"\nReporte: {OUT_MD}")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
