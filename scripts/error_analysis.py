"""CLI de análisis de errores (Fase 6 - AMBICIOSO).

Lee un `predictions.csv` ya generado por `scripts/evaluate.py` y el
`toi_summary.csv` con features físicas por TIC. Identifica los cuadrantes de
confusión (TP, TN, FN, FP) con el threshold dado y produce:

  - `top_fn.csv` / `top_fp.csv`: top-K casos peor calibrados de cada tipo,
    con features físicas asociadas.
  - `prob_histogram.png`: histograma de y_prob por clase verdadera (overlay).
  - `error_rate_by_feature.png`: 3 subplots (period, depth, tmag) con tasa
    de error por bin quantile. Sirve para detectar dónde falla más el modelo.
  - `top_fn_curves.png` / `top_fp_curves.png`: grid 1x5 con la curva del
    test para cada uno de los top-5 FN / FP, con título "TIC X - y_prob=Y.YY".
  - `error_analysis_summary.md`: tabla en markdown con conteos + top-10 FN +
    top-10 FP + observaciones automáticas.

Uso:

  python scripts/error_analysis.py \\
      --predictions experiments/2026-05-22_14-32-51_mamba_small/eval_test/predictions.csv \\
      --catalog data/splits/toi_summary.csv \\
      --output paper/results/error_analysis/mamba_small

  python scripts/error_analysis.py \\
      --predictions <path/to/predictions.csv> \\
      --catalog data/splits/toi_summary.csv \\
      --output paper/results/error_analysis/<model_name> \\
      --threshold 0.5 \\
      --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src/ en path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from exoplanet.data import LightCurveDataset  # noqa: E402

FEATURE_COLS = ("pl_orbper", "pl_trandep", "st_tmag")
QUADRANTS = ("TP", "TN", "FN", "FP")


# ---------------------------------------------------------------------------
# Carga y merge
# ---------------------------------------------------------------------------
def _load_predictions(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Aceptar tanto `y_prob` (single-model) como `y_prob_mean` (ensemble_eval.py).
    if "y_prob" not in df.columns and "y_prob_mean" in df.columns:
        df = df.rename(columns={"y_prob_mean": "y_prob"})
    required = {"tic_id", "y_true", "y_prob"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"predictions.csv debe tener {required} (o `y_prob_mean` en ensemble); "
            f"trae {set(df.columns)}"
        )
    if "y_pred" not in df.columns:
        df["y_pred"] = (df["y_prob"] >= threshold).astype(int)
    df["tic_id"] = df["tic_id"].astype(int)
    return df


def _load_catalog(path: Path) -> pd.DataFrame:
    """Carga toi_summary y deduplica por TIC (replica lógica de train_logreg.py)."""
    cat = pd.read_csv(path)
    if "tid" not in cat.columns:
        raise ValueError(f"Catalog debe tener columna 'tid'; trae {list(cat.columns)}")
    cat = cat.drop_duplicates(subset="tid", keep="first").reset_index(drop=True)
    cat = cat.rename(columns={"tid": "tic_id"})
    cat["tic_id"] = cat["tic_id"].astype(int)
    keep = ["tic_id"] + [c for c in FEATURE_COLS if c in cat.columns]
    return cat[keep]


def _merge(preds: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Left-join: conservamos TODAS las filas de predictions (los que no estén
    en el catálogo aparecen con NaN en features físicas)."""
    merged = preds.merge(catalog, on="tic_id", how="left")
    return merged


# ---------------------------------------------------------------------------
# Cuadrantes y ordenamientos
# ---------------------------------------------------------------------------
def _split_quadrants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "TP": df[(df["y_true"] == 1) & (df["y_pred"] == 1)],
        "TN": df[(df["y_true"] == 0) & (df["y_pred"] == 0)],
        "FN": df[(df["y_true"] == 1) & (df["y_pred"] == 0)],
        "FP": df[(df["y_true"] == 0) & (df["y_pred"] == 1)],
    }


def _top_fn(df_fn: pd.DataFrame, k: int) -> pd.DataFrame:
    """FN ordenados por y_prob DESC: los más "confidentes erróneos negativos".

    Un FN con y_prob=0.45 está al borde; uno con y_prob=0.05 fue catastrófico
    para el modelo. Para inspección manual queremos los del borde primero
    porque suelen ser los más informativos sobre el límite de decisión.
    """
    return df_fn.sort_values("y_prob", ascending=False).head(k).reset_index(drop=True)


def _top_fp(df_fp: pd.DataFrame, k: int) -> pd.DataFrame:
    """FP ordenados por y_prob ASC: los menos confidentes pero erróneos.

    Mismo argumento: y_prob~0.55 falla por poco; y_prob~0.95 falló mucho.
    Los del borde son más informativos para inspección visual.
    """
    return df_fp.sort_values("y_prob", ascending=True).head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_prob_histogram(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    pos = df.loc[df["y_true"] == 1, "y_prob"].to_numpy()
    neg = df.loc[df["y_true"] == 0, "y_prob"].to_numpy()
    bins = np.linspace(0, 1, 30)
    ax.hist(neg, bins=bins, alpha=0.55, label=f"FP (n={len(neg)})", color="steelblue")
    ax.hist(pos, bins=bins, alpha=0.55, label=f"CP (n={len(pos)})", color="crimson")
    ax.axvline(0.5, color="black", linestyle="--", lw=1, label="threshold 0.5")
    ax.set_xlabel("y_prob")
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de y_prob por clase verdadera")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _bin_quantile(series: pd.Series, n_bins: int) -> pd.Series:
    """qcut robusto. Si todos los valores son iguales o hay pocos únicos,
    cae al menor número de bins viable.
    """
    s = series.dropna()
    if s.empty:
        return pd.Series([], dtype="object")
    unique_vals = s.nunique()
    bins = min(n_bins, unique_vals)
    if bins < 2:
        return pd.Series([str(s.iloc[0])] * len(series), index=series.index)
    try:
        return pd.qcut(series, q=bins, duplicates="drop")
    except ValueError:
        return pd.qcut(series, q=bins, duplicates="drop")


def _plot_error_rate_by_feature(
    df: pd.DataFrame, output_path: Path, n_bins: int = 10
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)

    features = [c for c in FEATURE_COLS if c in df.columns]
    if not features:
        # Plot vacío con leyenda explicativa para que el script no crashee.
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.text(0.5, 0.5, "Sin features físicas en el catálogo", ha="center")
        ax.set_axis_off()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), dpi=120)
    if len(features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, features, strict=False):
        sub = df[[feat, "error"]].dropna()
        if sub.empty:
            ax.set_title(f"{feat} - sin datos")
            ax.set_axis_off()
            continue
        bins = _bin_quantile(sub[feat], n_bins)
        if bins.empty:
            ax.set_title(f"{feat} - bins vacíos")
            ax.set_axis_off()
            continue
        sub = sub.assign(_bin=bins)
        grouped = sub.groupby("_bin", observed=True)["error"].agg(["mean", "count"])
        # Label corto: midpoint del intervalo
        labels = [
            f"{interval.mid:.2g}" if hasattr(interval, "mid") else str(interval)
            for interval in grouped.index
        ]
        x = np.arange(len(grouped))
        ax.bar(x, grouped["mean"].values, color="indianred", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Tasa de error")
        ax.set_xlabel(f"{feat} (midpoint del bin)")
        ax.set_title(f"Error rate vs {feat}")
        # Anotar conteos arriba de cada barra
        for i, cnt in enumerate(grouped["count"].values):
            ax.text(i, grouped["mean"].iloc[i] + 0.02, f"n={int(cnt)}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_curves_grid(
    cases: pd.DataFrame,
    curve_loader: dict[int, np.ndarray],
    output_path: Path,
    title_prefix: str,
) -> Path:
    """Grid 1x5 (o menos si hay menos casos) con curva por caso."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(5, len(cases))
    if n == 0:
        # Plot vacío
        fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
        ax.text(0.5, 0.5, "Sin casos disponibles", ha="center")
        ax.set_axis_off()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), dpi=120, sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, cases.head(n).iterrows(), strict=False):
        tic = int(row["tic_id"])
        y_prob = float(row["y_prob"])
        curve = curve_loader.get(tic)
        if curve is None:
            ax.text(0.5, 0.5, f"TIC {tic} - sin curva", ha="center")
            ax.set_axis_off()
            continue
        t = np.arange(curve.shape[0])
        ax.plot(t, curve, color="steelblue", lw=0.5)
        ax.set_title(f"TIC {tic} - y_prob={y_prob:.2f}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(title_prefix, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Carga de curvas por TIC
# ---------------------------------------------------------------------------
def _try_load_curves(
    tics: list[int], split_csv: Path, processed_dir: Path
) -> dict[int, np.ndarray]:
    """Devuelve un dict tic_id -> numpy array de la curva (1, L) -> (L,).

    Si el split CSV o `processed_dir` no existen, devuelve dict vacío y deja
    que los plots dibujen "sin curva" - útil en dry-run o cuando los .pt no
    están todavía generados.
    """
    if not split_csv.exists():
        print(f"WARNING: split CSV {split_csv} no existe; no se cargan curvas.")
        return {}
    if not processed_dir.exists():
        print(
            f"WARNING: processed_dir {processed_dir} no existe; no se cargan curvas."
        )
        return {}
    try:
        dataset = LightCurveDataset(
            split_csv,
            processed_dir=processed_dir,
            augment=None,
            check_files=False,  # no abortar si falta alguno
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: no se pudo abrir el dataset ({e}); no se cargan curvas.")
        return {}
    tic_to_idx = {int(t): i for i, t in enumerate(dataset.tids)}
    out: dict[int, np.ndarray] = {}
    for tic in tics:
        idx = tic_to_idx.get(int(tic))
        if idx is None:
            continue
        try:
            sample = dataset[idx]
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: no pude cargar curva TIC {tic}: {e}")
            continue
        out[int(tic)] = sample["global_view"].squeeze().detach().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Reporte markdown
# ---------------------------------------------------------------------------
def _write_summary_md(
    output_path: Path,
    quadrant_counts: dict[str, int],
    top_fn: pd.DataFrame,
    top_fp: pd.DataFrame,
    n_total: int,
    threshold: float,
    auto_obs: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Análisis de errores\n")
    lines.append(f"- Total samples: {n_total}\n")
    lines.append(f"- Threshold: {threshold}\n\n")

    lines.append("## Conteos por cuadrante\n\n")
    lines.append("| Cuadrante | n |\n|---|---|\n")
    for q in QUADRANTS:
        lines.append(f"| {q} | {quadrant_counts.get(q, 0)} |\n")
    lines.append("\n")

    def _df_to_md(df: pd.DataFrame, title: str) -> None:
        lines.append(f"## {title}\n\n")
        if df.empty:
            lines.append("_Sin casos en este cuadrante._\n\n")
            return
        cols = [c for c in df.columns if c not in {"_bin", "error"}]
        lines.append("| " + " | ".join(cols) + " |\n")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        int_cols = {"tic_id", "y_true", "y_pred"}
        for _, row in df.iterrows():
            cells: list[str] = []
            for c in cols:
                val = row[c]
                if pd.isna(val):
                    cells.append("nan")
                elif c in int_cols:
                    cells.append(str(int(val)))
                elif isinstance(val, float | np.floating):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |\n")
        lines.append("\n")

    _df_to_md(top_fn, "Top FN (más confidentes erróneos negativos)")
    _df_to_md(top_fp, "Top FP (más confidentes erróneos positivos)")

    lines.append("## Observaciones automáticas\n\n")
    if not auto_obs:
        lines.append("- (sin observaciones)\n")
    else:
        for obs in auto_obs:
            lines.append(f"- {obs}\n")
    lines.append("\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _auto_observations(df: pd.DataFrame) -> list[str]:
    """Genera observaciones simples y defensivas: dónde hay mayor tasa de error.

    Para cada feature física: bins quantile y reporte del bin con peor tasa de
    error si esa tasa supera 1.5x la tasa global.
    """
    obs: list[str] = []
    if df.empty:
        return obs
    overall_error = float((df["y_true"] != df["y_pred"]).mean())
    obs.append(f"Tasa global de error: {overall_error:.3f} (n={len(df)}).")
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue
        sub = df[[feat, "y_true", "y_pred"]].dropna(subset=[feat])
        if len(sub) < 20:
            continue
        sub = sub.assign(error=(sub["y_true"] != sub["y_pred"]).astype(int))
        bins = _bin_quantile(sub[feat], 5)
        if bins.empty:
            continue
        sub = sub.assign(_bin=bins)
        grouped = sub.groupby("_bin", observed=True)["error"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= 5]  # bins con suficiente soporte
        if grouped.empty:
            continue
        worst = grouped["mean"].idxmax()
        worst_rate = grouped.loc[worst, "mean"]
        worst_n = int(grouped.loc[worst, "count"])
        if worst_rate >= max(0.15, 1.5 * overall_error):
            obs.append(
                f"El modelo falla más en {feat} bin {worst} "
                f"(error_rate={worst_rate:.3f}, n={worst_n})."
            )
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Análisis de errores: top-K FN/FP, histograma de y_prob, "
            "tasa de error por feature física, curvas y resumen markdown."
        )
    )
    p.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Ruta al predictions.csv (tic_id, y_true, y_prob[, y_pred]).",
    )
    p.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="Ruta al TOI summary (default proyecto: data/splits/toi_summary.csv).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directorio para los outputs.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold para construir y_pred si no viene en predictions.csv.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K FN y FP a reportar (default 10).",
    )
    p.add_argument(
        "--split-csv",
        type=str,
        default="data/splits/test_tics.csv",
        help=(
            "CSV del split desde donde cargar curvas para top_fn_curves / "
            "top_fp_curves. Default: data/splits/test_tics.csv."
        ),
    )
    p.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/global",
        help="Directorio con los .pt globales (default: data/processed/global).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    preds_path = Path(args.predictions)
    catalog_path = Path(args.catalog)
    output_dir = Path(args.output)

    if not preds_path.exists():
        print(f"ERROR: predictions.csv no existe: {preds_path}", file=sys.stderr)
        return 2
    if not catalog_path.exists():
        print(f"ERROR: catalog no existe: {catalog_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(preds_path, args.threshold)
    catalog = _load_catalog(catalog_path)
    merged = _merge(preds, catalog)
    print(f"predictions: {len(preds)} | catalog rows merged: {len(merged)}")

    quadrants = _split_quadrants(merged)
    quadrant_counts = {q: len(df) for q, df in quadrants.items()}
    print(
        "Cuadrantes: "
        + ", ".join(f"{q}={quadrant_counts[q]}" for q in QUADRANTS)
    )

    # Columnas que queremos en top_fn / top_fp
    base_cols = ["tic_id", "y_true", "y_prob", "y_pred"]
    feat_cols = [c for c in FEATURE_COLS if c in merged.columns]
    out_cols = base_cols + feat_cols

    top_fn = (
        _top_fn(quadrants["FN"], args.top_k)[out_cols]
        if not quadrants["FN"].empty
        else quadrants["FN"]
    )
    top_fp = (
        _top_fp(quadrants["FP"], args.top_k)[out_cols]
        if not quadrants["FP"].empty
        else quadrants["FP"]
    )
    top_fn.to_csv(output_dir / "top_fn.csv", index=False)
    top_fp.to_csv(output_dir / "top_fp.csv", index=False)
    print(f"  -> top_fn.csv ({len(top_fn)} filas)")
    print(f"  -> top_fp.csv ({len(top_fp)} filas)")

    # Plots
    _plot_prob_histogram(merged, output_dir / "prob_histogram.png")
    print("  -> prob_histogram.png")
    _plot_error_rate_by_feature(merged, output_dir / "error_rate_by_feature.png")
    print("  -> error_rate_by_feature.png")

    # Curvas top-5 (subset)
    fn_tics = top_fn["tic_id"].astype(int).tolist()[:5] if not top_fn.empty else []
    fp_tics = top_fp["tic_id"].astype(int).tolist()[:5] if not top_fp.empty else []
    all_tics = list(set(fn_tics + fp_tics))
    curves = _try_load_curves(all_tics, Path(args.split_csv), Path(args.processed_dir))

    _plot_curves_grid(
        top_fn,
        curves,
        output_dir / "top_fn_curves.png",
        title_prefix="Top FN - curvas de test",
    )
    print("  -> top_fn_curves.png")
    _plot_curves_grid(
        top_fp,
        curves,
        output_dir / "top_fp_curves.png",
        title_prefix="Top FP - curvas de test",
    )
    print("  -> top_fp_curves.png")

    # Markdown summary
    auto_obs = _auto_observations(merged)
    md_path = _write_summary_md(
        output_dir / "error_analysis_summary.md",
        quadrant_counts,
        top_fn,
        top_fp,
        n_total=len(merged),
        threshold=args.threshold,
        auto_obs=auto_obs,
    )
    print(f"  -> {md_path.name}")

    print(f"\nResultados en: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
