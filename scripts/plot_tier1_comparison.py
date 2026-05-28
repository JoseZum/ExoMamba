"""ROC comparativa Tier 1 (Fase 1.4 del plan AMBICIOSO).

Genera `paper/figures/roc_tier1.png` con las curvas ROC superpuestas de:

  * Random baseline (Fase 5.a)
  * Catalog LogReg (Fase 5.b)
  * CNN single-branch locked (Fase 6)
  * Mamba single-view locked (Fase 8)
  * Mamba ensemble (5 seeds, Fase 1.3)

Diseño:

  * Lee `eval_test/predictions.csv` de cada run (generado por
    `scripts/evaluate.py --split test`). Para el ensemble, lee
    `ensemble_predictions.csv` (generado por `scripts/ensemble_eval.py`).
  * Para LogReg, lee `experiments/logreg_baseline_test_predictions.csv`
    (generado por `scripts/train_logreg.py --split test`).
  * Si algún archivo falta, se omite del plot con un warning (no aborta).
    El plot se genera mientras quede al menos 1 modelo.
  * Reusa `plot_comparison_roc` de `src/exoplanet/evaluation/plots.py`.

Uso:

  python scripts/plot_tier1_comparison.py \\
    [--output paper/figures/roc_tier1.png] \\
    [--ensemble-dir paper/results/mamba_ensemble]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exoplanet.evaluation.plots import plot_comparison_roc  # noqa: E402


# Defaults: los paths que CLAUDE.md / AMBICIOSO.md fijan como locked.
DEFAULT_RUNS: dict[str, tuple[str, str]] = {
    # nombre_visible: (path_predictions_csv, columna_prob)
    "Random": ("experiments/2026-05-21_05-36-11_random_baseline/eval_test/predictions.csv", "y_prob"),
    "LogReg (catalog feats)": ("experiments/logreg_baseline_test_predictions.csv", "y_prob"),
    "CNN single-branch": ("experiments/2026-05-20_23-44-48_cnn_baseline/eval_test/predictions.csv", "y_prob"),
    "Mamba single-view": ("experiments/2026-05-22_14-32-51_mamba_small/eval_test/predictions.csv", "y_prob"),
    "Mamba ensemble (5 seeds)": ("paper/results/mamba_ensemble/ensemble_predictions.csv", "y_prob_mean"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROC comparativa Tier 1 sobre el split de test."
    )
    p.add_argument(
        "--output",
        type=str,
        default="paper/figures/roc_tier1.png",
        help="Path de salida del PNG. Default: paper/figures/roc_tier1.png",
    )
    p.add_argument(
        "--ensemble-dir",
        type=str,
        default="paper/results/mamba_ensemble",
        help=(
            "Directorio donde está `ensemble_predictions.csv`. Override del path "
            "default si el ensemble se guardó en otra carpeta."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Si está activo, falla cuando falta algún predictions.csv (en vez de omitir).",
    )
    return p.parse_args()


def _load_run(name: str, path_str: str, prob_col: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    path = PROJECT_ROOT / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "y_true" not in df.columns or prob_col not in df.columns:
        print(
            f"WARNING: {path} no contiene columnas requeridas (y_true, {prob_col}); omitiendo {name}.",
            file=sys.stderr,
        )
        return None
    y_true = df["y_true"].astype(int).values
    y_prob = df[prob_col].astype(float).values
    return (y_true, y_prob)


def main() -> int:
    args = _parse_args()

    runs_paths = dict(DEFAULT_RUNS)
    # Override ensemble path si el usuario lo cambió
    if args.ensemble_dir:
        ensemble_path = f"{args.ensemble_dir.rstrip('/')}/ensemble_predictions.csv"
        runs_paths["Mamba ensemble (5 seeds)"] = (ensemble_path, "y_prob_mean")

    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (path_str, prob_col) in runs_paths.items():
        data = _load_run(name, path_str, prob_col)
        if data is None:
            msg = f"Falta predictions.csv para '{name}' en {path_str}."
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 2
            print(f"WARNING: {msg} Omitiendo del plot.", file=sys.stderr)
            continue
        loaded[name] = data
        n = len(data[0])
        pos = int(data[0].sum())
        print(f"  loaded '{name}': n={n} (pos={pos}, neg={n - pos})")

    if not loaded:
        print(
            "ERROR: ningún modelo disponible para el plot. "
            "Corré primero `scripts/evaluate.py --split test` para cada modelo "
            "y `scripts/ensemble_eval.py` para el ensemble.",
            file=sys.stderr,
        )
        return 2

    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = plot_comparison_roc(loaded, output_path)
    print(f"\nROC comparativa guardada en: {out}")
    print(f"Modelos incluidos: {len(loaded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
