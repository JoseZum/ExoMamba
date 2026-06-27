"""Ensemble por promedio aritmético de y_prob (Fase 1.3 del plan AMBICIOSO).

Lee `predictions.csv` de N corridas ya evaluadas (típicamente las 5 seeds
de Mamba) y produce un ensemble que se reporta en la tabla principal del
paper. Cada `predictions.csv` viene de `scripts/evaluate.py --split <split>`.

Diseño:

  * Alineación por `tic_id` (inner join). Si algún tic_id falta en algún run,
    falla fuerte para no esconder un bug silencioso.
  * Promedia `y_prob` aritméticamente (NO geométrica) - es el agregador
    estándar para ensembles de clasificadores binarios calibrados.
  * Recalcula `y_pred = y_prob_mean >= threshold`. Default threshold=0.5.
  * Reusa `compute_classification_metrics` para AUC-ROC, AUC-PR, F1,
    recall, precision; calcula Brier score y accuracy adicionales.
  * Reusa plots de `src/exoplanet/evaluation/plots.py` (consistencia visual
    con eval individuales).
  * Imprime tabla comparando cada run vs ensemble (val_auc del eval/metrics.json).

Restricciones operativas:

  * NO entrena nada. Solo lee artefactos ya producidos.
  * NO toca el test sellado por sí mismo: solo agrega resultados de runs
    cuya eval_<split>/predictions.csv ya existe.

Uso:

  python scripts/ensemble_eval.py \\
    --runs experiments/run1,experiments/run2,experiments/run3 \\
    --split test \\
    --output-dir paper/results/mamba_ensemble \\
    [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Path bootstrap: el src layout requiere que `src/` esté en sys.path para
# importar `exoplanet.*` cuando se corre el script directamente sin instalar.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exoplanet.evaluation.plots import (  # noqa: E402
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from exoplanet.training.metrics import compute_classification_metrics  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Promedio aritmético de y_prob de N runs ya evaluadas. Lee "
            "<run>/eval_<split>/predictions.csv de cada una, alinea por "
            "tic_id, recalcula métricas y guarda resultados en --output-dir."
        )
    )
    p.add_argument(
        "--runs",
        type=str,
        required=True,
        help=(
            "Lista separada por comas de run dirs. Ej: "
            "experiments/2026-05-27_23-00-33_mamba_small_seed42,"
            "experiments/2026-05-28_00-49-39_mamba_small_seed123"
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split del que leer cada predictions.csv. Default: test.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio donde guardar ensemble_{metrics.json, predictions.csv, *.png}.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral para recalcular y_pred desde y_prob_mean. Default 0.5.",
    )
    return p.parse_args()


def _parse_run_list(runs_arg: str) -> list[Path]:
    runs = [Path(r.strip()) for r in runs_arg.split(",") if r.strip()]
    if len(runs) < 2:
        raise ValueError(
            f"Se requieren al menos 2 runs para hacer ensemble; recibidos: {len(runs)}."
        )
    for r in runs:
        if not r.exists():
            raise FileNotFoundError(f"Run dir no existe: {r}")
    return runs


def _load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    """Carga predictions.csv de `<run_dir>/eval_<split>/`.

    Falla fuerte si el archivo no existe (en cuyo caso hay que correr
    primero `scripts/evaluate.py --run <run_dir> --split <split>`).
    """
    pred_path = run_dir / f"eval_{split}" / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Falta predictions.csv en {pred_path}. "
            f"Corré primero: python scripts/evaluate.py --run {run_dir} --split {split}"
        )
    df = pd.read_csv(pred_path)
    expected = {"tic_id", "y_true", "y_prob", "y_pred"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"{pred_path} no tiene columnas {missing}. Cols vistas: {list(df.columns)}")
    df["tic_id"] = df["tic_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    df["y_pred"] = df["y_pred"].astype(int)
    return df


def _load_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    """Lee metrics.json del eval_<split> (para tabla comparativa)."""
    metrics_path = run_dir / f"eval_{split}" / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open(encoding="utf-8") as f:
        return json.load(f)


def _align_runs(dfs: list[pd.DataFrame], run_names: list[str]) -> pd.DataFrame:
    """Inner join por tic_id; falla si algún run omite un tic_id presente en el resto.

    Devuelve DataFrame con columnas: tic_id, y_true (verificado idéntico entre
    runs), y prob_<run> por cada run.
    """
    base = dfs[0][["tic_id", "y_true"]].copy()
    base["y_true_ref"] = base["y_true"]
    base = base.drop(columns="y_true")
    expected_tids = set(base["tic_id"].tolist())

    merged = base.copy()
    for name, df in zip(run_names, dfs, strict=True):
        this_tids = set(df["tic_id"].tolist())
        missing_here = expected_tids - this_tids
        extra_here = this_tids - expected_tids
        if missing_here or extra_here:
            raise RuntimeError(
                f"Desalineación de tic_id en run '{name}': "
                f"faltan {len(missing_here)} (ej: {list(missing_here)[:5]}), "
                f"sobran {len(extra_here)} (ej: {list(extra_here)[:5]}). "
                "El ensemble requiere conjuntos idénticos de tic_id por run."
            )
        col_prob = f"prob_{name}"
        sub = df[["tic_id", "y_true", "y_prob"]].rename(columns={"y_prob": col_prob})
        merged = merged.merge(sub, on="tic_id", how="inner", validate="one_to_one")
        # Verificación cruzada: y_true debe coincidir con la referencia (mismo TIC ⇒ misma label).
        mismatched = (merged["y_true"] != merged["y_true_ref"]).sum()
        if mismatched > 0:
            raise RuntimeError(
                f"y_true difiere entre runs para {mismatched} tic_ids en run '{name}'. "
                "Esto NO puede pasar: la etiqueta es propiedad de la estrella, no del modelo."
            )
        merged = merged.drop(columns="y_true")

    merged = merged.rename(columns={"y_true_ref": "y_true"})
    return merged


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true.astype(float)) ** 2))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _summarize_individual(run_dirs: list[Path], split: str) -> list[dict[str, Any]]:
    rows = []
    for r in run_dirs:
        m = _load_metrics(r, split)
        metrics = m.get("metrics", {})
        rows.append(
            {
                "run": r.name,
                "auc_roc": metrics.get("auc_roc"),
                "auc_pr": metrics.get("auc_pr"),
                "f1": metrics.get("f1"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "accuracy": metrics.get("accuracy"),
                "brier": metrics.get("brier_score"),
            }
        )
    return rows


def _print_comparison_table(
    individual: list[dict[str, Any]],
    ensemble_metrics: dict[str, Any],
) -> None:
    """Tabla de texto plano para que se vea bien en cualquier terminal."""
    print()
    print("=" * 110)
    print(
        f"{'run':50s} {'AUC':>8s} {'PR':>8s} {'F1':>8s} {'Rec':>8s} {'Prec':>8s} {'Acc':>8s} {'Brier':>8s}"
    )
    print("-" * 110)
    aucs = []
    for row in individual:
        auc = row["auc_roc"] if row["auc_roc"] is not None else float("nan")
        aucs.append(auc)
        print(
            f"{row['run']:50s} "
            f"{_fmt(row['auc_roc']):>8s} {_fmt(row['auc_pr']):>8s} "
            f"{_fmt(row['f1']):>8s} {_fmt(row['recall']):>8s} "
            f"{_fmt(row['precision']):>8s} {_fmt(row['accuracy']):>8s} "
            f"{_fmt(row['brier']):>8s}"
        )
    if aucs:
        mean_auc = np.nanmean(aucs)
        std_auc = np.nanstd(aucs)
        print("-" * 110)
        print(
            f"{'individual mean ± std':50s} "
            f"{mean_auc:>8.4f}±{std_auc:.4f}"
        )
    print("-" * 110)
    em = ensemble_metrics
    print(
        f"{'ENSEMBLE (mean y_prob)':50s} "
        f"{_fmt(em.get('auc_roc')):>8s} {_fmt(em.get('auc_pr')):>8s} "
        f"{_fmt(em.get('f1')):>8s} {_fmt(em.get('recall')):>8s} "
        f"{_fmt(em.get('precision')):>8s} {_fmt(em.get('accuracy')):>8s} "
        f"{_fmt(em.get('brier_score')):>8s}"
    )
    print("=" * 110)
    print()


def _fmt(x: Any) -> str:
    try:
        if x is None:
            return "-"
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "-"


def _write_ensemble_predictions(
    path: Path, merged: pd.DataFrame, y_prob_mean: np.ndarray, y_pred: np.ndarray
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "tic_id": merged["tic_id"].astype(int).values,
            "y_true": merged["y_true"].astype(int).values,
            "y_prob_mean": y_prob_mean,
            "y_pred": y_pred.astype(int),
        }
    )
    out.to_csv(path, index=False)
    return path


def main() -> int:
    args = _parse_args()

    run_dirs = _parse_run_list(args.runs)
    run_names = [r.name for r in run_dirs]

    print(f"Cargando predicciones de {len(run_dirs)} runs para split='{args.split}'...")
    dfs = [_load_predictions(r, args.split) for r in run_dirs]
    for name, df in zip(run_names, dfs, strict=True):
        print(f"  {name}: n_samples={len(df)}")

    merged = _align_runs(dfs, run_names)
    n_aligned = len(merged)
    print(f"Alineados por tic_id: n={n_aligned}")

    # Promedio aritmético de y_prob
    prob_cols = [f"prob_{name}" for name in run_names]
    y_prob_mean = merged[prob_cols].mean(axis=1).values
    y_true = merged["y_true"].values.astype(int)
    y_pred = (y_prob_mean >= args.threshold).astype(int)

    metrics = compute_classification_metrics(y_true, y_prob_mean, threshold=args.threshold)
    metrics["accuracy"] = _accuracy(y_true, y_pred)
    metrics["brier_score"] = _brier_score(y_true, y_prob_mean)

    individual = _summarize_individual(run_dirs, args.split)
    _print_comparison_table(individual, metrics)

    # WARNING explícito si el split es test
    if args.split == "test":
        print(
            "WARNING: este ensemble se construyó sobre el TEST SELLADO. "
            "Esto NO consume evaluaciones adicionales (reusa predictions.csv ya generados), "
            "pero la métrica reportada solo es válida si las predictions.csv vienen de "
            "una única evaluación de test por run.",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": args.split,
        "threshold": float(args.threshold),
        "n_runs": len(run_dirs),
        "run_dirs": [str(r) for r in run_dirs],
        "n_samples_aligned": int(n_aligned),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "auc_roc": float(metrics["auc_roc"]),
            "auc_pr": float(metrics["auc_pr"]),
            "f1": float(metrics["f1"]),
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "accuracy": float(metrics["accuracy"]),
            "brier_score": float(metrics["brier_score"]),
        },
        "individual_runs": individual,
    }
    metrics_path = output_dir / "ensemble_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pred_path = _write_ensemble_predictions(
        output_dir / "ensemble_predictions.csv", merged, y_prob_mean, y_pred
    )

    # Plots: mismas funciones que scripts/evaluate.py para consistencia visual
    plot_roc_curve(y_true, y_prob_mean, output_dir / "roc_curve.png", label="Ensemble")
    plot_pr_curve(y_true, y_prob_mean, output_dir / "pr_curve.png", label="Ensemble")
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    plot_calibration(y_true, y_prob_mean, output_dir / "calibration.png")

    print(f"\nResultados ensemble guardados en: {output_dir}")
    print(f"  metrics:      {metrics_path}")
    print(f"  predictions:  {pred_path}")
    print(f"  roc/pr/cm/calibration: {output_dir}/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
