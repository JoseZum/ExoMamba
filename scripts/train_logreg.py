"""
Baseline Tier 1 (Fase 5.b): Regresión Logística sobre features del catálogo TOI.

Contrato del proyecto (ver CLAUDE.md):
- Clases binarias estrictas: CP=1, FP=0. PC NO entra (los splits de Fase 4 ya lo
  excluyen). No reconstruimos la etiqueta desde `tfopwg_disp`: confiamos en la
  columna `label` de `train_tics.csv` / `val_tics.csv` / `test_tics.csv`.
- Universo idéntico al de CNN/Mamba: las train + val + test TICs producidas
  por Fase 4 (ya restringidas a las que tienen curva procesada).
- Test sellado: este script SOLO toca test cuando se invoca con `--split test`.
  Cada llamada con `--split test` cuenta como UNA evaluación del test sellado.

Ejecutar desde la raíz del repo:

    python scripts/train_logreg.py                  # eval por defecto en val
    python scripts/train_logreg.py --split val      # idem
    python scripts/train_logreg.py --split train    # debug
    python scripts/train_logreg.py --split test     # SOLO UNA VEZ por modelo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURES = ["pl_orbper", "pl_trandep", "st_tmag"]
LABEL_COLUMN = "label"
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "experiments"
# Output por split. Mantenemos `logreg_baseline.txt` como output histórico de val
# (no romper compatibilidad). Para train/test, distinguimos en el nombre.
OUTPUT_FILES = {
    "val": OUTPUT_DIR / "logreg_baseline.txt",
    "train": OUTPUT_DIR / "logreg_baseline_train.txt",
    "test": OUTPUT_DIR / "logreg_baseline_test.txt",
}
# predictions.csv se escribe junto al .txt cuando split != val (val ya tiene
# su archivo histórico sin predictions). Para test, esto es lo que consume
# plot_tier1_comparison.py.
PREDICTIONS_FILES = {
    "test": OUTPUT_DIR / "logreg_baseline_test_predictions.csv",
    "train": OUTPUT_DIR / "logreg_baseline_train_predictions.csv",
}


def _load_summary_dedup() -> pd.DataFrame:
    """Carga toi_summary y deduplica por TIC.

    Una misma estrella (TIC) puede tener múltiples TOIs en el catálogo. Sin
    deduplicar, el merge con los splits infla el train (sesgado hacia positivos
    porque las estrellas con planeta tienden a tener más TOIs). Tomamos la
    primera fila por TIC: las features del catálogo (period/depth/tmag) son
    consistentes a nivel de estrella para el alcance de este baseline.
    """
    summary = pd.read_csv(DATA_DIR / "toi_summary.csv")
    summary["tid"] = summary["tid"].astype(str)
    summary = summary.drop_duplicates(subset="tid", keep="first").reset_index(drop=True)
    return summary[["tid", *FEATURES]]


def _load_split_with_features(split_name: str, summary: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}_tics.csv"
    split = pd.read_csv(path)
    split["tid"] = split["tid"].astype(str)
    n_before = len(split)
    merged = split.merge(summary, on="tid", how="inner")
    # Defensa: el merge no debe agregar filas (summary ya deduplicado).
    if len(merged) != n_before:
        raise RuntimeError(
            f"El merge con toi_summary inflo {split_name}: {n_before} -> {len(merged)}. "
            "Esto indica TICs duplicados en summary y rompe la comparación con CNN/Mamba."
        )
    return merged


def load_split(split_name: str) -> pd.DataFrame:
    """Carga un split arbitrario (train/val/test) con sus features."""
    summary = _load_summary_dedup()
    return _load_split_with_features(split_name, summary)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibilidad: devuelve (train_df, val_df) como antes."""
    summary = _load_summary_dedup()
    train_df = _load_split_with_features("train", summary)
    val_df = _load_split_with_features("val", summary)
    return train_df, val_df


def _dropna_and_split_xy(
    df: pd.DataFrame, split_label: str
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    required = FEATURES + [LABEL_COLUMN]
    n_before = len(df)
    df = df.dropna(subset=required).copy()
    print(
        f"{split_label}: {n_before} -> {len(df)} "
        f"(perdidos por NaN en features: {n_before - len(df)})"
    )
    print(f"\nDistribución de clases ({split_label}):")
    print(df[LABEL_COLUMN].value_counts().sort_index())
    X = df[FEATURES]
    y = df[LABEL_COLUMN].astype(int)
    return X, y, len(df), n_before


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int]:
    """Compatibilidad: prepara train+val. Para nuevos splits arbitrarios, usar
    `prepare_train_xy` + `prepare_eval_xy` desde `evaluate_on_split`.
    """
    X_train, y_train, n_train, _ = _dropna_and_split_xy(train_df, "Train")
    X_val, y_val, n_val, _ = _dropna_and_split_xy(val_df, "Val")
    return X_train, y_train, X_val, y_val, n_train, n_val


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit SOLO en train
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(X_train_scaled, y_train)
    return scaler, model


def _compute_metrics(y_true: pd.Series, y_pred, y_prob) -> dict:
    """Métricas estándar. Igual contrato que evaluate_model original + AUC-PR."""
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["FP (0)", "CP (1)"]
        ),
    }


def evaluate_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Compatibilidad: evalúa el modelo en (X_val, y_val) ya preparados."""
    X_val_scaled = scaler.transform(X_val)  # transform, NO fit_transform
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    return _compute_metrics(y_val, y_pred, y_prob)


def evaluate_on_split(
    scaler: StandardScaler,
    model: LogisticRegression,
    split_name: str,
    summary: dict[str, Any] | None = None,
) -> dict:
    """Evalúa el modelo ya entrenado sobre un split arbitrario por nombre.

    Args:
        scaler: StandardScaler fiteado SOLO en train.
        model: LogisticRegression ya entrenado.
        split_name: "train" | "val" | "test".
        summary: dict mutable opcional donde se inyectan tic_id, y_true, y_prob,
            y_pred para que el caller los guarde en predictions.csv. Si es None,
            no se exportan.

    Returns:
        Métricas + tamaño efectivo + report.
    """
    df = load_split(split_name)
    label_for_print = split_name.capitalize()
    X, y, n_used, n_total = _dropna_and_split_xy(df, label_for_print)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = _compute_metrics(y_pred=y_pred, y_prob=y_prob, y_true=y)
    metrics["n_used"] = n_used
    metrics["n_total"] = n_total

    if summary is not None:
        # tic_id alineado con las filas conservadas tras dropna
        kept_tids = df.dropna(subset=FEATURES + [LABEL_COLUMN])["tid"].tolist()
        summary["tic_id"] = kept_tids
        summary["y_true"] = y.tolist()
        summary["y_prob"] = y_prob.tolist()
        summary["y_pred"] = y_pred.tolist()

    return metrics


def get_feature_importance(model: LogisticRegression) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURES,
        "coeficiente": model.coef_[0],
        "importancia_abs": abs(model.coef_[0]),
    }).sort_values(by="importancia_abs", ascending=False)


def save_report(
    metrics: dict,
    feature_importance: pd.DataFrame,
    train_rows: int,
    val_rows: int,
    *,
    split_name: str = "val",
    output_path: Path | None = None,
) -> Path:
    """Escribe el reporte de texto al disco.

    Mantiene firma original con default `split_name="val"` para compatibilidad
    de la versión histórica que escribe `experiments/logreg_baseline.txt`.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_path or OUTPUT_FILES[split_name]
    lines = []
    lines.append("Baseline Tier 1 - Catalog-feature Logistic Regression (Fase 5.b)\n")
    lines.append("=" * 64 + "\n\n")
    lines.append("Etiquetas: CP=1, FP=0 (PC excluido por los splits de Fase 4).\n")
    lines.append(f"Seed: {SEED}\n")
    lines.append(f"Split evaluado: {split_name}\n\n")
    lines.append("Features (todas del catálogo TOI, sin BLS):\n")
    for f in FEATURES:
        lines.append(f"  - {f}\n")
    lines.append(
        f"\nFilas usadas (post-dropna):\n  - Train: {train_rows}\n  - {split_name.capitalize()}:   {val_rows}\n"
    )
    lines.append(f"\nMétricas en {split_name}:\n")
    lines.append(f"  - AUC-ROC:   {metrics['auc_roc']:.4f}\n")
    if "auc_pr" in metrics:
        lines.append(f"  - AUC-PR:    {metrics['auc_pr']:.4f}\n")
    lines.append(f"  - F1:        {metrics['f1']:.4f}\n")
    lines.append(f"  - Recall:    {metrics['recall']:.4f}\n")
    lines.append(f"  - Precision: {metrics['precision']:.4f}\n")
    lines.append("\nClassification report:\n")
    lines.append(metrics["classification_report"])
    lines.append("\nCoeficientes (sobre features estandarizadas):\n")
    lines.append(feature_importance.to_string(index=False))
    lines.append("\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _write_predictions_csv(path: Path, payload: dict[str, Any]) -> Path:
    """Escribe predictions.csv con columnas (tic_id, y_true, y_prob, y_pred).

    Mismo schema que `scripts/evaluate.py` para que `ensemble_eval.py` y
    `plot_tier1_comparison.py` lean LogReg con el mismo loader.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tic_id", "y_true", "y_prob", "y_pred"])
        for tic, yt, yp, yh in zip(
            payload["tic_id"], payload["y_true"], payload["y_prob"], payload["y_pred"], strict=True
        ):
            w.writerow([int(tic), int(yt), float(yp), int(yh)])
    return path


def _print_warning_test_sealed() -> None:
    bar = "!" * 78
    print()
    print(bar)
    print("!!" + " " * 74 + "!!")
    print("!!  WARNING: EVALUANDO TEST SELLADO - SOLO UNA VEZ POR MODELO            !!")
    print("!!  Esta corrida consume la única evaluación de test del LogReg baseline. !!")
    print("!!  Re-evaluar invalida el reporte final (data leakage por uso múltiple). !!")
    print("!!" + " " * 74 + "!!")
    print(bar)
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Entrena LogReg baseline (Fase 5.b) y evalúa sobre el split pedido. "
            "Sin --split, replica el comportamiento histórico (eval en val)."
        )
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help=(
            "Split sobre el que evaluar el modelo entrenado. Default 'val'. "
            "'test' es sellado: imprime warning y solo debe correrse UNA vez."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # SIEMPRE entrenamos sobre train (no cambia con --split). Esta es la
    # única diferencia con la versión anterior: ahora también escribimos
    # predictions.csv cuando split != val.
    train_df = load_split("train")
    X_train, y_train, n_train, _ = _dropna_and_split_xy(train_df, "Train")
    scaler, model = train_model(X_train, y_train)
    feature_importance = get_feature_importance(model)

    if args.split == "test":
        _print_warning_test_sealed()

    pred_payload: dict[str, Any] = {}
    metrics = evaluate_on_split(scaler, model, args.split, summary=pred_payload)

    print(f"\nMétricas en {args.split}:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("\nImportancia de features (|coef| sobre features estandarizadas):")
    print(feature_importance.to_string(index=False))

    out_path = save_report(
        metrics,
        feature_importance,
        n_train,
        metrics["n_used"],
        split_name=args.split,
    )
    print(f"\nReporte guardado en: {out_path}")

    if args.split in PREDICTIONS_FILES:
        pred_path = _write_predictions_csv(PREDICTIONS_FILES[args.split], pred_payload)
        print(f"Predicciones guardadas en: {pred_path}")


if __name__ == "__main__":
    main()
