"""
Baseline Tier 1 (Fase 5.b): Regresión Logística sobre features del catálogo TOI.

Contrato del proyecto (ver CLAUDE.md):
- Clases binarias estrictas: CP=1, FP=0. PC NO entra (los splits de Fase 4 ya lo
  excluyen). No reconstruimos la etiqueta desde `tfopwg_disp`: confiamos en la
  columna `label` de `train_tics.csv` / `val_tics.csv`.
- Universo idéntico al de CNN/Mamba: las 1103 train + 237 val TICs producidas
  por Fase 4 (ya restringidas a las que tienen curva procesada).
- Test sellado: este script NUNCA toca `test_tics.csv`.

Ejecutar desde la raíz del repo:

    python scripts/train_logreg.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
OUTPUT_FILE = OUTPUT_DIR / "logreg_baseline.txt"


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _load_summary_dedup()
    train_df = _load_split_with_features("train", summary)
    val_df = _load_split_with_features("val", summary)
    return train_df, val_df


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int]:
    required = FEATURES + [LABEL_COLUMN]

    train_before, val_before = len(train_df), len(val_df)
    train_df = train_df.dropna(subset=required).copy()
    val_df = val_df.dropna(subset=required).copy()

    print(f"Train: {train_before} -> {len(train_df)} (perdidos por NaN en features: {train_before - len(train_df)})")
    print(f"Val:   {val_before} -> {len(val_df)} (perdidos por NaN en features: {val_before - len(val_df)})")

    print("\nDistribución de clases (train):")
    print(train_df[LABEL_COLUMN].value_counts().sort_index())
    print("\nDistribución de clases (val):")
    print(val_df[LABEL_COLUMN].value_counts().sort_index())

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL_COLUMN].astype(int)
    X_val = val_df[FEATURES]
    y_val = val_df[LABEL_COLUMN].astype(int)
    return X_train, y_train, X_val, y_val, len(train_df), len(val_df)


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


def evaluate_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    X_val_scaled = scaler.transform(X_val)  # transform, NO fit_transform
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    return {
        "auc_roc": roc_auc_score(y_val, y_prob),
        "f1": f1_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "classification_report": classification_report(
            y_val, y_pred, target_names=["FP (0)", "CP (1)"]
        ),
    }


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
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("Baseline Tier 1 — Catalog-feature Logistic Regression (Fase 5.b)\n")
    lines.append("=" * 64 + "\n\n")
    lines.append("Etiquetas: CP=1, FP=0 (PC excluido por los splits de Fase 4).\n")
    lines.append(f"Seed: {SEED}\n\n")
    lines.append("Features (todas del catálogo TOI, sin BLS):\n")
    for f in FEATURES:
        lines.append(f"  - {f}\n")
    lines.append(f"\nFilas usadas (post-dropna):\n  - Train: {train_rows}\n  - Val:   {val_rows}\n")
    lines.append("\nMétricas en validación:\n")
    lines.append(f"  - AUC-ROC:   {metrics['auc_roc']:.4f}\n")
    lines.append(f"  - F1:        {metrics['f1']:.4f}\n")
    lines.append(f"  - Recall:    {metrics['recall']:.4f}\n")
    lines.append(f"  - Precision: {metrics['precision']:.4f}\n")
    lines.append("\nClassification report:\n")
    lines.append(metrics["classification_report"])
    lines.append("\nCoeficientes (sobre features estandarizadas):\n")
    lines.append(feature_importance.to_string(index=False))
    lines.append("\n")
    OUTPUT_FILE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    train_df, val_df = load_data()
    X_train, y_train, X_val, y_val, n_train, n_val = prepare_xy(train_df, val_df)
    scaler, model = train_model(X_train, y_train)
    metrics = evaluate_model(scaler, model, X_val, y_val)
    feature_importance = get_feature_importance(model)

    print("\nMétricas en validación:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("\nImportancia de features (|coef| sobre features estandarizadas):")
    print(feature_importance.to_string(index=False))

    save_report(metrics, feature_importance, n_train, n_val)
    print(f"\nReporte guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
