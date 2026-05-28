"""
Entrenamiento de baseline con Regresión Logística.

Ejecutar desde la raíz del repositorio:

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "logreg_baseline.txt"


def normalize_tid(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que la columna tid tenga el mismo tipo en todos los archivos."""
    df = df.copy()

    if "tid" not in df.columns:
        raise KeyError(
            f"No se encontró la columna 'tid'. Columnas disponibles: {df.columns.tolist()}"
        )

    df["tid"] = df["tid"].astype(str)

    return df


def create_label_from_tfopwg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna label a partir de tfopwg_disp.

    FP = 0
    PC = 1
    CP = 1
    """

    df = df.copy()

    if "tfopwg_disp" not in df.columns:
        if "tfopwg_disp_x" in df.columns:
            df["tfopwg_disp"] = df["tfopwg_disp_x"]
        elif "tfopwg_disp_y" in df.columns:
            df["tfopwg_disp"] = df["tfopwg_disp_y"]
        else:
            raise KeyError(
                "No se encontró 'tfopwg_disp', 'tfopwg_disp_x' ni 'tfopwg_disp_y'. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )

    label_map = {
        "FP": 0,
        "PC": 1,
        "CP": 1,
    }

    df[LABEL_COLUMN] = (
        df["tfopwg_disp"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(label_map)
    )

    return df


def fix_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige el problema de label_x y label_y después de los merge.

    Prioridad:
    1. Si existe label, se usa.
    2. Si existe label_y, se renombra a label.
    3. Si existe label_x, se renombra a label.
    4. Si no existe ninguna, se crea desde tfopwg_disp.
    """

    df = df.copy()

    if LABEL_COLUMN in df.columns:
        return df

    if "label_y" in df.columns:
        df[LABEL_COLUMN] = df["label_y"]
        return df

    if "label_x" in df.columns:
        df[LABEL_COLUMN] = df["label_x"]
        return df

    df = create_label_from_tfopwg(df)

    return df


def fix_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige columnas duplicadas creadas por pandas después del merge.

    Si existen columnas como:
    - st_tmag_x / st_tmag_y
    - pl_orbper_x / pl_orbper_y

    crea columnas limpias:
    - st_tmag
    - pl_orbper
    """

    df = df.copy()

    for feature in FEATURES:
        if feature in df.columns:
            continue

        feature_x = f"{feature}_x"
        feature_y = f"{feature}_y"

        if feature_x in df.columns:
            df[feature] = df[feature_x]
        elif feature_y in df.columns:
            df[feature] = df[feature_y]
        else:
            raise KeyError(
                f"No se encontró la feature '{feature}', ni sus variantes "
                f"'{feature_x}' o '{feature_y}'. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y une los datos de entrenamiento y validación."""

    summary_path = DATA_DIR / "toi_summary.csv"
    train_path = DATA_DIR / "train_tics.csv"
    val_path = DATA_DIR / "val_tics.csv"
    labels_path = DATA_DIR / "tics_labeled.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {summary_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {val_path}")

    summary = pd.read_csv(summary_path)
    train_split = pd.read_csv(train_path)
    val_split = pd.read_csv(val_path)

    summary = normalize_tid(summary)
    train_split = normalize_tid(train_split)
    val_split = normalize_tid(val_split)

    train_df = train_split.merge(summary, on="tid", how="inner")
    val_df = val_split.merge(summary, on="tid", how="inner")

    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        labels = normalize_tid(labels)

        train_df = train_df.merge(labels, on="tid", how="inner")
        val_df = val_df.merge(labels, on="tid", how="inner")

    train_df = fix_label_column(train_df)
    val_df = fix_label_column(val_df)

    train_df = fix_feature_columns(train_df)
    val_df = fix_feature_columns(val_df)

    return train_df, val_df


def check_columns(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Verifica que existan las columnas necesarias."""

    required_columns = FEATURES + [LABEL_COLUMN]

    missing_train = [col for col in required_columns if col not in train_df.columns]
    missing_val = [col for col in required_columns if col not in val_df.columns]

    if missing_train or missing_val:
        raise KeyError(
            "Faltan columnas necesarias.\n"
            f"Faltan en train: {missing_train}\n"
            f"Faltan en val: {missing_val}\n"
            f"Columnas train: {train_df.columns.tolist()}\n"
            f"Columnas val: {val_df.columns.tolist()}"
        )


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int]:
    """Limpia valores faltantes y separa X e y."""

    check_columns(train_df, val_df)

    required_columns = FEATURES + [LABEL_COLUMN]

    train_before = len(train_df)
    val_before = len(val_df)

    train_df = train_df.dropna(subset=required_columns).copy()
    val_df = val_df.dropna(subset=required_columns).copy()

    train_after = len(train_df)
    val_after = len(val_df)

    print(f"Train antes de dropna: {train_before}")
    print(f"Train después de dropna: {train_after}")
    print(f"Val antes de dropna: {val_before}")
    print(f"Val después de dropna: {val_after}")

    print("\nDistribución de clases en train:")
    print(train_df[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribución de clases en val:")
    print(val_df[LABEL_COLUMN].value_counts().sort_index())

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL_COLUMN].astype(int)

    X_val = val_df[FEATURES]
    y_val = val_df[LABEL_COLUMN].astype(int)

    return X_train, y_train, X_val, y_val, train_after, val_after


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StandardScaler, LogisticRegression]:
    """Escala los datos de entrenamiento y entrena la regresión logística."""

    scaler = StandardScaler()

    # Se usa fit_transform SOLO en train para evitar data leakage.
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    return scaler, model


def evaluate_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Evalúa el modelo en validación."""

    # En validación se usa transform, no fit_transform.
    X_val_scaled = scaler.transform(X_val)

    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y_val, y_prob),
        "f1": f1_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "classification_report": classification_report(
            y_val,
            y_pred,
            target_names=["FP (0)", "PC/CP (1)"],
        ),
    }

    return metrics


def get_feature_importance(model: LogisticRegression) -> pd.DataFrame:
    """Devuelve los coeficientes del modelo como tabla ordenada."""

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "coeficiente": model.coef_[0],
            "importancia_abs": abs(model.coef_[0]),
        }
    )

    return importance.sort_values(by="importancia_abs", ascending=False)


def save_report(
    metrics: dict,
    feature_importance: pd.DataFrame,
    train_rows: int,
    val_rows: int,
) -> None:
    """Guarda los resultados del baseline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("Baseline de Regresión Logística\n")
    report.append("=" * 40 + "\n\n")

    report.append("Features usadas:\n")
    for feature in FEATURES:
        report.append(f"- {feature}\n")

    report.append("\nEtiqueta usada:\n")
    report.append("- FP = 0\n")
    report.append("- PC/CP = 1\n")

    report.append("\nFilas después de eliminar valores faltantes:\n")
    report.append(f"- Train: {train_rows}\n")
    report.append(f"- Val: {val_rows}\n")

    report.append("\nMétricas en validación:\n")
    report.append(f"- AUC-ROC: {metrics['auc_roc']:.4f}\n")
    report.append(f"- F1: {metrics['f1']:.4f}\n")
    report.append(f"- Recall: {metrics['recall']:.4f}\n")
    report.append(f"- Precision: {metrics['precision']:.4f}\n")

    report.append("\nClassification report:\n")
    report.append(metrics["classification_report"])

    report.append("\nCoeficientes del modelo:\n")
    report.append(feature_importance.to_string(index=False))
    report.append("\n")

    OUTPUT_FILE.write_text("".join(report), encoding="utf-8")


def main() -> None:
    """Función principal."""

    train_df, val_df = load_data()

    X_train, y_train, X_val, y_val, train_rows, val_rows = prepare_xy(
        train_df,
        val_df,
    )

    scaler, model = train_model(X_train, y_train)

    metrics = evaluate_model(
        scaler,
        model,
        X_val,
        y_val,
    )

    feature_importance = get_feature_importance(model)

    print("\nMétricas en validación:")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")

    print("\nClassification report:")
    print(metrics["classification_report"])

    print("\nImportancia de features según coeficientes:")
    print(feature_importance.to_string(index=False))

    save_report(
        metrics,
        feature_importance,
        train_rows,
        val_rows,
    )

    print(f"\nReporte guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()