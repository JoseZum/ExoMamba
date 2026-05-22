"""Métricas de clasificación binaria para el training loop."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula AUC-ROC, AUC-PR, F1, Recall y Precision para clasificación binaria.

    Args:
        y_true: array de 0/1.
        y_prob: probabilidades predichas para la clase positiva (1).
        threshold: umbral para convertir prob → predicción dura. Default 0.5.

    Returns:
        Dict con {auc_roc, auc_pr, f1, recall, precision, threshold}.
        Si solo hay una clase en y_true, las métricas que requieren ambas
        clases (AUC) se reportan como NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return {
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
            "f1": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "threshold": threshold,
        }

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
