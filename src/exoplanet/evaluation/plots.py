"""Plots de evaluación (Fase 9).

Funciones puras: reciben arrays de numpy + path de salida, guardan PNG y
devuelven el path. Diseñadas para alimentarse del output de
`evaluate_one_epoch` o del CSV `predictions.csv` que escribe
`scripts/evaluate.py`.

Estilo: matplotlib + seaborn, figsize=(8,6), dpi=120, paleta consistente
("colorblind" de seaborn). Sin display interactivo (backend Agg implícito al
no llamar plt.show).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Paleta y estilo compartidos
_PALETTE = sns.color_palette("colorblind")
_FIGSIZE = (8, 6)
_DPI = 120
sns.set_theme(style="whitegrid", context="notebook")


def _ensure_parent(output_path: Path | str) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | str,
    label: str | None = None,
) -> Path:
    """ROC curve con AUC en la leyenda. Diagonal aleatoria como referencia."""
    p = _ensure_parent(output_path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    legend_label = f"{label or 'model'} (AUC={roc_auc:.4f})"

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(fpr, tpr, color=_PALETTE[0], lw=2, label=legend_label)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | str,
    label: str | None = None,
) -> Path:
    """Precision-Recall curve. La línea de prior de la clase positiva sirve de baseline."""
    p = _ensure_parent(output_path)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    legend_label = f"{label or 'model'} (AP={pr_auc:.4f})"
    pos_prior = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(recall, precision, color=_PALETTE[1], lw=2, label=legend_label)
    ax.axhline(
        pos_prior,
        color="gray",
        lw=1,
        linestyle="--",
        label=f"Prior positivo ({pos_prior:.3f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | str,
    class_names: tuple[str, str] = ("FP", "CP"),
    normalize: bool = False,
) -> Path:
    """Matriz de confusión 2x2 con anotaciones. CP=positiva, FP=negativa."""
    p = _ensure_parent(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | str,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram: probabilidad predicha vs fracción observada de positivos.

    Un modelo bien calibrado cae sobre la diagonal. Saber si el modelo está
    sobre-confiado o sub-confiado importa para el agente (Fase 13).
    """
    p = _ensure_parent(output_path)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(mean_pred, frac_pos, marker="o", color=_PALETTE[2], lw=2, label="Modelo")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Calibración perfecta")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Probabilidad predicha media (por bin)")
    ax.set_ylabel("Fracción observada de positivos")
    ax.set_title(f"Reliability diagram ({n_bins} bins)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_comparison_roc(
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path | str,
) -> Path:
    """ROC comparativa de múltiples modelos en una sola figura.

    Args:
        runs: dict {nombre_modelo: (y_true, y_prob)}. Cada modelo aparece con
            su propio color y AUC en la leyenda.
        output_path: dónde guardar el PNG.

    Pensado para la tabla comparativa Tier 1 del paper.
    """
    p = _ensure_parent(output_path)
    if not runs:
        raise ValueError("Se requiere al menos un run para comparar.")

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    for i, (name, (y_true, y_prob)) in enumerate(runs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - Comparación de modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


__all__ = [
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_calibration",
    "plot_comparison_roc",
]
