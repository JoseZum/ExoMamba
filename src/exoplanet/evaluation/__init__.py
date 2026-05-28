"""Evaluación (Fase 9): métricas compartidas, plots y XAI.

Reexporta lo que el resto del proyecto (y `scripts/evaluate.py`) consume.
Mantener este `__init__` ligero — no importar nada pesado (mamba_ssm,
matplotlib backends interactivos) acá; los submódulos lo hacen.
"""

from exoplanet.evaluation.plots import (
    plot_calibration,
    plot_comparison_roc,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from exoplanet.evaluation.xai import (
    gradient_saliency,
    integrated_gradients,
    occlusion_sensitivity,
    plot_xai_overlay,
)
from exoplanet.training.metrics import compute_classification_metrics

__all__ = [
    "compute_classification_metrics",
    # plots
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_calibration",
    "plot_comparison_roc",
    # xai
    "gradient_saliency",
    "integrated_gradients",
    "occlusion_sensitivity",
    "plot_xai_overlay",
]
