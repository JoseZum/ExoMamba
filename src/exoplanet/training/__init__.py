"""Infraestructura de entrenamiento compartida (Fase 7)."""

from exoplanet.training.collate import collate_lightcurves
from exoplanet.training.config import dump_config, load_config
from exoplanet.training.loop import evaluate_one_epoch, train_one_epoch
from exoplanet.training.metrics import compute_classification_metrics
from exoplanet.training.registry import MODEL_REGISTRY, build_model
from exoplanet.training.runner import run_training

__all__ = [
    "collate_lightcurves",
    "compute_classification_metrics",
    "load_config",
    "dump_config",
    "build_model",
    "MODEL_REGISTRY",
    "train_one_epoch",
    "evaluate_one_epoch",
    "run_training",
]
