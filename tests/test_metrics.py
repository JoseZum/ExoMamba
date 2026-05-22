"""Pruebas de compute_classification_metrics con valores conocidos."""

from __future__ import annotations

import numpy as np

from exoplanet.training import compute_classification_metrics


def test_clasificador_perfecto() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.95])
    m = compute_classification_metrics(y_true, y_prob)
    assert m["auc_roc"] == 1.0
    assert m["f1"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_clasificador_inverso() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.95, 0.1, 0.2])
    m = compute_classification_metrics(y_true, y_prob)
    assert m["auc_roc"] == 0.0  # inverso perfecto
    assert m["recall"] == 0.0


def test_una_sola_clase_devuelve_nan() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6])
    m = compute_classification_metrics(y_true, y_prob)
    assert np.isnan(m["auc_roc"])
    assert np.isnan(m["f1"])


def test_threshold_custom() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.3, 0.4, 0.45, 0.6])
    # Con threshold=0.5: predicción es [0, 0, 0, 1] → recall=0.5
    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["recall"] == 0.5
    # Con threshold=0.35: predicción es [0, 1, 1, 1] → recall=1.0
    m2 = compute_classification_metrics(y_true, y_prob, threshold=0.35)
    assert m2["recall"] == 1.0
