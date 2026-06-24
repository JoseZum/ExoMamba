"""Estadística inferencial sobre el test sellado para el paper.

Calcula, a partir de los predictions.csv ya generados (sin re-evaluar modelos):

  1. Intervalos de confianza al 95 % por bootstrap estratificado (2000 resamples)
     para AUC-ROC y AUC-PR de cada modelo.
  2. Test de DeLong (Sun & Xu 2014, versión rápida) para comparar dos AUC-ROC
     correlacionadas sobre las MISMAS muestras (p-valor de H0: AUC_A == AUC_B).
  3. Bootstrap pareado de la diferencia de AUC (delta) con su IC 95 %.

Uso:
    python scripts/compute_stats.py

Salida:
    paper/results/statistics.json   — todas las cifras
    stdout                          — tabla legible
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"
EXP = ROOT / "experiments"
RNG = np.random.default_rng(42)
N_BOOT = 2000

# (nombre, ruta del csv, columna de probabilidad)
MODELS = {
    "random": (EXP / "2026-05-21_05-36-11_random_baseline/eval_test/predictions.csv", "y_prob"),
    "cnn_single": (EXP / "2026-05-20_23-44-48_cnn_baseline/eval_test/predictions.csv", "y_prob"),
    "mamba_locked": (EXP / "2026-05-22_14-32-51_mamba_small/eval_test/predictions.csv", "y_prob"),
    "mamba_best_seed789": (EXP / "2026-05-28_01-44-54_mamba_small_seed789/eval_test/predictions.csv", "y_prob"),
    "mamba_ensemble": (RESULTS / "mamba_ensemble/ensemble_predictions.csv", "y_prob_mean"),
    "astronet_ensemble": (RESULTS / "astronet_ensemble/ensemble_predictions.csv", "y_prob_mean"),
    "exomamba_v1_ensemble": (RESULTS / "exomamba_v1_ensemble/ensemble_predictions.csv", "y_prob_mean"),
}


def load(path: Path, prob_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={prob_col: "y_prob"})
    return df[["tic_id", "y_true", "y_prob"]].copy()


def boot_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot=N_BOOT):
    """IC 95 % por bootstrap estratificado (resamplea dentro de cada clase)."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    point = metric_fn(y_true, y_prob)
    stats = []
    for _ in range(n_boot):
        ip = RNG.choice(pos, size=len(pos), replace=True)
        ineg = RNG.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


# ----------------------------------------------------------------------------
# DeLong rápido (Sun & Xu 2014). Devuelve AUCs, varianza/covarianza y p-valor.
# ----------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, prob_a, prob_b):
    """p-valor (dos colas) de H0: AUC_a == AUC_b sobre las mismas muestras."""
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    var = l @ cov @ l.T
    var = float(var[0, 0])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float("inf"), "p_value": 0.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": float(aucs[0] - aucs[1]), "z": float(z), "p_value": float(p)}


def paired_delta_ci(y_true, prob_a, prob_b, n_boot=N_BOOT):
    """IC 95 % de (AUC_a - AUC_b) por bootstrap estratificado pareado."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, len(pos), True), RNG.choice(neg, len(neg), True)])
        deltas.append(roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_point": float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)),
            "ci_low": float(lo), "ci_high": float(hi)}


def main():
    data = {name: load(p, c) for name, (p, c) in MODELS.items() if p.exists()}
    out = {"n_boot": N_BOOT, "ci": {}, "pairwise": {}}

    print(f"\n=== Bootstrap CIs (95%, {N_BOOT} resamples estratificados) ===\n")
    print(f"{'modelo':<24} {'N':>4}  AUC-ROC [95% CI]            AUC-PR [95% CI]")
    for name, df in data.items():
        yt, yp = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
        roc = boot_ci(yt, yp, roc_auc_score)
        pr = boot_ci(yt, yp, average_precision_score)
        out["ci"][name] = {"n": int(len(df)), "auc_roc": roc, "auc_pr": pr}
        print(f"{name:<24} {len(df):>4}  "
              f"{roc['point']:.3f} [{roc['ci_low']:.3f}, {roc['ci_high']:.3f}]   "
              f"{pr['point']:.3f} [{pr['ci_low']:.3f}, {pr['ci_high']:.3f}]")

    def aligned(a, b):
        m = data[a].merge(data[b], on="tic_id", suffixes=("_a", "_b"))
        assert (m["y_true_a"] == m["y_true_b"]).all()
        return m["y_true_a"].to_numpy(), m["y_prob_a"].to_numpy(), m["y_prob_b"].to_numpy(), len(m)

    print("\n=== DeLong + bootstrap pareado (mismas muestras) ===\n")
    comparisons = [
        ("mamba_ensemble", "cnn_single"),
        ("mamba_ensemble", "mamba_locked"),
        ("mamba_ensemble", "astronet_ensemble"),
        ("mamba_best_seed789", "cnn_single"),
        ("astronet_ensemble", "cnn_single"),
        ("astronet_ensemble", "exomamba_v1_ensemble"),
    ]
    for a, b in comparisons:
        if a not in data or b not in data:
            continue
        yt, pa, pb, n = aligned(a, b)
        dl = delong_test(yt, pa, pb)
        ci = paired_delta_ci(yt, pa, pb)
        key = f"{a}__vs__{b}"
        out["pairwise"][key] = {"n": n, "delong": dl, "delta_ci": ci}
        print(f"{a} vs {b}  (N={n})")
        print(f"   AUC {dl['auc_a']:.3f} vs {dl['auc_b']:.3f} | "
              f"delta={ci['delta_point']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] | "
              f"DeLong z={dl['z']:.2f}, p={dl['p_value']:.2e}")

    (RESULTS / "statistics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGuardado: {RESULTS / 'statistics.json'}")


if __name__ == "__main__":
    main()
