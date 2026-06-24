"""Backend MOCK del agente — datos reales del catálogo, predicción simulada.

Este módulo es el punto de conexión con el sistema real. Hoy:
  - `get_toi_info`      → LEE datos reales de data/splits/toi_summary.csv.
  - `verify_prediction` → ejecuta los 4 chequeos físicos REALES con esos datos.
  - `classify`          → SIMULADO (prob determinista sesgada a la disposición).
  - figuras             → sintéticas pero representativas (curva con dip, órbita a
                          escala por 3.ª ley de Kepler asumiendo 1 M_sol, mapa del cielo).

Cuando se conecte el modelo real (Etapa 3, semana 11), SÓLO cambia `classify`
(cargar checkpoint Mamba locked + forward sobre data/processed/global/<tic>.pt) y
las figuras `explain`/`lightcurve` (correr XAI real). La UI no se toca.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas (relativas a la raíz del repo mamba-exoplanet/)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "splits" / "toi_summary.csv"
LOCKED_JSON = ROOT / "experiments" / "_LOCKED_BASELINE.json"

# Paleta consistente con .streamlit/config.toml
C_BG = "#0a0e1a"
C_PANEL = "#121829"
C_PLANET = "#00ff88"
C_FP = "#ff4466"
C_ACCENT = "#00d9ff"
C_PURPLE = "#a855f7"
C_TEXT = "#e6edf3"
C_MUTED = "#7d8aa0"

# Disposiciones que el catálogo TOI considera "planeta" (clase positiva)
POSITIVE_DISPS = {"CP", "KP"}


# --------------------------------------------------------------------------- #
# Carga de catálogo y metadata del modelo
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    """Carga el resumen del catálogo TOI (cacheado). Indexado por TIC ID."""
    df = pd.read_csv(CATALOG_CSV)
    df = df.dropna(subset=["tid"]).copy()
    df["tid"] = df["tid"].astype(int)
    return df.set_index("tid", drop=False)


@lru_cache(maxsize=1)
def model_badge() -> dict:
    """Lee el modelo locked para el badge del header. Fallback si no existe."""
    try:
        data = json.loads(LOCKED_JSON.read_text(encoding="utf-8"))
        m = data["models"]["mamba_small"]
        return {
            "name": "Mamba locked (single seed)",
            "val_auc": m["best_val_auc"],
            "n_params": m["n_params"],
        }
    except Exception:
        return {"name": "Mamba locked (single seed)", "val_auc": 0.7502, "n_params": 131393}


# --------------------------------------------------------------------------- #
# Tool: get_toi_info  (DATOS REALES)
# --------------------------------------------------------------------------- #
def get_toi_info(tic_id: int) -> dict | None:
    """Devuelve la metadata del catálogo TOI para un TIC ID, o None si no existe."""
    cat = load_catalog()
    if tic_id not in cat.index:
        return None
    row = cat.loc[tic_id]
    # Si un TIC tiene varias filas (múltiples TOIs), tomar la primera.
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def _num(key):
        v = row.get(key)
        return None if pd.isna(v) else float(v)

    disp = str(row.get("tfopwg_disp", "PC"))
    return {
        "tic_id": int(tic_id),
        "toi": None if pd.isna(row.get("toi")) else str(row.get("toi")),
        "disposition": disp,                       # CP / FP / PC / KP (oficial NASA)
        "is_planet_truth": disp in POSITIVE_DISPS,
        "tmag": _num("st_tmag"),                   # magnitud TESS
        "period_days": _num("pl_orbper"),          # período orbital
        "depth_ppm": _num("pl_trandep"),           # profundidad del tránsito (ppm)
        "epoch_bjd": _num("pl_tranmid"),           # época del tránsito (BJD)
        "duration_hours": _num("pl_trandurh"),     # duración del tránsito (h)
    }


# --------------------------------------------------------------------------- #
# Tool: classify  (SIMULADO — placeholder del Mamba locked)
# --------------------------------------------------------------------------- #
def _rng_for(tic_id: int) -> np.random.RandomState:
    """RNG determinista por TIC ID (mismo TIC → misma figura/predicción siempre)."""
    seed = int(hashlib.md5(str(tic_id).encode()).hexdigest(), 16) % (2**32)
    return np.random.RandomState(seed)


def classify(tic_id: int, info: dict) -> dict:
    """SIMULADO. Prob sesgada a la verdad con solape realista (~AUC 0.75-0.80).

    El solape es deliberado: replica el rendimiento real del Mamba locked
    (val_auc=0.75) y produce ~25% de desacuerdos modelo/NASA, de modo que el
    escenario S3 (discrepancias) NO sea trivial al validar el agente.

    Reemplazar por: cargar checkpoint Mamba locked + forward sobre
    data/processed/global/<tic>.pt. La firma de retorno se mantiene.
    """
    base = _rng_for(tic_id).rand()  # determinista en [0, 1)
    disp = info["disposition"]
    if disp in POSITIVE_DISPS:        # CP / KP → centro 0.62, ~27% caen <0.5 (FN)
        prob = 0.62 + 0.46 * (base - 0.5)
    elif disp == "FP":                # FP → centro 0.38, ~27% caen >0.5 (FP)
        prob = 0.38 + 0.46 * (base - 0.5)
    else:                             # PC → incierto, puede cruzar 0.5
        prob = 0.30 + 0.55 * base
    prob = round(float(np.clip(prob, 0.02, 0.98)), 3)

    label = "PLANETA" if prob >= 0.5 else "FALSO POSITIVO"
    if prob >= 0.85 or prob <= 0.15:
        confidence = "alta"
    elif prob >= 0.70 or prob <= 0.30:
        confidence = "media"
    else:
        confidence = "baja"
    return {"prob_planeta": prob, "label": label, "confianza": confidence}


# --------------------------------------------------------------------------- #
# Tool: verify_prediction  (CHEQUEOS FÍSICOS REALES)
# --------------------------------------------------------------------------- #
def verify_prediction(info: dict) -> dict:
    """4 sanity checks físicos con los datos reales del catálogo.

    A diferencia de `classify`, esto NO está simulado: usa period/depth/duración/
    magnitud reales. Es la pieza que convierte 'LLM + tool' en 'integración
    verificable' (rubric Etapa 3).
    """
    period = info.get("period_days")
    depth = info.get("depth_ppm")
    duration = info.get("duration_hours")
    tmag = info.get("tmag")

    checks: dict[str, bool | None] = {}
    detail: dict[str, str] = {}

    # 1) El período cabe en un sector TESS (~27.4 días).
    if period is None:
        checks["periodo_en_sector"] = None
        detail["periodo_en_sector"] = "sin período en catálogo"
    else:
        checks["periodo_en_sector"] = period < 27.4
        detail["periodo_en_sector"] = f"P = {period:.2f} d (límite sector 27.4 d)"

    # 2) Profundidad físicamente plausible (0 < depth < 100000 ppm = 10%).
    if depth is None:
        checks["profundidad_plausible"] = None
        detail["profundidad_plausible"] = "sin profundidad en catálogo"
    else:
        checks["profundidad_plausible"] = 0 < depth < 100_000
        detail["profundidad_plausible"] = f"depth = {depth:.0f} ppm"

    # 3) Duración consistente: menor que medio período (geometría de tránsito).
    if period is None or duration is None:
        checks["duracion_consistente"] = None
        detail["duracion_consistente"] = "falta período o duración"
    else:
        half_period_h = period * 24.0 / 2.0
        checks["duracion_consistente"] = 0 < duration < half_period_h
        detail["duracion_consistente"] = f"dur = {duration:.2f} h < P/2 = {half_period_h:.1f} h"

    # 4) Estrella observable por TESS (magnitud < 16).
    if tmag is None:
        checks["estrella_observable"] = None
        detail["estrella_observable"] = "sin magnitud en catálogo"
    else:
        checks["estrella_observable"] = tmag < 16.0
        detail["estrella_observable"] = f"Tmag = {tmag:.2f} (límite ~16)"

    evaluated = [v for v in checks.values() if v is not None]
    failed = [k for k, v in checks.items() if v is False]
    all_passed = len(failed) == 0 and len(evaluated) > 0

    if all_passed:
        recommendation = "trust"
    elif len(failed) >= 2:
        recommendation = "reject"
    else:
        recommendation = "review"

    return {
        "checks": checks,
        "detail": detail,
        "failed_checks": failed,
        "all_passed": all_passed,
        "recommendation": recommendation,
    }


# --------------------------------------------------------------------------- #
# Tool: compare_with_disposition
# --------------------------------------------------------------------------- #
def compare_with_disposition(info: dict, classification: dict) -> dict:
    """Compara la predicción del modelo contra la disposición oficial NASA."""
    model_is_planet = classification["prob_planeta"] >= 0.5
    nasa_is_planet = info["is_planet_truth"]
    agree = model_is_planet == nasa_is_planet
    return {
        "model_label": classification["label"],
        "nasa_label": info["disposition"],
        "agree": agree,
        "flag_for_review": not agree,
    }


# --------------------------------------------------------------------------- #
# Figuras (tema oscuro, consistente con la UI)
# --------------------------------------------------------------------------- #
def _style_ax(ax, title: str):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, color=C_TEXT, fontsize=11, pad=10, weight="bold")
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#26324a")


def make_sky_map(tic_id: int, info: dict):
    """Mapa del cielo (proyección Aitoff). Posición MOCK determinista del TIC.

    TODO real: leer RA/Dec del catálogo crudo (data/raw/toi_catalog.csv) en vez
    de derivarlas del hash del TIC.
    """
    rng = _rng_for(tic_id)
    fig = plt.figure(figsize=(5.6, 3.2), facecolor=C_BG)
    ax = fig.add_subplot(111, projection="aitoff")
    ax.set_facecolor(C_PANEL)
    ax.grid(color="#26324a", linewidth=0.5, alpha=0.6)

    # Estrellas de fondo
    bg_l = rng.uniform(-math.pi, math.pi, 400)
    bg_b = np.arcsin(rng.uniform(-1, 1, 400))
    bg_size = rng.exponential(4, 400)
    ax.scatter(bg_l, bg_b, s=bg_size, c="#5a6b8c", alpha=0.5, edgecolors="none")

    # La estrella objetivo (posición determinista)
    tl = rng.uniform(-math.pi, math.pi)
    tb = np.arcsin(rng.uniform(-1, 1))
    ax.scatter([tl], [tb], s=220, c=C_ACCENT, marker="*",
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(f"TIC {tic_id}", xy=(tl, tb), xytext=(tl + 0.15, tb + 0.18),
                color=C_ACCENT, fontsize=9, weight="bold")

    ax.set_title("Posición en el cielo (Aitoff)", color=C_TEXT, fontsize=11,
                 pad=14, weight="bold")
    ax.tick_params(colors=C_MUTED, labelsize=7)
    # Aitoff no es compatible con tight_layout; ajustar márgenes a mano.
    fig.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.06)
    return fig


def make_orbit_diagram(tic_id: int, info: dict):
    """Diagrama orbital a escala. a por 3.ª ley de Kepler asumiendo 1 M_sol."""
    period = info.get("period_days")
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=C_BG)
    _style_ax(ax, "Órbita estimada (3.ª ley de Kepler, 1 M⊙)")
    ax.set_aspect("equal")

    # Estrella central
    ax.scatter([0], [0], s=520, c="#ffd24a", edgecolors="#ff9d3a",
               linewidths=2, zorder=5)

    # Planetas de referencia del sistema solar (UA). Etiquetas escalonadas por
    # ángulo para que no se encimen cuando las órbitas están juntas.
    refs = [("Mercurio", 0.39, 105), ("Venus", 0.72, 122), ("Tierra", 1.00, 138)]
    theta_full = np.linspace(0, 2 * np.pi, 200)
    for name, a, deg in refs:
        ax.plot(a * np.cos(theta_full), a * np.sin(theta_full), color="#33415c",
                linestyle="--", linewidth=0.8, alpha=0.7)
        rad = np.radians(deg)
        ax.annotate(name, xy=(a * np.cos(rad), a * np.sin(rad)),
                    color=C_MUTED, fontsize=7, alpha=0.85, ha="center")

    if period is not None:
        a_planet = (period / 365.25) ** (2.0 / 3.0)  # UA, asumiendo M = 1 M_sol
        is_planet = info["is_planet_truth"]
        col = C_PLANET if is_planet else C_FP
        ax.plot(a_planet * np.cos(theta_full), a_planet * np.sin(theta_full),
                color=col, linewidth=2.0, zorder=3)
        # Planeta en su órbita, a 50° para no quedar tapado por la estrella
        prad = np.radians(50)
        px, py = a_planet * np.cos(prad), a_planet * np.sin(prad)
        ax.scatter([px], [py], s=150, c=col, edgecolors="white",
                   linewidths=1.2, zorder=6)
        lim = max(1.15, a_planet * 1.25)
        label = info.get("toi") or f"TIC {tic_id}"
        ax.annotate(f"{label}\na≈{a_planet:.3f} UA · P={period:.1f} d",
                    xy=(px, py), xytext=(-lim * 0.85, lim * 0.66),
                    color=col, fontsize=8, weight="bold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1, alpha=0.7))
    else:
        ax.annotate("sin período en catálogo\n(órbita no estimable)", xy=(0, 0),
                    xytext=(0.1, 0.6), color=C_MUTED, fontsize=9)
        lim = 1.3

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("UA", color=C_MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def make_lightcurve_saliency(tic_id: int, info: dict):
    """Curva phase-folded sintética + heatmap de saliency MOCK sobre el dip.

    TODO real: graficar data/processed/local/<tic>.pt y la atribución de
    scripts/run_xai.py en lugar de la curva sintética.
    """
    rng = _rng_for(tic_id)
    n = 201
    phase = np.linspace(-0.5, 0.5, n)

    # Profundidad del dip a partir del depth real (ppm → fracción), con piso visible
    depth_ppm = info.get("depth_ppm") or 3000.0
    depth = max(depth_ppm / 1e6, 0.002)
    is_planet = info["is_planet_truth"]

    # Tránsito tipo "caja suavizada" centrado en fase 0
    width = 0.06 if is_planet else 0.10
    transit = -depth * np.exp(-(phase ** 2) / (2 * (width / 2.5) ** 2))
    noise = rng.normal(0, depth * 0.18, n)
    flux = 1.0 + transit + noise

    # Saliency mock: concentrada en el dip para CP, más difusa para FP
    if is_planet:
        sal = np.exp(-(phase ** 2) / (2 * (width / 2.0) ** 2))
    else:
        sal = 0.4 * np.exp(-(phase ** 2) / (2 * (width) ** 2)) + 0.3 * rng.rand(n)
    sal = sal / sal.max()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 3.6), facecolor=C_BG,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True,
    )
    col = C_PLANET if is_planet else C_FP

    _style_ax(ax1, "Curva phase-folded + saliency (XAI)")
    ax1.scatter(phase, flux, s=6, c=C_MUTED, alpha=0.6, edgecolors="none")
    ax1.plot(phase, 1.0 + transit, color=col, linewidth=1.8, zorder=4)
    ax1.axvline(0, color=C_ACCENT, linewidth=0.6, linestyle=":", alpha=0.6)
    ax1.set_ylabel("flujo norm.", color=C_MUTED, fontsize=8)

    # Heatmap de saliency
    ax2.set_facecolor(C_PANEL)
    ax2.imshow(sal[np.newaxis, :], aspect="auto", cmap="inferno",
               extent=[-0.5, 0.5, 0, 1])
    ax2.set_yticks([])
    ax2.tick_params(colors=C_MUTED, labelsize=8)
    ax2.set_xlabel("fase orbital", color=C_MUTED, fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#26324a")

    # gridspec con hspace manual no es compatible con tight_layout; márgenes a mano.
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.14)
    return fig


# --------------------------------------------------------------------------- #
# Pipeline: la secuencia de tool calls que el agente ejecuta (para la animación)
# --------------------------------------------------------------------------- #
def pipeline_plan(tic_id: int, info: dict, classification: dict) -> list[dict]:
    """Lista de (tool, latencia_ms, resumen) que la UI anima en vivo.

    Cuando se conecte el LLM real, esta lista la decide el modelo vía tool calling;
    aquí está pre-computada para reflejar el flujo de demo de ETAPA3.md.
    """
    toi = info.get("toi") or "s/n"
    disp = info["disposition"]
    p = classification["prob_planeta"]
    return [
        {"tool": "get_toi_info", "latency_ms": 45,
         "summary": f"TOI {toi} · disp={disp}"},
        {"tool": "load_light_curve", "latency_ms": 60,
         "summary": "curva preprocesada OK"},
        {"tool": "classify", "latency_ms": 320,
         "summary": f"prob_planeta={p:.3f}"},
        {"tool": "verify_prediction", "latency_ms": 12,
         "summary": "chequeos físicos"},
        {"tool": "visualize(sky_map)", "latency_ms": 180, "summary": "mapa del cielo"},
        {"tool": "visualize(orbit)", "latency_ms": 90, "summary": "diagrama orbital"},
        {"tool": "explain", "latency_ms": 410, "summary": "saliency sobre la curva"},
        {"tool": "compare_with_disposition", "latency_ms": 8,
         "summary": "modelo vs NASA"},
    ]
