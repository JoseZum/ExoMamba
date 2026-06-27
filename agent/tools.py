"""Las 7 tools del agente - capa única que consume el modelo del proyecto.

Cada tool:
  - recibe argumentos simples (típicamente `tic_id`),
  - devuelve un dict JSON-serializable (las figuras devuelven rutas a PNG),
  - reporta `error` en vez de lanzar excepción (para que el LLM lo maneje).

`TOOL_SCHEMAS` expone el esquema de tool-use de Anthropic (lo consume agent/llm.py
en modo Claude). `dispatch(name, args)` ejecuta una tool midiendo latencia y es el
punto que registra el logger de sesiones.

Hoy la lógica vive en agent/mock.py (datos reales del catálogo + verifier real +
predicción simulada). Conectar el modelo real = reemplazar `mock.classify` y las
figuras; esta capa y sus firmas NO cambian.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from agent import mock, model_client

# Cache de figuras generadas por las tools (gitignored).
FIGCACHE = Path(__file__).resolve().parent / "_figcache"
FIGCACHE.mkdir(exist_ok=True)

# Dónde viven los tensores preprocesados (Tier 1). Puede no existir para todo TIC.
PROCESSED_GLOBAL = mock.ROOT / "data" / "processed" / "global"


def _save_fig(fig, name: str) -> str:
    path = FIGCACHE / name
    fig.savefig(path, dpi=110, facecolor=mock.C_BG, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Las 7 tools
# --------------------------------------------------------------------------- #
def get_toi_info(tic_id: int) -> dict:
    """Metadata del catálogo TOI para un TIC ID."""
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    return info


def get_star_info(tic_id: int) -> dict:
    """Datos ricos del candidato y su estrella: ubicacion (RA/Dec), distancia,
    temperatura y radio estelar, tamano y temperatura del planeta. REAL."""
    info = mock.get_star_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no esta en el catalogo crudo."}
    return info


def load_light_curve(tic_id: int) -> dict:
    """Confirma disponibilidad de la curva preprocesada. Real si existe el .pt."""
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    pt = PROCESSED_GLOBAL / f"{int(tic_id)}.pt"
    if pt.exists():
        return {
            "tic_id": int(tic_id),
            "available": True,
            "source": "processed",
            "length": 18000,
            "path": str(pt),
        }
    # No está preprocesada en esta máquina (data/processed/ es gitignored).
    return {
        "tic_id": int(tic_id),
        "available": False,
        "source": "synthetic_mock",
        "length": 18000,
        "note": "curva no preprocesada localmente; el análisis usa una vista sintética representativa",
    }


def classify(tic_id: int) -> dict:
    """Corre el modelo Mamba real sobre la curva. Cae al mock si el servicio de
    inferencia no está disponible (degradación elegante).

    1. Intenta el servicio de inferencia (`agent/inference/server.py` en Docker/WSL2),
       que carga el Mamba seed789 (test AUC 0.810) y hace forward real.
    2. Si el servicio no responde, o el TIC no tiene curva preprocesada del lado del
       servicio, usa la predicción mock determinista (marcada con `source="mock"`).
    """
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    real = model_client.classify(int(tic_id))
    if real is not None:
        return real
    return {**mock.classify(int(tic_id), info), "source": "mock"}


def verify_prediction(tic_id: int) -> dict:
    """4 chequeos físicos REALES sobre la metadata del catálogo."""
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    return mock.verify_prediction(info)


def compare_with_disposition(tic_id: int) -> dict:
    """Compara la predicción del modelo contra la disposición oficial NASA.

    Usa la MISMA predicción que `classify` (real con fallback a mock), para que el
    contraste con NASA sea coherente con el veredicto que ve el usuario.
    """
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    cls = classify(int(tic_id))
    if "error" in cls:
        return cls
    return mock.compare_with_disposition(info, cls)


_VIS_KINDS = {
    "sky_map": (mock.make_sky_map, "Posición de la estrella en el cielo (proyección Aitoff)."),
    "orbit_diagram": (mock.make_orbit_diagram, "Órbita estimada a escala (3.ª ley de Kepler)."),
    "lightcurve_xai": (mock.make_lightcurve_saliency, "Curva phase-folded con saliency (XAI)."),
}


def visualize(tic_id: int, kind: str) -> dict:
    """Genera una figura. kind ∈ {sky_map, orbit_diagram, lightcurve_xai}."""
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    if kind not in _VIS_KINDS:
        return {"error": f"kind inválido '{kind}'. Opciones: {list(_VIS_KINDS)}"}
    fn, caption = _VIS_KINDS[kind]
    path = _save_fig(fn(int(tic_id), info), f"{int(tic_id)}_{kind}.png")
    return {"tic_id": int(tic_id), "kind": kind, "png_path": path, "caption": caption}


def explain(tic_id: int) -> dict:
    """Saliency sobre la curva phase-folded (XAI). Devuelve PNG + región top."""
    info = mock.get_toi_info(int(tic_id))
    if info is None:
        return {"error": f"TIC {tic_id} no está en el catálogo TOI."}
    path = _save_fig(mock.make_lightcurve_saliency(int(tic_id), info),
                     f"{int(tic_id)}_explain.png")
    return {
        "tic_id": int(tic_id),
        "png_path": path,
        "method": "gradient_saliency (mock)",
        "top_regions": ["fase ≈ 0 (centro del tránsito)"],
        "caption": "La atribución se concentra en el dip central en un tránsito real.",
    }


# --------------------------------------------------------------------------- #
# Registro + dispatch
# --------------------------------------------------------------------------- #
REGISTRY = {
    "get_toi_info": get_toi_info,
    "get_star_info": get_star_info,
    "load_light_curve": load_light_curve,
    "classify": classify,
    "verify_prediction": verify_prediction,
    "compare_with_disposition": compare_with_disposition,
    "visualize": visualize,
    "explain": explain,
}


def dispatch(name: str, args: dict) -> dict:
    """Ejecuta una tool por nombre, midiendo latencia. Núcleo del logging."""
    if name not in REGISTRY:
        return {"result": {"error": f"tool desconocida '{name}'"}, "latency_ms": 0}
    t0 = time.perf_counter()
    try:
        result = REGISTRY[name](**args)
    except Exception as exc:  # robustez: nunca tumbar al agente por una tool
        result = {"error": f"{type(exc).__name__}: {exc}"}
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"result": result, "latency_ms": latency_ms}


# --------------------------------------------------------------------------- #
# Esquemas de tool-use (formato Anthropic) - los consume agent/llm.py modo Claude
# --------------------------------------------------------------------------- #
_TIC_PROP = {
    "tic_id": {"type": "integer", "description": "TIC ID de la estrella objetivo."}
}

TOOL_SCHEMAS = [
    {
        "name": "get_toi_info",
        "description": "Metadata del catálogo TOI: período, época, duración, profundidad, "
                       "magnitud y disposición oficial (CP/FP/PC/KP).",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "get_star_info",
        "description": "Datos ricos del candidato y su estrella: ubicacion en el cielo "
                       "(RA/Dec), distancia en parsecs, temperatura y radio estelar, y "
                       "radio/temperatura del planeta. Usalo para responder donde se "
                       "ubica, que tan lejos esta, que tipo de estrella es o si seria habitable.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "load_light_curve",
        "description": "Confirma que la curva de luz preprocesada del TIC está disponible "
                       "y reporta longitud y fuente.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "classify",
        "description": "Corre el modelo final del proyecto sobre la curva y devuelve la "
                       "probabilidad de planeta, la etiqueta y la confianza.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "verify_prediction",
        "description": "Chequeos físicos (período en sector, profundidad, duración vs período, "
                       "magnitud observable) antes de confiar en la predicción.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "compare_with_disposition",
        "description": "Compara la predicción del modelo con la disposición oficial NASA y "
                       "marca el caso para revisión si hay discrepancia.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
    {
        "name": "visualize",
        "description": "Genera una visualización del candidato. kind: 'sky_map', "
                       "'orbit_diagram' o 'lightcurve_xai'.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_TIC_PROP,
                "kind": {"type": "string", "enum": list(_VIS_KINDS),
                         "description": "Tipo de figura a generar."},
            },
            "required": ["tic_id", "kind"],
        },
    },
    {
        "name": "explain",
        "description": "Genera la explicabilidad (saliency/XAI) sobre la curva phase-folded "
                       "y reporta la región de mayor atribución.",
        "input_schema": {"type": "object", "properties": _TIC_PROP, "required": ["tic_id"]},
    },
]
