"""Cliente del servicio de inferencia Mamba - lado agente (Windows).

Solo usa la stdlib (urllib): el agente NO importa torch ni mamba-ssm. Habla por
HTTP con `agent/inference/server.py`, que corre en Docker o WSL2.

Contrato de diseño: **degradación elegante**. Si el servicio no está levantado,
no responde a tiempo, o devuelve un error, las funciones devuelven `None` y el
caller (agent/tools.py) cae al mock determinista. La demo nunca se rompe.

Para no pagar el timeout de conexión en cada predicción cuando el servicio está
caído, el estado de salud se cachea con un TTL corto: si se levanta el servicio,
el agente lo detecta en ~`_HEALTH_TTL` segundos sin reiniciar.

Config por entorno:
    MAMBA_SERVICE_URL   default http://127.0.0.1:8077
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

SERVICE_URL = os.environ.get("MAMBA_SERVICE_URL", "http://127.0.0.1:8077").rstrip("/")

# Timeouts cortos: el agente no debe colgarse esperando un servicio caído.
_HEALTH_TIMEOUT = 1.5
_CLASSIFY_TIMEOUT = 15.0

# Cache del health para no repetir el handshake en cada classify.
_HEALTH_TTL = 10.0
_HEALTH_CACHE: dict[str, Any] = {"ts": -1e9, "value": None}

# --------------------------------------------------------------------------- #
# Log en memoria de las operaciones HTTP contra el servicio de inferencia.
# El frontend lo lee para mostrar una "terminal" que evidencia la llamada real
# al Docker (GET /health, POST /classify) y de dónde salió la predicción.
# --------------------------------------------------------------------------- #
_LOG: list[dict[str, str]] = []
_LOG_MAX = 200


def _ts() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


def _log(line: str, level: str = "info") -> None:
    """Anexa una línea al log (level ∈ req|ok|warn|info) para la terminal del frontend."""
    _LOG.append({"ts": _ts(), "line": line, "level": level})
    if len(_LOG) > _LOG_MAX:
        del _LOG[:-_LOG_MAX]


def _fmt_auc(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "?"


def reset_log() -> None:
    """Limpia el log. El frontend lo llama antes de cada análisis."""
    _LOG.clear()


def get_log() -> list[dict[str, str]]:
    """Copia del log de operaciones HTTP (para renderizar la terminal)."""
    return list(_LOG)


def _get(path: str, timeout: float) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(f"{SERVICE_URL}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _post(path: str, payload: dict, timeout: float) -> dict[str, Any] | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVICE_URL}{path}", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _health_raw(timeout: float = _HEALTH_TIMEOUT) -> dict[str, Any] | None:
    h = _get("/health", timeout)
    return h if (isinstance(h, dict) and h.get("ok")) else None


def health(force: bool = False, log: bool = False) -> dict[str, Any] | None:
    """Metadata del servicio si está vivo (cacheada con TTL), None si no responde.

    `log=True` escribe la traza HTTP en la terminal del frontend. El header lo
    llama con log=False para no ensuciar el log con sus chequeos de cada rerun.
    """
    now = time.monotonic()
    if not force and (now - _HEALTH_CACHE["ts"]) < _HEALTH_TTL:
        if log:
            v = _HEALTH_CACHE["value"]
            if v is None:
                _log(f"servicio de inferencia no disponible en {SERVICE_URL}", "warn")
            else:
                _log(f"servicio vivo (cache): device={v.get('device', '?')} "
                     f"AUC_test={_fmt_auc(v.get('test_auc'))}", "ok")
        return _HEALTH_CACHE["value"]
    if log:
        _log(f"GET {SERVICE_URL}/health", "req")
    t0 = time.perf_counter()
    value = _health_raw()
    dt = (time.perf_counter() - t0) * 1000
    if log:
        if value is None:
            _log(f"sin respuesta del servicio ({dt:.0f} ms, timeout {_HEALTH_TIMEOUT}s)", "warn")
        else:
            _log(f"200 OK  modelo={value.get('model', 'mamba')}  "
                 f"device={value.get('device', '?')}  params={value.get('n_params', '?')}  "
                 f"AUC_test={_fmt_auc(value.get('test_auc'))}  ({dt:.0f} ms)", "ok")
    _HEALTH_CACHE.update(ts=now, value=value)
    return value


def service_available() -> bool:
    return health() is not None


def classify(tic_id: int, timeout: float = _CLASSIFY_TIMEOUT) -> dict[str, Any] | None:
    """Predicción del Mamba real. None si el servicio no respondió o no aplica.

    Devuelve None (→ fallback a mock) cuando: el servicio está caído, hubo error,
    o el TIC no tiene curva preprocesada del lado del servicio. No paga el timeout
    del POST si el health cacheado dice que el servicio está caído.
    """
    _log(f"clasificando TIC {int(tic_id)} con el Mamba real (servicio Docker)", "info")
    if health(log=True) is None:
        _log("respaldo: predicción simulada (mock determinista)", "warn")
        return None
    _log(f"POST {SERVICE_URL}/classify  payload={{\"tic_id\": {int(tic_id)}}}", "req")
    t0 = time.perf_counter()
    r = _post("/classify", {"tic_id": int(tic_id)}, timeout)
    dt = (time.perf_counter() - t0) * 1000
    if not isinstance(r, dict):
        _log(f"sin respuesta del servicio ({dt:.0f} ms) - respaldo mock", "warn")
        return None
    if "error" in r:
        _log(f"el servicio devolvió error: {r['error']} - respaldo mock", "warn")
        return None
    if "prob_planeta" not in r:
        _log(f"respuesta sin prob_planeta - respaldo mock ({dt:.0f} ms)", "warn")
        return None
    _log(f"200 OK  prob_planeta={r['prob_planeta']:.4f}  label={r.get('label', '?')}  "
         f"fuente={r.get('source', '?')}  device={r.get('device', '?')}  ({dt:.0f} ms)", "ok")
    return r
