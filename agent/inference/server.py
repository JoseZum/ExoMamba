"""Servidor de inferencia Mamba (FastAPI) - lado Linux (Docker o WSL2).

Expone el modelo Mamba real por HTTP para que el agente Streamlit lo consuma:

    GET  /health           -> metadata del modelo (device, params, test AUC)
    POST /classify {tic_id} -> {prob_planeta, label, confianza, source: mamba_real}

El agente (Windows) habla con este servicio vía `agent/model_client.py` y, si el
servicio no responde, cae al mock determinista. Así la demo nunca se rompe.

Levantar:

    # En WSL2 (venv del proyecto con mamba-ssm ya instalado) - vía instantánea:
    python -m agent.inference.server

    # En Docker (reproducible, GPU passthrough):
    docker compose -f agent/inference/docker-compose.yml up --build
"""

from __future__ import annotations

import os

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:  # mensaje útil si falta el stack del servicio
    raise ImportError(
        "Faltan dependencias del servicio. Instalá:\n"
        "  pip install -r agent/inference/requirements.txt\n"
        f"Error original: {e}"
    ) from e

from agent.inference.predictor import MambaPredictor

app = FastAPI(title="Mamba TOI Vetting - Inference Service", version="1.0")

_PREDICTOR: MambaPredictor | None = None


def get_predictor() -> MambaPredictor:
    """Singleton lazy: carga el modelo una sola vez."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = MambaPredictor()
    return _PREDICTOR


class ClassifyRequest(BaseModel):
    tic_id: int


@app.on_event("startup")
def _warmup() -> None:
    # Cargar el modelo al arrancar para fallar temprano si algo falta.
    p = get_predictor()
    print(f"[inference] modelo listo: {p.health()}", flush=True)


@app.get("/health")
def health() -> dict:
    try:
        return {"ok": True, **get_predictor().health()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"modelo no disponible: {exc}") from exc


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    predictor = get_predictor()
    try:
        return predictor.classify(req.tic_id)
    except FileNotFoundError as exc:
        # TIC sin curva preprocesada: error explícito, no es un 500 del servidor.
        return {"error": str(exc), "available": False, "tic_id": req.tic_id}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"fallo de inferencia: {exc}") from exc


def main() -> None:
    host = os.environ.get("MAMBA_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("MAMBA_SERVICE_PORT", "8077"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
