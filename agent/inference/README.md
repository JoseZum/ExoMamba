# Servicio de inferencia Mamba (modelo real del agente)

Reemplaza la predicción **mock** del agente por el **modelo Mamba real** (seed789,
el mejor por test AUC = **0.810**). Es un microservicio HTTP que carga el checkpoint
una vez y clasifica curvas por TIC ID.

## Por qué un servicio aparte

`mamba-ssm` compila kernels CUDA y **solo corre en Linux + GPU** (decisión del
proyecto: el modelo se entrena en WSL2). El agente Streamlit corre en Windows. La
solución limpia es separar la inferencia en un servicio Linux (Docker o WSL2) que el
agente consume por HTTP. Si el servicio no está arriba, el agente **cae al mock**
automáticamente (degradación elegante) - la demo nunca se rompe.

```
Windows (Streamlit)                       Linux (Docker o WSL2, GPU)
  agent/app.py                              agent/inference/server.py  (FastAPI)
  agent/tools.py ─ classify() ──HTTP──►       └─ MambaPredictor → best.pt (seed789)
  agent/model_client.py                       montado: data/processed/global, experiments
        │ (servicio caído)
        └─► fallback: agent/mock.classify (determinista)
```

## Endpoints

| Método | Ruta | Cuerpo | Respuesta |
|---|---|---|---|
| `GET` | `/health` | - | `{ok, model, run_dir, device, n_params, test_auc, ...}` |
| `POST` | `/classify` | `{"tic_id": 79748331}` | `{prob_planeta, label, confianza, source: "mamba_real", model_run, device}` |

## Opción A - Docker (reproducible, portable)

Requiere **NVIDIA Container Toolkit** en el host (GPU passthrough). Desde la raíz del repo:

```bash
docker compose -f agent/inference/docker-compose.yml up --build
```

El build instala torch 2.5.1+cu121, `causal-conv1d` y `mamba-ssm` (compila kernels,
tarda varios minutos la primera vez). Los checkpoints y curvas se **montan**
read-only, no se copian a la imagen. Probar:

```bash
curl http://127.0.0.1:8077/health
curl -X POST http://127.0.0.1:8077/classify -H "Content-Type: application/json" -d "{\"tic_id\": 79748331}"
```

## Opción B - WSL2 directo (vía instantánea si ya entrenaste ahí)

Si ya tenés el venv de WSL2 con `mamba-ssm` (donde se entrenaron los seeds), es lo
más rápido - no hay que compilar nada:

```bash
# dentro de WSL2, en la raíz del repo, con el venv del proyecto activado
pip install -r agent/inference/requirements.txt   # solo fastapi+uvicorn+pydantic
python -m agent.inference.server
```

## Conectar el agente (Windows)

El agente detecta el servicio solo. Por defecto apunta a `http://127.0.0.1:8077`
(override con `MAMBA_SERVICE_URL` en `agent/.env`). Con el servicio arriba:

- el badge del header muestra **"Mamba real · GPU"** con el AUC de test;
- cada veredicto indica **fuente: Mamba real**;
- el log de sesión registra `source: "mamba_real"`.

Sin el servicio, todo sigue funcionando en modo mock (badge **"(simulado)"**).

## Configuración (variables de entorno)

| Variable | Default | Dónde |
|---|---|---|
| `MAMBA_RUN_DIR` | `experiments/2026-05-28_01-44-54_mamba_small_seed789` | servicio |
| `MAMBA_SERVICE_PORT` | `8077` | servicio |
| `MAMBA_SERVICE_URL` | `http://127.0.0.1:8077` | agente (cliente) |

Para usar otro modelo (p. ej. el ensemble o el locked), cambiá `MAMBA_RUN_DIR` a su
run dir - debe tener `config.yaml` + `checkpoints/best.pt`.
