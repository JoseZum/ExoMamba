# Agente de vetting de TOIs - Etapa 3

Asistente conversacional que **consume el modelo del proyecto** (Mamba locked) como
herramienta de un LLM con tool calling, ejecuta chequeos físicos, explica su decisión
(XAI) y entrega un informe con visualizaciones. Registra cada sesión como evidencia.

## Modos

El agente corre en dos modos intercambiables (`Agent(mode="auto")`):

- **mock** (default sin API key): orquestación determinista. Funciona ya, sin costo.
- **claude**: loop real de tool calling con Anthropic SDK. Se activa solo: copiar
  `.env.example` → `.env`, poner la key, `pip install anthropic python-dotenv`.

La UI, el logging y la suite de validación son **idénticos** en ambos modos.

## Cómo correr

```powershell
# desde la raíz del repo (mamba-exoplanet/), con el entorno del proyecto
pip install -r agent/requirements.txt

# Demo (UI)
python -m streamlit run agent/app.py --server.address 127.0.0.1
#  → abrir http://127.0.0.1:8501   (usar 127.0.0.1, NO localhost)

# Suite de validación (genera eval/results/ + sesiones en logs/)
python -m agent.eval.run_eval
```

> **Windows:** se usa `python -m streamlit` (no `streamlit` suelto) porque el
> ejecutable no queda en el PATH con el Python de Microsoft Store. Y `localhost`
> resuelve a IPv6, por eso se fuerza `--server.address 127.0.0.1`.

## Casos de demo

| Input | Qué muestra | Escenario |
|---|---|---|
| `Analiza TIC 79748331` | CP → veredicto PLANETA, chequeos OK | S1 |
| `Analiza TIC 182943944` | FP → veredicto FALSO POSITIVO | S2 |
| `Analiza TIC 341420329` | PC → predicción incierta | - |
| `Analiza TIC 999999999` | TIC inexistente → no alucina | S4 |
| `¿hay vida en Marte?` | off-topic → responde dentro de scope | S6 |

## Estructura

```
agent/
├── app.py              # Streamlit UI (chat + panel)
├── llm.py              # Agent: orquestación mock/claude + informe
├── tools.py            # 7 tools + dispatch + schemas Anthropic
├── model_client.py     # cliente HTTP del servicio Mamba (stdlib, con fallback)
├── mock.py             # datos reales + verifier real + classify/figuras mock (fallback)
├── logs.py             # logger de sesiones → logs/
├── prompts/system.md   # rol, flujo, reglas de honestidad, límites
├── inference/          # servicio del modelo REAL (lado Linux)
│   ├── predictor.py    # MambaPredictor: best.pt seed789 + forward por TIC
│   ├── server.py       # FastAPI: /health, /classify
│   ├── Dockerfile      # imagen CUDA + mamba-ssm
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── README.md       # cómo levantar (Docker y WSL2)
├── eval/
│   ├── scenarios.py    # S1–S6 (TICs reales, deterministas)
│   ├── metrics.py      # 5 métricas del agente
│   ├── run_eval.py     # runner → results/
│   ├── results/        # SUMMARY.md + JSON [versionado]
│   └── FAILURE_ANALYSIS.md
├── ETHICS.md
├── requirements.txt
├── .env.example
└── logs/               # sesiones reales [gitignored]
```

## Modelo real (ya conectado)

`classify` corre el **Mamba seed789 real** (test AUC 0.810) a través de un servicio
de inferencia en Docker/WSL2. Levantarlo (ver `agent/inference/README.md`):

```bash
# Opción A - Docker (GPU passthrough):
docker compose -f agent/inference/docker-compose.yml up --build
# Opción B - WSL2 (venv con mamba-ssm):
pip install -r agent/inference/requirements.txt && python -m agent.inference.server
```

Con el servicio arriba, el badge muestra **"Mamba real · GPU"** y cada veredicto
indica `fuente: Mamba real`. Sin servicio, el agente cae al **mock determinista**
automáticamente (badge **"(simulado)"**) - la demo nunca se rompe.

## LLM real (lo único que falta)

1. **LLM:** `cp agent/.env.example agent/.env`, poner `ANTHROPIC_API_KEY`,
   `pip install anthropic python-dotenv`. El modo cambia a `claude` solo.
2. Reejecutar `python -m agent.eval.run_eval --claude` para medir los modos de fallo
   reales del LLM (ver `eval/FAILURE_ANALYSIS.md`).
3. **Figuras XAI reales** (opcional): reemplazar las figuras sintéticas de
   `mock.make_lightcurve_saliency` por la curva real + `scripts/run_xai.py`.
