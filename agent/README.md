# Agente de vetting de TOIs — Etapa 3

Asistente conversacional que **consume el modelo del proyecto** (Mamba locked) como
herramienta de un LLM con tool calling, ejecuta chequeos físicos, explica su decisión
(XAI) y entrega un informe con visualizaciones. Registra cada sesión como evidencia.

## Estado

| Componente | Estado | Nota |
|---|---|---|
| Frontend Streamlit (2 columnas) | ✅ | `app.py` |
| 7 tools + schemas JSON | ✅ | `tools.py` |
| Orquestador (mock + Claude) | ✅ | `llm.py` |
| Logger de sesiones JSON | ✅ | `logs.py` → `logs/` |
| System prompt | ✅ | `prompts/system.md` |
| Suite de validación S1–S6 + 5 métricas | ✅ | `eval/` (36 sesiones, todas las métricas en mock) |
| Ética + análisis de fallos | ✅ | `ETHICS.md`, `eval/FAILURE_ANALYSIS.md` |
| `get_toi_info`, `verify_prediction` | ✅ real | catálogo TOI real + física real |
| `classify` + figuras | 🟡 mock | placeholder del Mamba locked; se reemplaza sin tocar UI |
| LLM real (Claude Haiku 4.5) | ⬜ pendiente | falta solo `ANTHROPIC_API_KEY` en `agent/.env` |

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
| `Analiza TIC 341420329` | PC → predicción incierta | — |
| `Analiza TIC 999999999` | TIC inexistente → no alucina | S4 |
| `¿hay vida en Marte?` | off-topic → responde dentro de scope | S6 |

## Estructura

```
agent/
├── app.py              # Streamlit UI (chat + panel)
├── llm.py              # Agent: orquestación mock/claude + informe
├── tools.py            # 7 tools + dispatch + schemas Anthropic
├── mock.py             # datos reales + verifier real + classify/figuras mock
├── logs.py             # logger de sesiones → logs/
├── prompts/system.md   # rol, flujo, reglas de honestidad, límites
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

## Conexión del modelo/LLM reales (lo único que falta)

1. **LLM:** `cp agent/.env.example agent/.env`, poner `ANTHROPIC_API_KEY`,
   `pip install anthropic python-dotenv`. El modo cambia a `claude` solo.
2. **Modelo:** reemplazar `mock.classify` por el forward del Mamba locked (cargar
   `experiments/.../best.pt` y correr sobre `data/processed/global/<tic>.pt`).
   Las figuras de curva/saliency → leer la curva real + `scripts/run_xai.py`.
   Ni `tools.py`, ni `llm.py`, ni `app.py` cambian.
3. Reejecutar `python -m agent.eval.run_eval --claude` para medir los modos de fallo
   reales del LLM (ver `eval/FAILURE_ANALYSIS.md`).
