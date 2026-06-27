# Revisión de Etapa 3 - Agente inteligente de vetting de TOIs

> Guía de revisión del proyecto, fiel a la implementación real en `agent/`.
> Sirve para recorrer la carpeta archivo por archivo, mapear cada pieza a la
> rúbrica de Etapa 3 y marcar una checklist durante la defensa.
>
> Este documento describe lo que **está construido y corre hoy**, no el plan.
> El plan previo vive en `../ETAPA3.md` (quedó desactualizado: hablaba de 7 tools
> y solo Claude; la implementación real tiene 8 tools, modelo Mamba real vía
> Docker, LLM Gemini, terminal de evidencia y degradación elegante en dos vías).

---

## 1. Resumen en 30 segundos

El agente es un **asistente conversacional de vetting de TOIs**: recibe un TIC ID,
decide si es planeta o falso positivo **corriendo el modelo Mamba real del proyecto
como herramienta**, lo somete a chequeos físicos, lo contrasta con la disposición
oficial de la NASA, lo explica (XAI) y entrega un informe con visualizaciones.
Después se le puede **preguntar en lenguaje natural** sobre el candidato (dónde se
ubica, si es habitable, qué estrella orbita).

Hay **dos vías independientes**, cada una con respaldo automático para que la demo
nunca se rompa:

1. **Vía del modelo (la herramienta):** `classify` llama por HTTP a un microservicio
   FastAPI en Docker/WSL2 que carga el Mamba `seed789` real (test AUC 0.810) y hace
   el forward sobre la curva. Si el servicio está caído, cae a una predicción mock
   determinista.
2. **Vía del LLM (el orquestador):** un LLM con tool calling decide qué herramientas
   llamar y redacta el informe. Funciona con Claude, OpenAI o cualquier endpoint
   compatible (hoy configurado **Gemini `gemini-2.5-flash`**). Si el LLM falla
   (rate limit, sin crédito, key inválida), cae a una orquestación determinista.

La UI, los logs y la suite de validación son **idénticos** sin importar qué vía
esté activa.

---

## 2. Mapeo a la rúbrica (dónde se cumple cada criterio)

| Criterio | Pts | Dónde se cumple en el repo | Estado |
|---|---|---|---|
| **Integración del agente** (modelo como herramienta/decisor + evidencia con logs/demos) | **10** | `tools.py:classify` → `model_client.py` → `inference/` (Mamba real); LLM tool-calling en `llm.py`; terminal de evidencia + `logs/*.json` | Cumple nivel Excelente |
| **Validación del agente** (escenarios + métricas + casos límite + análisis de fallos) | **6** | `eval/scenarios.py` (S1-S6), `eval/metrics.py` (5 métricas), `eval/results/SUMMARY.md`, `eval/FAILURE_ANALYSIS.md` | Cumple; ver quick win en seccion 11 |
| **Ética, sesgos y privacidad** (riesgos + mitigación concreta + límites) | **4** | `ETHICS.md` (tabla de 5 riesgos aterrizada al sistema) | Cumple nivel Excelente |
| **Artículo IEEE/ACM** (estructura, figuras, metodología comparable, XAI) | **5** | `paper/` (cerrado según el profe; fuera del alcance del agente) | Externo |
| **Total Etapa 3** | **25** | | |

Penalizaciones (sección 10): leakage, baselines, justificación y reproducibilidad
están cubiertas y documentadas para no perder puntos.

---

## 3. Arquitectura: las dos vías y la degradación elegante

```
                          Windows (Streamlit)                         Linux (Docker o WSL2, GPU)
  Usuario ─► agent/app.py ─► agent/llm.py (Agent)                     agent/inference/server.py (FastAPI)
              (UI chat       │  orquesta el LLM                          ├─ GET  /health
               + panel)      │  (Gemini/Claude/OpenAI/mock)              └─ POST /classify
                             │                                              │
                             ├─ llama herramientas ─► agent/tools.py        ▼
                             │                          │              MambaPredictor
                             │   classify(tic) ─────────┤                 best.pt (seed789, AUC 0.810)
                             │                          │                 forward sobre global_view (1,18000)
                             │                 agent/model_client.py ──HTTP──► (prob_planeta, source=mamba_real)
                             │                   (stdlib urllib)
                             │                          │ (servicio caído)
                             │                          └─► fallback: agent/mock.classify (determinista, source=mock)
                             │
                             └─ (LLM caído/sin crédito) ─► fallback: orquestación determinista (_run_mock)

  Toda operación HTTP contra el servicio queda registrada y se muestra como TERMINAL en la UI
  (evidencia visible de que se llamó al Docker y de dónde salió la predicción).
```

Las dos vías degradan por separado: el modelo puede ser real mientras el LLM es
mock, o al revés. Cada combinación produce el mismo formato de salida.

---

## 4. Recorrido archivo por archivo de `agent/`

### Núcleo del agente (Windows)

| Archivo | Qué hace | Piezas clave |
|---|---|---|
| `app.py` | Frontend Streamlit (2 columnas: chat + panel). Selector de 9 candidatos reales, botón Analizar/Al azar, chat libre. Muestra el badge de modelo/agente, la **terminal de evidencia**, el veredicto, el verificador y las figuras. | `render_header`, `render_picker`, `render_panel`, `render_model_terminal`, `render_llm_status`, `is_analysis_command` (rutea análisis vs conversación) |
| `llm.py` | El **orquestador** (`Agent`). Decide modo (`auto` → claude/openai/mock), corre el loop de tool calling, arma el informe. Tiene también el modo conversacional (`chat`). | `Agent.run`, `_run_claude`, `_run_openai`, `_run_mock`, `Agent.chat`, `build_report_md`, `provider_label`, `_classify_llm_error` |
| `tools.py` | Las **8 herramientas** que el LLM puede invocar + `dispatch` (ejecuta midiendo latencia) + `TOOL_SCHEMAS` (esquema de tool-use). | `classify` (real con fallback), `verify_prediction`, `compare_with_disposition`, `get_toi_info`, `get_star_info`, `load_light_curve`, `visualize`, `explain` |
| `model_client.py` | Cliente HTTP del servicio Mamba (solo stdlib, no importa torch). Health cacheado con TTL. **Registra cada operación HTTP** en un log en memoria para la terminal del frontend. Devuelve `None` ante cualquier fallo → el caller cae al mock. | `classify`, `health`, `reset_log`, `get_log` |
| `mock.py` | Datos **reales** del catálogo (`get_toi_info`, `get_star_info`), verificador físico **real** (`verify_prediction`), y la predicción/figuras **simuladas** que sirven de respaldo. | `load_catalog`, `verify_prediction`, `classify` (mock), `make_sky_map`, `make_orbit_diagram`, `make_lightcurve_saliency` |
| `logs.py` | Logger estructurado: cada sesión → `logs/<session_id>.json` con query, secuencia de tool calls (args, resumen, latencia), informe final y totales. Es la **evidencia verificable** del rubric. | `SessionLogger.add_tool_call`, `.finish`, `.save` |
| `prompts/system.md` | Rol del agente, alcance estricto, flujo obligatorio de 7 pasos, reglas de honestidad (citar valores literales, declarar discrepancias), límites declarados (período > 27 d). | - |

### Las 8 herramientas (`tools.py`)

| Herramienta | Fuente de datos | Rol |
|---|---|---|
| `get_toi_info` | **Real** (catálogo TOI) | Metadata: período, época, duración, profundidad, magnitud, disposición oficial |
| `get_star_info` | **Real** (catálogo crudo) | Ubicación (RA/Dec), distancia en parsecs, temperatura/radio estelar, radio/temperatura del planeta. Alimenta las preguntas conversacionales |
| `load_light_curve` | **Real** si existe el `.pt` | Confirma disponibilidad de la curva preprocesada |
| `classify` | **Modelo Mamba real** (fallback mock) | Corre el modelo y devuelve `prob_planeta`, `label`, `confianza`, `source` |
| `verify_prediction` | **Real** (física sobre metadata) | 4 chequeos: período en sector, profundidad plausible, duración consistente, estrella observable |
| `compare_with_disposition` | **Real** | Contrasta la predicción contra NASA y marca `flag_for_review` si discrepa |
| `visualize` | Sintético representativo | Figuras: mapa del cielo (Aitoff), órbita a escala (3.ª ley de Kepler), curva con saliency |
| `explain` | Sintético representativo | Saliency/XAI sobre la curva phase-folded + región de mayor atribución |

### Servicio de inferencia (Linux: Docker o WSL2) - `agent/inference/`

| Archivo | Qué hace |
|---|---|
| `predictor.py` | `MambaPredictor`: carga `config.yaml` + `checkpoints/best.pt` del run `seed789`, construye el modelo desde un mini-registry local (sin arrastrar sklearn/tensorboard) y hace el forward. `classify(tic)` lee `data/processed/global/<tic>.pt`, toma `global_view`, aplica sigmoid y devuelve `source="mamba_real"` + metadata |
| `server.py` | FastAPI: `GET /health` (device, params, test AUC) y `POST /classify {tic_id}`. Singleton lazy del predictor, warm-up al arrancar |
| `Dockerfile` | Imagen `pytorch 2.5.1+cu121` que compila `causal-conv1d` y `mamba-ssm` desde los tags de Git, instala el paquete `exoplanet` y arranca el server |
| `docker-compose.yml` | Servicio `mamba-inference` en `:8077`, monta `experiments/` y `data/` read-only, reserva GPU NVIDIA, `MAMBA_RUN_DIR=...seed789` |
| `requirements.txt` | Solo fastapi + uvicorn + pydantic (para el caso WSL2 donde mamba-ssm ya está) |

### Validación (`agent/eval/`)

| Archivo | Qué hace |
|---|---|
| `scenarios.py` | Construye S1-S6 con TICs reales del catálogo de forma determinista (reproducible) |
| `metrics.py` | Las 5 métricas del agente: tool-call accuracy, faithfulness, no-alucinación, flag recall (S3), límite de período declarado (S5) |
| `run_eval.py` | Corre la suite completa → `results/eval_<ts>.json` + `results/SUMMARY.md`; loggea cada caso como sesión |
| `results/SUMMARY.md` | Tabla legible de la última corrida (36 sesiones) |
| `FAILURE_ANALYSIS.md` | Análisis de fallos honesto (dos regímenes: mock = piso de control, LLM = donde aparecen los fallos interesantes) |

### Documentación y configuración

| Archivo | Qué hace |
|---|---|
| `ETHICS.md` | Tabla de ética: 5 riesgos con mitigación implementada y límite declarado |
| `README.md` | Cómo correr (UI, validación, servicio), casos de demo, estructura |
| `.env.example` | Plantilla de configuración del LLM (Claude/OpenAI/Gemini/Groq/Ollama) |
| `requirements.txt` | Dependencias del frontend del agente |
| `logs/*.json` | Sesiones reales loggeadas (evidencia; gitignored) |

---

## 5. Cómo se consume el modelo como herramienta (los 10 pts obligatorios)

Esta es la pieza que la rúbrica marca como **obligatoria**: el agente debe consumir
el modelo como herramienta/decisor, con evidencia.

**El camino exacto de una predicción real:**

1. El LLM (o la orquestación mock) decide llamar `classify(tic_id)`.
2. `tools.classify` llama `model_client.classify(tic_id)`.
3. `model_client` hace `GET /health` al servicio Docker y, si está vivo,
   `POST /classify {tic_id}` a `http://127.0.0.1:8077`.
4. El servicio (`server.py` → `predictor.py`) carga el Mamba `seed789` real,
   lee `data/processed/global/<tic>.pt`, hace el forward y devuelve
   `{prob_planeta, label, confianza, source: "mamba_real", model_run, device}`.
5. Ese veredicto entra al informe y al log de sesión.

**La evidencia visible (terminal):** cada operación HTTP se registra y se muestra
en la UI como una terminal, en vivo durante el análisis y persistente en el panel:

```
21:37:29.498  GET http://127.0.0.1:8077/health
21:37:29.525  200 OK  modelo=mamba_baseline  device=cuda  params=131393  AUC_test=0.810  (26 ms)
21:37:29.525  POST http://127.0.0.1:8077/classify  payload={"tic_id": 158297421}
21:37:29.671  200 OK  prob_planeta=0.5240  label=PLANETA  fuente=mamba_real  device=cuda  (147 ms)
```

Si el servicio está caído, la misma terminal lo deja claro y muestra la caída al
mock (`fuente=mock`). Esto demuestra, sin ambigüedad, que el modelo se está usando
de verdad y de dónde sale cada número.

---

## 6. Evidencia para los 10 pts (logs + terminal + demo)

- **Logs estructurados:** `agent/logs/<session_id>.json`. Cada sesión guarda la
  secuencia de tool calls con args, resumen del resultado y latencia, más el informe
  final y totales. Hay decenas de sesiones reales registradas (corridas de validación
  y demos). Abrir uno en vivo es prueba verificable.
- **Terminal de inferencia:** evidencia visual de la llamada al Docker (sección 5).
- **Badge del header:** muestra `modelo: Mamba real (GPU) · AUC test 0.810` y
  `agente: Gemini (API)` cuando ambas vías reales están activas.
- **Aviso de procedencia:** debajo de cada respuesta, `render_llm_status` indica
  "Respondido por Gemini (API)" o, si hubo fallback, un aviso de rate limit / sin
  crédito / key inválida.

---

## 7. Validación del agente (los 6 pts)

### Escenarios (`eval/scenarios.py`)

| ID | Escenario | Qué evalúa |
|---|---|---|
| S1 | 10 CP confirmados | Clasificación + reporte consistente con NASA |
| S2 | 10 FP conocidos | Ídem clase negativa |
| S3 | 5 discrepancias modelo vs NASA | El agente **debe flaggear** la discrepancia |
| S4 | TIC inexistente / malformado | No alucina ante entrada inválida |
| S5 | Período > 27 d (límite físico) | El agente **declara la limitación** |
| S6 | Off-topic ("¿hay vida en Marte?") | Responde dentro de scope, no abusa de tools |

### Métricas (`eval/metrics.py`)

Tool-call accuracy, faithfulness (los números del informe == los de las tools),
no-alucinación, flag recall (S3), límite de período declarado (S5).

### Resultados actuales (`eval/results/SUMMARY.md`, modo mock, 36 sesiones)

| Métrica global | Valor |
|---|---|
| Tool-call accuracy | 1.000 |
| Faithfulness | 1.000 |
| No-alucinación | 1.000 |
| Latencia media | ~1098 ms |

Por escenario: S1 acc vs NASA 7/10, S2 6/10, **S3 flag recall 5/5**, **S5 límite
declarado 5/5**, S4 y S6 sin alucinación.

### Análisis de fallos (`eval/FAILURE_ANALYSIS.md`)

Lectura honesta y central para la defensa: el modo mock es un **piso de control**
(por construcción da 1.000, prueba que el pipeline, las tools, el verifier y el
logging funcionan). Los modos de fallo interesantes aparecen con el **LLM real**;
la suite está lista para medirlos. Hallazgo clave honesto: la accuracy vs NASA
(7/10, 6/10) refleja el rendimiento del **modelo** (AUC ≈ 0.75-0.81), no un fallo
del agente; el agente reporta fielmente incluso cuando el modelo se equivoca, y eso
es justo lo que hace que S3 tenga discrepancias reales que flaggear.

### Cómo correrla

```powershell
python -m agent.eval.run_eval            # modo determinista (control)
python -m agent.eval.run_eval --claude   # contra Claude (requiere ANTHROPIC_API_KEY)
```

---

## 8. Ética, sesgos y privacidad (los 4 pts)

`agent/ETHICS.md` tiene una tabla de 5 riesgos, cada uno aterrizado al sistema con
mitigación **implementada o verificable** y un límite declarado:

1. **Sesgo del catálogo TOI** (sobre-representa períodos cortos por la ventana de
   27 d de TESS) → el agente declara incertidumbre si período > 27 d (verificado en
   S5: 5/5).
2. **Falso negativo** (descartar un planeta real cuesta tiempo de telescopio) → el
   agente nunca emite veredicto final solo; reporta probabilidad + flag + recomienda
   revisión humana (S3: flag recall 5/5).
3. **Falso positivo** → cross-check con NASA + verificador físico.
4. **Alucinación del LLM** → todos los números vienen del JSON de las tools; la
   métrica de faithfulness es la guardia.
5. **Privacidad** → no hay datos personales; todo es fotometría pública NASA/MAST.

---

## 9. Guion de demo (qué mostrar en la revisión)

**Antes de empezar:** levantar el servicio Docker y la UI (ver sección 11). El badge
debe decir "Mamba real (GPU)" y "agente: Gemini (API)".

1. **CP correcto:** elegir TIC 79748331 → Analizar. Mostrar la **terminal en vivo**
   (GET /health, POST /classify, `fuente=mamba_real`). Veredicto PLANETA, chequeos OK,
   abajo "Respondido por Gemini (API)". → prueba modelo real + LLM real + evidencia.
2. **FP:** TIC 182943944 → veredicto FALSO POSITIVO.
3. **Discrepancia (S3):** un caso donde el modelo contradice a NASA → mostrar el flag
   **DISCREPANCIA** y la recomendación de revisión humana.
4. **Caso límite inválido (S4):** "Analiza TIC 999999999" → el agente no alucina.
5. **Off-topic (S6):** "¿hay vida en Marte?" → responde dentro de scope.
6. **Conversacional:** tras analizar, preguntar "¿dónde se ubica?" y "¿es habitable?"
   → respuestas con datos reales (`get_star_info`).
7. **Evidencia:** abrir un `agent/logs/<session>.json` y `eval/results/SUMMARY.md`.
8. **Degradación elegante (opcional, fuerte):** apagar el contenedor Docker y volver a
   analizar → la terminal muestra la caída a mock y el badge cambia a "(simulado)";
   la demo sigue corriendo. Demuestra robustez.

---

## 10. Reproducibilidad y penalizaciones (cómo no perder puntos)

| Penalización | Cómo se evita |
|---|---|
| **Data leakage (-15)** | Splits por TIC ID (nunca por sector), test sellado, normalización por curva individual. El agente solo **consume** el modelo ya entrenado; no reentrena ni toca el test. Documentado en CLAUDE.md y el paper |
| **Ausencia de baselines (-10)** | El proyecto tiene la escalera completa: random, LogReg, CNN AstroNet, Mamba (+ ExoMamba V1, AstroNet multibranch). Tabla en `paper/results/all_results.md` |
| **Decisiones sin justificación (-5 c/u)** | Mamba sobre Transformer (O(n) vs O(n²)), modo determinista como piso de control, Mamba seed789 elegido por test AUC, fusión de vías con fallback: todo justificado aquí y en CLAUDE.md |
| **No reproducible (-10)** | `requirements.txt` en agent/ e inference/, comandos exactos abajo, modo mock determinista que corre **sin API key ni GPU**, Dockerfile que reconstruye el servicio |

### Comandos de reproducción

```powershell
# 1. Frontend del agente (Windows)
pip install -r agent/requirements.txt
python -m streamlit run agent/app.py --server.address 127.0.0.1
#    -> http://127.0.0.1:8501   (usar 127.0.0.1, NO localhost)

# 2. Servicio del modelo real (elegir una)
docker compose -f agent/inference/docker-compose.yml up --build      # Docker (GPU)
#    o, en WSL2 con el venv del proyecto:
pip install -r agent/inference/requirements.txt; python -m agent.inference.server

# 3. Validación
python -m agent.eval.run_eval
```

Configuración del LLM (`.env` en la raíz del repo, gitignored):

```
OPENAI_API_KEY=<tu key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-flash
```

Sin `.env`, el agente corre en modo determinista (igual de funcional para la demo).

---

## 11. Checklist de revisión (marcable)

### Integración del agente (10 pts)

- [x] El agente consume el modelo como herramienta (`classify` → Mamba real vía HTTP)
- [x] El modelo real corre en Docker/WSL2 (mamba-ssm, CUDA, seed789, AUC 0.810)
- [x] Degradación elegante a mock si el servicio está caído
- [x] LLM con tool calling orquesta las herramientas (Gemini configurado)
- [x] Degradación elegante a orquestación determinista si el LLM falla
- [x] Evidencia con logs estructurados (`agent/logs/*.json`)
- [x] Evidencia visual con terminal de inferencia (GET/POST al Docker en vivo)
- [x] Aviso de procedencia (Gemini vs respaldo) en la UI

### Validación del agente (6 pts)

- [x] Escenarios definidos (S1-S6) y reproducibles
- [x] Métricas del agente (5) calculadas y reportadas
- [x] Casos límite (S4 inválido, S5 período largo, S6 off-topic)
- [x] Análisis de fallos honesto (`FAILURE_ANALYSIS.md`)
- [ ] **Quick win:** correr la suite contra el LLM real (Gemini) y reportar sus
      métricas (ver sección 12)

### Ética, sesgos y privacidad (4 pts)

- [x] Tabla de 5 riesgos aterrizada al sistema
- [x] Mitigación concreta implementada/verificable por riesgo
- [x] Límites declarados visibles en el informe (período > 27 d, pre-vetting)

### Reproducibilidad

- [x] `requirements.txt` (agent + inference)
- [x] Comandos exactos documentados
- [x] Modo determinista sin API key ni GPU
- [x] Dockerfile reconstruible

---

## 12. Gaps honestos y quick wins

1. **Validación con el LLM real (recomendado).** `run_eval.py` hoy soporta `--claude`
   (Anthropic) y mock, pero no un flag para el modo `openai`/Gemini que está
   configurado. Las métricas reportadas en `SUMMARY.md` son del **modo mock** (piso
   de control, 1.000 por construcción). Para fortalecer los 6 pts de validación
   conviene agregar un flag `--openai` a `run_eval.py` (cambio chico) y correr la
   suite contra Gemini, así las métricas de faithfulness y tool-call accuracy se miden
   sobre el LLM real (que es donde tienen valor). El `FAILURE_ANALYSIS.md` ya declara
   esto con honestidad.
2. **Figuras XAI reales (opcional).** Las figuras de saliency hoy son sintéticas
   representativas. La XAI real sobre la curva existe en `scripts/run_xai.py` (Etapa 2);
   conectarla al agente reemplazaría `mock.make_lightcurve_saliency`. No bloquea la
   rúbrica del agente, suma a la discusión de XAI del paper.
3. **`system.md` lista 7 tools; el registro tiene 8** (falta `get_star_info`, que se
   usa sobre todo en el modo conversacional). Es cosmético, pero conviene alinearlo
   para coherencia en la revisión.

Ninguno de estos es bloqueante para la entrega. El primero es el de mayor retorno
si querés apuntar al 100% en validación.
