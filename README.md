# Mamba-Exoplanet

> Selective State Space Models para *vetting* de exoplanetas en curvas de luz de TESS, comparados contra una escalera de baselines (Random, LogReg, CNN single-branch).

**Proyecto académico**  Inteligencia Artificial, Instituto Tecnológico de Costa Rica, Semestre I 2026.\
**Autores:** José Fabián Zumbado Ruiz, Jeremmy Aguilar Villanueva.
**Profesor:** Kenneth Obando Rodríguez.

## Objetivo

Evaluar si una arquitectura basada en **Mamba** (Gu & Dao, 2023) puede igualar o superar a clasificadores CNN 1D del estado del arte (familia AstroNet / ExoMiner) en la tarea binaria de distinguir **Confirmed Planets (CP)** de **False Positives (FP)** en curvas de luz de TESS a cadencia de 2 minutos, operando directamente sobre la señal cruda `PDCSAP_FLUX`.

## Resultados (test sellado)

| Modelo | Test AUC-ROC | Run / artefactos |
|---|---:|---|
| Random estratificado | 0.500 | `experiments/2026-05-21_05-36-11_random_baseline` |
| Catalog LogReg | 0.605 | `experiments/logreg_baseline_test.txt` |
| CNN single-branch | 0.604 | `experiments/2026-05-20_23-44-48_cnn_baseline` |
| Mamba locked | 0.763 | `experiments/2026-05-22_14-32-51_mamba_small` |
| **Mamba ensemble (5 seeds)** | **0.806** | `paper/results/mamba_ensemble/` |
| Mamba best seed (789) | 0.810 | `experiments/2026-05-28_01-44-54_mamba_small_seed789` |

El reporte técnico completo está en `paper/reporte_etapa2.md` y `paper/reporte_etapa2.tex`.

---

## Contexto: ¿de qué trata este proyecto?

> Para quien llega sin conocimiento previo de astronomía o ML.

### ¿Qué es un exoplaneta y cómo se detecta?

Un **exoplaneta** es un planeta que orbita una estrella distinta al Sol. No podemos fotografiarlos directamente, están demasiado lejos. Uno de los métodos indirectos más usados es el **método de tránsito**: cuando un planeta pasa por delante de su estrella desde nuestro punto de vista, tapa una pequeña fracción de la luz. El brillo de la estrella baja brevemente y luego vuelve a la normalidad.

```
Sin tránsito:  ─────────────────────────────
Con tránsito:  ───────────\____/───────────
```

<img src="public/transit_white.png" width="480" alt="Diagrama de tránsito"/>

Si ese bajón es pequeño, periódico y simétrico, hay evidencia de un planeta en órbita.

### ¿Qué es una curva de luz y por qué es la entrada del modelo?

Una **curva de luz** es la serie temporal del brillo de una estrella. TESS la mide cada 2 minutos durante ≈27 días por sector, produciendo una secuencia de ~18,000 puntos por estrella:

```
[1.0001, 0.9998, 1.0000, 0.9999, 0.9982, 0.9979, 0.9981, ...]
```

Esa secuencia es exactamente lo que recibe el modelo como input, sin ningún feature engineering adicional. La señal de tránsito es ese dip en los valores, apenas perceptible entre el ruido.

### ¿Qué son TESS y el TOI Catalog?

**TESS** (*Transiting Exoplanet Survey Satellite*, NASA, 2018) observa a cadencia de 2 minutos las ~200,000 estrellas enanas más brillantes del cielo, además de imágenes de campo completo. Es imposible revisar todo a mano, de ahí la necesidad de clasificadores automáticos.

El **TOI Catalog** (*TESS Objects of Interest*) es la tabla pública donde la NASA registra cada candidato detectado por TESS. Cada estrella tiene un identificador único (**TIC ID**) y un estado:

| Estado | Significado | Uso en este proyecto |
|---|---|---|
| `CP` - Confirmed Planet | Planeta confirmado por revisión científica | **Clase positiva** (label = 1) |
| `FP` - False Positive | Señal descartada: binaria eclipsante, artefacto, etc. | **Clase negativa** (label = 0) |
| `PC` - Planet Candidate | Sin confirmación aún | Excluido del entrenamiento supervisado |
| `KP` - Known Planet | Planeta confirmado por misiones previas | Excluido por decisión experimental |

El dataset etiquetado de este proyecto contiene 1,576 TICs (CP+FP) tras filtrar por disponibilidad en `lightkurve`.

TESS no observa el cielo completo a la vez: lo divide en regiones llamadas **sectores**, cada una observada durante ≈27 días. Una misma estrella puede aparecer en múltiples sectores, generando varias curvas de luz para el mismo TIC ID.

<img src="public/observation_sector.jpg" width="480" alt="Sectores de observación de TESS"/>

### Variables del TOI Catalog: cuáles usamos y por qué

El TOI Catalog tiene 85 columnas. **Ninguna entra al modelo como feature**: la entrada del modelo es siempre la serie temporal `PDCSAP_FLUX` de los archivos `.fits`. Las columnas del catálogo solo sirven para seleccionar qué estrellas descargar y asignar el label.

| Columna | Para qué |
|---|---|
| `tid` | Identificador único de la estrella. Se usa para pedir los `.fits` a MAST y para hacer el split por estrella |
| `tfopwg_disp` | Disposición (CP, FP, PC, KP). Define el label: CP = 1, FP = 0 |
| `pl_orbper` | Período orbital en días. Análisis exploratorio |
| `pl_tranmid` | Tiempo del centro del tránsito (T0). Análisis exploratorio |
| `pl_trandurh` | Duración del tránsito. Análisis exploratorio |
| `pl_trandep` | Profundidad del tránsito en ppm. Análisis exploratorio |
| `st_tmag` | Magnitud TESS. Análisis exploratorio de SNR |

Las otras columnas (coordenadas, errores, metadatos de catálogo) no aportan señal predictiva o introducen riesgo de leakage; se omiten.

### Data leakage por estrella: la trampa más común en este dominio

Una misma estrella puede haber sido observada por TESS en múltiples sectores, generando varias curvas con el mismo TIC ID. Si el split mezcla sector 1 de una estrella en train y sector 13 en test, el modelo aprende características propias de esa estrella (ruido estelar, variabilidad intrínseca) y hace overfitting al test. El resultado son métricas infladas que no reflejan generalización real. Por eso el split se hace **por TIC ID, nunca por sector**.

```
TIC 261136679 → train   (sectores 1, 2 y 13 van todos a train)
TIC 123456789 → test    (todos sus sectores van a test)
```

Ninguna estrella aparece en más de una partición.

---

## Estructura del repositorio

```
mamba-exoplanet/
├── configs/                # YAMLs por experimento (un archivo = un run reproducible)
├── data/
│   ├── raw/                # .fits descargados de MAST       (gitignored)
│   ├── processed/          # tensores listos para entrenar    (gitignored)
│   └── splits/             # TIC IDs de train/val/test        (versionado)
├── src/exoplanet/          # código fuente como paquete instalable
│   ├── data/               # descarga, preprocesamiento, Dataset, augment
│   ├── models/             # cnn_baseline, mamba
│   ├── training/           # loop, losses, schedulers, runner
│   ├── evaluation/         # métricas, plots, XAI
│   └── utils/              # seeds, logging, paths
├── scripts/                # CLIs reproducibles (un script por etapa del pipeline)
│   └── wsl2/               # helpers shell para entorno WSL2
├── notebooks/              # exploración (01_toi_eda.ipynb)
├── experiments/            # outputs de cada run               (gitignored)
├── tests/                  # pytest (49 tests)
├── docs/                   # documentación interna
├── public/                 # imágenes para el README.md
└── paper/                  # reporte LaTeX + figuras + tablas + resultados
```

---

## Instalación

**Requisitos previos:**

- Python **3.10 u 3.11** (probado con 3.11.9). Se recomienda la build oficial de [python.org](https://www.python.org/downloads/) sobre la versión de Microsoft Store.
- Git Bash o PowerShell en Windows; bash en Linux/macOS.
- ~2.5 GB libres en disco para el entorno (incluye PyTorch con CUDA).

> **Nota OneDrive:** si el repositorio queda dentro de una carpeta sincronizada por OneDrive, movelo a una ruta local (p. ej. `C:\dev\mamba-exoplanet\`) **antes** de crear el `.venv`. OneDrive intenta sincronizar miles de archivos del entorno virtual y puede corromper binarios de PyTorch.

### 1. Clonar y posicionarse

```bash
git clone <url-del-repo> mamba-exoplanet
cd mamba-exoplanet
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv .venv

# Activar - Git Bash en Windows:
source .venv/Scripts/activate
# Activar - PowerShell:
# .venv\Scripts\Activate.ps1
# Activar - Linux / macOS:
# source .venv/bin/activate
```

### 3. Instalar el paquete en modo editable

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Esto instala el paquete `exoplanet` y todas las dependencias declaradas en `pyproject.toml`, incluyendo `torch` (build CPU por defecto), `lightkurve`, `astropy`, `jupyterlab`, `pytest` y `ruff`.

### 4. Reinstalar PyTorch con CUDA (necesario para entrenar con GPU)

La build CPU de `torch` no usa la GPU. Para entrenar en la RTX 3050 hay que reemplazarla por la rueda CUDA. **Verificá primero la versión de CUDA del driver:**

```bash
nvidia-smi    # mirá "CUDA Version: XX.Y" en la esquina superior derecha
```

Luego desinstalá la build CPU e instalá la build que corresponda. Con driver 581+ (CUDA 13.0), usar la rueda CUDA 12.8:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Para otras versiones de CUDA, consultá <https://pytorch.org/get-started/locally/>.

Verificación:

```bash
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 5. Setup WSL2 para Mamba (necesario solo para entrenar Mamba)

`mamba-ssm` requiere compilar extensiones CUDA con `nvcc` y no tiene wheels pre-construidos para Windows nativo. El modelo Mamba se desarrolla y entrena en WSL2 con Ubuntu 24.04. El resto del pipeline (descarga, preprocesamiento, CNN, evaluación) corre en Windows nativo sin problema.

```powershell
# En PowerShell admin (Windows):
wsl --install -d Ubuntu-24.04
```

```bash
# Dentro de Ubuntu WSL2:
cd /mnt/c/Users/jfzum/Downloads/Proyecto-IA/mamba-exoplanet
chmod +x scripts/setup_wsl2.sh
./scripts/setup_wsl2.sh
```

El script `setup_wsl2.sh` es idempotente y hace todo: apt deps, nvcc, venv, torch+cuda, `pip install -e ".[dev,mamba]"`, y corre `verify_wsl2_env.py` al final. Detalles y troubleshooting en `docs/WSL2_SETUP.md`.

### 6. Verificación final

```bash
pytest -q                                                      # 49 tests deben pasar
python -c "import exoplanet; print(exoplanet.__version__)"     # → 0.1.0
```

---

## Reproducir el pipeline completo

Comandos marcados con **[WSL2]** requieren Linux + `mamba-ssm`; el resto corre en Windows nativo. Todos se ejecutan desde la raíz del repo con el venv activado.

### 1. Datos: descarga y preprocesamiento (una sola vez)

```bash
# Catálogo TOI + tics_labeled.csv + toi_summary.csv
python scripts/get_data.py

# Curvas de luz desde MAST (~3-4 h, ~9 GB; respeta cuotas y es idempotente)
python scripts/download_lightcurves.py --max-sectors 3 --shuffle

# Preprocesamiento Tier 1: tensores globales L=18000 por TIC
python scripts/preprocess_global.py

# Splits por TIC ID (70/15/15)
python scripts/make_splits.py

```

Salidas clave (versionadas): `data/splits/{train,val,test}_tics.csv`.
Salidas grandes (gitignored): `data/raw/`, `data/processed/`.

### 2. Entrenar baselines Tier 1

```bash
# Random estratificado (~5 s, CPU)
python scripts/train.py --config configs/random_baseline.yaml

# CNN single-branch (~30 min, CPU o GPU)
python scripts/train.py --config configs/cnn_baseline.yaml

# Mamba single  locked baseline  [WSL2, ~1 h]
python scripts/train.py --config configs/mamba_small.yaml

# Mamba multi-seed sweep  [WSL2, ~1 h × 5 = ~5 h]
for seed in 42 123 456 789 2024; do
    python scripts/train.py --config configs/mamba_small.yaml \
        --seed $seed --name-suffix "_seed${seed}"
done

# LogReg sobre features del catálogo (~10 s, CPU)
python scripts/train_logreg.py
```

### 3. Evaluar test sellado (una sola vez por modelo)

```bash
# Tier 1
python scripts/evaluate.py --run experiments/2026-05-20_23-44-48_cnn_baseline --split test
python scripts/evaluate.py --run experiments/2026-05-22_14-32-51_mamba_small --split test   # [WSL2]
python scripts/evaluate.py --run experiments/2026-05-21_05-36-11_random_baseline --split test

# Mamba 5 seeds
for run in experiments/2026-05-2{7,8}_*_mamba_small_seed*; do
    python scripts/evaluate.py --run "$run" --split test   # [WSL2]
done

# LogReg
python scripts/train_logreg.py --split test
```

Cada eval produce `<run_dir>/eval_test/{metrics.json, predictions.csv, roc_curve.png, pr_curve.png, confusion_matrix.png, calibration.png}`.

### 5. Ensembles (promedio de probabilidades sobre múltiples seeds)

```bash
# Mamba (5 seeds → AUC 0.806)
python scripts/ensemble_eval.py \
  --runs experiments/2026-05-27_23-00-33_mamba_small_seed42,experiments/2026-05-28_00-49-39_mamba_small_seed123,experiments/2026-05-28_01-26-18_mamba_small_seed456,experiments/2026-05-28_01-44-54_mamba_small_seed789,experiments/2026-05-28_02-17-54_mamba_small_seed2024 \
  --split test \
  --output-dir paper/results/mamba_ensemble
```

### 6. Curva ROC comparativa

```bash
python scripts/plot_tier1_comparison.py
# → paper/figures/roc_tier1.png
```

### 7. XAI sobre Mamba best seed

```bash
# [WSL2] Saliency + Integrated Gradients + Occlusion sobre 8 casos
# (top-2 por cuadrante TP/TN/FN/FP)
python scripts/run_xai.py \
  --run experiments/2026-05-28_01-44-54_mamba_small_seed789 \
  --split test \
  --output paper/figures/xai/mamba_seed789
# → 24 PNGs + _summary.png
```

### 8. Análisis de errores sobre Mamba ensemble

```bash
python scripts/error_analysis.py \
  --predictions paper/results/mamba_ensemble/ensemble_predictions.csv \
  --catalog data/splits/toi_summary.csv \
  --output paper/results/error_analysis/mamba_ensemble
# → top_{fn,fp}.csv, prob_histogram.png, error_rate_by_feature.png,
#   top_{fn,fp}_curves.png, error_analysis_summary.md
```

### 9. Compilar el reporte técnico

El reporte de Etapa 2 está en `paper/reporte_etapa2.md` y `paper/reporte_etapa2.tex`.

```bash
# LaTeX → PDF
cd paper
pdflatex reporte_etapa2.tex
pdflatex reporte_etapa2.tex   # segunda pasada para TOC y referencias
# → paper/reporte_etapa2.pdf

# Alternativa: Markdown → PDF con pandoc
pandoc paper/reporte_etapa2.md -o paper/reporte_etapa2.pdf \
  --pdf-engine=xelatex --toc --variable geometry:margin=2.5cm
```

### 10. Tests

```bash
pytest -q   # 49 tests, todos deben pasar
```

---

## Entorno de referencia

Tabla del entorno exacto usado para producir los resultados reportados. Necesaria para reproducibilidad.

| Parámetro | Pipeline general (Windows) | Modelo Mamba (WSL2) |
|---|---|---|
| OS | Windows 11 Home 26200 | Ubuntu 24.04 (WSL2) |
| Python | 3.11.9 | 3.12.x |
| PyTorch | 2.11.0+cu128 | 2.5.1+cu121 |
| CUDA Toolkit | 12.8 (via wheel) | 12.1 (nvcc nativo) |
| GPU | NVIDIA RTX 3050 4 GB | NVIDIA RTX 3050 4 GB (via WSL2) |
| Driver NVIDIA | 581.83 | 581.83 (host) |
| mamba-ssm | N/A | 2.2.6.post3 (pinned `<2.3`) |
| causal-conv1d | N/A | recompilado contra torch 2.5 |
| transformers | N/A | `<5` (pinned por compatibilidad) |
| Seeds multi-seed | n/a | {42, 123, 456, 789, 2024} |

### Hardware de referencia

| Componente | Especificación |
|---|---|
| GPU | NVIDIA RTX 3050 (4 GB VRAM  cuello de botella) |
| CPU | Intel Core i5-12450H (8 cores, 12 threads) |
| RAM | 40 GB |

Las restricciones de VRAM motivan el uso de mixed precision (FP16), `batch_size = 16` y gradient checkpointing en Mamba.

---

## Estado de entregas

### Etapa 2  Modelado, entrenamiento, XAI y evaluación (45 %)  entregada

- [x] **Baselines:** Random estratificado, Catalog LogReg, CNN single-branch.
- [x] **Modelo principal:** Mamba single-view, locked + 5-seed sweep + ensemble.
- [x] **Protocolo:** splits por TIC ID (70/15/15), test sellado, multi-seed como sustituto de K-fold.
- [x] **Métricas:** AUC-ROC, AUC-PR, F1, Recall, Precision, Brier; curvas ROC y PR; matriz de confusión; calibración.
- [x] **Análisis de errores:** top FN/FP, histograma `y_prob` por clase, tasa de error vs features físicas.
- [x] **XAI:** Gradient Saliency, Integrated Gradients, Occlusion Sensitivity sobre 8 casos (top-2 por cuadrante TP/TN/FN/FP) en Mamba best seed.
- [x] **Reproducibilidad:** configs YAML versionados, seeds fijos, `env_info.txt` + `git_info.txt` por run, 49 tests automatizados.
- [x] **Reporte técnico:** `paper/reporte_etapa2.{md,tex}` con figuras y tablas.

### Etapa 3  Agente, validación, ética y paper IEEE (25 %)  pendiente

- [ ] Agente LLM con *tool calling* sobre el modelo Mamba como herramienta de *vetting*.
- [ ] Validación del agente (escenarios, casos límite) + análisis ético.
- [ ] Artículo IEEE/ACM final.

---

## Cita

```bibtex
@misc{zumbado_aguilar_2026,
    title       = {Mamba State Space Models for Exoplanet Vetting in TESS Light Curves},
    author      = {Zumbado Ruiz, Jos\'e Fabi\'an and Aguilar Villanueva, Jeremmy},
    year        = {2026},
    institution = {Instituto Tecnol\'ogico de Costa Rica}
}
```

## Licencia

MIT. Ver `LICENSE` (pendiente).
