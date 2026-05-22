# Mamba-Exoplanet

> Selective State Space Models para detección de exoplanetas en curvas de luz de TESS, estudio comparativo frente a un baseline CNN 1D.

**Proyecto académico** Inteligencia Artificial, Instituto Tecnológico de Costa Rica, Semestre I 2026.
**Autores:** José Fabián Zumbado Ruiz, Jeremmy Aguilar Villanueva.
**Profesor:** Kenneth Obando Rodríguez.

## Objetivo

Evaluar si una arquitectura basada en **Mamba** (Gu & Dao, 2023) puede igualar o superar el rendimiento de clasificadores CNN 1D del estado del arte (familia AstroNet / ExoMiner) en la tarea binaria de distinguir **Confirmed Planets (CP)** de **False Positives (FP)** en curvas de luz de TESS a cadencia de 2 minutos, operando directamente sobre la señal cruda `PDCSAP_FLUX`.

Métricas objetivo: AUC-ROC ≥ 0.93, F1 (clase planeta) ≥ 0.85, mejora Mamba sobre CNN ≥ +3 p.p. de AUC, latencia de inferencia < 100 ms por estrella.

---

## Contexto: ¿de qué trata este proyecto?

> Para quien llega sin conocimiento previo de astronomía o ML.

---

### ¿Qué es un exoplaneta y cómo se detecta?

Un **exoplaneta** es un planeta que orbita una estrella distinta al Sol. No podemos fotografiarlos directamente, están demasiado lejos. Uno de los métodos indirectos más usados es el **método de tránsito**: cuando un planeta pasa por delante de su estrella desde nuestro punto de vista, tapa una pequeña fracción de la luz. El brillo de la estrella baja brevemente y luego vuelve a la normalidad.

```
Sin tránsito:  ─────────────────────────────
Con tránsito:  ───────────\____/───────────
```

<img src="public/transit_white.png" width="480" alt="Diagrama de tránsito"/>

Si ese bajón es pequeño, periódico y simétrico, hay evidencia de un planeta en órbita.

---

### ¿Qué es una curva de luz y por qué es la entrada del modelo?

Una **curva de luz** es la serie temporal del brillo de una estrella. TESS la mide cada 2 minutos durante ≈27 días por sector, produciendo una secuencia de ~18,000 puntos por estrella:

```
[1.0001, 0.9998, 1.0000, 0.9999, 0.9982, 0.9979, 0.9981, ...]
```

Esa secuencia es exactamente lo que recibe el modelo como input, sin ningún feature engineering adicional. La señal de tránsito es ese dip en los valores, apenas perceptible entre el ruido.

---

### ¿Qué son TESS y el TOI Catalog?

**TESS** (*Transiting Exoplanet Survey Satellite*, NASA, 2018) genera ≈27 GB de datos fotométricos por día observando más de 200,000 estrellas. Es imposible revisarlos a mano, de ahí la necesidad de clasificadores automáticos.

El **TOI Catalog** (*TESS Objects of Interest*) es la tabla pública donde la NASA registra cada candidato detectado por TESS. Cada estrella tiene un identificador único (**TIC ID**) y un estado:

| Estado | Significado | Uso en este proyecto |
|---|---|---|
| `CP` - Confirmed Planet | Planeta confirmado por revisión científica | **Clase positiva** (label = 1) |
| `FP` - False Positive | Señal descartada: binaria eclipsante, artefacto, etc. | **Clase negativa** (label = 0) |
| `PC` - Planet Candidate | Sin confirmación aún | Excluido del entrenamiento supervisado |
| `KP` - Known Planet | Planeta ya conocido de otras misiones | Excluido |

El dataset de este proyecto usa 724 CP y 1,242 FP (1,966 ejemplos etiquetados en total, conteo real al 2026-04-30).

TESS no observa el cielo completo a la vez: lo divide en regiones llamadas **sectores**, cada una observada durante ≈27 días. Una misma estrella puede aparecer en múltiples sectores, generando varias curvas de luz para el mismo TIC ID.

<img src="public/observation_sector.jpg" width="480" alt="Sectores de observación de TESS"/>

---

### Variables del TOI Catalog: cuales usamos y por que

El TOI Catalog tiene 85 columnas. **Ninguna entra al modelo como feature**: la entrada del modelo es siempre la serie temporal `PDCSAP_FLUX` de los archivos `.fits`. Las columnas del catálogo solo sirven para seleccionar qué estrellas descargar y con qué label.

**Variables que el pipeline usa activamente:**

| Columna | Para qué |
|---|---|
| `tid` | Identificador único de la estrella. Se usa para pedir los `.fits` a MAST y para hacer el split por estrella |
| `tfopwg_disp` | Disposición (CP, FP, PC, KP). Define el label: CP = 1, FP = 0 |
| `st_tmag` | Magnitud TESS. Estrellas con tmag > 15 tienen curvas muy ruidosas; se analiza en Fase 1 para decidir si hay que filtrar |
| `pl_orbper` | Periodo orbital en días. Confirma que el tránsito ocurre dentro de los 27 dias que cubre un sector (~18,000 puntos) |
| `pl_trandurh` | Duración del tránsito en horas. Confirma que la señal abarca suficientes puntos para ser detectable |
| `pl_trandep` | Profundidad del tránsito en ppm. Se analiza para entender si CP y FP tienen distribuciones distintas |
| `sectors` | Sectores en que fue observada la estrella. Informa cuántos `.fits` hay que descargar por TIC ID en Fase 2 |

**Por que se excluyen las otras 78 columnas:**

- Coordenadas y movimiento propio (`ra`, `dec`, `st_pmra`, `st_pmdec`, etc.): no tienen relacion causal con si una señal es planeta o falso positivo a efectos del modelo.
- Columnas de error (`*err1`, `*err2`, `*symerr`, `*lim`): metadatos de precisión de medición, irrelevantes para clasificación.
- Propiedades estelares (`st_teff`, `st_logg`, `st_rad`, `st_dist`): podrían usarse como features auxiliares en arquitecturas mas complejas, pero introducen riesgo de leakage y este proyecto evalua el modelo operando sobre señal cruda.
- Fechas y metadatos (`toi_created`, `rowupdate`, `release_date`, `toipfx`, `toidisplay`): administración del catalogo, sin valor predictivo.

---

### Data leakage por estrella: la trampa mas común en este dominio

Una misma estrella puede haber sido observada por TESS en múltiples sectores, generando varias curvas de luz con el mismo TIC ID. Si al dividir los datos se mete el sector 1 de una estrella en entrenamiento y su sector 13 en test, el modelo puede aprender características propias de esa estrella (ruido estelar, variabilidad intrínseca) y hacer overfitting en el test. El resultado son métricas infladas que no reflejan generalización real. Por eso, el split se hace siempre por TIC ID, nunca por sector.

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
│   ├── models/             # cnn_baseline.py, mamba.py
│   ├── training/           # loop, losses, schedulers
│   ├── evaluation/         # métricas, gráficos
│   └── utils/              # seeds, logging, paths
├── notebooks/              # exploración numerada (01_..., 02_..., 03_...)
├── scripts/                # CLIs reproducibles (download_data, train, evaluate)
├── experiments/            # outputs de cada run               (gitignored)
├── tests/                  # pytest
├── public/                 # imágenes para el README.md
└── paper/                  # LaTeX, figuras y tablas finales
```

## Instalación

**Requisitos previos:**

- Python **3.10 u 3.11** (probado con 3.11.9). Se recomienda la build oficial de [python.org](https://www.python.org/downloads/) sobre la versión de Microsoft Store, que a veces tiene problemas de permisos en `pip install -e`.
- Git Bash o PowerShell en Windows; bash en Linux/macOS.
- Aproximadamente **2.5 GB libres** en disco para el entorno completo (incluyendo PyTorch con CUDA).

> **Nota sobre OneDrive:** si el repositorio queda dentro de una carpeta sincronizada por OneDrive, mové el repo a una ruta local (p. ej. `C:\dev\mamba-exoplanet\`) **antes** de crear el `.venv`. OneDrive intenta sincronizar miles de archivos del entorno virtual y puede corromper binarios de PyTorch.

### Paso 1 - Clonar y posicionarse

```bash
git clone <url-del-repo> mamba-exoplanet
cd mamba-exoplanet
```

### Paso 2 - Crear y activar el entorno virtual

```bash
python -m venv .venv

# Activar - Git Bash en Windows:
source .venv/Scripts/activate
# Activar - PowerShell:
# .venv\Scripts\Activate.ps1
# Activar - Linux / macOS:
# source .venv/bin/activate
```

### Paso 3 - Instalar el paquete en modo editable

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Esto instala el paquete `exoplanet` y todas las dependencias declaradas en `pyproject.toml`, incluyendo `torch` (build **CPU** por defecto), `lightkurve`, `astropy`, `jupyterlab`, `pytest` y `ruff`.

### Paso 4 - Reinstalar PyTorch con CUDA (necesario para fases 5–9)

La build CPU de `torch` no usa la GPU. Para entrenar Mamba en la RTX 3050 hay que reemplazarla por la rueda CUDA. **Verificá primero la versión de CUDA de tu driver:**

```bash
nvidia-smi    # mirá "CUDA Version: XX.Y" en la esquina superior derecha
```

Luego desinstalá la build CPU e instalá la build que corresponde. Con driver 581+ (CUDA 13.0), usar la rueda CUDA 12.8:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Para otras versiones de CUDA, consultá <https://pytorch.org/get-started/locally/> y copiá el comando correspondiente.

Verificación de CUDA:

```bash
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Paso 5 - (Fases 8–9) Setup WSL2 para Mamba

`mamba-ssm` requiere compilar extensiones CUDA con `nvcc` y no tiene wheels pre-construidos para Windows nativo. **Decisión tomada: el modelo Mamba se desarrolla y entrena en WSL2 con Ubuntu 24.04.** Las fases 0–7 (exploración, preprocesamiento, CNN baseline) corren en Windows normalmente.

**Camino rápido (recomendado):** ver `docs/WSL2_SETUP.md` para guía completa. Resumen:

```powershell
# 1) En PowerShell admin (Windows):
wsl --install -d Ubuntu-24.04
```

```bash
# 2) Dentro de Ubuntu WSL2:
cd /mnt/c/Users/jfzum/Downloads/Proyecto-IA/mamba-exoplanet
chmod +x scripts/setup_wsl2.sh
./scripts/setup_wsl2.sh
```

El script `setup_wsl2.sh` es idempotente y hace todo: apt deps, nvcc, venv, torch+cuda, `pip install -e ".[dev,mamba]"`, y corre `verify_wsl2_env.py` al final.

**Camino manual (si el script falla):** ver `docs/WSL2_SETUP.md` sección "Troubleshooting".

### Paso 6 - Verificación final

```bash
pytest -q                                                      # smoke tests deben pasar
python -c "import exoplanet; print(exoplanet.__version__)"     # → 0.1.0
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import seaborn, tensorboard, einops, imbalanced_learn; print('deps extra OK')"
```

## Reproducir el pipeline de datos

Los pasos siguientes asumen que ya completaste la sección **Instalación** y que estás dentro de `mamba-exoplanet/` con el `.venv` activado.

### Fase 1 - Descargar el TOI Catalog y generar tics_labeled.csv

```bash
# Descarga el catálogo completo desde el NASA Exoplanet Archive (TAP)
python scripts/get_data.py
```

Esto produce:
- `data/raw/toi_catalog.csv` - catálogo completo (gitignored, ~7,800 filas)
- `data/splits/toi_summary.csv` - resumen ligero versionado

Para generar `data/splits/tics_labeled.csv` (1,968 filas con CP y FP etiquetados) y reproducir el EDA, abrí y ejecutá el notebook:

```bash
jupyter lab notebooks/01_toi_eda.ipynb
```

El notebook genera `data/splits/tics_labeled.csv`, que es el input de la Fase 2.

### Fase 2 - Descargar curvas de luz desde MAST

El script `scripts/download_lightcurves.py` lee `tics_labeled.csv` y descarga los archivos `_lc.fits` de cadencia 2 min (autor SPOC) para cada TIC ID. Mantiene un manifest en `data/splits/manifest.csv` y es **idempotente**: si se corta, podés volver a correrlo y sigue donde quedó.

**Piloto rápido (5 estrellas, ~30 segundos):**

```bash
python scripts/download_lightcurves.py --limit 5 --shuffle
```

**Descarga completa recomendada (1,968 estrellas, ~3-4 horas, ~9 GB de disco):**

```bash
mkdir -p logs
python scripts/download_lightcurves.py --max-sectors 3 --shuffle > logs/download.log 2>&1 &
```

Flags útiles:
- `--max-sectors N` - cap de sectores por TIC. Sin cap el dataset puede pasar de 30 GB. Para entrenamiento del paper, documentar la elección.
- `--shuffle` - mezcla el orden de descarga con seed=42 (determinístico). Útil para que el progreso parcial sea representativo del dataset.
- `--limit N` - procesa solo los primeros N TICs.
- `--no-retry-failed` - no reintenta TICs en estado `error` o `download_failed` (por defecto los reintenta).

**Monitoreo en otra terminal:**

```bash
tail -f logs/download.log
python -c "import pandas as pd; print(pd.read_csv('data/splits/manifest.csv')['status'].value_counts())"
```

Salida del script:
- `data/raw/lightcurves/mastDownload/TESS/...` - archivos `.fits` descargados (gitignored)
- `data/splits/manifest.csv` - una fila por TIC con `status`, `n_sectors_downloaded`, `total_size_mb`, `downloaded_at` y otras (versionado)

Estados posibles en el manifest:
- `ok` - descarga exitosa, terminal
- `no_data` - MAST no tiene SPOC 2-min para ese TIC, terminal (~10-20% del dataset esperado)
- `error` - excepción durante la consulta, reintentable
- `download_failed` - MAST devolvió resultados pero `download_all` falló, reintentable

### Fases siguientes (no disponibles todavía)

```bash
python scripts/train.py --config configs/cnn_baseline.yaml         # Fase 6
python scripts/smoke_train_mamba.py                                # Fase 8 preflight (WSL2)
python scripts/train.py --config configs/mamba_small.yaml          # Fase 8
python scripts/train.py --config configs/exomamba_v1.yaml          # Fase 10
python scripts/evaluate.py --run experiments/<run_id>
```

## Entorno de referencia

Esta tabla documenta el entorno exacto usado para producir los resultados del paper. Es necesaria para reproducibilidad.

| Parámetro | Fases 0–7 (Windows) | Fase 8–9 (WSL2) |
|---|---|---|
| OS | Windows 11 Home 26200 | Ubuntu 24.04 en WSL2 |
| Python | 3.11.9 | 3.11.x |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 (via wheel) | 12.8 (nvcc instalado) |
| GPU | NVIDIA RTX 3050 4 GB | NVIDIA RTX 3050 4 GB (via WSL2) |
| Driver NVIDIA | 581.83 | 581.83 |
| mamba-ssm | N/A | por determinar en Fase 8 |
| Commit hash | por completar al cierre | igual |
| Seeds globales | por definir en Fase 6 | igual |

> El commit hash y los seeds se fijan en Fase 6 y se copian al paper en Fase 10.

## Hardware de referencia

| Componente | Especificación |
|---|---|
| GPU | NVIDIA RTX 3050 (4 GB VRAM, cuello de botella) |
| CPU | Intel Core i5-12450H (8 cores, 12 threads) |
| RAM | 40 GB |

Las restricciones de VRAM motivan el uso de mixed precision (FP16), `batch_size = 16` y gradient checkpointing en Mamba.

## Roadmap

- [x] **Fase 0** - Setup del repositorio
- [ ] **Fase 1** - Exploración del TOI Catalog
- [ ] **Fase 2** - Pipeline de descarga (MAST + lightkurve)
- [ ] **Fase 3** - Preprocesamiento (normalización, NaN, longitud fija)
- [ ] **Fase 4** - Splits por TIC ID + `Dataset` PyTorch
- [ ] **Fase 5** - CNN 1D baseline
- [ ] **Fase 6** - Training loop (logs, seeds, checkpoints)
- [ ] **Fase 7** - Evaluación (métricas + curvas ROC/PR)
- [ ] **Fase 8** - Modelo Mamba
- [ ] **Fase 9** - Comparación rigurosa CNN vs Mamba
- [ ] **Fase 10** - Paper (figuras y tablas finales)

## Cita

```bibtex
@misc{zumbado_aguilar_2026,
    title       = {Mamba State Space Models for Exoplanet Detection in TESS Light Curves},
    author      = {Zumbado Ruiz, Jos\'e Fabi\'an and Aguilar Villanueva, Jeremmy},
    year        = {2026},
    institution = {Instituto Tecnol\'ogico de Costa Rica}
}
```

## Licencia

MIT. Ver `LICENSE` (pendiente).
