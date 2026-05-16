# Bitácora de investigación

Registro cronológico de decisiones, descubrimientos y cambios del proyecto.
Complementa el historial de git: git captura qué cambió en el código, esta bitácora captura por qué.

---

## 2026-04-29 | Fase 0: Setup del repositorio

### Lo que se hizo

- Estructura inicial del repositorio creada: `src/exoplanet/`, `data/`, `notebooks/`, `scripts/`, `configs/`, `experiments/`, `paper/`.
- `pyproject.toml` con todas las dependencias del proyecto (PEP 621).
- `.gitignore` configurado para excluir `data/raw/`, `data/processed/`, `experiments/`, archivos `.fits`, `.pt` y `.venv`.
- `README.md` inicial con objetivo, estructura y roadmap.
- Entorno virtual creado en `.venv/` con Python 3.11.9.
- Paquete instalado en modo editable con `pip install -e ".[dev]"`.

### Decisiones tomadas

**Entorno Mamba (Fases 8-9): WSL2 con Ubuntu 24.04.**
`mamba-ssm` requiere compilar extensiones CUDA con `nvcc` y no tiene wheels pre-construidos para Windows nativo. Intentar resolverlo en Fase 8 implicaría perder tiempo cerca del deadline. La decision se tomó en Fase 0 para evitar ese bloqueo. Fases 0-7 (exploración, preprocesamiento, CNN) corren en Windows normalmente.

**PyTorch con CUDA 12.8.**
El driver instalado (581.83) soporta hasta CUDA 13.0, compatible con la rueda `cu128`. Se reinstala `torch` con `--index-url https://download.pytorch.org/whl/cu128` para habilitar la RTX 3050. La build CPU que instala `pip install torch` por defecto no usa GPU.

**Dependencias adicionales añadidas a `pyproject.toml`:**
- `imbalanced-learn`: SMOTE para balanceo de clases (Fase 5+)
- `astroquery`: acceso a MAST para descarga de curvas de luz (Fase 2)
- `pyvo`: consultas TAP al NASA Exoplanet Archive (Fase 1)
- `einops`: operaciones de tensor para implementación de Mamba (Fase 8)
- `tensorboard`: logs de entrenamiento (Fase 6)
- `seaborn`: visualizaciones de métricas y comparaciones (Fases 7/9/10)

### Estado al cierre de esta sesión

- `torch 2.11.0+cu128` instalado, CUDA activo, RTX 3050 detectada.
- Smoke tests pasando (2/2).

---

## 2026-04-30 | Fase 1: Exploración del TOI Catalog

### Lo que se hizo

- Script `scripts/get_data.py` que descarga el TOI Catalog completo desde el NASA Exoplanet Archive via TAP y guarda `data/raw/toi_catalog.csv` (gitignored) y `data/splits/toi_summary.csv` (versionado).
- Notebook `notebooks/01_toi_eda.ipynb` con análisis de las 6 variables clave del catálogo.
- Archivo `data/splits/tics_labeled.csv` generado: solo CP y FP, con label binario (1/0).

### Descubrimientos del análisis

**Conteos reales (al 2026-04-30):**
Los números de la propuesta académica estaban desactualizados. Los valores reales del catálogo son:

| Disposición | Propuesta | Real |
|---|---|---|
| CP (Confirmed Planet) | ~638 | 726 |
| FP (False Positive) | ~1,400 | 1,242 |
| PC (Planet Candidate) | ~6,600 | 4,788 |
| KP (Known Planet) | no estimado | 591 |
| APC (Ambiguous PC) | no estimado | 481 |

Total etiquetado para entrenamiento (CP + FP): 1,968. Ratio real CP:FP = 1:1.71, no 1:2.2 como se asumía. Esto cambia el peso de la `weighted cross-entropy`.

**Magnitud TESS (st_tmag):**
Solo 2 CP y 16 FP tienen tmag > 15 (umbral donde el ruido vuelve las curvas poco confiables). No vale la pena filtrar por magnitud: el 99%+ del dataset está en rango útil.

**Periodo orbital (pl_orbper):**
El 8.8% de los CP (64 estrellas) tienen periodo orbital > 27 días. Algunos llegan hasta 1,134 días. TESS observa por sectores de ~27 días, por lo que estas estrellas probablemente no tienen tránsito completo visible en una sola curva. Decision pendiente para Fase 2: descartar estas 64 estrellas o mantenerlas y dejar que el preprocesamiento las maneje.

Solo el 2.7% de los FP tienen periodo > 27 días, lo que confirma que los FP son señales de corto período (binarias eclipsantes, artefactos).

**Profundidad y duración del tránsito:**
Las distribuciones de CP y FP se solapan casi completamente. Medianas: CP 1,964 ppm vs FP 2,870 ppm en profundidad; CP 2.56 h (77 puntos) vs FP 2.59 h (78 puntos) en duración. No existe un umbral simple que separe las clases. Esto confirma que el problema justifica deep learning.

Un tránsito promedio ocupa aproximadamente 77-78 puntos en una secuencia de 18,000. La señal es real pero pequeña en contexto.

**Columna `sectors` completamente vacía:**
La columna `sectors` del TOI Catalog tiene 100% de valores NaN. El catálogo no incluye en qué sectores fue observada cada estrella. En Fase 2 hay que consultar MAST directamente por TIC ID para obtener esa información. Se eliminó la columna `n_sectors` de `tics_labeled.csv` porque era siempre 0.

**NaNs en variables clave:**
Solo 17 NaNs en `pl_orbper` (0.9%). El resto de las variables clave están completas. No hay problema de NaNs en el catálogo.

### Cambios respecto a la propuesta original

- Conteos actualizados en `README.md` con los valores reales.
- Columna `sectors` descartada como fuente de información para Fase 2.
- Ninguna variable del TOI Catalog entra al modelo como feature. El catálogo solo sirve para seleccionar qué estrellas descargar y con qué label. Esto se documentó explícitamente en el README para evitar confusiones futuras.

### Decisiones pendientes que surgieron

- ¿Descartar los 64 CP con periodo > 27 días antes de Fase 2 o mantenerlos? Decisión antes de Fase 2.
- ¿Sumar KP (591 Known Planets) a la clase positiva para compensar el dataset pequeño? Decisión antes de Fase 4.

### Estado al cierre de esta sesión

- `data/splits/tics_labeled.csv`: 1,968 filas, columnas `tid / tfopwg_disp / st_tmag / pl_orbper / label`.
- Fase 1 completa.
- Listo para arrancar Fase 2 (descarga de curvas de luz desde MAST).

---

## 2026-05-04 | Fase 2: Pipeline de descarga MAST (script y piloto)

### Lo que se hizo

- Script `scripts/download_lightcurves.py` que lee `data/splits/tics_labeled.csv`, consulta MAST por cada TIC ID via `lightkurve`, descarga los archivos `_lc.fits` de cadencia 2 min (autor SPOC) y mantiene un manifest CSV con el resultado por TIC.
- Manifest `data/splits/manifest.csv` con columnas `tid / label / n_sectors_found / n_sectors_downloaded / sectors / total_size_mb / status / error / duration_s / downloaded_at`. Versionado.
- Piloto de 5 estrellas (orden mezclado con seed=42).

### Decisiones tomadas

**SPOC 2-min como única cadencia.**
La búsqueda usa `author="SPOC"` y `exptime=120`. Es la pipeline oficial de NASA que produce `PDCSAP_FLUX`, la misma señal que usan AstroNet y ExoMiner. TOIs que solo tienen FFI 30-min o QLP se marcan como `no_data` y se descartan del dataset. Mantenerlas implicaría manejar dos pipelines de preprocesamiento distintos, lo cual no aporta a Tier 1.

**El script descarga FITS, NO extrae PDCSAP_FLUX.**
La extracción de la serie PDCSAP_FLUX y el preprocesamiento (normalización, NaN handling, longitud fija) ocurren en Fase 3. Esta separación permite reanudar el pipeline desde cualquier punto sin re-descargar.

**Idempotencia con estados terminales.**
Estados `ok` y `no_data` son terminales: el script los saltea en corridas sucesivas. Estados `error` y `download_failed` se reintentan por defecto (típicamente fallos transitorios de MAST). Flag opcional `--no-retry-failed` para deshabilitar reintentos. Esta distinción evita el bug de "marcar como hecho cualquier cosa que esté en el manifest", que perdería data de TICs con fallos temporales.

**Reintento limpio sin duplicados.**
Cuando un TIC se reintenta, su fila vieja se borra del manifest antes de añadir la nueva. Garantiza una fila por TIC.

**Manifest se escribe cada 10 TICs, no al final.**
Una corrida de ~1,968 TICs toma horas. Si se corta, perdemos a lo sumo 10 descargas, no todo el progreso.

**Cap `--max-sectors 3` recomendado para descarga completa.**
El piloto reveló que un TIC tenía 32 sectores (63 MB). Sin cap, el dataset puede pasar de 30 GB. Con cap=3, ~9 GB. Se documenta en el script que capar sectores sesga el dataset (siempre se toman los primeros) y que para entrenamiento del paper hay que decidir explícitamente la política de selección. Para Tier 1 con longitud fija L=18,000 (un sector), 3 es suficiente y deja margen para elegir el mejor sector después.

**Mantener los 1,968 TOIs durante la descarga (incluye los 64 CP con período > 27 días).**
Descartarlos en Fase 2 perdería data irrecuperable. La decisión de filtrarlos o no se traslada a Fase 3 (preprocesamiento), donde se decide cómo manejar tránsitos parciales o ausentes. La descarga no cuesta nada extra por mantenerlos.

**`downloaded_at` en formato ISO-8601 UTC.**
Para reproducibilidad: deja constancia exacta de cuándo se obtuvo cada FITS desde MAST (los datos pueden actualizarse).

### Descubrimientos del piloto

**1 de 5 TICs sin SPOC 2-min (TIC 354400186).**
Era candidato del catálogo TOI pero MAST no devuelve curvas SPOC de cadencia 120s para esa estrella. Confirmamos que la pérdida del 10-20% del dataset por este motivo es esperable. El conteo final etiquetado se confirmará al terminar la descarga completa.

**Outlier de 32 sectores (TIC 272086159).**
Una sola estrella aportó 63 MB y tomó 60 s. Justifica el cap. Estrellas en la zona de visión continua de TESS pueden tener decenas de sectores acumulados.

**Patrón de archivos confirmado.**
Los FITS se guardan como `mastDownload/TESS/tess<fecha>-s<sector>-<tid:016d>-<scid>-s/tess<fecha>-s<sector>-<tid:016d>-<scid>-s_lc.fits`. El TIC aparece padded a 16 dígitos. El cálculo de `total_size_mb` con el patrón `**/*{tid:016d}*_lc.fits` funciona correctamente.

### Estado al cierre de esta sesión (piloto)

- `scripts/download_lightcurves.py` listo y validado con piloto de 5 TICs (4 ok, 1 no_data, ~76 MB).
- `data/splits/manifest.csv` con 5 filas.
- `data/raw/lightcurves/mastDownload/TESS/...` con 39 archivos FITS (~76 MB).
- Descarga completa (1,963 TICs restantes) **NO ejecutada todavía**: queda como tarea para correr en background del usuario. ETA estimado 3-4 horas, ~9 GB de disco.

---

## 2026-05-04 | Fase 2: Descarga completa finalizada

### Resultados finales (1,968 TICs procesados)

| Estado | Cantidad | % |
|---|---|---|
| ok | 1,705 | 86.6% |
| no_data | 259 | 13.2% |
| error | 4 | 0.2% |
| **Total** | **1,968** | |

- Sectores descargados: 4,182 (promedio 2.45 por TIC)
- Tamaño total en disco: 7.83 GB
- Ruta: `data/raw/lightcurves/mastDownload/TESS/`

### Interpretación de los estados

**no_data (259 TICs, 13.2%):** MAST confirmó que no existen curvas SPOC de cadencia 2 min para esas estrellas. Pérdida esperada y consistente con la estimación de Fase 0 (10-20%). No son fallos transitorios: el catálogo TOI incluye objetos observados por pipelines alternativas (QLP, FFI 30 min) que descartamos deliberadamente para mantener un preprocesamiento uniforme.

**error (4 TICs, 0.2%):** Fallos transitorios de red o MAST. Se pueden reintentar con `python scripts/download_lightcurves.py` sin flags adicionales (el script reintenta estados `error` por defecto). Si persisten, se descartan — 4 TICs no afectan el dataset.

**Dataset efectivo para Fase 3:** 1,705 TICs con al menos 1 sector SPOC 2-min descargado. La distribución exacta CP/FP dentro de los 1,705 se determina cruzando con `tics_labeled.csv` al iniciar Fase 3.

### Decisión: los 4 errores

Se recomienda un reintento rápido antes de iniciar Fase 3:

```bash
.venv\Scripts\python.exe scripts/download_lightcurves.py
```

Si siguen fallando, se documentan como pérdida permanente y Fase 3 procede con los 1,705.

### Estado al cierre

- `data/splits/manifest.csv`: 1,968 filas, descarga completa.
- `data/raw/lightcurves/`: 7.83 GB de archivos FITS.
- **Fase 2 completada.** Listo para Fase 3 (preprocesamiento base: extracción PDCSAP_FLUX, normalización, NaN handling, longitud fija L=18,000 → `data/processed/global/<tic>.pt`).

---

## 2026-05-04 | Fase 3: Preprocesamiento base (vista global)

### Lo que se hizo

- Script `scripts/preprocess_global.py` que toma los TICs con `status=ok` del manifest de Fase 2, extrae `PDCSAP_FLUX` directamente de los FITS con `astropy.io.fits` (más rápido que `lightkurve.read`), y produce un tensor `.pt` por TIC en `data/processed/global/<tid>.pt`.
- Manifest de salida `data/splits/processed_manifest.csv` con una fila por TIC: `tid, label, sector_chosen, valid_fraction, n_points_raw, status, error, duration_s, processed_at`. Versionado.

### Decisiones tomadas

**Estrategia "mejor sector" en lugar de concatenar.**
Se elige un único sector por TIC en vez de concatenar todos los sectores descargados (hasta 3 por el cap de Fase 2). Razón: el output final es L=18,000 y los sectores TESS de 2-min tienen ~20,000 puntos, así que un solo sector ya casi llena la ventana. Concatenar 2-3 sectores y después recortar a 18k significaría en la práctica usar solo el inicio de la concatenación — no es "más datos", es "el primer sector que apareció". Además, concatenar mete discontinuidades artificiales entre sectores (gaps de días/semanas entre observaciones) que la CNN o Mamba podrían aprender como ruido espurio. Mantenemos los otros sectores en disco por si en Tier 2 o trabajos futuros se decide procesarlos.

**Criterio de "mejor sector": mayor fracción de puntos válidos.**
Para cada sector candidato se calcula `valid_fraction = mean((QUALITY == 0) & isfinite(PDCSAP_FLUX))` sobre el flux crudo y se queda con el sector que maximiza esa fracción. Es objetivo, no requiere metadata externa, y favorece curvas limpias sobre curvas con muchos huecos. Se descartó la alternativa de "elegir el sector que cubra el tránsito según `pl_orbper` y epoch del catálogo" porque (a) requiere metadata que recién entra fuerte en Tier 2, (b) los TOIs sin period bien definido quedarían sin "mejor sector" y (c) introduce un acople innecesario con el catálogo en una fase que debería ser sobre fotometría pura.

**Manejo de NaNs en dos niveles.**
- *Gaps cortos* (≤5 puntos consecutivos, ~10 minutos de cadencia TESS) se interpolan linealmente. Es la cantidad típica de puntos perdidos por flags transitorios sin que la interpolación introduzca señal espuria.
- *Gaps largos* (>5 puntos) NO se interpolan: se dejan como NaN durante el cómputo de `valid_fraction` y luego se reemplazan con `1.0` (la mediana normalizada) al guardar el tensor. La `valid_mask` que acompaña el tensor marca esos puntos como `False` para que el modelo los pueda ignorar opcionalmente.

**Umbral de descarte: `valid_fraction < 0.5`.**
Si después de enmascarar `QUALITY != 0` e interpolar gaps cortos queda menos del 50% de puntos válidos, se descarta el TIC con `status=dropped_low_quality`. Es preferible perder ese TIC que meter una curva mayoritariamente sintética (rellena con 1.0) al modelo. El 50% es un valor conservador; si Fase 6/8 deja ver que descartamos demasiado, se baja a 0.4.

**Normalización por mediana de la propia curva (NO global).**
`flux_norm = flux / nanmedian(flux)`. Usar la mediana del propio sector evita el `data leakage` clásico: si normalizáramos con estadísticas globales del dataset (ej. la mediana del train), las curvas del test tendrían información del train codificada en su escala. Cada curva se normaliza independientemente. Esto está alineado con la sección 4.3 de la propuesta.

**Recorte centrado (no por inicio) y padding con 1.0.**
Como casi todos los sectores tienen ~20,000 puntos > 18,000, el caso típico es recortar. Se hace centrado: `start = (n - 18000) // 2`. Razón: los extremos de un sector TESS suelen tener peor calidad (descontinuidades por gaps de telemetría al inicio/fin del sector, momentum dumps), el centro es lo más limpio. Se descartan extremos por igual en lugar de tirar 2,000 puntos de un solo lado.
Si por excepción `n < 18,000`, se padea simétricamente con `1.0` (mediana post-normalización, no introduce sesgo) y la `valid_mask` marca esas posiciones como `False`.

**Output por TIC: dict con `flux`, `valid_mask`, `sector`, `valid_fraction`.**
Cada `.pt` guarda no solo el tensor de flux sino también la máscara de validez (importante para que el modelo distinga datos reales de padding/gaps), el sector elegido (trazabilidad: poder volver al FITS original) y la fracción válida final (útil para análisis de errores en Fase 9: ¿los TICs mal clasificados tienen menor calidad de curva?).

**Manifest se escribe cada 50 TICs.**
Más espaciado que Fase 2 (cada 10) porque el procesamiento es CPU-only y mucho más rápido por TIC (~décimas de segundo vs ~10 segundos en descarga).

### Resultados de la corrida completa

| Estado | Cantidad |
|---|---|
| ok | 1,576 |
| no_fits | 14 |
| dropped_low_quality | 1 |
| **Total únicos** | **1,591** |

**Dataset final Fase 3: 1,576 archivos `.pt`** en `data/processed/global/`.

- Distribución de clases: **605 CP (label=1) + 971 FP (label=0)**, ratio 1:1.60 (cercano al 1:1.71 esperado).
- Valid fraction promedio: **0.902** (90.2% de puntos reales por curva en promedio).
- Valid fraction min/max: 0.524 / 0.978. El piso 0.524 es el TIC más ruidoso que pasó el umbral.
- Tiempo total: ~22 min (CPU-only, ~0.8s por TIC).

### Bug encontrado: duplicados en `manifest.csv` de Fase 2

La corrida procesó 2,011 filas en vez de las 1,705 esperadas. Diagnóstico: el `manifest.csv` de Fase 2 tenía **116 tids duplicados** (~99 con doble fila `ok`). Causa probable: la lógica de retry borraba la fila vieja antes del nuevo intento, pero algún path (probablemente cuando MAST devolvía éxito tras un fallo previo escrito en otra corrida) no limpiaba bien.

**Impacto real: cero pérdida de datos.** Los TICs duplicados se procesaron 2 veces y los `.pt` se sobreescribieron entre sí (mismo tid → mismo archivo). El conteo real en disco (1,576 archivos) es el correcto.

**Mitigación aplicada:**
- `manifest.csv` deduplicado por `tid` con `keep='last'` (1,968 → 1,852 filas).
- `processed_manifest.csv` deduplicado igual (2,011 → 1,591 filas).
- TODO documentado para Fase 2: revisar la lógica de retry de `download_lightcurves.py` antes de futuras corridas para que no genere duplicados desde el inicio.

### Estado al cierre

- 1,576 archivos `.pt` listos en `data/processed/global/` con forma `{global_view: (1, 18000), valid_mask: (1, 18000), label, sector, valid_fraction, tid}`.
- `data/splits/processed_manifest.csv`: 1,591 filas únicas, versionado.
- **Fase 3 completada.** Listo para Fase 4: splits train/val/test **por TIC ID** (15/15/70) preservando estratificación de clase + `Dataset` PyTorch que devuelva el dict con `global_view` y campos opcionales para Tier 2.

---

## Fase 4 — Splits por TIC ID + Dataset PyTorch (2026-05-09)

### Objetivo

Particionar los 1,576 TICs preprocesados en `train / val / test` **por TIC ID** (nunca por sector ni por archivo) y exponer un `Dataset` PyTorch con la firma acordada en CLAUDE.md, listo para que las Fases 5–8 (baselines, CNN, Mamba) lo consuman sin redesign.

### Decisiones tomadas

**1. Split por TIC ID, nunca por sector.**
El `.pt` actual ya es uno por TIC (Fase 3 eligió el mejor sector), así que el riesgo concreto de leakage por múltiples sectores de la misma estrella ya está mitigado en preprocesamiento. Aun así, mantenemos el contrato "una estrella → un fold" como invariante explícito del pipeline: en Tier 2, si decidiéramos meter más de una vista por TIC, este invariante sigue protegiendo. Es la sección 4.1 de la propuesta.

**2. Estratificación por label en cada paso del split.**
Sin estratificar, con un dataset chico (1,576) y desbalance ~1:1.6, los folds podían quedar sesgados (ej. test con 50% más FP relativos que train) y eso contamina la comparación entre modelos. Con estratificación, cada fold preserva la proporción de clases del dataset original.

**3. Proporciones 70 / 15 / 15.**
Mismas que la propuesta original entregada en Etapa 1. No se cambian: cualquier desviación obligaría a actualizar el documento entregado al curso.

**4. Seed 42, fija y versionada.**
Misma seed para todas las corridas de splits. Si en algún momento hace falta regenerar (ej. agregar TICs nuevos), se vuelve a correr con la misma seed y se obtiene un superset reproducible. No se hace búsqueda de "buena" semilla — eso sería overfit al test.

**5. Split en dos pasos:**
  - Paso 1: `train (70%) vs temp (30%)`, estratificado por label.
  - Paso 2: del `temp`, `val (50%) vs test (50%)` → 15% / 15% del total, también estratificado.
  Es la receta canónica de sklearn para tres splits estratificados.

**6. Verificación dura de existencia de `.pt`.**
El script chequea que cada `tid` del `processed_manifest.csv` con `status=ok` tenga un `.pt` físico en disco. Si falta, lo excluye y avisa. Evita inconsistencias futuras donde alguien borra un `.pt` y los splits siguen apuntando a él.

**7. CSVs versionados con columnas `(tid, label)`.**
Mínimo necesario para reproducibilidad. El Dataset es quien resuelve el path al `.pt` desde el `tid`. Si el día de mañana cambiamos la ubicación de los `.pt`, los splits siguen siendo válidos.

**8. `test_tics.csv` SELLADO hasta Fase 9.**
Política operativa, no enforced en código. Se documenta en el output del script y en la bitácora. Toda decisión de hiperparámetro / arquitectura / early stopping va contra `val_tics.csv`. El test se evalúa una sola vez al final, para reportar (sección 4.2 de la propuesta).

**9. Dataset devuelve dict con `local_view = None` y `scalar_features = None`.**
Cumple la firma de CLAUDE.md desde el inicio. Tier 1 (CNN, Mamba puro) lee solo `global_view` + `label`. Tier 2 (ExoMamba V1/V2) llenará los `None` cuando Fase 3.b genere los artefactos. La interfaz no cambia entre tiers — cambia solo qué claves están pobladas.

  Nota operativa: el `default_collate` de PyTorch no maneja `None`. La Fase 6/7 (training loop) deberá usar un `collate_fn` propio o leer solo las claves necesarias por modelo. Es decisión del training loop, no del Dataset.

### Resultados de la corrida

```
[INFO] TICs elegibles: 1576 (CP=603, FP=973)
[INFO] Split: train=0.7, val=0.15, test=0.15
[INFO] Seed: 42

=== Distribución por fold ===
  train  | n=1103 (69.99%) | CP= 422 | FP= 681 | ratio FP:CP = 1.61:1
  val    | n= 236 (14.97%) | CP=  90 | FP= 146 | ratio FP:CP = 1.62:1
  test   | n= 237 (15.04%) | CP=  91 | FP= 146 | ratio FP:CP = 1.60:1

[OK] Sin overlap de TICs entre folds.
```

| Fold | n | % | CP | FP | Ratio FP:CP |
|---|---:|---:|---:|---:|---:|
| train | 1,103 | 69.99 | 422 | 681 | 1.61 : 1 |
| val   |   236 | 14.97 |  90 | 146 | 1.62 : 1 |
| test  |   237 | 15.04 |  91 | 146 | 1.60 : 1 |
| **Total** | **1,576** | 100.00 | **603** | **973** | 1.61 : 1 |

Discrepancia menor con Fase 3 (que reportó 605 / 971): Fase 3 contaba archivos `.pt` antes del merge contra `tics_labeled.csv`. Aquí el conteo es post-merge — 2 TICs probablemente quedaron sin label limpio en el labels CSV; pendiente de confirmar si afecta más adelante. No bloquea Fase 5+.

### Artefactos generados

- `scripts/make_splits.py` — CLI reproducible, parámetros expuestos (`--seed`, `--train`, `--val`).
- `src/exoplanet/data/dataset.py` — `LightCurveDataset(split_csv, processed_dir)`, devuelve dict por sample.
- `src/exoplanet/data/__init__.py` — expone `LightCurveDataset` para `from exoplanet.data import LightCurveDataset`.
- `tests/test_dataset.py` — 3 smoke tests (no vacío, schema del dict, shape `(1, 18000)` y dtype `float32`).
- `data/splits/train_tics.csv` — 1,103 filas, versionado.
- `data/splits/val_tics.csv` — 236 filas, versionado.
- `data/splits/test_tics.csv` — 237 filas, versionado y SELLADO.

### Verificación

```
$ pytest -q
.....  [100%]
5 passed in 8.05s
```

5/5 tests pasan: 2 originales (importación del paquete y subpaquetes) + 3 nuevos (Dataset).

### Estado al cierre

- Splits reproducibles, sin overlap de TICs, ratios de clase preservados (~1:1.61 en los 3 folds).
- `test_tics.csv` no se toca hasta Fase 9.
- `LightCurveDataset` listo y testeado, firma compatible con Tier 1 y Tier 2.
- **Fase 4 completada.** Listo para Fase 5: escalera de baselines (5.a random estratificado → 5.b LogReg sobre features del catálogo).
