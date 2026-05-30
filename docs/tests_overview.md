# Suite de pruebas — `tests/`

Documentación de la batería de pruebas unitarias y de integración del repositorio. Pensada como insumo para la sección "Testing" del reporte de Etapa 2.

## Estrategia general

La suite cubre tres niveles distintos:

1. **Smoke / contratos** (`test_smoke.py`, `test_seeds.py`): garantizan que el paquete carga y que la reproducibilidad (semillas) funciona. 
2. **Componentes aislados** (`test_augment.py`, `test_collate.py`, `test_metrics.py`, `test_cnn_baseline.py`): pruebas unitarias puras sobre funciones del paquete, sin tocar disco ni datos. 
3. **End-to-end** (`test_dataset.py`, `test_training_smoke.py`): Prueban el camino de disco, dataset, dataloader, modelo y loop. Usan `pytest.skip()` si los `.pt` preprocesados o los splits no existen, para que `pytest -q` siga pasando en un clon fresco del repo.

---

## 1. `test_smoke.py` — Importación del paquete

**Propósito.** Confirmar que el paquete `exoplanet` y sus 5 subpaquetes cargan sin error. Es el test más simple y el primero que debe pasar.

| Test | Verifica |
|---|---|
| `test_package_imports` | `import exoplanet` funciona y la versión declarada es `"0.1.0"` |
| `test_subpackages_import` | `exoplanet.{data, models, training, evaluation, utils}` son importables uno por uno |

Si esto falla, el `pip install -e .` está roto o hay un import circular. 

---

## 2. `test_seeds.py` — Reproducibilidad de semillas

**Propósito.** Validar que `set_seed()` configura **ambos** generadores aleatorios (PyTorch y NumPy) de forma consistente. Esto hace que los resultados del paper sean reproducibles.

| Test | Verifica |
|---|---|
| `test_torch_reproducible` | Llamar `set_seed(42)` dos veces produce la misma secuencia de `torch.randn(10)` |
| `test_numpy_reproducible` | Lo mismo con `np.random.rand(10)` y `set_seed(123)` |
| `test_seeds_distintas_dan_resultados_distintos` | Sanity: seeds 1 y 2 generan secuencias **distintas** (evita el bug en que `set_seed` no haga nada) |

---

## 3. `test_augment.py` — Pipeline de data augmentation

**Propósito.** Verificar las 4 transformaciones de augmentation definidas en la propuesta original, más el orquestador `Compose`. Solo se aplican al split de train.

La idea de estas transformaciones es generar pequeñas variaciones realistas de una misma curva para que el modelo no memorice posiciones o valores exactos.

Todas las pruebas usan una curva sintética creada con `_make_curve(length=18000, channels=1)`. Esta función genera un tensor con forma `(1, 18000)`, donde `1` representa el canal de la señal y `18000` representa la longitud fija de la secuencia. La curva está centrada alrededor de `1.0`, con una pequeña variación aleatoria, imitando un flujo TESS normalizado.

### Contrato funcional usado por todas las augmentations

Las pruebas no solo verifican que cada transformación corra, sino que cumpla un contrato básico para poder usarse de forma segura dentro del `DataLoader` y del entrenamiento.

1. **Pureza, es decir, no mutar el tensor de entrada.**
La transformación debe devolver una nueva curva modificada, pero sin alterar directamente la curva original. Esto es importante porque, si una función modifica el tensor original por accidente, podría corromper los datos cargados por el `Dataset` o causar errores difíciles de detectar durante el entrenamiento.
2. **Shape y dtype invariantes.**
La salida debe conservar la misma forma y el mismo tipo de dato que la entrada. Por ejemplo, si entra un tensor `(1, 18000)` de tipo `float32`, debe salir otro tensor `(1, 18000)` también de tipo `float32` para que la salida siga siendo compatible con el modelo y con el `DataLoader`.
3. **Determinismo bajo `torch.Generator`.**
Como las transformaciones usan aleatoriedad, las pruebas verifican que al usar la misma semilla se obtenga exactamente el mismo resultado. Esto es fundamental para la reproducibilidad del experimento, ya que permite repetir una corrida bajo las mismas condiciones.
4. **Casos identidad.**
Algunas configuraciones deberían no hacer nada. Por ejemplo, `sigma=0` no debe agregar ruido, `max_shift=0` no debe desplazar la curva, `low=high=1.0` no debe cambiar la amplitud y `prob=0` no debe invertir la curva. Estos casos se prueban para asegurar que las funciones manejen correctamente configuraciones límite.
5. **Validación de entradas inválidas.**
Las pruebas también revisan que configuraciones incorrectas, como rangos invertidos o tipos de transformación desconocidos, generen un `ValueError`. Esto evita que el pipeline falle silenciosamente o produzca datos incorrectos sin avisar.

### Bloque `temporal_shift` (±N puntos)

| Test | Verifica |
|---|---|
| `test_temporal_shift_preserva_shape_y_dtype` | Shape `(1, 18000)` y `float32` se mantienen |
| `test_temporal_shift_no_muta_input` | El input queda intacto |
| `test_temporal_shift_reproducible_con_generator` | Dos `torch.Generator` con misma seed deberían dar la misma salida |
| `test_temporal_shift_max_shift_cero_es_clon` | `max_shift=0` devuelve tensor **nuevo** igual al input (no la misma referencia) |
| `test_temporal_shift_1d_tensor` | Acepta también curvas de forma `(L,)` sin canal explícito |

### Bloque `gaussian_noise` (N(0, σ) sumado)

| Test | Verifica |
|---|---|
| `test_gaussian_noise_preserva_shape_y_dtype` | Shape y dtype invariantes |
| `test_gaussian_noise_no_muta_input` | Pureza |
| `test_gaussian_noise_rango_razonable` | Con σ=0.001, ningún punto se desvía >10σ del original (cota laxa pero atrapa errores de escala) |
| `test_gaussian_noise_reproducible` | Determinismo con `torch.Generator` |
| `test_gaussian_noise_sigma_cero_es_clon` | `sigma=0` es clon (no-op no destructivo) |

### Bloque `time_reverse` (inversión temporal con probabilidad `p`)

| Test | Verifica |
|---|---|
| `test_time_reverse_prob_uno_invierte` | `prob=1.0` siempre invierte; salida = `torch.flip(x, dims=[-1])` |
| `test_time_reverse_prob_cero_no_invierte` | `prob=0.0` nunca invierte, devuelve clon |
| `test_time_reverse_no_muta_input` | Pureza |
| `test_time_reverse_reproducible` | Determinismo con `prob=0.5` y generador |

### Bloque `amplitude_scale` (multiplicación por factor uniforme en `[low, high]`)

| Test | Verifica |
|---|---|
| `test_amplitude_scale_preserva_shape_y_dtype` | Invariantes |
| `test_amplitude_scale_no_muta_input` | Pureza |
| `test_amplitude_scale_factor_en_rango` | Sobre 20 seeds, la mediana del ratio `out/x` cae dentro de `[low, high] ± 1e-5` |
| `test_amplitude_scale_low_igual_high` | `low=high=1.0` es identidad (`torch.allclose`, no `equal`, por float) |
| `test_amplitude_scale_low_mayor_que_high_lanza_error` | Rango invertido levanta `ValueError` |

### Bloque `Compose` y registry YAML

| Test | Verifica |
|---|---|
| `test_compose_aplica_en_orden` | Aplica `[gaussian_noise, amplitude_scale(1,1)]`; como el segundo es identidad, la salida iguala aplicar solo el primero con misma seed. Confirma orden + propagación del `generator` |
| `test_compose_reproducible` | Dos pipelines construidos desde el mismo dict YAML producen salida bit-a-bit idéntica |
| `test_build_augment_pipeline_tipo_desconocido` | YAML con `{"type": "no_existe"}` levanta `ValueError` matching "desconocida" |
| `test_build_augment_pipeline_entrada_invalida` | YAML sin clave `type` levanta `ValueError` matching "inválida" |
| `test_compose_vacio` | `Compose([])` es no-op (devuelve input). Permite YAMLs que desactiven augmentation |

---

## 4. `test_collate.py` — Función de batch para el DataLoader

**Propósito.** Validar el `collate_fn` que arma batches a partir de muestras heterogéneas. Cada muestra es un `dict` con campos opcionales (`local_view`, `scalar_features` pueden ser `None` en Tier 1, tensor en Tier 2).

Las pruebas usan un helper `_sample(tic, label, with_local, with_scalars)` que fabrica dicts sintéticos.

| Test | Verifica |
|---|---|
| `test_batch_con_todos_none` | Batch de 3 muestras Tier 1 (sin local, sin scalars): `global_view` se apila como `(3, 1, 18000)`; `local_view` y `scalar_features` del batch quedan `None`; `label` y `tic_id` se vuelven listas correctas |
| `test_batch_con_local_y_scalars` | Batch Tier 2: `local_view` apilado como `(2, 1, 200)`, `scalar_features` como `(2, 5)` |
| `test_batch_mixto_lanza_error` | Si una muestra trae `local_view` y otra no, levanta `ValueError("Batch mixto")` (evita silently ignoring) |
| `test_batch_vacio_lanza_error` | Batch vacío levanta `ValueError("Batch vacío")` |

El test de "batch mixto" es importante: previene un bug sutil donde el DataLoader mezcla TICs con y sin local_view y el modelo recibe un batch incoherente.

---

## 5. `test_metrics.py` — Métricas de clasificación

**Propósito.** Verificar que `compute_classification_metrics(y_true, y_prob, threshold=0.5)` calcula AUC-ROC, F1, Recall y Precision correctamente en casos canónicos.

| Test | Verifica |
|---|---|
| `test_clasificador_perfecto` | Con `y_prob` perfectamente separable de `y_true`, todas las métricas = 1.0 |
| `test_clasificador_inverso` | Si `y_prob` predice exactamente lo contrario, `auc_roc = 0.0` y `recall = 0.0` |
| `test_una_sola_clase_devuelve_nan` | Con `y_true = [1,1,1,1]` (sin clase 0), AUC y F1 son `NaN` (no excepción, ni cero falso) — comportamiento documentado para que el reporting maneje el caso degenerado |
| `test_threshold_custom` | Mismo `y_prob`, dos `threshold` distintos producen recall distinto (`0.5` vs `1.0`). Demuestra que el threshold se respeta |

---

## 6. `test_cnn_baseline.py` — Modelo CNN baseline

**Propósito.** Sanity checks sobre el CNN inspirado en AstroNet (Fase 6, single-branch sobre global_view).

| Test | Verifica |
|---|---|
| `test_forward_output_shape` | Batch de 4 → logits shape `(4,)` y dtype `float32` |
| `test_backward_pasa` | El gradiente se propaga: tras `loss.backward()`, todos los parámetros entrenables tienen `.grad is not None` |
| `test_params_count_razonable` | Conteo de parámetros entre 10K y 1M (cabe en 4 GB VRAM de la RTX 3050) |
| `test_kwargs_arquitectura_funciona` | Configurable: `channels=(8,16)`, `hidden_dim=16`, `dropout=0` con `length=1000` también funciona. Confirma que la arquitectura no tiene tamaños hardcoded |

Los 4 son cheap y rápidos (no requieren datos reales).

---

## 7. `test_dataset.py` — `LightCurveDataset` (integración con disco)

**Propósito.** Validar `LightCurveDataset` leyendo `.pt` reales del disco. Skip-si-no-hay-datos: si `data/splits/train_tics.csv` o `data/processed/global/` no existen, todos los tests del módulo se saltan limpiamente.

### Fixture `dataset` (Tier 1, solo global)

| Test | Verifica |
|---|---|
| `test_dataset_non_empty` | `len(dataset) > 0` |
| `test_sample_schema` | Cada muestra es dict con exactamente las claves `{tic_id, global_view, local_view, scalar_features, label}`. `tic_id` es int, `label` ∈ {0, 1}. En Tier 1, `local_view` y `scalar_features` son `None` |
| `test_global_view_shape_and_dtype` | `global_view` es tensor `float32` shape `(1, 18000)` — confirma normalización a longitud fija |

### Fixture `dataset_with_local` (Tier 2)

Skip adicional si no hay `.pt` en `data/processed/local/`.

| Test | Verifica |
|---|---|
| `test_local_view_optional_load` | En splits Tier 1 con `local_dir` seteado: si el `.pt` local existe para un TIC, devuelve tensor `(1, 201)`; si no existe, devuelve `None` (no lanza excepción). Recorre hasta 50 muestras para encontrar al menos un cargado |
| `test_tier2_splits_all_load_local` | En splits Tier 2 (que por construcción solo contienen TICs con local_view válido), **todos** los samples deben tener `local_view` como tensor `(1, 201)`, ninguno `None` |

Estos dos tests juntos validan el contrato Tier 1 vs Tier 2 definido en `CLAUDE.md`: Tier 1 usa todos los TICs con global_view, Tier 2 es un subconjunto estricto con global + local válidos.

---

## 8. `test_training_smoke.py` — End-to-end del training loop

**Propósito.** Correr un entrenamiento mínimo (config `configs/smoke.yaml`: 1 epoch, batch=4, subset=16, arquitectura mini) y verificar que todos los artefactos de un run se generan correctamente.

Como los anteriores, skip si faltan splits o `.pt`. El output va a `tmp_path` para no contaminar `experiments/`.

| Test | Verifica |
|---|---|
| `test_smoke_devuelve_run_dir` | `run_training()` devuelve dict con clave `run_dir` y el directorio existe en disco |
| `test_smoke_artefactos_creados` | En el run_dir existen los 6 artefactos obligatorios: `config.yaml`, `env_info.txt`, `git_info.txt`, `train.log`, `metrics.csv`, `checkpoints/last.pt` |
| `test_smoke_metrics_csv_tiene_columnas_esperadas` | `metrics.csv` tiene las columnas `{epoch, train_loss, val_loss, val_auc_roc, val_f1, lr}` y al menos 1 fila |

Es el test más caro (un entrenamiento real), pero es la única forma de detectar regresiones en el `runner` (config loading + DataLoader build + loop + checkpoint + logging) en una sola corrida.

---

## Resumen para el reporte

- **7 archivos**, **41 tests**.
- Cobertura por capa: paquete (2), reproducibilidad (3), augmentation (21), batching (4), métricas (4), modelo (4), dataset (5), training loop (3).
- Política de skip-si-no-hay-datos en tests de integración: el repo clonado en limpio pasa `pytest -q` sin necesidad de regenerar el preprocesamiento.
- Contratos validados: reproducibilidad de seeds, pureza de augmentations (no mutación), forma/dtype de tensores, schema del dataset, integridad de artefactos de run.
- Lo que **no** cubre la suite: GPU/CUDA, FP16, mamba-ssm (que vive en WSL2 y no entra en CI), evaluación XAI, error analysis, ensemble. Esos paths se validan corriéndolos manualmente (`scripts/*.py`) según el README.
