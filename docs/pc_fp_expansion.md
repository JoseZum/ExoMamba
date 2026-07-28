# Expansión a PC-vs-FP (triage operacional)

## Por qué

El paper actual resuelve **CP-vs-FP**: planeta confirmado contra falso positivo.
Es una formulación limpia y el paper la declara honestamente como limitación,
pero también es la más fácil: los planetas confirmados pasaron por seguimiento
adicional, así que sus curvas son más nítidas y menos representativas de lo que un
astrónomo ve en el momento del vetting.

El problema operacionalmente relevante es **PC-vs-FP**: dado un candidato sin
confirmar, ¿vale la pena hacerle seguimiento? Es la tarea que resuelven Yu et al.
(2019) y DART-Vetter, así que reformularlo así vuelve el trabajo directamente
comparable con la literatura de triage en vez de ser un estudio aislado.

Además hay 4,788 TOIs con disposición PC en el catálogo que hoy se descartan por
completo: es la mayor fuente de señal desaprovechada del proyecto.

## Estado actual del dataset

| Formulación | TICs etiquetables | Utilizables (est.) | Balance neg:pos |
|---|---:|---:|---:|
| `cp_fp` (paper actual) | 1,852 | 1,576 | 2.01:1 |
| `pc_fp` | 5,892 | ~5,000 | 0.27:1 |
| `triage` (CP+KP+PC vs FP+FA) | 7,143 | ~6,100 | 0.23:1 |

Los "utilizables" aplican la atrición histórica del 14.9 % (TICs sin curva
descargable en MAST, o descartados por calidad en el preprocesado).

Ojo con el **balance invertido**: en `cp_fp` los negativos duplican a los
positivos; en `pc_fp` los positivos cuadruplican a los negativos. El `pos_weight`
de la BCE se calcula solo desde los conteos de train, así que se ajusta sin tocar
código, pero conviene revisar que la métrica principal siga siendo la adecuada
(con esa proporción, AUC-PR se vuelve más informativa que AUC-ROC).

## Costo

Medido sobre las descargas ya hechas (4.6 MB y 6.1 s por TIC de media):

| | Descarga | Disco | Preprocesado |
|---|---:|---:|---:|
| `pc_fp` | ~10 h | ~27 GB | ~1.2 h |
| `triage` | ~12 h | ~32 GB | ~1.4 h |

La descarga es de MAST y depende de la red, no de la GPU; se puede dejar corriendo
por partes porque `download_lightcurves.py` reintenta lo fallido y es incremental.

El entrenamiento es el costo real: con ~3.2x los datos, cada epoch tarda ~3.2x más.
Reproducir el barrido completo del paper (3 modelos x 5 semillas) en la RTX 3050 de
4 GB pasa de ~100 a ~320 GPU-horas. Si eso no es viable, la vía honesta es reportar
PC-vs-FP solo para Mamba y el CNN baseline con 3 semillas.

## Receta

Los splits sellados del paper **no se tocan**: todo lo nuevo lleva prefijo propio.

```bash
# 1. Etiquetas. Mirar primero los conteos sin escribir nada:
python scripts/make_labels.py --task pc_fp --dry-run
python scripts/make_labels.py --task pc_fp
#    -> data/splits/tics_labeled_pc_fp.csv

# 2. Descarga de curvas (incremental y reintentable; se puede cortar y retomar).
#    Ajustar la fuente de TICs al nuevo CSV de etiquetas antes de correrlo.
python scripts/download_lightcurves.py

# 3. Preprocesado a la vista global de 18,000 cadencias.
python scripts/preprocess_global.py

# 4. Splits por TIC, con prefijo para no pisar los sellados.
python scripts/make_splits.py \
    --labels data/splits/tics_labeled_pc_fp.csv \
    --out-prefix pc_fp_
#    -> data/splits/pc_fp_{train,val,test}_tics.csv

# 5. Entrenar apuntando los configs a los nuevos CSV, y evaluar.
python scripts/train.py --config configs/mamba_small_pc_fp.yaml --seed 42
python scripts/evaluate.py --run experiments/<run_dir> --split test
```

El paso 5 necesita un config nuevo (copiar `configs/mamba_small.yaml` y apuntar
`data.train_csv` / `data.val_csv` / `data.test_csv` a los archivos `pc_fp_*`).

## Qué falta en el pipeline

`download_lightcurves.py` y `preprocess_global.py` todavía leen rutas fijas
(`data/splits/tics_labeled.csv` y `data/splits/manifest.csv`). Para la expansión
hay que parametrizarlos igual que se hizo con `make_splits.py`, o correrlos contra
una copia del archivo de etiquetas. Es el único trabajo de plomería pendiente.

## Qué cambiaría en el paper

No es un cambio de números: es un cambio de reclamo. Con PC-vs-FP funcionando:

- La sección de Limitations pierde su punto más débil ("label selection bias").
- La comparación con Yu et al. (2019) y DART-Vetter pasa de ser contextual a ser
  una comparación real de tarea, aunque no de dataset.
- El encuadre puede subir de *controlled feasibility study* a resultado
  operacionalmente relevante, que es lo que separa un paper de workshop de uno de
  journal de astronomía.

Un AUC que se sostenga por encima de ~0.75 en PC-vs-FP sería el umbral a partir del
cual vale la pena apuntar a un journal con revisión más exigente.

## Nota sobre las etiquetas en conflicto

Dos estrellas del dataset actual (TIC 207468071 y TIC 441738827) alojan a la vez un
TOI confirmado y uno falso positivo. El etiquetado histórico, hecho en un notebook,
se quedó con el falso positivo, así que ambas figuran como negativas pese a tener un
planeta confirmado: su curva contiene un tránsito real.

`scripts/make_labels.py` lo resuelve explícitamente con `--on-conflict positive-wins`
(por defecto). El impacto en los resultados publicados es nulo — las dos cayeron en
train y val, no en el test sellado — pero en `pc_fp` y `triage` los conflictos suben
a 4 y 14 respectivamente, así que la regla deja de ser irrelevante.
