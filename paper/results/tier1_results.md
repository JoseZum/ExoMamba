# Tier 1 — Resultados finales

> Generado automáticamente. Reemplazar `{...}` por valores reales después de correr `scripts/evaluate.py --split test` sobre cada modelo.

## Modelos evaluados

| Modelo | Fase | Val_AUC | Params | Notas |
|---|---|---|---|---|
| Random baseline | 5.a | 0.500 | 0 | Prior estratificado |
| Catalog LogReg | 5.b | {pending} | ~10 | Partner branch |
| CNN baseline | 6 | 0.6795 | 62,881 | AstroNet-inspired |
| Mamba small | 8 | **0.7502** | 131,393 | **+7.07pp sobre CNN** |

## Performance contra test sellado (N=237)

| Modelo | AUC-ROC | AUC-PR | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|
| CNN baseline | {test_auc_cnn} | ... | ... | ... | ... | ... |
| Mamba small | {test_auc_mamba} | ... | ... | ... | ... | ... |

## Curvas ROC comparativas
![ROC](figures/roc_comparison.png)

## Análisis de errores
{Pending — análisis de falsos positivos y falsos negativos del Mamba}

## Conclusión
{Pending}
