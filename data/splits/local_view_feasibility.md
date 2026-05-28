# Análisis de factibilidad — local_view (Fase 3.b / Tier 2)

Generado por `scripts/analyze_local_view_feasibility.py`.

## Catálogo TOI

- Total TOIs (rows en `toi_summary.csv`): **7,931**
- TICs únicos: **7,622**
- TICs con al menos una TOI con (period, epoch, duration) válidos: **7,535**

Columnas usadas: `pl_orbper`, `pl_tranmid` (BJD), `pl_trandurh` (horas).

## Filtros aplicados

1. Los 3 campos no-NaN y > 0.
2. `pl_orbper` <= **27.4** días (1 sector TESS).
3. `pl_trandurh / 24 < pl_orbper / 2` (descarta duraciones degeneradas).

## Conteos por split

| Split | N total | Con metadata | Period > 27.4d | Dur degen | **N Tier 2 OK** | OK · CP | OK · FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1103 | 1093 | 42 | 0 | **1051** | 396 | 655 |
| val | 236 | 234 | 8 | 0 | **226** | 84 | 142 |
| test | 237 | 235 | 13 | 0 | **222** | 82 | 140 |

## Distribución de clases dentro de Tier 2 OK

| Split | OK CP | OK FP | Ratio CP/(CP+FP) | Total Tier 1 CP | Total Tier 1 FP |
|---|---:|---:|---:|---:|---:|
| train | 396 | 655 | 0.377 | 422 | 681 |
| val | 84 | 142 | 0.372 | 90 | 146 |
| test | 82 | 140 | 0.369 | 91 | 146 |

## Veredicto HARD STOP

**N_train_tier2 = 1051 >= 500** → Tier 2 es VIABLE. Se puede continuar con `scripts/preprocess_local.py`.
