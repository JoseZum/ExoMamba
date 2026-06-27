# Ética, sesgos y privacidad del agente de vetting

> Input para la sección de ética del paper (Etapa 3, 4 pts). Cada fila está
> **aterrizada al sistema concreto**, no es un disclaimer genérico. La mitigación
> de cada riesgo está implementada o es verificable en el código/logs.

## Tabla de riesgos

| Riesgo | Mitigación concreta (implementada) | Límite declarado |
|--------|-----------------------------------|------------------|
| **Sesgo del catálogo TOI** - TESS observa ~27 d por sector, sobre-representando planetas de período corto; el modelo lo hereda. | El agente declara incertidumbre cuando el TOI tiene período > 27 d (`build_report_md` inserta el aviso; verificado en S5: 5/5 casos lo declaran). El paper reporta accuracy estratificada por período. | "No confiable para candidatos de período largo (> 27 d)" - visible en cada informe afectado. |
| **Falso negativo en contexto científico** - descartar un planeta real cuesta tiempo de telescopio. | El agente nunca emite veredicto final solo: reporta probabilidad + flag de discrepancia con NASA + recomienda revisión humana. `compare_with_disposition` marca `flag_for_review` y el informe lo expone (S3: flag recall 5/5). | "Diseñado como asistente de pre-vetting, no reemplazo del juicio experto." |
| **Falso positivo** - gastar seguimiento en una binaria eclipsante. | Cross-check con la disposición oficial NASA + verificador físico (`verify_prediction`) que marca cuando profundidad/duración/período son inconsistentes con un planeta. | Recomendación `review`/`reject` cuando los chequeos físicos fallan. |
| **Alucinación del LLM** - Claude podría inventar números al redactar. | Todos los números del informe provienen del JSON de las tools; el system prompt instruye citar literalmente y no calcular. La métrica de **faithfulness** verifica que la probabilidad del informe == la que devolvió `classify` (mock: 36/36). En modo Claude esta métrica es la guardia. | "Los números mostrados son los devueltos por las tools, no inferidos por el LLM." |
| **Datos sensibles / privacidad** | No hay datos personales: todo es fotometría pública NASA/MAST y metadata del catálogo TOI. No se recolecta input del usuario más allá del TIC ID. | "Sin datos sensibles; fuentes 100 % públicas." |

## Notas de despliegue

- Los logs de sesión (`agent/logs/`) guardan TIC ID, tool calls y el informe - nada
  sensible. Están gitignored por higiene, no por privacidad.
- El agente no toma acciones irreversibles: solo lee catálogo/modelo y produce un
  informe. No escribe en bases de datos científicas ni dispara observaciones.
- El modo determinista (sin API key) permite auditar el comportamiento del agente
  sin depender de un proveedor externo - útil para reproducibilidad del paper.
