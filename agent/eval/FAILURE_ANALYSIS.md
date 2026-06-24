# Análisis de fallos del agente

> Input para la sección "Agent failure modes" del paper (Etapa 3, parte de los 6 pts
> de validación). Honesto sobre qué medimos y qué falta medir.

## Contexto: dos regímenes

El agente corre en dos modos. La distinción es central para leer estos resultados:

- **Modo determinista (mock):** la orquestación de tools es fija (orden pre-definido)
  y el informe se construye mecánicamente desde el JSON de las tools. En este modo,
  por construcción, tool-call accuracy = faithfulness = no-alucinación = **1.000**
  (36/36 sesiones). Es el **piso de control**: prueba que el pipeline, las tools, el
  verifier y el logging funcionan sin la variabilidad del LLM.
- **Modo Claude (LLM real):** la orquestación la decide Claude vía tool calling. Aquí
  aparecen los modos de fallo interesantes. **Pendiente de ejecución** hasta cargar
  `ANTHROPIC_API_KEY`; la suite (`run_eval.py --claude`) ya está lista para medirlos.

## Hallazgos reales (modo determinista, 36 sesiones)

1. **El pipeline es robusto en los 6 escenarios.** S1–S6 corren sin excepciones;
   las entradas inválidas (S4) y off-topic (S6) se manejan sin alucinar (0 tools
   llamadas en off-topic, solo `get_toi_info` en TIC inexistente).
2. **El agente flaggea el 100 % de las discrepancias (S3: 5/5).** Cuando el modelo
   contradice la disposición NASA, el informe lo declara y recomienda revisión humana.
3. **El agente declara el límite físico en el 100 % de los casos de período largo
   (S5: 5/5).**
4. **Limitación del modelo subyacente, no del agente:** la accuracy del informe vs
   NASA es 7/10 (S1, CP) y 6/10 (S2, FP). Esto refleja el rendimiento real del modelo
   (AUC ≈ 0.75), no un fallo del agente: el agente reporta fielmente lo que el modelo
   predice, incluso cuando el modelo se equivoca. Que estos errores existan es lo que
   hace que S3 tenga casos reales que flaggear.

## Modos de fallo anticipados (a medir con Claude)

La suite está diseñada para atrapar estos fallos cuando el LLM maneje la orquestación
(cada uno tiene su métrica asociada):

| Fallo anticipado | Métrica que lo detecta | Mitigación ya en el código |
|------------------|------------------------|----------------------------|
| El LLM redondea/inventa la probabilidad ("95 %" en vez de 0.943). | faithfulness < 1 | system prompt: "citá valores literales, nunca calcules". |
| El LLM omite el flag de discrepancia en el informe final. | flag recall (S3) < 1 | `compare_with_disposition` + instrucción explícita; el verifier es 2.ª línea. |
| El LLM no llama `verify_prediction` o `explain`. | tool-call accuracy < 1 | system prompt fija el flujo recomendado de 7 pasos. |
| El LLM responde off-topic usando tools indebidamente. | tool-call accuracy (S6) | system prompt acota el dominio; S6 lo verifica. |
| El LLM omite declarar el límite de período > 27 d. | long_period_declared (S5) < 1 | aviso inyectado en el informe por código como red de seguridad. |

## Cómo reproducir

```bash
python -m agent.eval.run_eval            # modo mock (este reporte)
python -m agent.eval.run_eval --claude   # modo Claude (cuando haya API key)
```

Los resultados se guardan en `agent/eval/results/` (JSON con detalle por caso +
`SUMMARY.md`). Las sesiones individuales quedan en `agent/logs/<session_id>.json`.
