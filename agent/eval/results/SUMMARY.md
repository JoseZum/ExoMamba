# Resultados de validación del agente - suite S1–S6

- **Modo:** mock
- **Sesiones totales:** 36
- **Tool-call accuracy (global):** 1.000
- **Faithfulness (global):** 1.000
- **Tasa de no-alucinación (global):** 1.000
- **Latencia media:** 1098.4 ms

## Por escenario

| ID | Escenario | N | Tool-call acc | Faithfulness | No-alucina | Lat (ms) | Extra |
|----|-----------|---|---------------|--------------|-----------|----------|-------|
| S1 | 10 CP confirmados | 10 | 10/10 | 10/10 | 10/10 | 1224 | acc vs NASA 7/10 |
| S2 | 10 FP conocidos | 10 | 10/10 | 10/10 | 10/10 | 1095 | acc vs NASA 6/10 |
| S3 | 5 discrepancias modelo vs NASA | 5 | 5/5 | 5/5 | 5/5 | 1650 | flag recall 5/5 |
| S4 | TIC inexistente / malformado | 3 | 3/3 | 3/3 | 3/3 | 0 |  |
| S5 | período > 27 d (límite físico) | 5 | 5/5 | 5/5 | 5/5 | 1622 | límite declarado 5/5 |
| S6 | off-topic | 3 | 3/3 | 3/3 | 3/3 | 0 |  |

_Generado por `python -m agent.eval.run_eval`. El modo mock usa el orquestador determinista; con ANTHROPIC_API_KEY se reejecuta contra Claude._