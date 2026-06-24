"""Agente de vetting de TOIs (Etapa 3).

Asistente conversacional que consume el modelo final del proyecto (Mamba locked)
como herramienta de un LLM con tool calling, explica sus decisiones (XAI) y entrega
un informe con visualizaciones.

Estado actual: frontend con backend mock (datos reales del catálogo TOI, predicción
del modelo simulada). El LLM real (Claude/Haiku) y el checkpoint Mamba se conectan
reemplazando `classify` y el loop de `llm.py` — la UI no cambia.
"""

__version__ = "0.1.0"
