"""Servicio de inferencia Mamba para el agente (Etapa 3).

Este subpaquete vive del lado Linux (Docker o WSL2), porque `mamba-ssm` solo
corre con CUDA + nvcc. Expone el modelo Mamba real (seed789, test AUC 0.810) por
HTTP para que el agente Streamlit (Windows) lo consuma vía `agent/model_client.py`.

No importar desde Windows: `predictor` importa `mamba_ssm` de forma transitiva.
"""
