"""Modelos del proyecto: random baseline (Fase 5.a), CNN baseline (Fase 6),
Mamba puro (Fase 8), ExoMamba (Tier 2).

Nota: MambaBaseline se importa de forma normal pero su __init__ requiere
mamba-ssm (solo Linux/WSL2). Importar la clase NO falla en Windows;
instanciarla sí.
"""

from exoplanet.models.base import BaseModel
from exoplanet.models.cnn_baseline import CNNBaseline
from exoplanet.models.mamba_baseline import MambaBaseline
from exoplanet.models.random_baseline import RandomBaseline

__all__ = ["BaseModel", "CNNBaseline", "MambaBaseline", "RandomBaseline"]
