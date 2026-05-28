"""Modelos del proyecto: random baseline (Fase 5.a), CNN baseline (Fase 6),
Mamba puro (Fase 8), ExoMamba V1 (Tier 2 — Fase 10), AstroNet multibranch
(Tier 2 — Fase 11).

Nota: MambaBaseline y ExoMambaV1 se importan de forma normal pero su
__init__ requiere mamba-ssm (solo Linux/WSL2). Importar la clase NO falla
en Windows; instanciarla sí.

AstroNetMultibranch es pure-torch, instancia y corre en Windows.
"""

from exoplanet.models.astronet_multibranch import AstroNetMultibranch
from exoplanet.models.base import BaseModel
from exoplanet.models.cnn_baseline import CNNBaseline
from exoplanet.models.exomamba_v1 import ExoMambaV1
from exoplanet.models.mamba_baseline import MambaBaseline
from exoplanet.models.random_baseline import RandomBaseline

__all__ = [
    "AstroNetMultibranch",
    "BaseModel",
    "CNNBaseline",
    "ExoMambaV1",
    "MambaBaseline",
    "RandomBaseline",
]
