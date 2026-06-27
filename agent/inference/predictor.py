"""Predictor Mamba real - carga el checkpoint ganador y corre forward por TIC.

Reemplaza la simulación de `agent/mock.classify` por el modelo de verdad:

  - Modelo: Mamba single-view (seed789), el mejor por test AUC (0.810) según
    `paper/results/all_results.md`. El proyecto ya lo designó para el agente.
  - Carga: reusa `build_model` (registry) + el formato de checkpoint del proyecto
    (`{"model_state", "epoch", "metrics"}`), exactamente como `scripts/evaluate.py`.
  - Forward: lee `data/processed/global/<tic>.pt`, toma `global_view` (1, 18000),
    lo pasa por el modelo y devuelve la probabilidad de planeta.

mamba-ssm SOLO corre en Linux/CUDA, por eso este módulo vive en `agent/inference/`
y se ejecuta dentro del contenedor Docker o del venv de WSL2 - nunca en Windows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

# Importamos los modelos directamente (paquete ligero: torch puro, mamba lazy).
# NO importamos `exoplanet.training`, que arrastra sklearn/tensorboard/etc.: la
# inferencia solo necesita construir el modelo y leer el YAML del run.
from exoplanet.models.astronet_multibranch import AstroNetMultibranch
from exoplanet.models.cnn_baseline import CNNBaseline
from exoplanet.models.exomamba_v1 import ExoMambaV1
from exoplanet.models.mamba_baseline import MambaBaseline
from exoplanet.models.random_baseline import RandomBaseline

# Repo root = .../mamba-exoplanet (agent/inference/predictor.py -> parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Mini-registry local (mismo mapeo que exoplanet.training.registry, sin la dep).
_MODEL_REGISTRY = {
    "mamba_baseline": MambaBaseline,
    "cnn_baseline": CNNBaseline,
    "exomamba_v1": ExoMambaV1,
    "astronet_multibranch": AstroNetMultibranch,
    "random_baseline": RandomBaseline,
}


def _load_config(path: Path) -> dict[str, Any]:
    """Lee el config.yaml del run (equivale a exoplanet.training.config.load_config)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"El config {path} no es un dict YAML")
    return cfg


def _build_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Construye el modelo desde `{type, params}` (equivale al registry del proyecto)."""
    model_type = model_cfg.get("type")
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"model.type='{model_type}' no soportado por el predictor.")
    return _MODEL_REGISTRY[model_type](**(model_cfg.get("params") or {}))

# Mejor modelo Mamba por test AUC (0.810). Override con env MAMBA_RUN_DIR.
DEFAULT_RUN_DIR = "experiments/2026-05-28_01-44-54_mamba_small_seed789"


def _confidence(prob: float) -> str:
    """Mismo mapeo de confianza que el mock, para coherencia de la UI/logs."""
    if prob >= 0.85 or prob <= 0.15:
        return "alta"
    if prob >= 0.70 or prob <= 0.30:
        return "media"
    return "baja"


class MambaPredictor:
    """Carga una vez el Mamba locked y clasifica curvas por TIC ID."""

    def __init__(
        self,
        run_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        rd = run_dir or os.environ.get("MAMBA_RUN_DIR", DEFAULT_RUN_DIR)
        self.run_dir = (REPO_ROOT / rd) if not Path(rd).is_absolute() else Path(rd)
        cfg_path = self.run_dir / "config.yaml"
        ckpt_path = self.run_dir / "checkpoints" / "best.pt"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.yaml no encontrado en {self.run_dir}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoints/best.pt no encontrado en {self.run_dir}")

        self.cfg = _load_config(cfg_path)

        # Device: cuda si hay (mamba-ssm lo requiere de facto). Permite override.
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # processed_dir: del config del run, override por arg/env, resuelto al repo.
        pd_cfg = self.cfg.get("data", {}).get("processed_dir", "data/processed/global")
        pd = processed_dir or os.environ.get("PROCESSED_DIR", pd_cfg)
        self.processed_dir = (REPO_ROOT / pd) if not Path(pd).is_absolute() else Path(pd)

        # Modelo desde el registry local + state dict del checkpoint (patrón de evaluate.py).
        self.model = _build_model(self.cfg["model"]).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.checkpoint_epoch = ckpt.get("epoch")
        # AUC de test si el run lo tiene evaluado (solo metadata para el badge).
        self.test_auc = self._read_test_auc()

    def _read_test_auc(self) -> float | None:
        import json

        mpath = self.run_dir / "eval_test" / "metrics.json"
        if not mpath.exists():
            return None
        try:
            return float(json.loads(mpath.read_text())["metrics"]["auc_roc"])
        except Exception:
            return None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.cfg["model"]["type"],
            "run_dir": str(self.run_dir.relative_to(REPO_ROOT)),
            "device": str(self.device),
            "n_params": self.n_params,
            "checkpoint_epoch": self.checkpoint_epoch,
            "test_auc": self.test_auc,
            "processed_dir": str(self.processed_dir.relative_to(REPO_ROOT)),
        }

    @torch.no_grad()
    def classify(self, tic_id: int) -> dict[str, Any]:
        """Corre el Mamba real sobre la curva del TIC. Devuelve el mismo esquema
        que `mock.classify` + metadata de procedencia (`source`)."""
        pt_path = self.processed_dir / f"{int(tic_id)}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"curva preprocesada no disponible para TIC {tic_id} en {self.processed_dir}"
            )
        payload = torch.load(pt_path, weights_only=False)
        gv = payload["global_view"]                      # (1, L)
        if not torch.is_tensor(gv):
            raise TypeError(f"global_view de TIC {tic_id} no es tensor")
        gv = gv.float().unsqueeze(0).to(self.device)     # (1, 1, L)

        logit = self.model({"global_view": gv})          # (1,)
        prob = float(torch.sigmoid(logit).item())
        prob = round(prob, 3)

        return {
            "prob_planeta": prob,
            "label": "PLANETA" if prob >= 0.5 else "FALSO POSITIVO",
            "confianza": _confidence(prob),
            "source": "mamba_real",
            "model_run": str(self.run_dir.name),
            "device": str(self.device),
        }
