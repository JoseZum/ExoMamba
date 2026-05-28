"""CLI de entrenamiento.

Uso:
  python scripts/train.py --config configs/cnn_baseline.yaml
  python scripts/train.py --config configs/smoke.yaml

Overrides opcionales (para sweeps multi-seed sin duplicar YAML):
  --seed 123                  Override de experiment.seed
  --name-suffix _seed123      Sufijo concatenado a experiment.name (afecta el run_dir)
"""

from __future__ import annotations

import argparse
import sys

from exoplanet.training import load_config, run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar un modelo desde un config YAML.")
    p.add_argument("--config", type=str, required=True, help="Ruta al YAML del experimento.")
    p.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    p.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Sufijo concatenado a experiment.name (útil para multi-seed).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.name_suffix:
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + args.name_suffix
    summary = run_training(cfg)
    print(f"\nRun dir: {summary['run_dir']}")
    print(f"Mejor val_auc: {summary['best_val_auc']:.4f} (epoch {summary['best_epoch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
