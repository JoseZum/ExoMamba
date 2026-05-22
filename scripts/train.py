"""CLI de entrenamiento.

Uso:
  python scripts/train.py --config configs/cnn_baseline.yaml
  python scripts/train.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import sys

from exoplanet.training import load_config, run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar un modelo desde un config YAML.")
    p.add_argument("--config", type=str, required=True, help="Ruta al YAML del experimento.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    summary = run_training(cfg)
    print(f"\nRun dir: {summary['run_dir']}")
    print(f"Mejor val_auc: {summary['best_val_auc']:.4f} (epoch {summary['best_epoch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
