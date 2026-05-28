#!/usr/bin/env bash
# One-shot helper: corre eval test sobre Mamba locked + 5 seeds en WSL2.
# Cada corrida cuenta como UNA evaluación del test sellado.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUNS=(
    experiments/2026-05-22_14-32-51_mamba_small
    experiments/2026-05-27_23-00-33_mamba_small_seed42
    experiments/2026-05-28_00-49-39_mamba_small_seed123
    experiments/2026-05-28_01-26-18_mamba_small_seed456
    experiments/2026-05-28_01-44-54_mamba_small_seed789
    experiments/2026-05-28_02-17-54_mamba_small_seed2024
)

for run in "${RUNS[@]}"; do
    echo "==== $run ===="
    python scripts/evaluate.py --run "$run" --split test 2>&1 | tail -30
done
