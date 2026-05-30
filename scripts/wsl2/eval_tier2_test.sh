#!/usr/bin/env bash
# Eval test sealed sobre los 6 runs Tier 2 (ExoMamba V1 + AstroNet, 3 seeds c/u).
# Cada corrida cuenta como UNA evaluación del test sellado por modelo+seed.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUNS=(
    experiments/2026-05-28_16-25-39_exomamba_v1_seed42
    experiments/2026-05-28_16-53-09_exomamba_v1_seed123
    experiments/2026-05-28_17-20-26_exomamba_v1_seed789
    experiments/2026-05-28_17-37-31_astronet_multibranch_seed42
    experiments/2026-05-28_17-47-33_astronet_multibranch_seed123
    experiments/2026-05-28_18-01-15_astronet_multibranch_seed789
)

for run in "${RUNS[@]}"; do
    echo "==== $run ===="
    python scripts/evaluate.py --run "$run" --split test 2>&1 | tail -30
done
