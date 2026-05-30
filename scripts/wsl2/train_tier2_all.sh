#!/usr/bin/env bash
# Entrena ExoMamba V1 + AstroNet multibranch en 3 seeds cada uno (6 runs total).
# Secuencial: la RTX 3050 4GB solo aguanta un training a la vez.
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SEEDS=(42 123 789)

echo "############################################################"
echo "# ExoMamba V1 — 3 seeds"
echo "############################################################"
for seed in "${SEEDS[@]}"; do
    echo
    echo "--- ExoMamba V1 seed=$seed ---"
    python scripts/train.py \
        --config configs/exomamba_v1.yaml \
        --seed "$seed" \
        --name-suffix "_seed${seed}" 2>&1 | tail -5
done

echo
echo "############################################################"
echo "# AstroNet multibranch — 3 seeds"
echo "############################################################"
for seed in "${SEEDS[@]}"; do
    echo
    echo "--- AstroNet seed=$seed ---"
    python scripts/train.py \
        --config configs/astronet_multibranch.yaml \
        --seed "$seed" \
        --name-suffix "_seed${seed}" 2>&1 | tail -5
done

echo
echo "############################################################"
echo "# DONE — 6 runs completados"
echo "############################################################"
ls -1d experiments/2026-05-28_*_exomamba_v1_seed* 2>/dev/null || true
ls -1d experiments/2026-05-28_*_astronet_multibranch_seed* 2>/dev/null || true
