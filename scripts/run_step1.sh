#!/usr/bin/env bash
# Step 1: FedAvg baseline for three levels of non-IID data, three seeds each.
#   alpha 0.1  = very uneven data
#   alpha 0.5  = moderately uneven
#   alpha 100  = almost IID (everyone sees every class)
set -euo pipefail

for alpha in 0.1 0.5 100; do
  for seed in 0 1 2; do
    echo "=== alpha=${alpha} seed=${seed} ==="
    python scripts/run.py --config configs/baseline.yaml --set data.alpha=${alpha} --set seed=${seed}
  done
done

python scripts/plot.py results/baseline_alpha*_seed* --output results/step1_accuracy.png
