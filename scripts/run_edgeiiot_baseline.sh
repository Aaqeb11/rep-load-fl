#!/usr/bin/env bash
# Week 1: FedAvg baseline on Edge-IIoTset for three levels of non-IID data, three seeds each.
#   alpha 0.1  = very uneven data
#   alpha 0.5  = moderately uneven
#   alpha 100  = almost IID
# Needs the prepared data first (scripts/prepare_edgeiiot.py).
set -euo pipefail

for alpha in 0.1 0.5 100; do
  for seed in 0 1 2; do
    echo "=== alpha=${alpha} seed=${seed} ==="
    python scripts/run.py --config configs/edgeiiot_baseline.yaml --set data.alpha=${alpha} --set seed=${seed}
  done
done

python scripts/plot.py results/edgeiiot_baseline_alpha*_seed* --metric test_macro_f1 \
    --output results/edgeiiot_baseline_macro_f1.png
python scripts/plot.py results/edgeiiot_baseline_alpha*_seed* --metric test_accuracy \
    --output results/edgeiiot_baseline_accuracy.png
