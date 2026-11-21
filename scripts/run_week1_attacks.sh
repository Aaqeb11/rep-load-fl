#!/usr/bin/env bash
# Week 1 attack grid on Edge-IIoTset (alpha 0.5, 20% malicious, 3 seeds):
#   5 attack settings x 3 aggregators x 3 seeds = 45 runs.
# Each attacked run is compared with the SAME aggregator's no-attack run.
set -euo pipefail

for aggregation in fedavg krum trimmed_mean; do
  for attack in none label_flip scaling free_rider on_off; do
    for seed in 0 1 2; do
      name="week1_${attack}_${aggregation}"
      if [ -f "results/${name}_alpha0.5_seed${seed}/rounds.csv" ] && \
         [ "$(wc -l < "results/${name}_alpha0.5_seed${seed}/rounds.csv")" -gt 100 ]; then
        echo "=== skip ${name} seed=${seed} (already done) ==="
        continue
      fi
      echo "=== ${name} seed=${seed} ==="
      python scripts/run.py --config configs/edgeiiot_baseline.yaml \
        --set run_name=${name} \
        --set methods.aggregation=${aggregation} \
        --set attack.type=${attack} \
        --set seed=${seed}
    done
  done
done
