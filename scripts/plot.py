"""Plot test accuracy per round for one or more runs.

Runs that differ only by seed are grouped: the line is the average across
seeds and the shaded band is the min-max range.

Example:
    python scripts/plot.py results/baseline_alpha0.1_seed* results/baseline_alpha0.5_seed* \
        results/baseline_alpha100_seed* --output results/step1_accuracy.png
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def group_name(run_dir):
    """'baseline_alpha0.5_seed2' -> 'baseline_alpha0.5' (drop the seed)."""
    return re.sub(r"_seed\d+$", "", os.path.basename(os.path.normpath(run_dir)))


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy curves.")
    parser.add_argument("run_dirs", nargs="+", help="Result folders containing rounds.csv")
    parser.add_argument("--output", default="results/accuracy.png")
    args = parser.parse_args()

    # Collect the accuracy column of every run, grouped by run name without seed.
    groups = {}
    for run_dir in args.run_dirs:
        csv_path = os.path.join(run_dir, "rounds.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {run_dir}: no rounds.csv")
            continue
        rounds = pd.read_csv(csv_path).set_index("round")["test_accuracy"]
        groups.setdefault(group_name(run_dir), []).append(rounds)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, runs in sorted(groups.items()):
        table = pd.concat(runs, axis=1)
        mean = table.mean(axis=1)
        ax.plot(mean.index, mean.values, label=f"{name} ({len(runs)} seeds)")
        if len(runs) > 1:
            ax.fill_between(mean.index, table.min(axis=1), table.max(axis=1), alpha=0.2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("FedAvg test accuracy per round")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
