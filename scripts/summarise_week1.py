"""Summarise the Week 1 attack grid into tables and charts.

Reads the 45 runs made by scripts/run_week1_attacks.sh
(results/week1_<attack>_<aggregation>_alpha0.5_seed<s>/) and writes:

    <out>/summary.md          the tables below, ready to paste into an update
    <out>/runs.csv            one row per run (every number used in the tables)
    <out>/macro_f1_curves.png macro-F1 per round, one panel per attack
    <out>/attacker_influence.png  how much of the model the attackers got

Usage:
    python scripts/summarise_week1.py
    python scripts/summarise_week1.py --results results --meta data/edgeiiot/meta.json

How the numbers are made (so they can be explained in the thesis):
- "Final" values are the average of the last 10 rounds (91-100), because
  macro-F1 jumps around from round to round.
- "mean ± sd" is over the 3 seeds (sd = sample standard deviation).
- Every attacked run is compared with the SAME aggregator's no-attack run.
- A run has "collapsed" if its final macro-F1 is below 0.1 (the lazy
  "always Normal" model scores about 0.06).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ATTACKS = ["none", "label_flip", "scaling", "free_rider", "on_off"]
AGGREGATIONS = ["fedavg", "krum", "trimmed_mean"]
SEEDS = [0, 1, 2]
LAST_ROUNDS = 10
COLLAPSE_THRESHOLD = 0.1

PRETTY = {
    "none": "No attack", "label_flip": "Label flip", "scaling": "Scaling (s = −1)",
    "free_rider": "Free-riding", "on_off": "On-off",
    "fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trimmed mean",
}
# Validated categorical palette (dataviz reference palette, first three slots).
COLORS = {"fedavg": "#2a78d6", "krum": "#eb6834", "trimmed_mean": "#1baf7a"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Class names in label order, used if meta.json is not available.
DEFAULT_CLASS_NAMES = [
    "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP", "Fingerprinting", "MITM",
    "Normal", "Password", "Port_Scanning", "Ransomware", "SQL_injection", "Uploading",
    "Vulnerability_scanner", "XSS",
]


# ----------------------------------------------------------------------------
# Loading one run
# ----------------------------------------------------------------------------
def run_folder(results_dir, attack, aggregation, seed):
    return os.path.join(results_dir, f"week1_{attack}_{aggregation}_alpha0.5_seed{seed}")


def load_run(results_dir, attack, aggregation, seed):
    """Everything saved for one run."""
    folder = run_folder(results_dir, attack, aggregation, seed)
    with open(os.path.join(folder, "config.yaml")) as f:
        config = yaml.safe_load(f)
    confusions = {}
    with open(os.path.join(folder, "confusion.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            confusions[entry["round"]] = np.array(entry["confusion"], dtype=float)
    return {
        "config": config,
        "rounds": pd.read_csv(os.path.join(folder, "rounds.csv")),
        "nodes": pd.read_csv(os.path.join(folder, "node_rounds.csv")),
        "confusions": confusions,
    }


# ----------------------------------------------------------------------------
# Measures (each takes one run)
# ----------------------------------------------------------------------------
def final_value(run, column):
    """Average of a rounds.csv column over the last LAST_ROUNDS rounds."""
    rounds = run["rounds"]
    last_round = rounds["round"].max()
    recent = rounds[rounds["round"] > last_round - LAST_ROUNDS]
    return float(recent[column].mean())


def attack_success_rate(run, source, target):
    """Share of the source class's test rows predicted as the target class,
    averaged over the last LAST_ROUNDS rounds. E.g. Backdoor rows called Normal."""
    last_round = max(run["confusions"])
    rates = []
    for round_number, confusion in run["confusions"].items():
        if round_number > last_round - LAST_ROUNDS:
            rates.append(confusion[source, target] / confusion[source].sum())
    return float(np.mean(rates))


def class_recall(run, class_number):
    """Share of a class's test rows predicted correctly (last LAST_ROUNDS rounds)."""
    return attack_success_rate(run, class_number, class_number)


def per_class_f1(run):
    """F1 of every class, from the confusion table summed over the last rounds."""
    last_round = max(run["confusions"])
    total = None
    for round_number, confusion in run["confusions"].items():
        if round_number > last_round - LAST_ROUNDS:
            total = confusion if total is None else total + confusion
    correct = np.diag(total)
    predicted = total.sum(axis=0)
    actual = total.sum(axis=1)
    f1 = []
    for i in range(len(correct)):
        precision = correct[i] / predicted[i] if predicted[i] > 0 else 0.0
        recall = correct[i] / actual[i] if actual[i] > 0 else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return np.array(f1)


def attack_rounds(config):
    """Rounds in which malicious nodes attack (all rounds, except for on_off)."""
    attack = config["attack"]
    num_rounds = config["federation"]["num_rounds"]
    if attack["type"] != "on_off":
        return list(range(1, num_rounds + 1))
    cycle = attack["honest_rounds"] + attack["attack_rounds"]
    return [r for r in range(1, num_rounds + 1) if (r - 1) % cycle >= attack["honest_rounds"]]


def attacker_share_per_round(run, column):
    """For each attack round: the attackers' total influence (column="influence")
    or their share of the training rows (column="num_examples")."""
    nodes = run["nodes"]
    shares = []
    for round_number in attack_rounds(run["config"]):
        this_round = nodes[nodes["round"] == round_number]
        malicious = this_round[this_round["is_malicious"] == 1]
        if column == "influence":
            shares.append(malicious["influence"].sum())
        else:
            shares.append(malicious["num_examples"].sum() / this_round["num_examples"].sum())
    return np.array(shares)


def attacker_chosen_by_krum(run):
    """(rounds where Krum's single chosen update came from an attacker,
        rounds where at least one attacker was selected)."""
    nodes = run["nodes"]
    chosen = 0
    present = 0
    for round_number in attack_rounds(run["config"]):
        malicious = nodes[(nodes["round"] == round_number) & (nodes["is_malicious"] == 1)]
        if len(malicious) > 0:
            present += 1
        if (malicious["influence"] > 0.5).any():
            chosen += 1
    return chosen, present


def rounds_with_many_attackers(run, at_least):
    """How many rounds had at least `at_least` malicious nodes among the selected."""
    malicious = run["nodes"][run["nodes"]["is_malicious"] == 1]
    per_round = malicious.groupby("round").size()
    return int((per_round >= at_least).sum())


def on_off_gap(attacked_run, clean_run, in_window):
    """Average macro-F1 difference (clean − attacked) over the on-off attack windows
    (in_window=True) or over the honest rounds after the first window (False)."""
    windows = set(attack_rounds(attacked_run["config"]))
    first_window = min(windows)
    clean = clean_run["rounds"].set_index("round")["test_macro_f1"]
    attacked = attacked_run["rounds"].set_index("round")["test_macro_f1"]
    gaps = []
    for round_number in attacked.index:
        if round_number < first_window:
            continue
        if (round_number in windows) == in_window:
            gaps.append(clean[round_number] - attacked[round_number])
    return float(np.mean(gaps))


# ----------------------------------------------------------------------------
# Small helpers for tables
# ----------------------------------------------------------------------------
def mean_sd(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def fmt(values, digits=3):
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def add_end_labels(ax, labels, x, min_gap=0.045):
    """Direct labels at the right end of the lines, nudged apart so they don't overlap
    (identity is never carried by colour alone)."""
    labels = sorted(labels)                      # lowest first
    placed = []
    for y, text in labels:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT_SECONDARY, annotation_clip=False)


def chart_macro_f1_curves(runs, out_path):
    """One panel per attack: macro-F1 per round, seed mean as a line and the
    seed min-max as a light band. On-off attack windows are shaded."""
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(20, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, attack in zip(axes, ATTACKS):
        style_axes(ax)
        if attack == "on_off":
            config = runs[(attack, "fedavg", 0)]["config"]
            for round_number in attack_rounds(config):
                ax.axvspan(round_number - 0.5, round_number + 0.5, color="#efeeea", linewidth=0)
        end_labels = []          # (y at the last round, text) for the direct labels
        for aggregation in AGGREGATIONS:
            curves = [runs[(attack, aggregation, s)]["rounds"].set_index("round")["test_macro_f1"]
                      for s in SEEDS]
            table = pd.concat(curves, axis=1)
            collapsed = any(final_value(runs[(attack, aggregation, s)], "test_macro_f1")
                            < COLLAPSE_THRESHOLD for s in SEEDS)
            if collapsed:
                # Averaging a healthy seed with collapsed ones draws a level no run had,
                # so show every seed as its own thin line instead.
                for s, curve in zip(SEEDS, curves):
                    ax.plot(curve.index, curve.values, color=COLORS[aggregation], linewidth=1.2,
                            label=PRETTY[aggregation] if s == SEEDS[0] else None)
                    end_labels.append((curve.values[-1], f"{PRETTY[aggregation]}, seed {s}"))
            else:
                mean = table.mean(axis=1)
                ax.fill_between(table.index, table.min(axis=1), table.max(axis=1),
                                color=COLORS[aggregation], alpha=0.12, linewidth=0)
                ax.plot(mean.index, mean.values, color=COLORS[aggregation], linewidth=2,
                        label=PRETTY[aggregation])
                end_labels.append((mean.values[-1], PRETTY[aggregation]))
        add_end_labels(ax, end_labels, x=table.index[-1])
        ax.set_title(PRETTY[attack], color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("Round", color=TEXT_SECONDARY, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("Test macro-F1", color=TEXT_SECONDARY, fontsize=9)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    fig.suptitle("Macro-F1 per round, Edge-IIoTset (α = 0.5, 20% malicious). "
                 "Line = mean of 3 seeds, band = min–max (seeds drawn separately where a run collapsed); "
                 "grey = on-off attack windows",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_attacker_influence(influence, data_share, out_path):
    """Grouped bars: attackers' average share of the model per attack and aggregator,
    with a marker for their share of the training data (what FedAvg 'should' give)."""
    attacks = [a for a in ATTACKS if a != "none"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    width = 0.26
    for i, aggregation in enumerate(AGGREGATIONS):
        positions = np.arange(len(attacks)) + (i - 1) * (width + 0.02)
        heights = [influence[(a, aggregation)] for a in attacks]
        bars = ax.bar(positions, heights, width=width, color=COLORS[aggregation],
                      label=PRETTY[aggregation])
        for bar, value in zip(bars, heights):
            ax.annotate(f"{value:.0%}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=8, color=TEXT_SECONDARY, zorder=4,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
    for j, attack in enumerate(attacks):
        ax.hlines(data_share[attack], j - 0.45, j + 0.45, colors=TEXT_PRIMARY,
                  linestyles=(0, (3, 2)), linewidth=1, zorder=3)
    ax.set_xticks(np.arange(len(attacks)))
    ax.set_xticklabels([PRETTY[a] for a in attacks], color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel("Attackers' share of the new model", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("How much of the model the attackers got (average per attack round; "
                 "dashed = their share of the data)", color=TEXT_PRIMARY, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results", help="folder holding the week1_* runs")
    parser.add_argument("--meta", default="data/edgeiiot/meta.json",
                        help="meta.json from prepare_edgeiiot.py (for class names)")
    parser.add_argument("--out", default=None, help="output folder (default: <results>/week1_summary)")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.results, "week1_summary")
    os.makedirs(out_dir, exist_ok=True)

    class_names = DEFAULT_CLASS_NAMES
    if os.path.exists(args.meta):
        with open(args.meta) as f:
            class_names = json.load(f)["class_names"]

    # Load all 45 runs.
    runs = {}
    for attack in ATTACKS:
        for aggregation in AGGREGATIONS:
            for seed in SEEDS:
                runs[(attack, aggregation, seed)] = load_run(args.results, attack, aggregation, seed)

    attack_settings = runs[("label_flip", "fedavg", 0)]["config"]["attack"]
    source = class_names.index(attack_settings["source_class"])
    target = class_names.index(attack_settings["target_class"])

    # One row per run with every number the tables use.
    rows = []
    for (attack, aggregation, seed), run in runs.items():
        row = {
            "attack": attack, "aggregation": aggregation, "seed": seed,
            "final_macro_f1": final_value(run, "test_macro_f1"),
            "final_accuracy": final_value(run, "test_accuracy"),
            "collapsed": final_value(run, "test_macro_f1") < COLLAPSE_THRESHOLD,
            "asr": attack_success_rate(run, source, target),
            "source_recall": class_recall(run, source),
            "rounds_with_3plus_attackers": rounds_with_many_attackers(run, 3),
        }
        if attack != "none":
            row["attacker_influence"] = float(attacker_share_per_round(run, "influence").mean())
            row["attacker_data_share"] = float(attacker_share_per_round(run, "num_examples").mean())
            clean = runs[("none", aggregation, seed)]
            row["macro_f1_drop"] = final_value(clean, "test_macro_f1") - row["final_macro_f1"]
        if attack == "on_off":
            clean = runs[("none", aggregation, seed)]
            row["on_off_gap_windows"] = on_off_gap(run, clean, in_window=True)
            row["on_off_gap_honest"] = on_off_gap(run, clean, in_window=False)
        if aggregation == "krum" and attack != "none":
            chosen, present = attacker_chosen_by_krum(run)
            row["krum_chose_attacker"] = chosen
            row["krum_rounds_attacker_present"] = present
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    def values(attack, aggregation, column):
        selected = table[(table["attack"] == attack) & (table["aggregation"] == aggregation)]
        return selected[column].tolist()

    sections = ["# Week 1 results: attacks × defences on Edge-IIoTset\n",
                "α = 0.5, 20 nodes (10 per round), 100 rounds, 20% malicious, 3 seeds. "
                f"Final = average of rounds {101 - LAST_ROUNDS}–100; ± = sd over the 3 seeds.\n"]

    # Table 1: final macro-F1
    header = ["Attack"] + [PRETTY[g] for g in AGGREGATIONS]
    body = []
    for attack in ATTACKS:
        cells = [PRETTY[attack]]
        for aggregation in AGGREGATIONS:
            cell = fmt(values(attack, aggregation, "final_macro_f1"))
            collapsed = sum(values(attack, aggregation, "collapsed"))
            if collapsed:
                cell += f" ({collapsed}/3 collapsed)"
            cells.append(cell)
        body.append(cells)
    sections.append("## 1. Final macro-F1\n\n" + markdown_table(header, body) + "\n")

    # Table 2: drop vs the same aggregator's clean run, and accuracy
    body = []
    for attack in ATTACKS[1:]:
        body.append([PRETTY[attack]] + [f"{np.mean(values(attack, g, 'macro_f1_drop')):+.3f}"
                                        for g in AGGREGATIONS])
    sections.append("## 2. Macro-F1 drop vs the same aggregator's no-attack run "
                    "(positive = attack hurt)\n\n" + markdown_table(header, body) + "\n")
    body = [[PRETTY[a]] + [f"{np.mean(values(a, g, 'final_accuracy')):.4f}" for g in AGGREGATIONS]
            for a in ATTACKS]
    sections.append("## 3. Final accuracy (for reference; misleading on this imbalanced data)\n\n"
                    + markdown_table(header, body) + "\n")

    # Table 4: attacker influence vs data share
    influence = {}
    data_share = {}
    body = []
    for attack in ATTACKS[1:]:
        data_share[attack] = float(np.mean([np.mean(values(attack, g, "attacker_data_share"))
                                            for g in AGGREGATIONS]))
        cells = [PRETTY[attack], f"{data_share[attack]:.1%}"]
        for aggregation in AGGREGATIONS:
            influence[(attack, aggregation)] = float(np.mean(values(attack, aggregation,
                                                                    "attacker_influence")))
            cells.append(f"{influence[(attack, aggregation)]:.1%}")
        body.append(cells)
    sections.append("## 4. Attackers' share of the new model (average per attack round)\n\n"
                    + markdown_table(["Attack", "Their data share"] + header[1:], body)
                    + "\n\nOn-off counts only its attack rounds.\n")

    # Table 5: Krum's choices
    body = []
    for attack in ATTACKS[1:]:
        chosen = int(sum(values(attack, "krum", "krum_chose_attacker")))
        present = int(sum(values(attack, "krum", "krum_rounds_attacker_present")))
        body.append([PRETTY[attack], f"{chosen} / {present}", f"{chosen / present:.0%}"])
    sections.append("## 5. Krum: rounds where its single chosen update came from an attacker\n\n"
                    + markdown_table(["Attack", "Chosen / rounds with an attacker present (3 seeds)",
                                      "Share"], body) + "\n")

    # Table 6: label flip
    body = []
    for aggregation in AGGREGATIONS:
        body.append([PRETTY[aggregation],
                     fmt(values("none", aggregation, "asr")), fmt(values("label_flip", aggregation, "asr")),
                     fmt(values("none", aggregation, "source_recall")),
                     fmt(values("label_flip", aggregation, "source_recall"))])
    sections.append(f"## 6. Label flip ({class_names[source]} → {class_names[target]})\n\n"
                    + markdown_table(["Aggregator", "ASR, no attack", "ASR, label flip",
                                      f"{class_names[source]} recall, no attack",
                                      f"{class_names[source]} recall, label flip"], body) + "\n")

    # Table 7: on-off
    body = []
    for aggregation in AGGREGATIONS:
        body.append([PRETTY[aggregation], fmt(values("on_off", aggregation, "on_off_gap_windows")),
                     fmt(values("on_off", aggregation, "on_off_gap_honest"))])
    sections.append("## 7. On-off: macro-F1 gap to the clean run (positive = attack hurt)\n\n"
                    + markdown_table(["Aggregator", "During attack windows", "During honest rounds"],
                                     body) + "\n")

    # Table 8: rare classes with no attack
    f1_by_aggregation = {}
    for aggregation in AGGREGATIONS:
        f1_by_aggregation[aggregation] = np.mean([per_class_f1(runs[("none", aggregation, s)])
                                                  for s in SEEDS], axis=0)
    body = []
    for i, name in enumerate(class_names):
        body.append([name] + [f"{f1_by_aggregation[g][i]:.2f}" for g in AGGREGATIONS])
    sections.append("## 8. Per-class F1 with no attack (mean of 3 seeds)\n\n"
                    + markdown_table(["Class"] + header[1:], body) + "\n")

    # Sanity check on selection: rounds with 3+ attackers (expected about 29 of 100).
    many = table[table["attack"] != "none"]["rounds_with_3plus_attackers"].mean()
    sections.append(f"Rounds with 3 or more attackers among the 10 selected: {many:.1f} of 100 on "
                    "average (hypergeometric expectation: 29.0).\n")

    summary = "\n".join(sections)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)
    print(summary)

    chart_macro_f1_curves(runs, os.path.join(out_dir, "macro_f1_curves.png"))
    chart_attacker_influence(influence, data_share, os.path.join(out_dir, "attacker_influence.png"))
    print(f"Saved tables, runs.csv and charts to {out_dir}/")


if __name__ == "__main__":
    main()
