"""Save what happened in every round to CSV files.

rounds.csv - one row per round:
    round, test_accuracy, test_loss, test_macro_f1, train_loss, selected_nodes,
    max_train_seconds, round_seconds

confusion.jsonl - one line per round: the test confusion table
    (rows = true class, columns = predicted class). Kept so that attack success
    rates (e.g. "how many DDoS rows were called Normal?") can be computed later
    without rerunning anything.

node_rounds.csv - one row per node that trained, per round:
    round, partition_id, num_examples, train_loss, train_seconds, train_cpu_seconds,
    is_malicious, attacking, influence

    is_malicious / attacking are ground truth for evaluation (1 = yes).
    influence is the node's share of the new global model (0 = filtered out).

workload.csv - only with the real-time workload model on: one row per node (all
    nodes, selected or not) per round: its load, speed, slack, simulated training
    time, whether training was on time, and its real-time jobs and misses (with
    and without training). rounds.csv then also gets deadline_miss_rate (all
    nodes), deadline_miss_rate_without_fl (same jobs, no training anywhere) and
    late_updates (selected nodes whose training missed the round window).

The per-node file is what Step 3 uses to measure how much CPU time training
takes away from each fog node's real-time tasks.

Two kinds of time are logged:
- train_seconds:     wall-clock time. Inflated when several simulated nodes
                     share the laptop's CPU at once.
- train_cpu_seconds: CPU time the node's own process used. Closer to "how much
                     computing training needs", so better for Step 3.
Round 1 is slower for every node: each fresh worker process warms up first.
"""

import csv
import json
import os
import time

ROUND_COLUMNS = [
    "round", "test_accuracy", "test_loss", "test_macro_f1", "train_loss", "selected_nodes",
    "max_train_seconds", "round_seconds",
    "deadline_miss_rate", "deadline_miss_rate_without_fl", "late_updates",
]
NODE_COLUMNS = [
    "round", "partition_id", "num_examples", "train_loss", "train_seconds", "train_cpu_seconds",
    "is_malicious", "attacking", "influence", "on_time", "slack_s",
]
WORKLOAD_COLUMNS = [
    "round", "partition_id", "selected", "busy", "load", "mips", "slack_s", "training_s",
    "training_on_time", "rt_jobs", "rt_missed", "rt_missed_without_training",
]


class RoundRecorder:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "rounds.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(ROUND_COLUMNS)

        self.confusion_path = os.path.join(output_dir, "confusion.jsonl")
        open(self.confusion_path, "w").close()      # start with an empty file

        self.node_csv_path = os.path.join(output_dir, "node_rounds.csv")
        with open(self.node_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(NODE_COLUMNS)

        self.workload_path = os.path.join(output_dir, "workload.csv")   # created on first use

        # Filled in during training, written out when the round's test result arrives.
        self.round_started_at = time.time()
        self.train_loss = None
        self.selected_partitions = []
        self.max_train_seconds = None
        self.workload_summary = None

    def start_round(self):
        self.round_started_at = time.time()

    def record_training(self, server_round, updates, influence=None):
        """Called once per round with the NodeUpdate of every node that trained.

        influence: {node_id: share of the new model} from the aggregator.
        """
        influence = influence or {}
        # One row per node goes straight into node_rounds.csv.
        with open(self.node_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for update in sorted(updates, key=lambda u: u.partition_id):
                writer.writerow([
                    server_round,
                    update.partition_id,
                    update.num_examples,
                    round(update.train_loss, 4),
                    round(update.train_seconds, 3),
                    round(update.train_cpu_seconds, 3),
                    int(update.is_malicious),
                    int(update.attacking),
                    round(influence.get(update.node_id, 0.0), 4),
                    int(update.on_time),
                    "" if update.slack is None else round(update.slack, 2),
                ])

        # The round summary is kept until the test result arrives.
        self.selected_partitions = sorted(update.partition_id for update in updates)
        self.train_loss = sum(update.train_loss for update in updates) / len(updates)
        # The slowest node decides how long the round takes (the straggler).
        self.max_train_seconds = max(update.train_seconds for update in updates)

    def record_workload(self, server_round, rows):
        """Called once per round (workload model on) with one row per node."""
        new_file = not os.path.exists(self.workload_path)
        with open(self.workload_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=WORKLOAD_COLUMNS)
            if new_file:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

        jobs = sum(row["rt_jobs"] for row in rows)
        missed = sum(row["rt_missed"] for row in rows)
        missed_without = sum(row["rt_missed_without_training"] for row in rows)
        late = sum(1 for row in rows if row["selected"] and row["training_on_time"] == 0)
        self.workload_summary = {
            "deadline_miss_rate": missed / jobs if jobs else 0.0,
            "deadline_miss_rate_without_fl": missed_without / jobs if jobs else 0.0,
            "late_updates": late,
        }

    def record_test(self, server_round, result):
        """Called after the global model is tested; writes the full round row.

        `result` is the dictionary returned by model.test().
        """
        seconds = time.time() - self.round_started_at if server_round > 0 else 0.0
        row = [
            server_round,
            round(result["accuracy"], 4),
            round(result["loss"], 4),
            round(result["macro_f1"], 4),
            "" if self.train_loss is None else round(self.train_loss, 4),
            json.dumps(self.selected_partitions),
            "" if self.max_train_seconds is None else round(self.max_train_seconds, 3),
            round(seconds, 2),
        ]
        if self.workload_summary is None or server_round == 0:
            row += ["", "", ""]
        else:
            row += [round(self.workload_summary["deadline_miss_rate"], 5),
                    round(self.workload_summary["deadline_miss_rate_without_fl"], 5),
                    self.workload_summary["late_updates"]]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        with open(self.confusion_path, "a") as f:
            line = {"round": server_round, "confusion": result["confusion"].tolist()}
            f.write(json.dumps(line) + "\n")

        message = (f"[round {server_round:>3}] test accuracy = {result['accuracy']:.4f}  "
                   f"macro-F1 = {result['macro_f1']:.4f}  test loss = {result['loss']:.4f}")
        if self.workload_summary is not None and server_round > 0:
            message += (f"  deadline misses = {self.workload_summary['deadline_miss_rate']:.2%}"
                        f"  late updates = {self.workload_summary['late_updates']}")
        print(message)
