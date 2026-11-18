"""Save what happened in every round to CSV files.

rounds.csv - one row per round:
    round, test_accuracy, test_loss, train_loss, selected_nodes,
    max_train_seconds, round_seconds

node_rounds.csv - one row per node that trained, per round:
    round, partition_id, num_examples, train_loss, train_seconds, train_cpu_seconds

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
    "round", "test_accuracy", "test_loss", "train_loss", "selected_nodes",
    "max_train_seconds", "round_seconds",
]
NODE_COLUMNS = [
    "round", "partition_id", "num_examples", "train_loss", "train_seconds", "train_cpu_seconds",
]


class RoundRecorder:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "rounds.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(ROUND_COLUMNS)

        self.node_csv_path = os.path.join(output_dir, "node_rounds.csv")
        with open(self.node_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(NODE_COLUMNS)

        # Filled in during training, written out when the round's test result arrives.
        self.round_started_at = time.time()
        self.train_loss = None
        self.selected_partitions = []
        self.max_train_seconds = None

    def start_round(self):
        self.round_started_at = time.time()

    def record_training(self, server_round, updates):
        """Called once per round with the NodeUpdate of every node that trained."""
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
                ])

        # The round summary is kept until the test result arrives.
        self.selected_partitions = sorted(update.partition_id for update in updates)
        self.train_loss = sum(update.train_loss for update in updates) / len(updates)
        # The slowest node decides how long the round takes (the straggler).
        self.max_train_seconds = max(update.train_seconds for update in updates)

    def record_test(self, server_round, accuracy, loss):
        """Called after the global model is tested; writes the full round row."""
        seconds = time.time() - self.round_started_at if server_round > 0 else 0.0
        row = [
            server_round,
            round(accuracy, 4),
            round(loss, 4),
            "" if self.train_loss is None else round(self.train_loss, 4),
            json.dumps(self.selected_partitions),
            "" if self.max_train_seconds is None else round(self.max_train_seconds, 3),
            round(seconds, 2),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[round {server_round:>3}] test accuracy = {accuracy:.4f}  test loss = {loss:.4f}")
