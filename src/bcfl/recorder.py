"""Save what happened in every round to a CSV file.

One row per round:
    round, test_accuracy, test_loss, train_loss, selected_nodes, round_seconds
"""

import csv
import json
import os
import time

COLUMNS = ["round", "test_accuracy", "test_loss", "train_loss", "selected_nodes", "round_seconds"]


class RoundRecorder:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "rounds.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

        # Filled in during training, written out when the round's test result arrives.
        self.round_started_at = time.time()
        self.train_loss = None
        self.selected_partitions = []

    def start_round(self):
        self.round_started_at = time.time()

    def record_training(self, selected_partitions, train_loss):
        self.selected_partitions = sorted(selected_partitions)
        self.train_loss = train_loss

    def record_test(self, server_round, accuracy, loss):
        """Called after the global model is tested; writes the full row."""
        seconds = time.time() - self.round_started_at if server_round > 0 else 0.0
        row = [
            server_round,
            round(accuracy, 4),
            round(loss, 4),
            "" if self.train_loss is None else round(self.train_loss, 4),
            json.dumps(self.selected_partitions),
            round(seconds, 2),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[round {server_round:>3}] test accuracy = {accuracy:.4f}  test loss = {loss:.4f}")
