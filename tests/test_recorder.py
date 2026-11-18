import csv

import torch

from bcfl.aggregation import NodeUpdate
from bcfl.recorder import RoundRecorder


def make_update(partition_id, seconds, num_examples=100, loss=1.0):
    return NodeUpdate(node_id=partition_id, partition_id=partition_id,
                      weights={"layer": torch.zeros(1)}, num_examples=num_examples,
                      train_loss=loss, train_seconds=seconds)


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_one_node_row_per_trained_node_per_round(tmp_path):
    recorder = RoundRecorder(tmp_path)
    recorder.record_training(1, [make_update(4, 2.5), make_update(1, 3.0)])
    recorder.record_test(1, accuracy=0.5, loss=1.2)
    recorder.record_training(2, [make_update(7, 1.0)])
    recorder.record_test(2, accuracy=0.6, loss=1.1)

    node_rows = read_rows(tmp_path / "node_rounds.csv")
    assert [(r["round"], r["partition_id"], r["train_seconds"]) for r in node_rows] == [
        ("1", "1", "3.0"), ("1", "4", "2.5"), ("2", "7", "1.0"),
    ]


def test_round_row_keeps_the_slowest_node_time(tmp_path):
    recorder = RoundRecorder(tmp_path)
    recorder.record_test(0, accuracy=0.1, loss=2.3)          # round 0: no training yet
    recorder.record_training(1, [make_update(0, 2.0), make_update(1, 5.0), make_update(2, 1.0)])
    recorder.record_test(1, accuracy=0.4, loss=1.5)

    round_rows = read_rows(tmp_path / "rounds.csv")
    assert round_rows[0]["max_train_seconds"] == ""
    assert round_rows[1]["max_train_seconds"] == "5.0"
    assert round_rows[1]["selected_nodes"] == "[0, 1, 2]"
