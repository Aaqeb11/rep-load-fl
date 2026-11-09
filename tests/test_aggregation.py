import numpy as np
import torch

from bcfl.aggregation import NodeUpdate, fedavg
from bcfl.selection import random_selection


def make_update(node_id, value, num_examples):
    weights = {"layer": torch.full((2, 2), float(value))}
    return NodeUpdate(node_id=node_id, partition_id=node_id, weights=weights,
                      num_examples=num_examples, train_loss=0.0)


def test_fedavg_weights_by_number_of_examples():
    # Node A (value 0) has 100 samples, node B (value 10) has 300.
    # Expected average: 0 * 0.25 + 10 * 0.75 = 7.5
    updates = [make_update(1, 0.0, 100), make_update(2, 10.0, 300)]
    result = fedavg(updates, scores={1: 1.0, 2: 1.0})
    assert torch.allclose(result["layer"], torch.full((2, 2), 7.5))


def test_fedavg_of_identical_models_is_unchanged():
    updates = [make_update(i, 3.0, 50) for i in range(5)]
    result = fedavg(updates, scores={})
    assert torch.allclose(result["layer"], torch.full((2, 2), 3.0))


def test_random_selection_picks_the_right_number_of_distinct_nodes():
    rng = np.random.default_rng(0)
    chosen = random_selection(list(range(100, 120)), num_to_pick=10, rng=rng, context={})
    assert len(chosen) == 10
    assert len(set(chosen)) == 10
    assert all(100 <= node < 120 for node in chosen)


def test_random_selection_keeps_huge_node_ids_exact():
    # Flower node ids can be bigger than numpy's int64; they must not be rounded.
    huge_ids = [17692838096047403008, 16775115643702890496, 4098738847234375680]
    chosen = random_selection(huge_ids, num_to_pick=3, rng=np.random.default_rng(0), context={})
    assert sorted(chosen) == sorted(huge_ids)
