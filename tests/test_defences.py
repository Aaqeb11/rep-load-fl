import torch

from bcfl.aggregation import NodeUpdate, krum, trimmed_mean


def make_update(node_id, value, num_examples=100):
    weights = {"layer": torch.full((3,), float(value))}
    return NodeUpdate(node_id=node_id, partition_id=node_id, weights=weights,
                      num_examples=num_examples, train_loss=0.0)


def honest_and_attackers():
    # 8 honest nodes close to 1.0, and 2 attackers far away at -10.
    honest = [make_update(i, 1.0 + 0.01 * i) for i in range(8)]
    attackers = [make_update(8, -10.0), make_update(9, -10.0)]
    return honest + attackers


def test_krum_picks_an_honest_update_and_ignores_attackers():
    new_weights, influence = krum(honest_and_attackers(), scores={}, settings={"krum_f": 2})
    assert torch.all(new_weights["layer"] > 0.9)            # an honest model was chosen
    assert influence[8] == 0.0 and influence[9] == 0.0      # attackers had no influence
    assert abs(sum(influence.values()) - 1.0) < 1e-9


def test_trimmed_mean_removes_the_extremes():
    new_weights, influence = trimmed_mean(honest_and_attackers(), scores={}, settings={"trim_k": 2})
    # Both -10 values are among the 2 lowest per coordinate, so they are trimmed away.
    assert torch.all(new_weights["layer"] > 0.9)
    assert influence[8] == 0.0 and influence[9] == 0.0
    assert abs(sum(influence.values()) - 1.0) < 1e-6


def test_trimmed_mean_with_no_trimming_is_the_plain_average():
    updates = [make_update(i, value) for i, value in enumerate([1.0, 2.0, 6.0])]
    new_weights, _ = trimmed_mean(updates, scores={}, settings={"trim_k": 0})
    assert torch.allclose(new_weights["layer"], torch.full((3,), 3.0))


def test_krum_matches_the_worked_example():
    # Squared distances to the 2 nearest neighbours (f = 1, n = 5):
    # 1.05 has the smallest total (0.005), the attacker -10 the largest.
    updates = [make_update(i, value) for i, value in enumerate([1.0, 1.1, 0.9, 1.05, -10.0])]
    new_weights, influence = krum(updates, scores={}, settings={"krum_f": 1})
    assert torch.allclose(new_weights["layer"], torch.full((3,), 1.05))
    assert influence[3] == 1.0 and influence[4] == 0.0


def test_trimmed_mean_cuts_k_from_each_end():
    # k = 1: drop -50 (lowest) and 100 (highest), average 1, 2 and 6 -> 3.
    updates = [make_update(i, value) for i, value in enumerate([1.0, 2.0, 6.0, 100.0, -50.0])]
    new_weights, influence = trimmed_mean(updates, scores={}, settings={"trim_k": 1})
    assert torch.allclose(new_weights["layer"], torch.full((3,), 3.0))
    assert influence[3] == 0.0 and influence[4] == 0.0
    assert abs(influence[0] - 1 / 3) < 1e-6
