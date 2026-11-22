import torch
from torch.utils.data import DataLoader, TensorDataset

from bcfl.attacks import (
    flip_labels,
    free_rider_weights,
    get_attack_settings,
    get_malicious_partitions,
    is_attacking_this_round,
    scale_update,
)


def config_with(attack):
    return {"attack": attack, "federation": {"num_nodes": 20}, "seed": 0}


def test_twenty_percent_of_twenty_nodes_are_malicious_and_fixed():
    config = config_with({"type": "scaling", "fraction": 0.2})
    first = get_malicious_partitions(config)
    assert len(first) == 4
    assert first == get_malicious_partitions(config)        # same answer in every process


def test_no_attack_means_no_malicious_nodes():
    assert get_malicious_partitions(config_with({"type": "none", "fraction": 0.2})) == []


def test_on_off_cycle_is_10_honest_then_5_attacking():
    settings = get_attack_settings(config_with({"type": "on_off"}))
    attacking_rounds = [r for r in range(1, 31) if is_attacking_this_round(settings, r)]
    assert attacking_rounds == [11, 12, 13, 14, 15, 26, 27, 28, 29, 30]


def test_scaling_by_minus_one_reverses_the_update():
    global_weights = {"layer": torch.tensor([0.40])}
    local_weights = {"layer": torch.tensor([0.42])}        # honest change: +0.02
    sent = scale_update(global_weights, local_weights, scale=-1.0)
    assert torch.allclose(sent["layer"], torch.tensor([0.38]))   # change: -0.02


def test_label_flip_relabels_only_the_source_class():
    labels = torch.tensor([0, 7, 0, 3, 7])                  # class 0 = source, 7 = Normal
    loader = DataLoader(TensorDataset(torch.zeros(5, 2), labels), batch_size=2)
    flipped = flip_labels(loader, source_number=0, target_number=7)
    assert flipped == 2
    assert loader.dataset.tensors[1].tolist() == [7, 7, 7, 3, 7]


def test_plain_free_rider_sends_the_global_model_back():
    global_weights = {"layer": torch.tensor([1.0, 2.0])}
    sent = free_rider_weights(global_weights, noise_std=0.0, seed=0)
    assert torch.equal(sent["layer"], global_weights["layer"])
