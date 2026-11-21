"""Malicious fog nodes: who they are, and what they do instead of honest training.

Configured in the YAML file, for example:

    attack:
      type: scaling          # none | label_flip | scaling | free_rider | on_off
      fraction: 0.2          # share of the fog nodes that are malicious (0.2 of 20 = 4)
      source_class: Backdoor # label_flip: this attack class is relabelled as...
      target_class: Normal   #             ...this class
      scale: -1.0            # scaling / on_off: sent = global + scale * (local - global)
      honest_rounds: 10      # on_off: behave honestly for this many rounds...
      attack_rounds: 5       #         ...then attack for this many, and repeat
      noise_std: 0.0         # free_rider: tiny noise added to hide the empty update

The coordinator never reads which nodes are malicious. Nodes only report it
in their metrics so that results can be evaluated afterwards (ground truth).
"""

import numpy as np
import torch

ATTACK_TYPES = ["none", "label_flip", "scaling", "free_rider", "on_off"]


def get_attack_settings(config):
    """The attack section of the config, with defaults for anything missing."""
    settings = {
        "type": "none",
        "fraction": 0.0,
        "source_class": "Backdoor",
        "target_class": "Normal",
        "scale": -1.0,
        "honest_rounds": 10,
        "attack_rounds": 5,
        "noise_std": 0.0,
    }
    settings.update(config.get("attack", {}))
    if settings["type"] not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack type '{settings['type']}'. Options: {ATTACK_TYPES}")
    return settings


def get_malicious_partitions(config):
    """Which fog nodes (partition ids) are malicious. Fixed for the whole run.

    Drawn with its own random generator from the run's seed, so every process
    (each simulated node and the coordinator) gets exactly the same answer.
    """
    settings = get_attack_settings(config)
    if settings["type"] == "none" or settings["fraction"] <= 0:
        return []

    num_nodes = config["federation"]["num_nodes"]
    num_malicious = round(settings["fraction"] * num_nodes)

    # "+ 1000" keeps this draw separate from the other uses of the seed.
    rng = np.random.default_rng(config["seed"] + 1000)
    chosen = rng.choice(num_nodes, size=num_malicious, replace=False)
    return sorted(int(partition) for partition in chosen)


def is_attacking_this_round(settings, server_round):
    """Does a malicious node attack in this round?

    For on_off with 10 honest and 5 attack rounds:
        rounds 1-10 honest, 11-15 attack, 16-25 honest, 26-30 attack, ...
    """
    if settings["type"] == "none":
        return False
    if settings["type"] != "on_off":
        return True                          # the other attacks attack every round

    cycle_length = settings["honest_rounds"] + settings["attack_rounds"]
    position_in_cycle = (server_round - 1) % cycle_length    # 0, 1, 2, ... cycle_length-1
    return position_in_cycle >= settings["honest_rounds"]

def flip_labels(loader, source_number, target_number):
    """label_flip: relabel every source-class row of this node as the target class.

    The loader's dataset was just built for this round, so changing it here
    does not affect any other node or any later round.
    Returns how many rows were flipped.
    """
    labels = loader.dataset.tensors[1]
    is_source = labels == source_number
    labels[is_source] = target_number
    return int(is_source.sum())


def scale_update(global_weights, local_weights, scale):
    """scaling: sent = global + scale * (local - global), layer by layer.

    scale = -1: reverse the learning direction (a sign flip)
    scale = -5: reverse it and make it five times bigger
    """
    sent = {}
    for layer_name in global_weights:
        change = local_weights[layer_name].float() - global_weights[layer_name].float()
        sent[layer_name] = global_weights[layer_name].float() + scale * change
    return sent


def free_rider_weights(global_weights, noise_std, seed):
    """free_rider: send back the global model (optionally with tiny noise) instead of training."""
    if noise_std <= 0:
        return {name: tensor.clone() for name, tensor in global_weights.items()}

    generator = torch.Generator().manual_seed(seed)
    sent = {}
    for layer_name, tensor in global_weights.items():
        noise = torch.randn(tensor.shape, generator=generator) * noise_std
        sent[layer_name] = tensor.float() + noise
    return sent
