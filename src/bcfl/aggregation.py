"""AGGREGATE step: combine the nodes' updates into one new model.

Every aggregator has the same shape:

    aggregator(updates, scores, settings) -> (new_weights, influence)

- updates:   list of NodeUpdate objects (one per node that trained)
- scores:    {node_id: score} from the scoring step
- settings:  extra parameters, e.g. {"krum_f": 2, "trim_k": 2}
- new_weights: the new global model, {layer name: tensor}
- influence:   {node_id: share of the new model that came from this node}.
               The shares add up to 1. A node that was filtered out gets 0.
               Used to measure whether attackers (e.g. free-riders) got through.

To add a new aggregator, write a function with this shape and add it to AGGREGATORS.
"""

from dataclasses import dataclass

import torch


@dataclass
class NodeUpdate:
    """Everything the coordinator gets back from one node after training."""

    node_id: int            # Flower's id for the node
    partition_id: int       # which slice of the data this node holds
    weights: dict           # the trained model: {layer name: tensor}
    num_examples: int       # how many training samples the node used
    train_loss: float
    train_seconds: float = 0.0       # wall-clock time the node spent training this round
    train_cpu_seconds: float = 0.0   # CPU time it used (not inflated by other nodes sharing the CPU)
    # Ground truth, for evaluation only (the methods never look at these):
    is_malicious: bool = False       # is this one of the attacker nodes?
    attacking: bool = False          # did it actually attack this round?


def fedavg(updates, scores, settings=None):
    """Federated averaging (McMahan et al., 2017).

    The new model is a weighted average of the nodes' models, where each node's
    weight is (its number of samples) / (total samples across chosen nodes).
    Nodes with more data count for more.

    Scores are ignored here: plain FedAvg trusts everyone.
    """
    total_examples = sum(update.num_examples for update in updates)
    influence = {u.node_id: u.num_examples / total_examples for u in updates}

    new_weights = {}
    for layer_name in updates[0].weights:
        weighted_sum = torch.zeros_like(updates[0].weights[layer_name], dtype=torch.float32)
        for update in updates:
            weighted_sum += influence[update.node_id] * update.weights[layer_name].float()
        new_weights[layer_name] = weighted_sum

    return new_weights, influence

def flatten(weights):
    """Turn a model's weights into one long vector, so we can measure distances."""
    return torch.cat([weights[name].float().flatten() for name in sorted(weights)])


def krum(updates, scores, settings):
    """Krum (Blanchard et al., 2017): keep the ONE update that looks most "central".

    For each update, add up its squared distances to its (n - f - 2) closest
    neighbours. Attackers sit far from the honest crowd, so they get big totals.
    The update with the smallest total becomes the new global model.

    f = the number of attackers we expect among the n updates (settings["krum_f"]).
    Krum needs n > 2f + 2; if there are fewer updates, f is reduced to fit.
    """
    n = len(updates)
    f = min(settings["krum_f"], (n - 3) // 2)
    neighbours = max(n - f - 2, 1)

    vectors = [flatten(u.weights) for u in updates]

    totals = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                distances.append(float(torch.sum((vectors[i] - vectors[j]) ** 2)))
        distances.sort()
        totals.append(sum(distances[:neighbours]))

    best = totals.index(min(totals))
    chosen = updates[best]

    new_weights = {name: tensor.float().clone() for name, tensor in chosen.weights.items()}
    influence = {u.node_id: (1.0 if u is chosen else 0.0) for u in updates}
    return new_weights, influence



AGGREGATORS = {
    "fedavg": fedavg,
    "krum": krum,
}

def get_aggregator(name):
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation method '{name}'. Options: {list(AGGREGATORS)}")
    return AGGREGATORS[name]
