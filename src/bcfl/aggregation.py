"""AGGREGATE step: combine the nodes' updates into one new model.

Every aggregator has the same shape:

    aggregator(updates, scores) -> new model weights (dict of name -> tensor)

- updates: list of NodeUpdate objects (one per node that trained)
- scores:  {node_id: score} from the scoring step

To add a new aggregator (Krum, trimmed mean, ... in Step 2), write a function
with this shape and add it to AGGREGATORS.
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


def fedavg(updates, scores):
    """Federated averaging (McMahan et al., 2017).

    The new model is a weighted average of the nodes' models, where each node's
    weight is (its number of samples) / (total samples across chosen nodes).
    Nodes with more data count for more.

    Scores are ignored here: plain FedAvg trusts everyone.
    """
    total_examples = sum(update.num_examples for update in updates)

    new_weights = {}
    for layer_name in updates[0].weights:
        weighted_sum = torch.zeros_like(updates[0].weights[layer_name], dtype=torch.float32)
        for update in updates:
            share = update.num_examples / total_examples
            weighted_sum += share * update.weights[layer_name].float()
        new_weights[layer_name] = weighted_sum

    return new_weights


AGGREGATORS = {
    "fedavg": fedavg,
}


def get_aggregator(name):
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation method '{name}'. Options: {list(AGGREGATORS)}")
    return AGGREGATORS[name]
