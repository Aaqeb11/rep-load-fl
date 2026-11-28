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
from flwr.app import ArrayRecord, RecordDict

# Krum and trimmed mean: the maths comes from Flower's own, tested implementations
# (flwr 1.42, pinned in requirements.txt). Flower ships them as whole strategies;
# we call only the functions inside them, so our pipeline (selection -> scoring ->
# aggregation, influence logging) stays the same for every method.
from flwr.serverapp.strategy.fedtrimmedavg import trim_mean as flower_trim_mean
from flwr.serverapp.strategy.multikrum import select_multikrum as flower_select_multikrum


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
    # Real-time workload (only when the workload model is on):
    on_time: bool = True             # did training finish within the round window?
    slack: float = None              # the node's slack (seconds) when it was selected


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


def krum(updates, scores, settings):
    """Krum (Blanchard et al., 2017): keep the ONE update that looks most "central".

    For each update, add up its squared distances to its (n - f - 2) closest
    neighbours. Attackers sit far from the honest crowd, so they get big totals.
    The update with the smallest total becomes the new global model.

    f = the number of attackers we expect among the n updates (settings["krum_f"]).
    Krum needs n > 2f + 2; if there are fewer updates, f is reduced to fit.

    Flower does the scoring and picking (select_multikrum with 1 node to select
    is plain Krum). We only translate our updates into Flower's format and back.
    """
    n = len(updates)
    f = min(settings["krum_f"], (n - 3) // 2)
    f = max(f, 0)

    # Flower works on RecordDicts holding an ArrayRecord, like a node's reply.
    contents = []
    for update in updates:
        contents.append(RecordDict({"arrays": ArrayRecord(torch_state_dict=update.weights)}))

    selected = flower_select_multikrum(contents, num_malicious_nodes=f, num_nodes_to_select=1)

    # Find which of our updates Flower picked (same object it handed back).
    chosen = None
    for update, content in zip(updates, contents):
        if content is selected[0]:
            chosen = update

    new_weights = {name: tensor.float().clone() for name, tensor in chosen.weights.items()}
    influence = {u.node_id: (1.0 if u is chosen else 0.0) for u in updates}
    return new_weights, influence


def trimmed_mean(updates, scores, settings):
    """Trimmed mean (Yin et al., 2018): for every single weight, drop the k highest
    and k lowest values across the nodes, and average the rest.

    An attacker's extreme values get trimmed away, one weight at a time.
    k = settings["trim_k"] values removed from EACH end (needs n > 2k).

    Flower computes the trimmed average itself (trim_mean, as in FedTrimmedAvg).
    Flower does not report WHICH values it kept, so influence is counted here.
    """
    n = len(updates)
    k = min(settings["trim_k"], (n - 1) // 2)

    # Flower takes a fraction of the nodes to cut from each end, not a count.
    # It cuts int(fraction * n) values. Adding 0.25 keeps a rounding error from
    # turning 2.0 into 1.999... (which int() would make 1). Still below 0.5.
    cut_fraction = (k + 0.25) / n if k > 0 else 0.0

    new_weights = {}
    for layer_name in updates[0].weights:
        # Stack the nodes' versions of this layer: shape [n, ...layer shape...]
        stacked = torch.stack([u.weights[layer_name].float() for u in updates])
        averaged = flower_trim_mean(stacked.numpy(), cut_fraction)
        new_weights[layer_name] = torch.from_numpy(averaged).float()

    influence = trimmed_mean_influence(updates, k)
    return new_weights, influence


def trimmed_mean_influence(updates, k):
    """Each node's share of a trimmed-mean model.

    For every weight, sort the n values, keep the middle n - 2k, and count whose
    values were kept. Each kept value contributes 1 / (n - 2k) of that weight,
    so a node's share is (its kept values) / ((n - 2k) * number of weights).
    """
    n = len(updates)
    kept = n - 2 * k
    kept_counts = torch.zeros(n)        # how often each node's value was kept
    total_weights = 0

    for layer_name in updates[0].weights:
        stacked = torch.stack([u.weights[layer_name].float() for u in updates])
        _, order = torch.sort(stacked, dim=0)       # order[i] = which node is i-th smallest
        middle_nodes = order[k:n - k].flatten()
        kept_counts += torch.bincount(middle_nodes, minlength=n).float()
        total_weights += stacked[0].numel()

    shares = kept_counts / (kept * total_weights)
    return {updates[i].node_id: float(shares[i]) for i in range(n)}


AGGREGATORS = {
    "fedavg": fedavg,
    "krum": krum,
    "trimmed_mean": trimmed_mean,
}


def get_aggregator(name):
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation method '{name}'. Options: {list(AGGREGATORS)}")
    return AGGREGATORS[name]
