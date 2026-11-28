"""SELECT step: which fog nodes train this round?

Every selector has the same shape:

    selector(node_ids, num_to_pick, rng, context) -> list of chosen node ids

- node_ids:     all fog nodes that are online
- num_to_pick:  how many should train this round
- rng:          a numpy random generator (so runs are repeatable)
- context:      a dictionary with extra information:
                  "slack": {node_id: seconds} - expected idle CPU time this round
                           minus the time training would take (workload model on)
                  "scoring_state": the scorer's memory (e.g. reputations)

To add a new selector, write a function with this shape and add it to SELECTORS.
"""


def random_selection(node_ids, num_to_pick, rng, context):
    """Pick nodes at random, ignoring trust and load. This is the Step 1 baseline."""
    node_ids = sorted(node_ids)
    # Pick positions, not ids: Flower's node ids are huge numbers that numpy
    # would round off if we passed them in directly.
    positions = rng.choice(len(node_ids), size=num_to_pick, replace=False)
    return [node_ids[position] for position in positions]


def slack_selection(node_ids, num_to_pick, rng, context):
    """Load-aware baseline: only pick nodes that can afford training right now.

    A node "fits" if its slack is at least 0 (training is expected to fit in its
    idle CPU time this round). Pick at random among the nodes that fit, so the
    same few idle nodes are not chosen every round. If fewer than num_to_pick
    fit, take all of them and fill up with the nodes that have the most slack.
    Ignores trust: this is the "load only" baseline (like ELSA-FL).
    """
    slack = context["slack"]
    fits = sorted(node for node in node_ids if slack[node] >= 0)
    if len(fits) >= num_to_pick:
        positions = rng.choice(len(fits), size=num_to_pick, replace=False)
        return [fits[position] for position in positions]

    does_not_fit = [node for node in node_ids if slack[node] < 0]
    does_not_fit.sort(key=lambda node: slack[node], reverse=True)     # most slack first
    return fits + does_not_fit[:num_to_pick - len(fits)]


SELECTORS = {
    "random": random_selection,
    "slack": slack_selection,
}


def get_selector(name):
    if name not in SELECTORS:
        raise ValueError(f"Unknown selection method '{name}'. Options: {list(SELECTORS)}")
    return SELECTORS[name]
