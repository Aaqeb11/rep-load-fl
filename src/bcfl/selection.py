"""SELECT step: which fog nodes train this round?

Every selector has the same shape:

    selector(node_ids, num_to_pick, rng, context) -> list of chosen node ids

- node_ids:     all fog nodes that are online
- num_to_pick:  how many should train this round
- rng:          a numpy random generator (so runs are repeatable)
- context:      a dictionary with extra information. Empty for now; later it
                will hold reputation scores (Step 4) and slack values (Step 3).

To add a new selector, write a function with this shape and add it to SELECTORS.
"""


def random_selection(node_ids, num_to_pick, rng, context):
    """Pick nodes at random, ignoring trust and load. This is the Step 1 baseline."""
    node_ids = sorted(node_ids)
    # Pick positions, not ids: Flower's node ids are huge numbers that numpy
    # would round off if we passed them in directly.
    positions = rng.choice(len(node_ids), size=num_to_pick, replace=False)
    return [node_ids[position] for position in positions]


SELECTORS = {
    "random": random_selection,
}


def get_selector(name):
    if name not in SELECTORS:
        raise ValueError(f"Unknown selection method '{name}'. Options: {list(SELECTORS)}")
    return SELECTORS[name]
