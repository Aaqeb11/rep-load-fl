"""SCORE step: how much should we trust each node's update?

Every scorer has the same shape:

    scorer(updates, state) -> {node_id: score}

- updates: list of NodeUpdate objects from this round (see aggregation.py)
- state:   a dictionary the scorer can use to remember things between rounds
           (for example, each node's reputation history in Step 4)

Scores are numbers where 1.0 means "fully trusted" and 0.0 means "ignore".

To add a new scorer, write a function with this shape and add it to SCORERS.
"""


def no_scoring(updates, state):
    """Trust every node equally. This is the Step 1 baseline."""
    return {update.node_id: 1.0 for update in updates}


SCORERS = {
    "none": no_scoring,
}


def get_scorer(name):
    if name not in SCORERS:
        raise ValueError(f"Unknown scoring method '{name}'. Options: {list(SCORERS)}")
    return SCORERS[name]
