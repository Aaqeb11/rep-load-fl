"""The coordinator's rules: our custom Flower strategy.

Flower's FedAvg strategy already handles sending messages and running rounds.
We only replace the two places where our research happens:

- configure_train()  -> SELECT: which nodes train this round
- aggregate_train()  -> SCORE + AGGREGATE: how much to trust each update,
                        and how to combine them

The actual logic lives in selection.py, scoring.py and aggregation.py, so each
can be swapped through the config file without touching this file.
"""

import time

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from bcfl.aggregation import NodeUpdate, get_aggregator
from bcfl.scoring import get_scorer
from bcfl.selection import get_selector


class RepLoadStrategy(FedAvg):
    def __init__(self, config, recorder):
        federation = config["federation"]
        super().__init__(
            fraction_evaluate=0.0,          # testing is done centrally by the coordinator
            min_train_nodes=federation["nodes_per_round"],
            min_available_nodes=federation["num_nodes"],
        )
        self.num_nodes = federation["num_nodes"]
        self.nodes_per_round = federation["nodes_per_round"]
        self.recorder = recorder
        self.rng = np.random.default_rng(config["seed"])

        methods = config["methods"]
        self.select = get_selector(methods["selection"])
        self.score = get_scorer(methods["scoring"])
        self.aggregate = get_aggregator(methods["aggregation"])

        # Memory the scorer keeps between rounds (reputation history in Step 4).
        self.scoring_state = {}

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid):
        self.recorder.start_round()

        # Wait until every simulated node has connected.
        while len(list(grid.get_node_ids())) < self.num_nodes:
            time.sleep(0.5)
        node_ids = list(grid.get_node_ids())

        # Information the selector may use. Empty in Step 1; later it will hold
        # reputation scores (Step 4) and each node's slack (Step 3).
        context = {"scoring_state": self.scoring_state}

        chosen = self.select(node_ids, self.nodes_per_round, self.rng, context)

        # One "train" message per chosen node, carrying the current global model.
        config["server-round"] = server_round
        messages = []
        for node_id in chosen:
            content = RecordDict({"arrays": arrays, "config": config})
            messages.append(Message(content=content, message_type=MessageType.TRAIN, dst_node_id=node_id))
        return messages

    # ------------------------------------------------------------------
    # SCORE + AGGREGATE
    # ------------------------------------------------------------------
    def aggregate_train(self, server_round: int, replies):
        valid_replies, _failed = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        # Turn each reply into a simple NodeUpdate object.
        updates = []
        for reply in valid_replies:
            metrics = reply.content["metrics"]
            updates.append(NodeUpdate(
                node_id=reply.metadata.src_node_id,
                partition_id=int(metrics["partition-id"]),
                weights=reply.content["arrays"].to_torch_state_dict(),
                num_examples=int(metrics["num-examples"]),
                train_loss=float(metrics["train-loss"]),
                train_seconds=float(metrics["train-seconds"]),
                train_cpu_seconds=float(metrics["train-cpu-seconds"]),
            ))

        scores = self.score(updates, self.scoring_state)
        new_weights = self.aggregate(updates, scores)

        self.recorder.record_training(server_round, updates)
        average_train_loss = sum(u.train_loss for u in updates) / len(updates)

        return ArrayRecord(torch_state_dict=new_weights), MetricRecord({"train-loss": average_train_loss})
