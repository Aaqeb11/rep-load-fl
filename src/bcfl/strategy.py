"""The coordinator's rules: our custom Flower strategy.

Flower's FedAvg strategy already handles sending messages and running rounds.
We only replace the two places where our research happens:

- configure_train()  -> SELECT: which nodes train this round
- aggregate_train()  -> SCORE + AGGREGATE: how much to trust each update,
                        and how to combine them

The actual logic lives in selection.py, scoring.py and aggregation.py, so each
can be swapped through the config file without touching this file.

When the workload model is on (config "workload: enabled: true"), each round also:
- tells the selector every node's slack (spare CPU time after training),
- simulates every node's real-time jobs (workload.py) and counts deadline misses,
- drops updates that did not finish within the round window.
"""

import time

import numpy as np
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from bcfl.aggregation import NodeUpdate, get_aggregator
from bcfl.attacks import get_attack_settings
from bcfl.model import build_model
from bcfl.scoring import get_scorer
from bcfl.selection import get_selector
from bcfl.workload import FogWorkload, get_workload_settings


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

        # Robust defences need to know how many attackers to expect per round.
        # Mentor's choice: give them the TRUE attacker fraction (fair to the baselines),
        # and use the same value in their no-attack runs so drops are comparable.
        expected_attackers = round(methods.get("assumed_malicious_fraction", 0.2) * self.nodes_per_round)
        self.aggregation_settings = {"krum_f": expected_attackers, "trim_k": expected_attackers}

        # Memory the scorer keeps between rounds (reputation history).
        self.scoring_state = {}

        # Real-time workload (Week 2). Off unless the config turns it on.
        self.workload = None
        if get_workload_settings(config)["enabled"]:
            num_weights = sum(p.numel() for p in build_model(config).parameters())
            self.workload = FogWorkload(config, num_weights)
        self.attack = get_attack_settings(config)

        # Filled in by discover_nodes() before the first round.
        self.partition_of = {}       # Flower node id -> partition id
        self.num_examples_of = {}    # Flower node id -> training rows it holds

    # ------------------------------------------------------------------
    # Ask every node once who it is
    # ------------------------------------------------------------------
    def discover_nodes(self, grid, node_ids):
        """Send one "query" message to every node: which data slice do you hold, and
        how many rows? (A real fog node would report this when it joins.)"""
        messages = []
        for node_id in node_ids:
            content = RecordDict({"config": ConfigRecord({})})
            messages.append(Message(content=content, message_type=MessageType.QUERY, dst_node_id=node_id))
        for reply in grid.send_and_receive(messages):
            if reply.has_error():
                raise RuntimeError(f"Node did not answer the query: {reply.error}")
            metrics = reply.content["metrics"]
            node_id = reply.metadata.src_node_id
            self.partition_of[node_id] = int(metrics["partition-id"])
            self.num_examples_of[node_id] = int(metrics["num-examples"])

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid):
        self.recorder.start_round()

        # Wait until every simulated node has connected.
        while len(list(grid.get_node_ids())) < self.num_nodes:
            time.sleep(0.5)
        node_ids = list(grid.get_node_ids())

        if self.workload is not None and not self.partition_of:
            self.discover_nodes(grid, node_ids)

        # Information the selector may use: reputation state, and (with the
        # workload on) each node's slack = expected idle CPU seconds this round
        # minus the seconds training would take. Negative = training won't fit.
        context = {"scoring_state": self.scoring_state, "slack": {}}
        if self.workload is not None:
            for node_id in node_ids:
                context["slack"][node_id] = self.workload.expected_slack(
                    server_round, self.partition_of[node_id], self.num_examples_of[node_id])
        self.current_slack = context["slack"]

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
                is_malicious=bool(metrics["is-malicious"]),
                attacking=bool(metrics["attacking"]),
            ))

        # Real-time workload: simulate this round on every node, then keep only
        # the updates that finished within the round window.
        on_time_updates = updates
        if self.workload is not None:
            on_time_updates = self.simulate_workload(server_round, updates)

        scores = self.score(updates, self.scoring_state)
        if on_time_updates:
            new_weights, influence = self.aggregate(on_time_updates, scores, self.aggregation_settings)
        else:
            new_weights, influence = None, {}     # nothing arrived in time: keep the old model

        self.recorder.record_training(server_round, updates, influence)
        average_train_loss = sum(u.train_loss for u in updates) / len(updates)

        if new_weights is None:
            return None, MetricRecord({"train-loss": average_train_loss})
        return ArrayRecord(torch_state_dict=new_weights), MetricRecord({"train-loss": average_train_loss})

    def simulate_workload(self, server_round, updates):
        """Run the real-time workload of every node for this round and record it.

        Returns the updates whose training finished within the round window.
        The simulated CPU follows what each node really did: a free-rider that skipped
        training spends no CPU on it. (This is the simulated world, not a method:
        selection, scoring and aggregation never see it.)
        """
        trained = {u.partition_id: u for u in updates}
        rows = []
        on_time_updates = []
        for node_id, partition_id in sorted(self.partition_of.items(), key=lambda item: item[1]):
            update = trained.get(partition_id)
            training_mi = 0.0
            if update is not None:
                skipped_training = update.attacking and self.attack["type"] == "free_rider"
                if not skipped_training:
                    training_mi = self.workload.training_mi(self.num_examples_of[node_id])
            result = self.workload.run_round(server_round, partition_id, training_mi)

            if update is not None:
                update.on_time = result["training_on_time"] is not False
                update.slack = self.current_slack.get(node_id)
                if update.on_time:
                    on_time_updates.append(update)

            rows.append({
                "round": server_round,
                "partition_id": partition_id,
                "selected": int(update is not None),
                "busy": int(self.workload.busy[server_round, partition_id]),
                "load": round(float(self.workload.load[server_round, partition_id]), 3),
                "mips": round(float(self.workload.mips[partition_id]), 1),
                "slack_s": round(float(self.current_slack.get(node_id, 0.0)), 2),
                "training_s": round(float(result["training_seconds"]), 2),
                "training_on_time": "" if result["training_on_time"] is None else int(result["training_on_time"]),
                "rt_jobs": result["jobs"],
                "rt_missed": result["missed"],
                "rt_missed_without_training": result["missed_without_training"],
            })
        self.recorder.record_workload(server_round, rows)
        return on_time_updates
