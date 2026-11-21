"""What runs on each fog node.

Flower starts one copy of this ClientApp for every simulated fog node.
When the coordinator sends a "train" message, the node:
  1. loads the current global model from the message,
  2. trains it on its own slice of the data (or, if it is malicious, attacks),
  3. sends the trained model back, with how many samples it used
     and how long training took.
"""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from bcfl.attacks import (
    flip_labels,
    free_rider_weights,
    get_attack_settings,
    get_malicious_partitions,
    is_attacking_this_round,
    scale_update,
)
from bcfl.data import get_dataset_info, get_node_loader
from bcfl.model import build_model, get_device, train


def build_client_app(config):
    """Create the ClientApp. `config` is the experiment config dictionary."""
    app = ClientApp()

    attack = get_attack_settings(config)
    malicious_partitions = get_malicious_partitions(config)

    @app.train()
    def handle_train(msg: Message, context: Context) -> Message:
        # Each simulated node runs in its own process; one thread each keeps the
        # laptop responsive when many nodes train at the same time.
        torch.set_num_threads(1)

        # Flower tells each node which slice of the data it holds.
        partition_id = int(context.node_config["partition-id"])
        server_round = int(msg.content["config"]["server-round"])

        # Is this node malicious, and does it attack in this round?
        is_malicious = partition_id in malicious_partitions
        attacking = is_malicious and is_attacking_this_round(attack, server_round)
        if attacking:
            print(f"[node {partition_id}] round {server_round}: attacking ({attack['type']})")

        # 1. Load the global model sent by the coordinator.
        global_weights = msg.content["arrays"].to_torch_state_dict()
        model = build_model(config)
        model.load_state_dict(global_weights)

        training = config["training"]
        loader = get_node_loader(config, partition_id)

        # label_flip poisons the data BEFORE training.
        if attacking and attack["type"] == "label_flip":
            class_names = get_dataset_info(config)["class_names"]
            flip_labels(
                loader,
                source_number=class_names.index(attack["source_class"]),
                target_number=class_names.index(attack["target_class"]),
            )

        # 2. Train (timed). Step 3 needs this to know how much CPU time training
        #    takes away from the node's real-time tasks.
        started_at = time.perf_counter()        # wall-clock time
        cpu_started_at = time.process_time()     # CPU time used by this process only

        if attacking and attack["type"] == "free_rider":
            # Skip training entirely: no CPU spent, nothing learned.
            train_loss = 0.0
        else:
            train_loss = train(
                model,
                loader,
                epochs=training["local_epochs"],
                learning_rate=training["learning_rate"],
                device=get_device(training["device"]),
            )

        train_seconds = time.perf_counter() - started_at
        train_cpu_seconds = time.process_time() - cpu_started_at

        # 3. Decide what to send back.
        model.to("cpu")
        weights_to_send = model.state_dict()

        if attacking and attack["type"] in ("scaling", "on_off"):
            weights_to_send = scale_update(global_weights, weights_to_send, attack["scale"])
        elif attacking and attack["type"] == "free_rider":
            weights_to_send = free_rider_weights(
                global_weights, attack["noise_std"], seed=config["seed"] + server_round,
            )

        reply = RecordDict({
            "arrays": ArrayRecord(torch_state_dict=weights_to_send),
            "metrics": MetricRecord({
                "num-examples": len(loader.dataset),      # attackers report their true count
                "train-loss": train_loss,
                "train-seconds": train_seconds,
                "train-cpu-seconds": train_cpu_seconds,
                "partition-id": partition_id,
                # Ground truth for evaluation only; the coordinator's methods never use it.
                "is-malicious": int(is_malicious),
                "attacking": int(attacking),
            }),
        })
        return Message(content=reply, reply_to=msg)

    return app
