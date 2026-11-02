"""What runs on each fog node.

Flower starts one copy of this ClientApp for every simulated fog node.
When the coordinator sends a "train" message, the node:
  1. loads the current global model from the message,
  2. trains it on its own slice of the data,
  3. sends the trained model back, with how many samples it used.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from bcfl.data import get_node_loader
from bcfl.model import SimpleCNN, get_device, train


def build_client_app(config):
    """Create the ClientApp. `config` is the experiment config dictionary."""
    app = ClientApp()

    @app.train()
    def handle_train(msg: Message, context: Context) -> Message:
        # Each simulated node runs in its own process; one thread each keeps the
        # laptop responsive when many nodes train at the same time.
        torch.set_num_threads(1)

        # Flower tells each node which slice of the data it holds.
        partition_id = int(context.node_config["partition-id"])

        # 1. Load the global model sent by the coordinator.
        model = SimpleCNN()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

        # 2. Train it on this node's own data.
        training = config["training"]
        loader = get_node_loader(config, partition_id)
        train_loss = train(
            model,
            loader,
            epochs=training["local_epochs"],
            learning_rate=training["learning_rate"],
            device=get_device(training["device"]),
        )

        # 3. Send the trained model back.
        model.to("cpu")
        reply = RecordDict({
            "arrays": ArrayRecord(torch_state_dict=model.state_dict()),
            "metrics": MetricRecord({
                "num-examples": len(loader.dataset),
                "train-loss": train_loss,
                "partition-id": partition_id,
            }),
        })
        return Message(content=reply, reply_to=msg)

    return app
