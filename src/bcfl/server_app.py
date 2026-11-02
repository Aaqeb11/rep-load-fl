"""What runs on the coordinator: start from a fresh model and run all rounds."""

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from bcfl.data import get_test_loader
from bcfl.model import SimpleCNN, get_device, test
from bcfl.strategy import RepLoadStrategy


def build_server_app(config, recorder):
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(config["seed"])
        initial_model = SimpleCNN()

        strategy = RepLoadStrategy(config, recorder)
        test_loader = get_test_loader(config)
        device = get_device(config["training"]["device"])

        def evaluate_global_model(server_round, arrays):
            """Test the combined model on the held-out test set after every round."""
            model = SimpleCNN()
            model.load_state_dict(arrays.to_torch_state_dict())
            accuracy, loss = test(model, test_loader, device)
            recorder.record_test(server_round, accuracy, loss)
            return MetricRecord({"accuracy": accuracy, "loss": loss})

        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(torch_state_dict=initial_model.state_dict()),
            num_rounds=config["federation"]["num_rounds"],
            evaluate_fn=evaluate_global_model,
        )

    return app
