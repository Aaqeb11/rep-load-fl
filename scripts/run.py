"""Run one experiment.

Examples:
    python scripts/run.py --config configs/smoke.yaml
    python scripts/run.py --config configs/baseline.yaml
    python scripts/run.py --config configs/baseline.yaml --set data.alpha=0.1 --set seed=1

Results go to results/<run name>/:
    config.yaml   the exact settings used
    split.json    how many samples of each class every node got
    rounds.csv    accuracy and loss after every round
"""

import argparse
import json
import os

import yaml
from flwr.simulation import run_simulation

from bcfl.client_app import build_client_app
from bcfl.config import apply_overrides, load_config
from bcfl.data import describe_split, load_dataset
from bcfl.recorder import RoundRecorder
from bcfl.server_app import build_server_app


def main():
    parser = argparse.ArgumentParser(description="Run one federated learning experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set data.alpha=0.1")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)

    # Make the data path absolute, so every simulated node finds the same folder.
    config["data"]["data_dir"] = os.path.abspath(config["data"]["data_dir"])

    run_name = f"{config['run_name']}_alpha{config['data']['alpha']}_seed{config['seed']}"
    output_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")

    # Download the dataset once, before the simulated nodes start.
    load_dataset(config["data"]["dataset"], config["data"]["data_dir"], download=True)

    # Save the exact settings and the data split for this run.
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(os.path.join(output_dir, "split.json"), "w") as f:
        json.dump(describe_split(config), f)

    recorder = RoundRecorder(output_dir)

    run_simulation(
        server_app=build_server_app(config, recorder),
        client_app=build_client_app(config),
        num_supernodes=config["federation"]["num_nodes"],
        backend_config={
            "client_resources": {
                "num_cpus": config["simulation"]["cpus_per_node"],
                "num_gpus": 0.0,
            }
        },
    )
    print(f"Done. Per-round results: {recorder.csv_path}")


if __name__ == "__main__":
    main()
