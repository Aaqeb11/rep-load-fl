# rep-load-fl

Code for the dissertation **"Blockchain-Anchored Reputation for Workload-Aware Scheduling and Aggregation of Federated Learning in Hierarchical Fog Networks"** (BITS Pilani Dubai).

Parent fog nodes choose child fog nodes each round using two inputs: on-chain reputation and current real-time task load. They aggregate the children's updates weighted by reputation, and they verify each other's aggregation through a permissioned blockchain (Hyperledger Fabric).

## Design rule

Selection, scoring and aggregation are **separate, swappable functions**, so every experiment changes only one of them:

| Module | Job | Step 1 default |
|---|---|---|
| `src/bcfl/selection.py` | Choose which fog nodes train this round | random |
| `src/bcfl/scoring.py` | Turn each node's history into a reputation score | none (all 1.0) |
| `src/bcfl/aggregation.py` | Combine the updates | FedAvg |

## Setup

```bash
conda create -n replfl python=3.11 -y
conda activate replfl
pip install -r requirements.txt
pip install -e .          # makes `bcfl` importable from scripts and tests
pytest                    # unit tests
```

## Running experiments

```bash
# 1. Quick check on fake data (about a minute, no download)
python scripts/run.py --config configs/smoke.yaml

# 2. One real run: CIFAR-10, 20 nodes, 10 per round, 50 rounds
#    (downloads CIFAR-10 into ./data the first time, about 170 MB)
python scripts/run.py --config configs/baseline.yaml

# 3. Change any setting without editing the file
python scripts/run.py --config configs/baseline.yaml --set data.alpha=0.1 --set seed=1

# 4. Full Step 1: alpha 0.1 / 0.5 / 100, three seeds each, then plot
bash scripts/run_step1.sh

# 5. Plot any set of runs (seeds are averaged, shaded band = min to max)
python scripts/plot.py results/baseline_alpha0.5_seed* --output results/alpha05.png
```

Each run writes to `results/<run_name>_alpha<α>_seed<s>/`:

| File | Contents |
|---|---|
| `config.yaml` | The exact settings used |
| `split.json` | Samples of each class held by every node |
| `rounds.csv` | Per round: test accuracy and loss, average training loss, selected nodes, round time |

On an Apple Silicon Mac you can try `--set training.device=mps`. If it is slower or errors with many parallel nodes, stay on `cpu`.

## How one round works

1. **Select** (`selection.py`): the coordinator picks which nodes train.
2. **Train** (`client_app.py`): each chosen node trains the current model on its own data.
3. **Score** (`scoring.py`): the coordinator rates each update.
4. **Aggregate** (`aggregation.py`): the updates are combined into the new global model.
5. **Record** (`recorder.py`): the new model is tested and a row is written to `rounds.csv`.

`strategy.py` is the Flower strategy that calls steps 1, 3 and 4. `server_app.py` and `client_app.py` are the coordinator and fog-node programs, and `scripts/run.py` starts the simulation.

## Roadmap

- [x] **Step 1: baseline.** Flower + PyTorch, 20–30 fog nodes, CIFAR-10 with a Dirichlet non-IID split, FedAvg, accuracy curves.
- [ ] **Step 2: attacks.** Label flipping, scaled poisoning, free-riding and on-off attacks. Baselines: Krum and trimmed mean.
- [ ] **Step 3: workload model.** Real-time task queues with deadlines, random selection, deadline misses and round time.
- [ ] Reputation scoring and reputation- and load-aware selection (RQ1, RQ2).
- [ ] Fabric anchoring and peer verification of aggregation (RQ3, RQ4).

## Layout

```
src/bcfl/   library code (data, model, selection, scoring, aggregation, strategy)
configs/    experiment configs (YAML)
scripts/    entry points to run experiments and plot results
results/    run outputs (git-ignored, except summary files)
docs/       research context and notes
tests/      unit tests
```

See `docs/PROJECT_CONTEXT.md` for the research questions and the novelty analysis.
