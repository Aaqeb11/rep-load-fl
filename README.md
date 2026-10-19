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

## Roadmap

- [ ] **Step 1: baseline.** Flower + PyTorch, 20–30 fog nodes, CIFAR-10 with a Dirichlet non-IID split, FedAvg, accuracy curves.
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
