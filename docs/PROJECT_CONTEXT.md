# BCFL dissertation: project context (as of 19 Oct 2026)

## 1. The mentor's direction (email, 12 Oct 2026)

**Title:** Blockchain-Anchored Reputation for Workload-Aware Scheduling and Aggregation of Federated Learning in Hierarchical Fog Networks.

**Idea:**
- Parent fog nodes choose child fog nodes each round using two inputs: on-chain reputation and current real-time task load.
- Parents aggregate the children's updates, weighted by reputation.
- Parents verify each other's aggregation through a permissioned blockchain (Hyperledger Fabric).
- The system must resist poisoning, free-riding and on-off attacks under non-IID data, while real-time tasks keep meeting their deadlines.

**Research questions:**
- **RQ1:** Does selecting by reputation plus load keep deadlines better than selecting by reputation alone? Main metric: deadline-miss rate. Compare against a load-only selector (ELSA-style) and a reputation-only selector (Rjoub- or Zhao-style).
- **RQ2:** Does reputation-weighted aggregation resist poisoning and on-off attacks better than FedAvg, Krum and trimmed mean under non-IID data? Reputation weighting on its own is already published, so the new part is on-off robustness and these baselines.
- **RQ3:** Can parent nodes detect a peer that deviates (for example, one that aggregates incorrectly)?
- **RQ4:** What latency and storage overhead does the chain add, and what does it give that a central database doesn't?

**Out of scope:** a new consensus protocol, encryption as the core contribution, incentive mechanisms.

**Code plan:**
1. **Baseline:** Flower + PyTorch, 20–30 fog nodes, CIFAR-10 with a Dirichlet non-IID split, FedAvg, accuracy curves.
2. **Attacks:** label flipping, scaled poisoning, free-riding, on-off. Baselines: Krum and trimmed mean.
3. **Workload model:** real-time task queues with deadlines; random trainer selection; measure deadline misses and round time.

Keep selection, scoring and aggregation as separate functions. Keep everything in Git.

## 2. Novelty search: does any paper select FL trainers by reputation and real-time load together?

**Answer as of 19 Oct 2026: no.** The literature splits into two families, and nobody joins them.

| Family | Papers | What's missing |
|---|---|---|
| Load-aware selection | ELSA-FL (Lackinger et al., TU Wien, 2026; picks clients with low background inference load, under an energy budget); Lackinger et al., "Inference load-aware orchestration for hierarchical FL", IEEE LCN 2024 (**not yet read**); Han et al., INFOCOM 2024; Luo et al., IEEE TMC 2024; FedFog (arXiv 2507.03952); FedQueue (arXiv 2605.02125) | No trust or reputation, no attacks, no blockchain |
| Trust-aware selection | Rjoub et al., *Inf. Syst. Frontiers* 26:1261, 2024 (trust plus *current free* CPU, RAM, bandwidth and disk; trust comes from resource-use anomalies, not update quality; no deadlines); Zhao et al., *Chinese J. Aeronautics* 37(9), 2024 (reputation plus CPU capability, consortium chain, reputation-weighted aggregation); Liu et al., BHFL, *Future Internet* 17(2):75, 2025 (hierarchical, Fabric 2.3, DQN selection, score-weighted edge aggregation); Tan et al., SCS, *IEEE TBD* 2022; FREB, *IEEE TSC* 2024 (reputation only) | Load means static capability or free capacity; no deadline-bound real-time tasks; weak attack evaluation (noise or label flipping only; no on-off attacks; no Krum or trimmed-mean baselines) |

**Claim to use with the mentor:**
> Load-aware selection (ELSA-FL, LCN'24) avoids busy clients but assumes every client is honest. Trust-aware selection (Rjoub 2024, Zhao 2024, BHFL 2025) uses trust plus static or free capacity, has no notion of deadline-bound real-time tasks, and is weakly tested against poisoning and on-off attacks. We combine reputation built from update behaviour and recorded on-chain with fog nodes' live real-time task queues, so selection both meets deadlines and resists poisoning and on-off attacks, and parent fog nodes verify each other's aggregation.

**Already published, so not novel on its own:**
- reputation-weighted aggregation;
- reputation stored on a permissioned chain;
- hierarchical FL on Fabric.

**Still open:**
- selection by load from deadline-bound real-time tasks;
- trainers that are fog nodes and also serve real-time work;
- peer verification of aggregation;
- on-off and free-riding attack evaluation.

**Known flaws in competitor papers (useful for critique):**
- **Zhao 2024:** Table 3 shows malicious-node reputation *rising* with attack intensity (0.025 → 0.215), which contradicts the text.
- **BHFL:** devices compute their own score (step 5); the CIFAR-10 claims don't match the MNIST-only figure captions.
- **Rjoub:** the reward in Eq. 15 contains neither trust nor energy, despite what the text says.

## 3. Application domain (tentative)

**First choice: Industrial IoT.**
- Fog gateways in plants run control and alarm tasks with millisecond deadlines.
- The FL task is intrusion detection, using Edge-IIoTset or TON_IoT.
- It fits the Trustworthy Edge base paper and a permissioned consortium (plants or operators), and it suits his InfoSec background.

**Fallback: vehicles and smart transport** (roadside units as fog nodes). This area is more crowded.

Steps 1–3 stay on CIFAR-10. The domain dataset gets swapped in later.

## 4. Base and related papers already studied

- **Base papers:** Trustworthy Edge (SD-IIoT, trust-weighted FedAvg); RTH²S (real-time scheduling across fog tiers: laxity, priority tags, iFogSim).
- **Reviewed:** BFLP, TFFL, SecureBFL, BDFL-VM, the SDV trust paper, Yang & Li; C2SecFL was done earlier.
- **Recurring finding:** reputation is computed off-chain by a single party and the chain only records it. None of these papers tested consensus on a real multi-machine network. On-off attackers evade the reputation rules in Trustworthy Edge and the SDV paper.
