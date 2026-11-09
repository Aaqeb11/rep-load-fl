# Step 2: attack and defence definitions (draft for agreement)

This note fixes how every attack is defined and measured **before** any Step 2
experiment is run. Items marked **[decide]** need the supervisor's agreement.

## 1. Common setup

| Setting | Value |
|---|---|
| Fog nodes | 20, of which 10 are selected at random each round (Step 2 keeps random selection) |
| Data | CIFAR-10, Dirichlet split; **α = 0.5** main setting, **α = 0.1** stress case |
| Rounds | 100 (150 for α = 0.1 if accuracy has not levelled off) |
| Seeds | 3 per configuration |
| Malicious fraction *m* | **20 %** by default (4 of 20 nodes); sweep 10 % and 30 % (2 and 6 nodes) |
| Which nodes are malicious | Fixed for the whole run, drawn with the run's seed, independent of the data split **[decide: see 3.1]** |
| One attack per run | All malicious nodes perform the same attack |

**Threat model.**
- Attackers know the current global model, their own data and the training procedure.
- They do **not** see other nodes' updates.
- They do **not** know which defence is used, and they do not collude beyond running the same attack.
- They report their true sample count. Inflating the count, which FedAvg trusts blindly, is left for later.

Because selection is random, the number of attackers in a given round varies. With *m* = 20 %, a round contains on average 2 attackers out of 10, and sometimes 0 or 4. This number is logged every round.

**Ground truth for later steps.** Every row of `node_rounds.csv` gets two extra columns:
- `is_malicious`: is this one of the attacker nodes?
- `attacking`: did it actually attack in this round? (For an on-off attacker in its honest phase, this is false.)

These are only for evaluation. The coordinator never sees them. They let Step 4 measure detection precision and recall.

## 2. Defences compared in Step 2

| Aggregator | Rule | Parameter |
|---|---|---|
| FedAvg | Sample-weighted average (no defence) | none |
| Krum (Blanchard et al., 2017) | Keep the single update with the smallest summed distance to its *n − f − 2* nearest neighbours | *f* = expected attackers among the selected = round(*m* × 10) → 2 at 20 % (needs *n* > 2*f* + 2: 10 > 6 ✓) |
| Trimmed mean (Yin et al., 2018) | For every parameter, drop the *k* largest and *k* smallest values, average the rest | *k* = round(*m* × 10) → 2 per side at 20 % |
| Coordinate-wise median (optional) | For every parameter, take the median | none |

Krum and trimmed mean are given the **true** expected attacker fraction. This is favourable to the baselines, so any advantage of our method is not caused by mis-tuned baselines.

## 3. Attacks

### 3.1 Label flipping (data poisoning)

Attackers train normally, but on wrong labels. Two variants:

| Variant | Definition |
|---|---|
| **Targeted** | Every training image of **source class 3 (cat)** is relabelled as **target class 5 (dog)**. Other classes are untouched. **[decide: class pair]** |
| **Untargeted** | Every label *y* is replaced by *9 − y* (airplane↔truck, car↔ship, …) |

**Issue under non-IID data:** with a Dirichlet split, a randomly chosen attacker may hold very few cat images, so the targeted attack would be weak by accident. Two options **[decide]**:
- **(a) Random attackers:** realistic. Report how many source-class images the attackers actually hold.
- **(b) Worst case:** the attackers are the 4 nodes holding the most source-class images.

**Success measures:**
- Targeted attack success rate: **ASR = (test images of class 3 predicted as class 5) / (test images of class 3)**. Also report clean accuracy on all classes, and on the source class.
- Untargeted: **accuracy drop = accuracy(no-attack run) − accuracy(attacked run)**, same α and seed, averaged over the last 10 rounds.

### 3.2 Scaled poisoning (model poisoning)

The attacker trains honestly, then sends a **scaled and reversed** update:

> sent model = global model + *s* × (local model − global model)

| Parameter | Value |
|---|---|
| Scaling factor *s* | **−5** by default (reverse the learning direction and amplify it); also run *s* = −1 (pure sign flip) once **[decide]** |
| Reported sample count | true count |

**Success measure:** accuracy drop against the no-attack run, averaged over the last 10 rounds. Also report the **lowest** accuracy reached and whether training **diverges** (accuracy stuck at about 10 %).

### 3.3 Free-riding

The free-rider **does not train**. It returns the global model it received, sending a zero update, while reporting its true sample count. That way it receives the improved model each round without spending compute.

| Variant | Definition |
|---|---|
| **Plain** | Returns the global model unchanged |
| **Disguised** | Returns the global model plus small Gaussian noise, σ = 10⁻³ per parameter, so the update is not exactly zero **[decide: include or not]** |

Free-riders report their measured training time, which is close to 0. This matters in Step 3: a free-rider spends none of its capacity on training.

**Success measures:**
- **Weight captured:** the share of the aggregation weight given to free-riders, averaged over rounds. Under FedAvg this is their share of the samples.
- **Undetected participation rate:** the fraction of a free-rider's participations in which it received non-zero weight. For Krum, that means it was the chosen update; for trimmed mean, that its values were not all trimmed.
- **Accuracy impact:** accuracy against the no-attack run. This is expected to be small, mainly slower convergence.

### 3.4 On-off attack

The attacker alternates between honest behaviour, which builds reputation, and attacking.

| Parameter | Value |
|---|---|
| Cycle | **10 rounds honest, then 5 rounds attacking**, repeated (15-round cycle; about 6–7 cycles in 100 rounds) **[decide]** |
| Phase | Starts honest. All on-off attackers are **synchronised** (worst case) **[decide: or staggered]** |
| Attack used in the "on" phase | Scaled poisoning, *s* = −5 (section 3.2) |
| Cycle clock | Global round number. An attacker that isn't selected during its "on" window simply misses that attack (with 10 of 20 selected, it is chosen about half the time) |

**Success measures, per attack window:**
- **Window drop:** accuracy at the last round before the window minus the lowest accuracy inside the window.
- **Recovery time:** rounds after the window ends until accuracy is back within 1 percentage point of its pre-window level.

Report the mean over all windows and seeds. In Step 4 we add: rounds until the attacker's reputation falls below the selection threshold, and how quickly it regains reputation during the honest phase.

## 4. Experiment grid and compute

At about 14 minutes per 100-round run on the laptop:

| Block | Configurations | Runs (× 3 seeds) | Time |
|---|---|---|---|
| Main: α = 0.5, *m* = 20 % | 6 attack settings (none, LF-targeted, LF-untargeted, scaling, free-ride, on-off) × 3 aggregators | 54 | ≈ 12.5 h |
| Fraction sweep: *m* = 10 %, 30 % | scaling + on-off × 3 aggregators × 2 fractions | 36 | ≈ 8.5 h |
| Stress: α = 0.1, *m* = 20 % | scaling + on-off × 3 aggregators | 18 | ≈ 4–6 h |

That's about 25 hours in total, run overnight in blocks. The main block alone answers whether Krum and trimmed mean help against each attack under non-IID data.

## 5. Decisions needed

1. Targeted label-flip pair: cat (3) → dog (5)?
2. Attacker choice for targeted flipping: random (a) or worst case (b)?
3. Scaling factor: *s* = −5 as default, plus one run at *s* = −1?
4. Free-riding: plain only, or plain + disguised (σ = 10⁻³)?
5. On-off cycle: 10 honest / 5 attacking, synchronised?
6. Is it acceptable to give Krum and trimmed mean the true attacker fraction?
