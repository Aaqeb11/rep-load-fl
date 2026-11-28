"""WORKLOAD: the real-time jobs each fog node must run, and what FL training costs it.

The simulated fog nodes all run on one laptop, so their real timings say nothing
about a factory gateway. Instead, the coordinator keeps a small "digital twin" of
every node's CPU for every round:

  1. Each node has a speed (MIPS) and, each round, a load level: the share of its
     CPU that real-time jobs need ("normal" or "busy", switching between rounds).
  2. Real-time jobs arrive during the round, each with a size (MI) and a deadline.
  3. If the node was selected, training is one more job, due at the end of the round.
  4. Real-time jobs run earliest-deadline-first (EDF); training shares the CPU
     according to "training_policy". We count the jobs that miss their deadline,
     and whether training finished in time.

Where the numbers come from (config section "workload", see configs/*.yaml):
  RTH2S = Kaur et al., "Real-Time Scheduling on Hierarchical Heterogeneous Fog
          Networks", IEEE Transactions on Services Computing, 2023.
  - Node speed 1,000-2,000 MIPS ........................ RTH2S tier-1 fog nodes
  - Job sizes: small 155-250 MI, medium 1,850-3,000 MI .. RTH2S Table 2
  - Small/medium mix 41:34 ............................. RTH2S job mix (its large jobs,
                                                         25%, run on higher tiers)
  - Deadlines inside 250 ms - 10 s, divided by the
    deadline factor (DF) ............................... RTH2S synthetic workload
  - EDF scheduling, success ratio ...................... RTH2S
  Our assumptions (to confirm with the supervisor; all are config values):
  - Jobs arrive as a Poisson process (random times), at the rate that fills the
    node's load level on average.
  - Load switches between normal (20-50%) and busy (80-95%) from round to round.
  - Training needs about 6 operations per model weight per training row
    (forward + backward pass), i.e. 6 x weights x rows x epochs instructions.
  - One FL round is a 60 s window; training not finished by then is too late.
  - Deadlines are firm: a job still unfinished at its deadline is dropped (a late
    sensor reading is useless).
  - How training shares the CPU ("training_policy"):
      fair_share: training is an ordinary process; whenever real-time jobs are also
                  waiting, the CPU is split half-half (like a normal OS scheduler).
      edf:        training is scheduled as one more EDF job, due at the end of the
                  round, so real-time jobs (earlier deadlines) almost always go first.
"""

from dataclasses import dataclass

import numpy as np

DEFAULTS = {
    "enabled": False,
    "round_window_s": 60.0,
    "mips": [1000, 2000],                 # RTH2S tier-1
    "small_job_mi": [155, 250],           # RTH2S Table 2 (sensing)
    "medium_job_mi": [1850, 3000],        # RTH2S Table 2 (control, object recognition)
    "small_job_share": 0.55,              # 41 / (41 + 34)
    "small_deadline_s": [0.25, 2.0],      # tight; inside RTH2S's 250 ms - 10 s
    "medium_deadline_s": [4.0, 10.0],     # loose; inside RTH2S's 250 ms - 10 s
    "deadline_factor": 1.0,               # RTH2S DF: all deadlines divided by this
    "normal_load": [0.2, 0.5],
    "busy_load": [0.8, 0.95],
    "p_normal_to_busy": 0.2,              # per round
    "p_busy_to_normal": 0.5,              # per round
    "ops_per_weight_per_row": 6,
    "training_policy": "fair_share",      # fair_share | edf (see simulate_round)
}


def get_workload_settings(config):
    """The workload section of the config, with defaults for anything missing."""
    settings = dict(DEFAULTS)
    settings.update(config.get("workload", {}))
    return settings


@dataclass
class Job:
    arrival: float      # seconds after the round starts
    size_mi: float      # millions of instructions
    deadline: float     # absolute time (seconds after the round starts)


# ----------------------------------------------------------------------------
# The scheduler
# ----------------------------------------------------------------------------
def simulate_round(jobs, mips, round_window, training_mi=0.0, training_policy="edf"):
    """Run one round on one node's CPU and count deadline misses.

    Real-time jobs use preemptive EDF: the waiting job with the earliest deadline
    runs; a newly arrived job with an earlier deadline interrupts it.
    Firm deadlines: a job still unfinished at its deadline is dropped and counted
    as missed. Training (if training_mi > 0) is one job, released at time 0 and
    due at the end of the round window; if it is not done by then, it is aborted.

    training_policy decides how training shares the CPU with real-time jobs:
        "edf":        training is one more EDF job (deadline = end of the window).
        "fair_share": while both training and real-time work are waiting, each
                      gets half of the CPU; otherwise whoever is waiting gets all.

    Returns a dictionary:
        jobs            number of real-time jobs
        missed          how many of them missed their deadline
        training_done_at  time training finished (None if no training or too late)
        training_on_time  True if training finished within the round window
    """
    upcoming = sorted(jobs, key=lambda job: job.arrival)
    waiting = []            # each entry: {"deadline", "remaining" (seconds), "training"}
    if training_mi > 0:
        waiting.append({"deadline": round_window, "remaining": training_mi / mips, "training": True})

    time_now = 0.0
    next_index = 0
    missed = 0
    training_done_at = None

    while next_index < len(upcoming) or waiting:
        # 1. Jobs that have arrived by now join the waiting list.
        while next_index < len(upcoming) and upcoming[next_index].arrival <= time_now:
            job = upcoming[next_index]
            waiting.append({"deadline": job.deadline, "remaining": job.size_mi / mips, "training": False})
            next_index += 1

        # 2. Drop jobs whose deadline has passed (firm deadlines).
        still_waiting = []
        for job in waiting:
            if job["deadline"] <= time_now:
                if not job["training"]:
                    missed += 1
            else:
                still_waiting.append(job)
        waiting = still_waiting

        # 3. Nothing to do: jump to the next arrival.
        if not waiting:
            if next_index < len(upcoming):
                time_now = upcoming[next_index].arrival
            continue

        # 4. Decide who gets the CPU, and at what share of its speed.
        next_arrival = upcoming[next_index].arrival if next_index < len(upcoming) else float("inf")
        real_time = [j for j in waiting if not j["training"]]
        training = [j for j in waiting if j["training"]]
        if training_policy == "fair_share" and real_time and training:
            # Two processes share the CPU: the most urgent real-time job and training,
            # each at half speed.
            running = [(min(real_time, key=lambda j: j["deadline"]), 0.5), (training[0], 0.5)]
        else:
            # EDF over everything waiting (training included, with its own deadline).
            running = [(min(waiting, key=lambda j: j["deadline"]), 1.0)]

        # 5. Run until something changes: a running job finishes, a deadline passes,
        #    or a new job arrives (which might be more urgent).
        run_until = next_arrival
        for job, share in running:
            run_until = min(run_until, time_now + job["remaining"] / share, job["deadline"])
        for job, share in running:
            job["remaining"] -= (run_until - time_now) * share
        time_now = run_until

        for job, share in running:
            if job["remaining"] <= 1e-9:
                waiting.remove(job)
                if job["training"]:
                    training_done_at = time_now

    return {
        "jobs": len(jobs),
        "missed": missed,
        "training_done_at": training_done_at,
        "training_on_time": training_done_at is not None,
    }


# ----------------------------------------------------------------------------
# The fog nodes over a whole run
# ----------------------------------------------------------------------------
class FogWorkload:
    """Speeds, load levels and real-time jobs of every fog node, for every round.

    Everything is drawn from the run's seed and does NOT depend on which nodes are
    selected, so every selection method faces exactly the same workload.
    """

    def __init__(self, config, num_weights):
        self.settings = get_workload_settings(config)
        self.num_nodes = config["federation"]["num_nodes"]
        self.num_rounds = config["federation"]["num_rounds"]
        self.seed = config["seed"]
        self.num_weights = num_weights
        self.local_epochs = config["training"]["local_epochs"]
        self.round_window = self.settings["round_window_s"]

        rng = np.random.default_rng(self.seed + 2000)       # own random sequence
        low, high = self.settings["mips"]
        self.mips = rng.uniform(low, high, size=self.num_nodes)
        self.busy, self.load = self._draw_load_levels(rng)

    def _draw_load_levels(self, rng):
        """busy[round][node] (True/False) and load[round][node] (share of CPU), rounds 1..N.

        Each node switches between normal and busy like a two-state Markov chain;
        round 1 starts from the long-run share of busy rounds.
        """
        s = self.settings
        busy_share = s["p_normal_to_busy"] / (s["p_normal_to_busy"] + s["p_busy_to_normal"])
        busy = np.zeros((self.num_rounds + 1, self.num_nodes), dtype=bool)
        load = np.zeros((self.num_rounds + 1, self.num_nodes))
        busy[1] = rng.random(self.num_nodes) < busy_share
        for round_number in range(2, self.num_rounds + 1):
            for node in range(self.num_nodes):
                was_busy = busy[round_number - 1, node]
                switch = s["p_busy_to_normal"] if was_busy else s["p_normal_to_busy"]
                busy[round_number, node] = (not was_busy) if rng.random() < switch else was_busy
        for round_number in range(1, self.num_rounds + 1):
            for node in range(self.num_nodes):
                low, high = s["busy_load"] if busy[round_number, node] else s["normal_load"]
                load[round_number, node] = rng.uniform(low, high)
        return busy, load

    def jobs_for(self, server_round, partition_id):
        """The real-time jobs arriving at one node during one round."""
        s = self.settings
        # Its own generator per (round, node): the same jobs whatever is selected.
        rng = np.random.default_rng([self.seed, 3000, server_round, partition_id])

        small_mean = sum(s["small_job_mi"]) / 2
        medium_mean = sum(s["medium_job_mi"]) / 2
        mean_job_mi = s["small_job_share"] * small_mean + (1 - s["small_job_share"]) * medium_mean
        # Jobs per second so that, on average, they fill `load` of the CPU.
        rate = self.load[server_round, partition_id] * self.mips[partition_id] / mean_job_mi

        jobs = []
        time_now = rng.exponential(1 / rate)
        while time_now < self.round_window:
            if rng.random() < s["small_job_share"]:
                size = rng.uniform(*s["small_job_mi"])
                relative_deadline = rng.uniform(*s["small_deadline_s"])
            else:
                size = rng.uniform(*s["medium_job_mi"])
                relative_deadline = rng.uniform(*s["medium_deadline_s"])
            relative_deadline /= s["deadline_factor"]
            jobs.append(Job(arrival=time_now, size_mi=size, deadline=time_now + relative_deadline))
            time_now += rng.exponential(1 / rate)
        return jobs

    def training_mi(self, num_examples):
        """Instructions (in millions) one round of local training needs."""
        ops = self.settings["ops_per_weight_per_row"] * self.num_weights * num_examples * self.local_epochs
        return ops / 1e6

    def expected_slack(self, server_round, partition_id, num_examples):
        """What a node can report before the round: its expected idle CPU time in the
        window minus the time training would take. Negative = training probably won't fit."""
        idle_seconds = self.round_window * (1 - self.load[server_round, partition_id])
        training_seconds = self.training_mi(num_examples) / self.mips[partition_id]
        return idle_seconds - training_seconds

    def run_round(self, server_round, partition_id, training_mi):
        """Simulate one node's round, with and without the training job.

        The difference between the two is the deadline misses caused by FL.
        """
        jobs = self.jobs_for(server_round, partition_id)
        mips = self.mips[partition_id]
        policy = self.settings["training_policy"]
        without = simulate_round(jobs, mips, self.round_window)
        if training_mi <= 0:
            with_training = without
        else:
            with_training = simulate_round(jobs, mips, self.round_window, training_mi, policy)
        return {
            "jobs": with_training["jobs"],
            "missed": with_training["missed"],
            "missed_without_training": without["missed"],
            "training_on_time": with_training["training_on_time"] if training_mi > 0 else None,
            "training_seconds": training_mi / mips,
        }
