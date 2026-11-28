"""Tests for the real-time workload model. Each EDF case can be checked by hand.

All cases use a 1,000 MIPS node, so a job of X MI takes X / 1000 seconds.
"""

import numpy as np

from bcfl.workload import FogWorkload, Job, simulate_round


def test_a_job_that_fits_finishes_on_time():
    result = simulate_round([Job(arrival=0.0, size_mi=500, deadline=1.0)], mips=1000, round_window=10)
    assert result["jobs"] == 1 and result["missed"] == 0


def test_a_job_too_big_for_its_deadline_misses():
    # Needs 1 s, has 0.5 s.
    result = simulate_round([Job(arrival=0.0, size_mi=1000, deadline=0.5)], mips=1000, round_window=10)
    assert result["missed"] == 1


def test_an_urgent_job_interrupts_a_less_urgent_one():
    # A (0.8 s of work, due at 2.0) starts at 0. B arrives at 0.1 (0.5 s, due at 0.7).
    # EDF: B runs 0.1-0.6 (on time), A finishes at 1.3 (on time).
    jobs = [Job(0.0, 800, 2.0), Job(0.1, 500, 0.7)]
    assert simulate_round(jobs, mips=1000, round_window=10)["missed"] == 0


def test_without_preemption_order_the_urgent_job_would_miss():
    # Same as above, but B is due at 0.55: even running at once (0.1-0.6) it is late.
    jobs = [Job(0.0, 800, 2.0), Job(0.1, 500, 0.55)]
    assert simulate_round(jobs, mips=1000, round_window=10)["missed"] == 1


def test_training_can_push_a_job_past_its_deadline():
    # Round window 10 s, training needs 9.5 s (due at 10). A job arrives at 5 needing
    # 1 s, due at 10.2. Training is more urgent (10 < 10.2), so it runs until 9.5;
    # the job then runs 9.5-10.5 and misses 10.2. Without training it runs 5-6.
    jobs = [Job(5.0, 1000, 10.2)]
    with_training = simulate_round(jobs, mips=1000, round_window=10, training_mi=9500,
                                   training_policy="edf")
    without = simulate_round(jobs, mips=1000, round_window=10)
    assert with_training["missed"] == 1 and with_training["training_on_time"]
    assert without["missed"] == 0


def test_fair_share_training_can_make_an_urgent_job_miss():
    # Training (4 s) and a job (1 s of work, due at 1.5) both start at 0.
    # EDF: the job runs first and finishes at 1.0. Fair share: each gets half the
    # CPU, so the job would need until 2.0 and misses at 1.5.
    jobs = [Job(0.0, 1000, 1.5)]
    edf = simulate_round(jobs, mips=1000, round_window=10, training_mi=4000, training_policy="edf")
    fair = simulate_round(jobs, mips=1000, round_window=10, training_mi=4000, training_policy="fair_share")
    assert edf["missed"] == 0 and fair["missed"] == 1
    assert fair["training_on_time"]          # training: 1.5 s at half speed, then full speed


def test_training_that_does_not_fit_the_window_is_late():
    result = simulate_round([], mips=1000, round_window=10, training_mi=12000)
    assert result["training_on_time"] is False


def make_workload(**overrides):
    config = {"federation": {"num_nodes": 20, "num_rounds": 100}, "seed": 0,
              "training": {"local_epochs": 1}, "workload": {"enabled": True, **overrides}}
    return FogWorkload(config, num_weights=20879)


def test_workload_is_the_same_whatever_is_selected():
    # Jobs depend only on (seed, round, node), never on what else happened.
    first = make_workload().jobs_for(7, 3)
    second = make_workload().jobs_for(7, 3)
    assert [(j.arrival, j.size_mi) for j in first] == [(j.arrival, j.size_mi) for j in second]


def test_jobs_fill_about_the_requested_load():
    workload = make_workload()
    used = []
    for server_round in range(1, 101):
        for node in range(20):
            jobs = workload.jobs_for(server_round, node)
            busy_seconds = sum(j.size_mi for j in jobs) / workload.mips[node]
            used.append(busy_seconds / workload.round_window - workload.load[server_round, node])
    assert abs(np.mean(used)) < 0.03          # on average, within 3 percentage points


def test_busy_rounds_take_the_long_run_share():
    # p(normal->busy) = 0.2, p(busy->normal) = 0.5, so about 0.2 / 0.7 = 29% busy rounds.
    workload = make_workload()
    assert 0.22 < workload.busy[1:].mean() < 0.36


def test_training_cost_follows_six_ops_per_weight_per_row():
    # 6 x 20,879 weights x 72,000 rows = 9,019.7 million instructions.
    assert abs(make_workload().training_mi(72000) - 9019.728) < 0.01
