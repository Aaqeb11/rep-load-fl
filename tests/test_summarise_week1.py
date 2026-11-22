"""Checks for scripts/summarise_week1.py (loaded from its file, as scripts/ is not a package)."""

import importlib.util
import os

import numpy as np

from bcfl.attacks import get_attack_settings, is_attacking_this_round

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "summarise_week1.py")
spec = importlib.util.spec_from_file_location("summarise_week1", SCRIPT)
summarise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarise)


def test_on_off_windows_match_the_attack_code():
    # The summary must count exactly the rounds in which nodes really attacked.
    config = {"attack": {"type": "on_off", "fraction": 0.2, "honest_rounds": 10, "attack_rounds": 5},
              "federation": {"num_nodes": 20, "num_rounds": 100}, "seed": 0}
    settings = get_attack_settings(config)
    expected = [r for r in range(1, 101) if is_attacking_this_round(settings, r)]
    assert summarise.attack_rounds(config) == expected
    assert len(expected) == 30


def test_attack_success_rate_reads_the_right_cell():
    # 3 classes; class 0's rows: 6 correct, 4 predicted as class 2 -> ASR(0 -> 2) = 0.4
    confusion = np.array([[6, 0, 4], [0, 5, 0], [1, 0, 9]], dtype=float)
    run = {"confusions": {r: confusion for r in range(1, 101)}}
    assert abs(summarise.attack_success_rate(run, source=0, target=2) - 0.4) < 1e-9
    assert abs(summarise.class_recall(run, 0) - 0.6) < 1e-9
