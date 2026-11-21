"""T25: single-ship baselines."""
import numpy as np
import pytest

from tsunami import baseline_detector as bd
from tsunami.config import load_config

CFG = load_config()
K_LIT = CFG["single_ship"]["literal_k_sigma"]
FAR = CFG["scoring"]["false_alarm_rate"]


def quiet(n, S, T, sigma, rng):
    return (rng.normal(size=(n, S, T)) * sigma[None, :, None]).astype(np.float32)


def test_large_single_ship_signal_triggers_both_rules():
    rng = np.random.default_rng(0)
    S, T = 300, 120
    sigma = rng.uniform(0.05, 0.12, S).astype(np.float32)
    k_cal = bd.calibrated_threshold(bd.single_ship_stat(quiet(1000, S, T, sigma, rng), sigma), FAR)
    d = quiet(1, S, T, sigma, rng)
    d[0, 17, 60:] += 20 * sigma[17]                           # one ship, wave arrives at minute 60
    stat = bd.single_ship_stat(d, sigma)
    assert bd.alarm_minute(stat, k_cal)[0] == 60
    assert 0 <= bd.literal_alarm_minute(d, sigma, k=K_LIT)[0] <= 60


def test_literal_rule_false_alarm_rate_is_near_one_for_a_fleet():
    """Thousands of ship-minutes: some ship almost always exceeds 3 sigma on a quiet day."""
    rng = np.random.default_rng(1)
    S, T = 2000, 120
    sigma = np.full(S, 0.08, np.float32)
    stat = bd.single_ship_stat(quiet(50, S, T, sigma, rng), sigma)
    assert bd.false_alarm_rate(stat, K_LIT) > 0.99


def test_calibrated_rule_holds_far_on_held_out():
    rng = np.random.default_rng(2)
    S, T = 200, 120
    sigma = rng.uniform(0.05, 0.12, S).astype(np.float32)
    k = bd.calibrated_threshold(bd.single_ship_stat(quiet(3000, S, T, sigma, rng), sigma), FAR)
    rate = bd.false_alarm_rate(bd.single_ship_stat(quiet(3000, S, T, sigma, rng), sigma), k)
    assert rate == pytest.approx(FAR, abs=0.006)
    assert k > K_LIT                                           # calibrated k is stricter than 3 sigma


def test_stat_is_running_max_and_ignores_missing():
    d = np.zeros((1, 2, 4), np.float32)
    d[0, 0, 1] = 5.0
    d[0, 1, 3] = 9.0
    valid = np.ones((2, 4), bool)
    valid[1, 3] = False
    stat = bd.single_ship_stat(d, np.ones(2, np.float32), valid)
    np.testing.assert_array_equal(stat[0], [0, 5, 5, 5])
