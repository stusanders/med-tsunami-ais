"""Single-ship baseline detectors (T25). PLAN.md Phase 4; docs/interfaces.md section 11.

Run on exactly the same observations d (E, S, T) as the network detector (T15).
sigma_i must be each ship's TOTAL noise std after filtering, i.e.
sqrt(noise.ship_sigma**2 + noise.shared_sigma**2): "its own noise" as a single ship would measure it.

  (i)  Inazu's literal rule: alarm when any ship's |d| exceeds k = 3 times its own noise.
       Its false-alarm rate is measured on quiet days, not set.
  (ii) Calibrated single-ship rule: same statistic, threshold k set per stratum on the quiet days
       so it shares the network detector's 1 % false-alarm rate.
"""
import numpy as np

from . import backend


def single_ship_stat(d, sigma_i, valid=None):
    """(E, T): running max over ships and minutes up to t of |d_i| / sigma_i (missing reports ignored)."""
    xp = backend.array_module(d)
    z = xp.abs(xp.asarray(d)) / xp.asarray(sigma_i)[None, :, None]
    if valid is not None:
        z = z * xp.asarray(valid)[None]
    best = backend.to_numpy(z.max(axis=1))
    return np.maximum.accumulate(best, axis=-1)


def alarm_minute(stat, k):
    """(E,) first minute with stat > k, -1 if never. Alarm time = 60 (m+1) s."""
    above = np.asarray(stat) > k
    return np.where(above.any(axis=-1), above.argmax(axis=-1), -1)


def literal_alarm_minute(d, sigma_i, valid=None, k=3.0):
    """Inazu's literal three-sigma rule on any single ship."""
    return alarm_minute(single_ship_stat(d, sigma_i, valid), k)


def calibrated_threshold(stat_quiet, far):
    """k such that a fraction `far` of quiet 2-hour windows alarm."""
    return float(np.quantile(np.asarray(stat_quiet).max(axis=-1), 1.0 - far))


def false_alarm_rate(stat_quiet, k):
    """Measured fraction of quiet windows with at least one alarm."""
    return float(np.mean(np.asarray(stat_quiet).max(axis=-1) > k))
