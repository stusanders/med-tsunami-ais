"""Observation operators (T12). docs/interfaces.md section 9.

Truth comes from library-format runs that are already block-averaged and baseline-filtered,
so signal + filtered noise = filter(truth + noise) exactly (the filter is linear).
"""
import numpy as np

from .geometry import project_normal


def gather_truth(recf_run, draw):
    """(120, R, 3) filtered run -> (120, S, 3) at the active ships (numpy, CPU)."""
    return np.asarray(recf_run[:, np.asarray(draw.pool_idx), :])


def ais_signal(truth_s, draw, response=1.0):
    """(S, 120) heading-normal (starboard-positive) current felt by each ship, m/s.

    response < 1 models ships that only partly follow the water (T20).
    """
    theta = np.asarray(draw.heading)[None, :]
    return (response * project_normal(truth_s[:, :, 1], truth_s[:, :, 2], theta)).T.astype(np.float32)


def gps_signal(truth_s, draw):
    """(S, 120) sea-surface height at each ship, m."""
    return truth_s[:, :, 0].T.astype(np.float32)


def gaps(draw, cfg, rng, drop_frac=0.0, off_mask=None):
    """valid (S, 120) bool, one mask per traffic draw, shared by all its noise draws.

    drop_frac: fraction of ship-minutes missing at random. off_mask (S,): ships silent all window.
    """
    S, T = len(draw.sog), int(cfg["library"]["n_minutes"])
    valid = rng.random((S, T)) >= drop_frac if drop_frac > 0 else np.ones((S, T), bool)
    if off_mask is not None:
        valid &= ~np.asarray(off_mask, bool)[:, None]
    return valid


def observe(signal, noise):
    """d (E, S, 120) = signal (S, 120) + noise (E, S, 120)."""
    return (signal[None] + noise).astype(np.float32)
