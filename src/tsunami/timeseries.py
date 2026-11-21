"""Time handling shared by the library build (T09/T14) and the observations (T12).

Raw receiver samples are at t = 30, 60, ..., 7200 s (index k <-> t = 30 (k+1)).
Minute m (0..119) is the mean of raw samples 2m and 2m+1 and covers (60m, 60(m+1)] s.
"""
import numpy as np

from .backend import array_module


def block_average(x, factor=2, axis=1):
    """Mean of consecutive groups of `factor` samples along `axis` (30 s -> 1 min)."""
    m = array_module(x)
    x = m.moveaxis(x, axis, 0)
    n = x.shape[0] // factor
    y = x[: n * factor].reshape((n, factor) + x.shape[1:]).mean(axis=1)
    return m.moveaxis(y, 0, axis)


def baseline_filter(x, window, axis=1, history=None):
    """Causal trailing-mean removal: y[m] = x[m] - mean(x[m-W], ..., x[m-1]).

    Never uses x[m] or any later sample. Samples before index 0 come from `history`
    (the W samples preceding x, same shape as x except `window` along `axis`), or
    are zero if history is None (correct for templates: no signal before origin).
    Linear, so filter(signal + noise) = filter(signal) + filter(noise).
    Accumulates in float64, returns x's dtype.
    """
    m = array_module(x)
    xd = m.moveaxis(x, axis, 0).astype(np.float64)
    if history is None:
        hist = m.zeros((window,) + xd.shape[1:], dtype=np.float64)
    else:
        hist = m.moveaxis(history, axis, 0).astype(np.float64)
        assert hist.shape[0] == window, "history must hold exactly `window` samples"
    padded = m.concatenate([hist, xd], axis=0)
    c = m.concatenate([m.zeros((1,) + xd.shape[1:]), m.cumsum(padded, axis=0)], axis=0)
    n = xd.shape[0]
    trailing_mean = (c[window : window + n] - c[0:n]) / window
    y = (xd - trailing_mean).astype(x.dtype)
    return m.moveaxis(y, 0, axis)


def minute_end_seconds(n_minutes):
    """Time (s after origin) at the end of each minute: 60, 120, ..."""
    return 60.0 * (np.arange(n_minutes) + 1)
