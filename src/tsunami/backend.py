"""NumPy/CuPy switch. Import `xp` from here; never import cupy directly elsewhere.

Set TSUNAMI_BACKEND=numpy to force NumPy even when a GPU is present.
"""
import os

import numpy as np

GPU = False
xp = np
if os.environ.get("TSUNAMI_BACKEND", "auto").lower() != "numpy":
    try:
        import cupy as _cp

        _cp.cuda.runtime.getDeviceCount()
        xp = _cp
        GPU = True
    except Exception:  # no cupy or no device
        xp = np
        GPU = False

DTYPE = np.float32


def array_module(a):
    """Return numpy or cupy, whichever owns array `a`."""
    if GPU:
        import cupy as _cp

        return _cp.get_array_module(a)
    return np


def to_numpy(a):
    """Copy to host if needed."""
    if GPU and array_module(a) is not np:
        return a.get()
    return np.asarray(a)


def to_xp(a, dtype=DTYPE):
    """Move to the active backend as float32 (or `dtype`)."""
    return xp.asarray(a, dtype=dtype)


def sync():
    """Block until queued GPU work finishes (no-op on CPU). Use before stopping timers."""
    if GPU:
        xp.cuda.Stream.null.synchronize()
