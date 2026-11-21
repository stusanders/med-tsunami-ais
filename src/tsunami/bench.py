"""GPU-synchronised benchmark harness (T08, Stream C).

Contract (docs/interfaces.md section 15):
    bench.Timer(name, sync=backend.sync)           # context manager; .seconds
    bench.record(name, seconds, cell_updates=None, bytes_moved=None, **meta) -> dict
    bench.write_json(records, path)
"""
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import backend

# Published peak memory bandwidth (GB/s, decimal 10^9 B/s) for common Colab / workstation GPUs.
KNOWN_GPU_PEAK_BANDWIDTH_GB_S = {
    "T4": 320.0,
    "L4": 300.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE-40GB": 1555.0,
    "A100": 1555.0,
    "V100": 900.0,
    "P100": 732.0,
}


class Timer:
    """Context manager that synchronises the active backend before starting and stopping."""

    def __init__(self, name, sync=backend.sync):
        self.name = str(name)
        self.sync = sync
        self.seconds = None
        self._t0 = None

    def __enter__(self):
        if self.sync is not None:
            self.sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync is not None:
            self.sync()
        self.seconds = time.perf_counter() - self._t0
        return False


def _to_jsonable(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    return val


def device_info():
    """Return a dict describing the active compute backend and device."""
    info = {
        "backend": "cupy" if backend.GPU else "numpy",
        "gpu": bool(backend.GPU),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    if backend.GPU:
        try:
            dev = backend.xp.cuda.Device(0)
            props = backend.xp.cuda.runtime.getDeviceProperties(dev.id)
            name = props["name"].decode("utf-8", errors="replace") if isinstance(props["name"], bytes) else str(props["name"])
            info["device_name"] = name
            info["vram_gb"] = round(float(props["totalGlobalMem"]) / 1e9, 2)
            # Match published peak memory bandwidth if recognised
            peak = None
            for key, bw in KNOWN_GPU_PEAK_BANDWIDTH_GB_S.items():
                if key in name:
                    peak = bw
                    break
            if peak is not None:
                info["peak_bandwidth_gb_s"] = peak
        except Exception as exc:
            info["device_error"] = str(exc)
    return info


def record(name, seconds, cell_updates=None, bytes_moved=None, *, peak_bandwidth_gb_s=None, **meta):
    """Build a JSON-ready benchmark record with throughput and memory-bandwidth metrics.

    Memory bandwidth uses decimal GB/s (bytes / seconds / 1e9) to match GPU datasheets.
    """
    sec = float(seconds)
    out = {
        "name": str(name),
        "seconds": sec,
        "backend": "cupy" if backend.GPU else "numpy",
    }
    if cell_updates is not None:
        cu = int(cell_updates)
        out["cell_updates"] = cu
        out["cell_updates_per_s"] = (float(cu) / sec) if sec > 0.0 else None
    if bytes_moved is not None:
        bm = int(bytes_moved)
        out["bytes_moved"] = bm
        bw_gb_s = (float(bm) / sec / 1e9) if sec > 0.0 else None
        out["bandwidth_gb_s"] = bw_gb_s
        if peak_bandwidth_gb_s is not None and bw_gb_s is not None:
            out["peak_bandwidth_gb_s"] = float(peak_bandwidth_gb_s)
            out["bandwidth_pct_peak"] = 100.0 * bw_gb_s / float(peak_bandwidth_gb_s)
    for k, v in meta.items():
        out[str(k)] = _to_jsonable(v)
    return out


def write_json(records, path):
    """Write benchmark records (list or dict) to `path` as formatted JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(records)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def read_json(path):
    """Read a benchmark JSON file back into Python objects."""
    return json.loads(Path(path).read_text())
