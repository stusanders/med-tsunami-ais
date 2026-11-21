"""Load config.yaml as a plain dict. Keys carry their unit as a suffix (_m, _s, _deg, _kn)."""
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def magnitudes(cfg):
    s = cfg["sources"]
    n = int(round((s["mw_max"] - s["mw_min"]) / s["mw_step"])) + 1
    return np.round(s["mw_min"] + s["mw_step"] * np.arange(n), 4)


def repo_path(rel):
    """Resolve a path from config relative to the repo (or worktree) root."""
    return REPO_ROOT / rel
