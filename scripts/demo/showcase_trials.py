"""Shared loader for the T17 showcase trials used by scenes 8, 9, 9b (prefix "showcase86") and 9c ("showcase80").

Files (written by scripts/demo/make_showcase86_data.py into outputs/demo_data/):
  <prefix>_meta.json       trial, ais, gps, gauges, ships_above_noise, typical, provenance
  <prefix>_snapshots.npz   eta (121, ny, nx) float16 per minute, lon, lat, t_min
  <prefix>_traffic.npz     per-ship lon, lat, depth_m, sigma_total, true_drift, obs_drift, snr_t (S, 120);
                           stat_ais, stat_gps (120,), thr_ais, thr_gps; g_lon, g_lat, g_trigger_s, g_is_buoy; t_min

Everything is read at build time (the statistic arrays may be rewritten when the full library run finishes).
Missing files -> None (scenes draw cm.placeholder). For layout work before the files exist, set T27_STANDIN=1:
the old Mw 8.0 showcase files are then mapped onto the same contract, with obviously fake 'typical' values.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import common as cm
import numpy as np

DATA = Path(os.environ.get("T27_SHOWCASE_DIR", cm.DEMO_DATA))


def load(prefix: str = "showcase86") -> dict:
    """{'meta', 'snap', 'traffic', 'standin'}; each data entry None if its file is missing."""
    out = {"standin": False}
    p = DATA / f"{prefix}_meta.json"
    out["meta"] = json.loads(p.read_text()) if p.exists() else None
    for key, name in (("snap", "snapshots"), ("traffic", "traffic")):
        p = DATA / f"{prefix}_{name}.npz"
        out[key] = dict(np.load(p, allow_pickle=False)) if p.exists() else None
    if out["meta"] is None and os.environ.get("T27_STANDIN") == "1":
        return _standin(prefix)
    return out


def snapshot_base(snap, grid):
    """(wet, depth) on the snapshot lon/lat, from the snapshot file if present, else looked up in the model grid."""
    if "wet" in snap and "depth" in snap:
        return snap["wet"].astype(bool), snap["depth"].astype(float)
    ix = np.clip(np.rint((snap["lon"] - grid.lon[0]) / (grid.lon[1] - grid.lon[0])).astype(int), 0, len(grid.lon) - 1)
    iy = np.clip(np.rint((snap["lat"] - grid.lat[0]) / (grid.lat[1] - grid.lat[0])).astype(int), 0, len(grid.lat) - 1)
    wet = grid.wet[np.ix_(iy, ix)].astype(bool)
    depth = np.where(wet, grid.depth[np.ix_(iy, ix)], np.nan).astype(float)
    return wet, depth


def nice_ceil(x):
    """Round up to 1, 2 or 5 x 10^k."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    k = math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if x <= m * 10 ** k * (1 + 1e-9):
            return m * 10 ** k
    return 10 ** (k + 1)


def fmt_min(m):
    """Minutes for labels: integer when whole, else one decimal (script style: 5.5 min, 84 min; triggers are 30 s steps)."""
    if m is None:
        return None
    if abs(m - round(m)) > 0.05:
        return f"{m:.1f}"
    return f"{round(m):.0f}"


def pct(frac):
    return f"{100 * frac:.0f}%"


def stroke(w=4):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=cm.BG)]


# --------------------------------------------------------------------------------------
# Stand-in (layout only; never used unless T27_STANDIN=1)
# --------------------------------------------------------------------------------------
def _standin(prefix):
    from tsunami.geometry import haversine_m
    p = DATA / "showcase_meta.json"
    if not p.exists():
        return {"standin": True, "meta": None, "snap": None, "traffic": None}
    old = json.loads(p.read_text())
    tr = dict(np.load(DATA / "showcase_traffic.npz"))
    sn = dict(np.load(DATA / "showcase_snapshots.npz"))
    sh, inst = old["showcase"], old["instruments"]
    top = old["network"]["top3"]
    shift = 0.0 if prefix == "showcase86" else 0.0
    loc = haversine_m(top[0]["lon_deg"], top[0]["lat_deg"], sh["lon_deg"], sh["lat_deg"], 6371000.0) / 1000
    is_b = tr["gauge_rtype"] != 1
    stat = tr["net_stat"].astype(float)
    thr = float(tr["threshold"])
    if prefix == "showcase86":           # pretend it confirms at ~84 min
        stat = stat * 0 + np.linspace(0, 1.3 * thr, len(stat))
    stat_gps = np.r_[0.5, np.full(len(stat) - 1, 3.0)] * 50.0
    snr = tr["snr_t"]
    meta = {
        "trial": {"run_id": "standin", "region": sh["region"], "mw": 8.6 if prefix == "showcase86" else 8.1,
                  "lon": sh["lon_deg"] + shift, "lat": sh["lat_deg"], "regime": "calm",
                  "range_cap_m": old["range_cap_m"], "n_ships": old["n_ships"],
                  "n_shelf_ships": int(np.sum(tr["depth"] < 200)) if prefix == "showcase86" else 0},
        "ais": {"detected": prefix == "showcase86", "confirm_s": 77 * 60.0 if prefix == "showcase86" else None,
                "confirm_min": 77 if prefix == "showcase86" else None, "threshold": thr,
                "best_lon": top[0]["lon_deg"], "best_lat": top[0]["lat_deg"], "best_mw": top[0]["mw"],
                "loc_err_km": loc, "top3": top},
        "gps": {"detected": True, "confirm_s": 60.0, "confirm_min": 1, "threshold": 50.0},
        "gauges": {"n_gauges": int((~is_b).sum()), "n_buoys": int(is_b.sum()),
                   "first_gauge_s": inst["first_gauge_s"], "first_gauge_name": inst["first_gauge_code"],
                   "first_buoy_s": inst["first_buoy_s"],
                   "trigger_s": dict(zip(tr["gauge_code"].tolist(), tr["gauge_trigger_s"].tolist()))},
        "ships_above_noise": {"definition": "standin: peak snr_t > 1", "n_above": int((snr.max(1) > 1).sum()),
                              "n_total": len(snr)},
        "typical": {"ais_det_frac": 0.11, "ais_med_confirm_min": 99.0, "ais_med_loc_km": 99.0,
                    "first_gauge_med_min": 9.9, "gps_det_frac": 0.11, "gps_med_confirm_min": 9.0},
    }
    traffic = {"lon": tr["lon"], "lat": tr["lat"], "depth_m": tr["depth"], "sigma_total": tr["sigma_total"],
               "true_drift": tr["true_signal"], "obs_drift": tr["cleaned_drift"], "snr_t": snr,
               "stat_ais": stat, "stat_gps": stat_gps, "thr_ais": np.float64(thr), "thr_gps": np.float64(50.0),
               "g_lon": tr["gauge_lon"], "g_lat": tr["gauge_lat"], "g_trigger_s": tr["gauge_trigger_s"],
               "g_is_buoy": is_b, "t_min": np.arange(1, 121)}
    snap = {"eta": sn["eta"], "lon": sn["lon"], "lat": sn["lat"], "t_min": sn["times_min"]}
    return {"standin": True, "meta": meta, "snap": snap, "traffic": traffic}
