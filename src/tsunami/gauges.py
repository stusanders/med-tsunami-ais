"""Tide gauges and deep-sea buoys (T06). Contract: docs/interfaces.md section 12.

`load_gauges` returns the gauge and buoy section of the receiver pool, in pool order (gauges first, then buoys).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import load_config, repo_path
from .grid import _unit_xyz
from .solver_reference import sponge_mask

GAUGE, BUOY = 1, 2


def _snap(grid, lon_deg, lat_deg, radius_m):
    """Nearest wet cell centre for each point; returns (cell_j, cell_i, distance_m)."""
    jj, ii = np.nonzero(grid.wet)
    tree = cKDTree(_unit_xyz(grid.lon[ii], grid.lat[jj]))
    chord, k = tree.query(_unit_xyz(lon_deg, lat_deg))
    return jj[k].astype(np.int32), ii[k].astype(np.int32), 2.0 * radius_m * np.arcsin(np.clip(chord / 2.0, 0, 1))


def read_ioc(cfg):
    """IOC stations kept for the model: normal status, a sea-level sensor, lon/lat inside the area of interest."""
    df = pd.read_csv(repo_path(cfg["paths"]["ioc_gauges"]))
    g, d = cfg["gauges"], cfg["domain"]["area_of_interest"]
    keep = (df["active"].astype(bool) & (df["ioc_status"] == g["require_ioc_status"])
            & ~df["sensors"].isin(g["exclude_sensors"])
            & df["lon"].between(*d["lon_deg"]) & df["lat"].between(*d["lat_deg"]))
    return df[keep].sort_values("station_code", key=lambda s: s.str.lower()).reset_index(drop=True)


def read_buoys(cfg):
    """Ionian buoys from T26a (Stream C). Returns an empty frame if the file is not there yet."""
    path = repo_path(cfg["paths"]["buoys"])
    if not path.exists():
        return pd.DataFrame(columns=["code", "name", "lon_deg", "lat_deg"])
    return pd.read_csv(path)


def load_gauges(grid, cfg=None):
    """Table of gauges (rtype 1) then buoys (rtype 2) snapped to their nearest wet cell centre.

    Columns: code, name, lon_deg, lat_deg (snapped cell centre), cell_j, cell_i, rtype, threshold_m, delay_s,
    plus orig_lon_deg, orig_lat_deg, snap_dist_m. Stations farther than gauges.max_snap_dist_m from a wet cell are dropped
    (Black Sea, Marmara), any station outside the area of interest or snapping into the sponge, and any second station snapping
    to a cell already used; all are listed in `.attrs["dropped"]` as (code, reason).
    """
    cfg = cfg or load_config()
    R, gc = cfg["physics"]["earth_radius_m"], cfg["gauges"]
    ioc, buoys = read_ioc(cfg), read_buoys(cfg)
    rows = pd.concat([
        pd.DataFrame({"code": ioc["station_code"], "name": ioc["name"], "orig_lon_deg": ioc["lon"], "orig_lat_deg": ioc["lat"],
                      "rtype": GAUGE}),
        pd.DataFrame({"code": buoys["code"], "name": buoys["name"], "orig_lon_deg": buoys["lon_deg"],
                      "orig_lat_deg": buoys["lat_deg"], "rtype": BUOY})], ignore_index=True)
    cj, ci, dist = _snap(grid, rows["orig_lon_deg"].values, rows["orig_lat_deg"].values, R)
    rows["cell_j"], rows["cell_i"], rows["snap_dist_m"] = cj, ci, dist
    aoi, sponge = cfg["domain"]["area_of_interest"], sponge_mask(grid.depth.shape, cfg)
    in_aoi = lambda lon, lat: aoi["lon_deg"][0] <= lon <= aoi["lon_deg"][1] and aoi["lat_deg"][0] <= lat <= aoi["lat_deg"][1]
    dropped, keep, used = [], np.zeros(len(rows), bool), set()
    for k, r in rows.iterrows():
        if not in_aoi(r["orig_lon_deg"], r["orig_lat_deg"]):
            dropped.append((r["code"], "outside the area of interest"))
        elif sponge[r["cell_j"], r["cell_i"]] or not in_aoi(grid.lon[r["cell_i"]], grid.lat[r["cell_j"]]):
            dropped.append((r["code"], "snapped cell is in the sponge or outside the area of interest"))
        elif r["rtype"] == GAUGE and r["snap_dist_m"] > gc["max_snap_dist_m"]:
            dropped.append((r["code"], "no wet cell within max_snap_dist_m"))
        elif (r["cell_j"], r["cell_i"]) in used:
            dropped.append((r["code"], "duplicate of a station in the same cell"))
        else:
            keep[k] = True
            used.add((r["cell_j"], r["cell_i"]))
    out = rows[keep].reset_index(drop=True)
    out["lon_deg"] = grid.lon[out["cell_i"].values]
    out["lat_deg"] = grid.lat[out["cell_j"].values]
    is_buoy = (out["rtype"] == BUOY).values
    out["threshold_m"] = np.where(is_buoy, cfg["buoys"]["threshold_m"], gc["threshold_m"])
    out["delay_s"] = np.where(is_buoy, cfg["buoys"]["telemetry_delay_s"], gc["telemetry_delay_s"])
    out = out[["code", "name", "lon_deg", "lat_deg", "cell_j", "cell_i", "rtype", "threshold_m", "delay_s",
               "orig_lon_deg", "orig_lat_deg", "snap_dist_m"]]
    out.attrs["dropped"] = dropped
    return out


def trigger_times(gb30, table, eta0_gb=None, sample_interval_s=None):
    """First time each gauge or buoy raises an alarm, in seconds after origin; inf if never.

    gb30 (E, 240, R_gb): raw eta at 30 s. A receiver triggers at the first sample k with |eta(t) - eta(0)| > threshold_m, i.e.
    at t = 30 (k + 1) s, and the alarm reaches the warning centre `delay_s` later.

    eta0_gb (E, R_gb): the source's initial surface eta(0) at each receiver's snapped cell. A real gauge or buoy sits on the
    seabed or on land that moves with the earthquake, so it does not see the static offset of the initial field; only the
    wave that arrives afterwards counts. eta(0) is not in gb30 (sample k is at t = 30 (k + 1) s). None keeps the raw-eta
    behaviour (tests only; it makes receivers inside the source area trigger at the first sample).
    """
    dt = sample_interval_s if sample_interval_s is not None else load_config()["solver"]["sample_interval_s"]
    gb30 = np.asarray(gb30)
    if eta0_gb is not None:
        gb30 = gb30 - np.asarray(eta0_gb, dtype=gb30.dtype)[:, None, :]
    exceed = np.abs(gb30) > table["threshold_m"].values.astype(gb30.dtype)[None, None, :]    # same dtype: no float32/64 rounding
    first = exceed.argmax(axis=1)
    t = dt * (first + 1) + table["delay_s"].values[None, :]
    return np.where(exceed.any(axis=1), t, np.inf)


def eta0_at_receivers(eta0, table):
    """eta(0) at the gauge and buoy cells: (ny, nx) or (E, ny, nx) -> (R_gb,) or (E, R_gb)."""
    eta0 = np.asarray(eta0)
    j, i = table["cell_j"].values, table["cell_i"].values
    return eta0[..., j, i]


def table_from_pool(pool, cfg):
    """The gauge and buoy table rebuilt from the receiver pool alone (same rows, same order, same columns as load_gauges).

    Lets a runtime without the raw IOC and buoy files run triggers: the pool already stores each receiver's snapped cell.
    """
    gb = np.nonzero(np.asarray(pool.rtype) > 0)[0]
    is_buoy = np.asarray(pool.rtype)[gb] == BUOY
    g, b = cfg["gauges"], cfg["buoys"]
    return pd.DataFrame({"code": np.asarray(pool.code)[gb], "name": np.asarray(pool.code)[gb], "lon_deg": np.asarray(pool.lon)[gb],
                         "lat_deg": np.asarray(pool.lat)[gb], "cell_j": np.asarray(pool.cell_j)[gb], "cell_i": np.asarray(pool.cell_i)[gb],
                         "rtype": np.asarray(pool.rtype)[gb].astype(int), "threshold_m": np.where(is_buoy, b["threshold_m"], g["threshold_m"]),
                         "delay_s": np.where(is_buoy, b["telemetry_delay_s"], g["telemetry_delay_s"])})
