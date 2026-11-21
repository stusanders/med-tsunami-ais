"""Model grid (T01): GEBCO bathymetry block-averaged to 2 arc-min, masks and spherical metrics.

Contract: docs/interfaces.md section 2. Indexing is [j, i] = [lat, lon], row 0 = south.
Latitude and longitude are stored in degrees; trig converts to radians through units.deg2rad.
"""
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.spatial import cKDTree

from .config import repo_path
from .units import deg2rad


@dataclass
class Grid:
    lon: np.ndarray          # (nx,) f64 deg
    lat: np.ndarray          # (ny,) f64 deg, ascending
    dlon_deg: float
    dlat_deg: float
    depth: np.ndarray        # (ny, nx) f32 m, positive down, land 0
    wet: np.ndarray          # (ny, nx) bool
    strait: np.ndarray       # (ny, nx) bool
    dx: np.ndarray           # (ny,) f64 m
    dy: float                # m
    cos_c: np.ndarray        # (ny,) f64
    cos_v: np.ndarray        # (ny+1,) f64
    coast_dist: np.ndarray   # (ny, nx) f32 m
    depth_band: np.ndarray   # (ny, nx) int8
    radius_m: float          # Earth radius used for the metrics (not in the contract table; solver helpers may use it)


# ----------------------------------------------------------------------------- building blocks

def block_mean(a, factor):
    """Mean over non-overlapping factor x factor blocks; trailing rows or columns that do not fill a block are dropped."""
    ny, nx = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    a = a[:ny, :nx].astype(np.float64)
    return a.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def box_mask(lon, lat, box):
    """True inside a config box {lon_deg: [lo, hi], lat_deg: [lo, hi]} (cell centres)."""
    lo, hi = box["lon_deg"]
    la, lb = box["lat_deg"]
    return ((lon[None, :] >= lo) & (lon[None, :] <= hi) & (lat[:, None] >= la) & (lat[:, None] <= lb))


def ocean_connected(wet, lon, lat, seed_lon_deg, seed_lat_deg):
    """Keep only wet cells 4-connected to the seed cell (drops Qattara, Dead Sea, inland lakes)."""
    lab, _ = label(wet)
    j = int(np.argmin(np.abs(lat - seed_lat_deg)))
    i = int(np.argmin(np.abs(lon - seed_lon_deg)))
    if lab[j, i] == 0:
        raise ValueError("ocean seed cell is not wet")
    return lab == lab[j, i]


def _unit_xyz(lon_deg, lat_deg):
    lam, phi = deg2rad(np.asarray(lon_deg, float)), deg2rad(np.asarray(lat_deg, float))
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def distance_to_land_m(land, lon, lat, radius_m):
    """Great-circle distance in m from each cell centre to the nearest land cell centre (0 on land).

    A KD-tree on 3-D unit-sphere coordinates gives the chord; the arc is 2 R asin(chord / 2). This keeps the metric
    exact in both directions at every latitude (a plain distance transform would need a single dx).
    """
    LON, LAT = np.meshgrid(lon, lat)
    land_xyz = _unit_xyz(LON[land], LAT[land])
    tree = cKDTree(land_xyz)
    chord, _ = tree.query(_unit_xyz(LON.ravel(), LAT.ravel()))
    arc = 2.0 * radius_m * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return np.where(land, 0.0, arc.reshape(land.shape))


def depth_bands(depth, wet, edges_m):
    """0: depth < edges[0], 1: edges[0] to edges[1], 2: > edges[1], -1: land."""
    band = np.full(depth.shape, -1, dtype=np.int8)
    band[wet & (depth < edges_m[0])] = 0
    band[wet & (depth >= edges_m[0]) & (depth <= edges_m[1])] = 1
    band[wet & (depth > edges_m[1])] = 2
    return band


def metrics(lat, dlat_deg, dlon_deg, radius_m):
    """dx per row, dy, cos(phi) at centres and at v faces (cos_v[j] = south face of row j)."""
    cos_c = np.cos(deg2rad(lat))
    lat_v = np.concatenate([lat - dlat_deg / 2.0, [lat[-1] + dlat_deg / 2.0]])
    cos_v = np.cos(deg2rad(lat_v))
    dx = radius_m * cos_c * deg2rad(dlon_deg)
    dy = radius_m * deg2rad(dlat_deg)
    return dx, float(dy), cos_c, cos_v


# ----------------------------------------------------------------------------- build / load

def build_grid(cfg, save=True):
    d, R = cfg["domain"], cfg["physics"]["earth_radius_m"]
    res_deg = d["resolution_arcmin"] / 60.0
    ds = xr.open_dataset(repo_path(cfg["paths"]["gebco"]))
    glon, glat = ds["lon"].values, ds["lat"].values
    g_res = float(np.median(np.diff(glon)))
    factor = int(round(res_deg / g_res))
    # Align the coarse cells to the domain's south-west corner on the GEBCO (cell-centre) grid.
    j0 = int(round((d["lat_min_deg"] - glat[0]) / g_res + 0.5))
    i0 = int(round((d["lon_min_deg"] - glon[0]) / g_res + 0.5))
    elev = ds["elevation"].values[j0:, i0:]
    # Work on the whole GEBCO window east and north of the domain corner so that the coast distance of cells on the
    # domain edge sees land just outside the box (Adriatic head, Turkish coast north of 42 N, ...).
    coarse = block_mean(elev, factor)
    ny_ext, nx_ext = coarse.shape
    lon_e = d["lon_min_deg"] + res_deg * (np.arange(nx_ext) + 0.5)
    lat_e = d["lat_min_deg"] + res_deg * (np.arange(ny_ext) + 0.5)
    depth_raw = -coarse                                  # positive down
    wet = depth_raw > 0
    for box in d.get("exclude_boxes", {}).values():
        wet &= ~box_mask(lon_e, lat_e, box)
    seed = d["ocean_seed"]
    wet = ocean_connected(wet, lon_e, lat_e, seed["lon_deg"], seed["lat_deg"])
    dist_e = distance_to_land_m(~wet, lon_e, lat_e, R)

    nx = int(round((d["lon_max_deg"] - d["lon_min_deg"]) / res_deg))
    ny = int(round((d["lat_max_deg"] - d["lat_min_deg"]) / res_deg))
    sl = (slice(0, ny), slice(0, nx))
    lon, lat, wet = lon_e[:nx], lat_e[:ny], wet[sl]
    depth = np.where(wet, np.maximum(depth_raw[sl], d["min_wet_depth_m"]), 0.0).astype(np.float32)
    strait = np.zeros((ny, nx), bool)
    for box in d.get("straits", {}).values():
        strait |= box_mask(lon, lat, box)
    dx, dy, cos_c, cos_v = metrics(lat, res_deg, res_deg, R)
    grid = Grid(lon=lon, lat=lat, dlon_deg=res_deg, dlat_deg=res_deg, depth=depth, wet=wet, strait=strait,
                dx=dx, dy=dy, cos_c=cos_c, cos_v=cos_v, coast_dist=dist_e[sl].astype(np.float32),
                depth_band=depth_bands(depth, wet, d["depth_band_edges_m"]), radius_m=float(R))
    if save:
        path = repo_path(cfg["paths"]["grid"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(grid, f.name) for f in fields(Grid)})
    return grid


def load_grid(cfg):
    with np.load(repo_path(cfg["paths"]["grid"])) as z:
        kw = {f.name: z[f.name] for f in fields(Grid)}
    for k in ("dlon_deg", "dlat_deg", "dy", "radius_m"):
        kw[k] = float(kw[k])
    return Grid(**kw)
