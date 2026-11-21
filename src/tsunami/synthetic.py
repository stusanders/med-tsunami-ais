"""Synthetic stand-ins that match the T01/T04 contract (docs/interfaces.md sections 2 and 4).

Used by Stream A's tests and development until B's grid.py and ships.py are merged.
Not for production runs.
"""
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt

SHIP, GAUGE, BUOY = 0, 1, 2


def synthetic_grid(depth, lon0_deg=20.0, lat0_deg=35.0, dlon_deg=1 / 30, dlat_deg=1 / 30,
                   radius_m=6371000.0, min_wet_depth_m=10.0):
    """Grid-like namespace from a depth array (ny, nx), positive down, <= 0 = land.

    lon0/lat0 are the centre of cell (0, 0). Rows ascend northwards.
    """
    depth = np.asarray(depth, dtype=np.float64)
    ny, nx = depth.shape
    wet = depth > 0
    d = np.where(wet, np.maximum(depth, min_wet_depth_m), 0.0).astype(np.float32)
    lon = lon0_deg + dlon_deg * np.arange(nx)
    lat = lat0_deg + dlat_deg * np.arange(ny)
    lat_v = lat0_deg + dlat_deg * (np.arange(ny + 1) - 0.5)
    cos_c = np.cos(np.deg2rad(lat))
    cos_v = np.cos(np.deg2rad(lat_v))
    dx = radius_m * cos_c * np.deg2rad(dlon_deg)
    dy = radius_m * np.deg2rad(dlat_deg)
    if wet.all():
        coast = np.full((ny, nx), np.inf, dtype=np.float32)
    else:
        coast = distance_transform_edt(wet, sampling=(dy, float(dx.mean()))).astype(np.float32)
    band = np.full((ny, nx), -1, dtype=np.int8)
    band[wet & (d < 50)] = 0
    band[wet & (d >= 50) & (d <= 200)] = 1
    band[wet & (d > 200)] = 2
    return SimpleNamespace(lon=lon, lat=lat, dlon_deg=dlon_deg, dlat_deg=dlat_deg, depth=d, wet=wet,
                           strait=np.zeros((ny, nx), bool), dx=dx, dy=dy, cos_c=cos_c, cos_v=cos_v,
                           coast_dist=coast, depth_band=band, radius_m=radius_m)


def synthetic_pool(grid, lon_deg, lat_deg, rtype, lane_dir=None, code=None):
    """ReceiverPool-like namespace. Gauges and buoys are snapped to their nearest cell centre."""
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    rtype = np.asarray(rtype, dtype=np.int8)
    R = len(lon_deg)
    ci = np.clip(np.rint((lon_deg - grid.lon[0]) / grid.dlon_deg), 0, len(grid.lon) - 1).astype(np.int32)
    cj = np.clip(np.rint((lat_deg - grid.lat[0]) / grid.dlat_deg), 0, len(grid.lat) - 1).astype(np.int32)
    snapped = rtype > 0
    lon = np.where(snapped, grid.lon[ci], lon_deg)
    lat = np.where(snapped, grid.lat[cj], lat_deg)
    if lane_dir is None:
        lane_dir = np.zeros(R, np.float32)
    lane_dir = np.where(snapped, np.nan, lane_dir).astype(np.float32)
    if code is None:
        code = np.array(["" if t == SHIP else f"R{k}" for k, t in enumerate(rtype)], dtype="<U24")
    return SimpleNamespace(lon=lon, lat=lat, depth=grid.depth[cj, ci], lane_dir=lane_dir, rtype=rtype,
                           code=code, cell_j=cj, cell_i=ci, coast_dist=grid.coast_dist[cj, ci],
                           weight=np.where(snapped, 0.0, 1.0).astype(np.float32),
                           area_id=np.where(snapped, -1, 0).astype(np.int32))
