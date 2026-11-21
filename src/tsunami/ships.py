"""Receiver pool and traffic draws (T04). Contract: docs/interfaces.md sections 4 and 5.

The library is recorded once at the pool's ~20,000 candidate ship positions (plus every gauge and buoy). A traffic draw is
a random subset of the ship part of the pool, so one library serves every traffic draw.

Pool construction:
  1. EMODnet vessel density (hours / km2 / month) is regridded to the model grid and turned into the expected number of
     vessels present in each cell, E_cell = density / hours_per_month * cell_area_km2.
  2. Eligible cells: depth >= min_ship_depth_m, coast distance <= pool_max_coast_dist_m, not in a strait.
  3. Traffic areas are the scoring cells (geometry.cell_index at scoring.cell_m). Each area gets pool points in
     proportion to its expected ship count (at least pool_min_per_area), placed in cells in proportion to E_cell and jittered
     uniformly inside the cell. Pool weights are E_area / n_points_area, so the weights in an area add up to the
     expected number of ships there.
  4. Lane direction at each point is the axis perpendicular to the dominant density gradient (structure tensor of the
     smoothed log density), folded to [0, pi).
"""
from dataclasses import dataclass, fields

import numpy as np
from scipy.ndimage import gaussian_filter

from . import gauges as gauges_mod
from .config import repo_path
from .geometry import cell_index, fold_axis
from .units import knots_to_ms

SHIP, GAUGE, BUOY = 0, 1, 2
AREA_STRIDE = 4096          # area_id = cell_j * AREA_STRIDE + cell_i of geometry.cell_index (both are far below this)
PURPOSE_POOL = 8            # seed purpose code for pool construction (codes 1 to 7 are in docs/interfaces.md)


# ----------------------------------------------------------------------------- data classes

@dataclass
class ReceiverPool:
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    lane_dir: np.ndarray
    rtype: np.ndarray
    code: np.ndarray
    cell_j: np.ndarray
    cell_i: np.ndarray
    coast_dist: np.ndarray
    weight: np.ndarray
    area_id: np.ndarray

    def __len__(self):
        return len(self.lon)

    @property
    def n_ships(self):
        return int((self.rtype == SHIP).sum())


@dataclass
class TrafficDraw:
    pool_idx: np.ndarray     # (S,) int64
    heading: np.ndarray      # (S,) f32 rad, compass
    sog: np.ndarray          # (S,) f32 m/s
    seed: int


# ----------------------------------------------------------------------------- density and lanes

def regrid_density(cfg, grid):
    """EMODnet density (hours / km2 / month) on the model grid (ny, nx) float32, 0 where there is no data.

    Area-averaged from the 1 km EPSG:3035 cells (about 12 of them per model cell).
    """
    import rasterio                                   # only needed to build the pool, not on the GPU runtime
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    with rasterio.open(repo_path(cfg["paths"]["vessel_density"])) as src:
        ny, nx = grid.depth.shape
        dst = np.full((ny, nx), np.nan, np.float32)
        tr = from_origin(grid.lon[0] - grid.dlon_deg / 2, grid.lat[-1] + grid.dlat_deg / 2, grid.dlon_deg, grid.dlat_deg)
        reproject(rasterio.band(src, 1), dst, src_nodata=src.nodata, dst_nodata=np.nan, dst_transform=tr,
                  dst_crs="EPSG:4326", resampling=Resampling.average)
    return np.nan_to_num(np.flipud(dst), nan=0.0).astype(np.float32)      # flip: row 0 = south


def cell_area_km2(grid):
    return np.broadcast_to((grid.dx * grid.dy / 1.0e6)[:, None], grid.depth.shape)


def expected_ships(density, grid, cfg):
    """Expected vessels present in each cell: density / hours_per_month * area (km2). Zero on land."""
    return np.where(grid.wet, density / cfg["traffic"]["hours_per_month"] * cell_area_km2(grid), 0.0)


def lane_axis(density, grid, cfg):
    """Lane direction per cell (rad, compass bearing, folded to [0, pi)) and structure-tensor coherence in [0, 1].

    Land cells are filled from the neighbouring sea by normalised convolution, so the coastline does not make a fake lane.
    """
    t = cfg["traffic"]
    f = np.log1p(density).astype(np.float64)
    wet = grid.wet.astype(np.float64)
    sg, st = t["lane_grad_sigma_cells"], t["lane_tensor_sigma_cells"]
    fs = gaussian_filter(f * wet, sg) / np.maximum(gaussian_filter(wet, sg), np.finfo(float).tiny)
    fs = np.where(grid.wet, fs, gaussian_filter(f * wet, st) / np.maximum(gaussian_filter(wet, st), np.finfo(float).tiny))
    gx = np.gradient(fs, axis=1) / grid.dx[:, None]          # d/d(east), per metre
    gy = np.gradient(fs, axis=0) / grid.dy                   # d/d(north)
    jxx, jyy, jxy = (gaussian_filter(a, st) for a in (gx * gx, gy * gy, gx * gy))
    theta_grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)        # math angle (from east) of the strongest change = across the lane
    # across-lane vector (east, north) = (cos, sin); the lane vector is its perpendicular (-sin, cos); bearing = atan2(east, north)
    lane_bearing = np.arctan2(-np.sin(theta_grad), np.cos(theta_grad))
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0.0))
    coherence = disc / np.maximum(tr / 2, np.finfo(float).tiny)
    return fold_axis(lane_bearing).astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


def area_id(lon_deg, lat_deg, cfg, grid):
    s, d, R = cfg["scoring"], cfg["domain"], cfg["physics"]["earth_radius_m"]
    cj, ci = cell_index(np.asarray(lon_deg), np.asarray(lat_deg), s["cell_m"], s["cell_ref_lat_deg"],
                        d["lon_min_deg"], d["lat_min_deg"], R)
    return (cj * AREA_STRIDE + ci).astype(np.int32)


def _bilinear(field, grid, lon_deg, lat_deg):
    """Bilinear interpolation between cell centres (what the solver's sampler uses for ships)."""
    x = np.clip((lon_deg - grid.lon[0]) / grid.dlon_deg, 0, len(grid.lon) - 1 - 1e-9)
    y = np.clip((lat_deg - grid.lat[0]) / grid.dlat_deg, 0, len(grid.lat) - 1 - 1e-9)
    i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - i0, y - j0
    return ((1 - fy) * ((1 - fx) * field[j0, i0] + fx * field[j0, i0 + 1])
            + fy * ((1 - fx) * field[j0 + 1, i0] + fx * field[j0 + 1, i0 + 1]))


# ----------------------------------------------------------------------------- pool

def _eligible(grid, cfg):
    t = cfg["traffic"]
    return (grid.depth >= t["min_ship_depth_m"]) & (grid.coast_dist <= t["pool_max_coast_dist_m"]) & ~grid.strait & grid.wet


def build_pool(cfg, grid, save=True):
    """Ships first (pool_size points), then gauges, then buoys (from gauges.load_gauges)."""
    t = cfg["traffic"]
    rng = np.random.default_rng([cfg["meta"]["seed"], PURPOSE_POOL, 0])
    density = regrid_density(cfg, grid)
    lanes, _ = lane_axis(density, grid, cfg)
    E = np.where(_eligible(grid, cfg), expected_ships(density, grid, cfg), 0.0)

    jj, ii = np.nonzero(E > 0)
    e_cell = E[jj, ii]
    centre_area = area_id(grid.lon[ii], grid.lat[jj], cfg, grid)
    areas, inv = np.unique(centre_area, return_inverse=True)
    e_area = np.bincount(inv, weights=e_cell)
    n_extra = t["pool_size"] - t["pool_min_per_area"] * len(areas)
    share = max(n_extra, 0) * e_area / e_area.sum()
    n_pts = t["pool_min_per_area"] + np.floor(share).astype(int)
    short = t["pool_size"] - n_pts.sum()                          # hand the rounding remainder to the largest fractions
    if short > 0:
        n_pts[np.argsort(-(share - np.floor(share)))[:short]] += 1

    order = np.argsort(inv, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(areas)))])
    lon_l, lat_l, cj_l, ci_l, aid_l = [], [], [], [], []
    for a in range(len(areas)):
        cells = order[bounds[a]:bounds[a + 1]]
        p = e_cell[cells] / e_cell[cells].sum()
        pick = cells[rng.choice(len(cells), size=n_pts[a], p=p)]
        lon = grid.lon[ii[pick]].copy()
        lat = grid.lat[jj[pick]].copy()
        todo = np.ones(len(pick), bool)
        for _ in range(t["pool_jitter_tries"]):
            if not todo.any():
                break
            k = np.nonzero(todo)[0]
            tl = grid.lon[ii[pick[k]]] + (rng.random(len(k)) - 0.5) * grid.dlon_deg
            ta = grid.lat[jj[pick[k]]] + (rng.random(len(k)) - 0.5) * grid.dlat_deg
            ok = ((area_id(tl, ta, cfg, grid) == areas[a]) & (_bilinear(grid.depth, grid, tl, ta) >= t["min_ship_depth_m"])
                  & (_bilinear(grid.coast_dist, grid, tl, ta) <= t["pool_max_coast_dist_m"]))
            lon[k[ok]], lat[k[ok]] = tl[ok], ta[ok]
            todo[k[ok]] = False                                  # points never placed keep the cell centre
        lon_l.append(lon); lat_l.append(lat)
        cj_l.append(jj[pick]); ci_l.append(ii[pick]); aid_l.append(np.full(len(pick), areas[a]))
    lon, lat = np.concatenate(lon_l), np.concatenate(lat_l)
    cj, ci, aid = np.concatenate(cj_l).astype(np.int32), np.concatenate(ci_l).astype(np.int32), np.concatenate(aid_l)
    n_ships = len(lon)
    w_area = dict(zip(areas, e_area / n_pts))
    weight = np.array([w_area[a] for a in aid], np.float32)

    gt = gauges_mod.load_gauges(grid, cfg)
    ng = len(gt)
    cat = lambda a, b, dt: np.concatenate([np.asarray(a, dt), np.asarray(b, dt)])
    pool = ReceiverPool(
        lon=cat(lon, gt["lon_deg"].values, np.float64), lat=cat(lat, gt["lat_deg"].values, np.float64),
        depth=cat(_bilinear(grid.depth, grid, lon, lat), grid.depth[gt["cell_j"], gt["cell_i"]], np.float32),
        lane_dir=cat(lanes[cj, ci], np.full(ng, np.nan), np.float32),
        rtype=cat(np.full(n_ships, SHIP), gt["rtype"].values, np.int8),
        code=cat(np.full(n_ships, ""), gt["code"].values, "<U24"),
        cell_j=cat(cj, gt["cell_j"].values, np.int32), cell_i=cat(ci, gt["cell_i"].values, np.int32),
        coast_dist=cat(_bilinear(grid.coast_dist, grid, lon, lat), grid.coast_dist[gt["cell_j"], gt["cell_i"]], np.float32),
        weight=cat(weight, np.zeros(ng), np.float32), area_id=cat(aid, np.full(ng, -1), np.int32))
    if save:
        path = repo_path(cfg["paths"]["receiver_pool"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(pool, f.name) for f in fields(ReceiverPool)})
    return pool


def load_pool(cfg):
    with np.load(repo_path(cfg["paths"]["receiver_pool"])) as z:
        return ReceiverPool(**{f.name: z[f.name] for f in fields(ReceiverPool)})


# ----------------------------------------------------------------------------- traffic draws

def sample_traffic(pool, cfg, rng, seed=-1):
    """One traffic draw: Poisson counts per area, points chosen without replacement in proportion to weight.

    N_a ~ Poisson(sum of pool weight in area a), capped at the number of pool points in the area. Within an area the
    points are chosen by the Gumbel top-k trick, which is exactly sampling without replacement with probability
    proportional to weight. Heading is lane_dir or lane_dir + pi with equal probability. Speed is uniform in
    traffic.speed_kn (PLACEHOLDER until T05); ships slower than min_speed_kn are dropped.
    """
    t = cfg["traffic"]
    ships = np.nonzero(pool.rtype == SHIP)[0]
    aid, w = pool.area_id[ships], pool.weight[ships].astype(np.float64)
    areas, inv = np.unique(aid, return_inverse=True)
    lam = np.bincount(inv, weights=w)
    n_pool = np.bincount(inv)
    n_draw = np.minimum(rng.poisson(lam), n_pool)
    key = np.log(w) + rng.gumbel(size=len(ships))
    order = np.lexsort((-key, inv))                              # by area, then descending key
    start = np.concatenate([[0], np.cumsum(n_pool)[:-1]])
    rank = np.arange(len(ships)) - start[inv[order]]
    chosen = np.sort(ships[order][rank < n_draw[inv[order]]])
    S = len(chosen)
    flip = rng.random(S) < 0.5
    heading = np.mod(pool.lane_dir[chosen].astype(np.float64) + np.where(flip, np.pi, 0.0), 2 * np.pi)
    lo, hi = t["speed_kn"]
    sog = knots_to_ms(rng.uniform(lo, hi, S))
    fast = sog >= knots_to_ms(t["min_speed_kn"])
    return TrafficDraw(chosen[fast].astype(np.int64), heading[fast].astype(np.float32), sog[fast].astype(np.float32), int(seed))


def traffic_draw(pool, cfg, index, purpose=1):
    """Reproducible draw number `index` (purpose 1 = traffic, 5 = quiet-day traffic), seeded as in docs/interfaces.md."""
    return sample_traffic(pool, cfg, np.random.default_rng([cfg["meta"]["seed"], purpose, index]), seed=index)


def apply_range_cap(draw, pool, cap_m):
    """Subset of the draw with coast_dist <= cap_m (the 40 km fleet is always inside the 100 km fleet)."""
    keep = pool.coast_dist[draw.pool_idx] <= cap_m
    return TrafficDraw(draw.pool_idx[keep], draw.heading[keep], draw.sog[keep], draw.seed)
