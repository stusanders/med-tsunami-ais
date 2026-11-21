"""Slow, readable, single-scenario linear shallow-water solver (T02).

Spherical Arakawa C grid, forward-backward stepping, no Coriolis, no dispersion.
Layout and equations: docs/interfaces.md section 3. This is the reference for
the batched solver (T09) and the serial baseline of the benchmark (T19).
Plain NumPy, float32 fields, one Python loop over time.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GridOps:
    """Everything the time step needs, precomputed once from the grid."""
    hu: np.ndarray       # (ny, nx+1) depth on u faces, 0 on closed faces
    hv: np.ndarray       # (ny+1, nx) depth on v faces, 0 on closed faces
    mu: np.ndarray       # (ny, nx+1) 1 on open u faces
    mv: np.ndarray       # (ny+1, nx) 1 on open v faces
    wet: np.ndarray      # (ny, nx) 1 on wet cells
    dx: np.ndarray       # (ny, 1) m
    dy: float            # m
    cos_c: np.ndarray    # (ny, 1)
    cos_v: np.ndarray    # (ny+1, 1)
    damp_c: np.ndarray   # (ny, nx) sponge factor per step at centres (1 = no damping)
    damp_u: np.ndarray   # (ny, nx+1)
    damp_v: np.ndarray   # (ny+1, nx)


def face_depths(grid):
    """Face depth = mean of the two neighbouring cells if both wet, else 0 (closed).

    Outer domain faces are closed.
    """
    h = grid.depth.astype(np.float64)
    wet = grid.wet
    ny, nx = h.shape
    hu = np.zeros((ny, nx + 1))
    open_u = wet[:, :-1] & wet[:, 1:]
    hu[:, 1:-1] = np.where(open_u, 0.5 * (h[:, :-1] + h[:, 1:]), 0.0)
    hv = np.zeros((ny + 1, nx))
    open_v = wet[:-1, :] & wet[1:, :]
    hv[1:-1, :] = np.where(open_v, 0.5 * (h[:-1, :] + h[1:, :]), 0.0)
    return hu, hv


def time_step(grid, cfg):
    """dt from CFL, shrunk so the sample interval is a whole number of steps."""
    g = cfg["physics"]["g_m_s2"]
    s = cfg["solver"]
    dt_cfl = s["cfl"] * min(float(np.min(grid.dx)), float(grid.dy)) / np.sqrt(g * float(grid.depth.max()))
    n_sub = int(np.ceil(s["sample_interval_s"] / dt_cfl))
    return s["sample_interval_s"] / n_sub, n_sub


def _sponge_rate(dist_cells, width, max_rate):
    """Quadratic ramp: 0 at `width` cells from the edge, max_rate at the edge."""
    if width <= 0:
        return np.zeros_like(dist_cells)
    frac = np.clip((width - dist_cells) / width, 0.0, 1.0)
    return max_rate * frac**2


EDGES = ("west", "east", "south", "north")


def _edge_distance(jj, ii, ny, nx, edges):
    """Distance (cells) from the nearest of the listed outer edges; edge cell = 0, inf if no edges."""
    d = np.full(np.broadcast(jj, ii).shape, np.inf)
    for e in edges:
        assert e in EDGES, e
        d = np.minimum(d, {"west": ii, "east": nx - 1 - ii, "south": jj, "north": ny - 1 - jj}[e])
    return np.maximum(d, 0.0)


def sponge_factors(shape_c, dt, cfg):
    """exp(-gamma dt) at centres, u faces and v faces, on the edges in solver.sponge_edges."""
    ny, nx = shape_c
    width = cfg["solver"]["sponge_width_cells"]
    rate = cfg["solver"]["sponge_max_rate_per_s"]
    edges = cfg["solver"].get("sponge_edges", EDGES)
    jc, ic = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ju, iu = np.meshgrid(np.arange(ny), np.arange(nx + 1) - 0.5, indexing="ij")
    jv, iv = np.meshgrid(np.arange(ny + 1) - 0.5, np.arange(nx), indexing="ij")
    dists = [_edge_distance(j, i, ny, nx, edges) for j, i in ((jc, ic), (ju, iu), (jv, iv))]
    return tuple(np.exp(-_sponge_rate(np.minimum(d, 1e9), width, rate) * dt) for d in dists)


def sponge_mask(shape_c, cfg):
    """(ny, nx) bool: cells where the sponge damps (no receivers, gauges or buoys allowed there)."""
    ny, nx = shape_c
    jc, ic = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    d = _edge_distance(jc, ic, ny, nx, cfg["solver"].get("sponge_edges", EDGES))
    return d < cfg["solver"]["sponge_width_cells"]


def grid_ops(grid, cfg, dtype=np.float32):
    dt, _ = time_step(grid, cfg)
    hu, hv = face_depths(grid)
    damp_c, damp_u, damp_v = sponge_factors(grid.depth.shape, dt, cfg)
    f = lambda a: np.asarray(a, dtype=dtype)
    return GridOps(hu=f(hu), hv=f(hv), mu=f(hu > 0), mv=f(hv > 0), wet=f(grid.wet),
                   dx=f(grid.dx[:, None]), dy=float(grid.dy), cos_c=f(grid.cos_c[:, None]),
                   cos_v=f(grid.cos_v[:, None]), damp_c=f(damp_c), damp_u=f(damp_u), damp_v=f(damp_v))


def step(eta, u, v, ops, dt, g):
    """One forward-backward step: eta from the old velocities, then u, v from the new eta."""
    flux_u = ops.hu * u                     # (ny, nx+1)
    flux_v = ops.hv * v * ops.cos_v         # (ny+1, nx)
    div = (flux_u[:, 1:] - flux_u[:, :-1]) / ops.dx \
        + (flux_v[1:, :] - flux_v[:-1, :]) / (ops.dy * ops.cos_c)
    eta = (eta - dt * div) * ops.wet

    u = u.copy()
    v = v.copy()
    u[:, 1:-1] -= dt * g * (eta[:, 1:] - eta[:, :-1]) / ops.dx
    v[1:-1, :] -= dt * g * (eta[1:, :] - eta[:-1, :]) / ops.dy
    u *= ops.mu
    v *= ops.mv

    return eta * ops.damp_c, u * ops.damp_u, v * ops.damp_v


# ---------------------------------------------------------------- receivers

@dataclass
class Sampler:
    idx: np.ndarray   # (R, 4) int64 flat indices into ny*nx
    w: np.ndarray     # (R, 4) float32 weights


def make_sampler(grid, pool):
    """Bilinear weights between 4 centres for ships; the snapped cell for gauges and buoys."""
    ny, nx = grid.depth.shape
    fi = (np.asarray(pool.lon) - grid.lon[0]) / grid.dlon_deg
    fj = (np.asarray(pool.lat) - grid.lat[0]) / grid.dlat_deg
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, ny - 2)
    a = np.clip(fi - i0, 0.0, 1.0)
    b = np.clip(fj - j0, 0.0, 1.0)
    idx = np.stack([j0 * nx + i0, j0 * nx + i0 + 1, (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1], axis=1)
    w = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], axis=1)
    snapped = np.asarray(pool.rtype) > 0
    cell = np.asarray(pool.cell_j, np.int64) * nx + np.asarray(pool.cell_i, np.int64)
    idx[snapped] = cell[snapped, None]
    w[snapped] = [1.0, 0.0, 0.0, 0.0]
    return Sampler(idx=idx, w=w.astype(np.float32))


def centre_velocities(u, v):
    """Average face velocities to cell centres (works for (..., ny, nx+1) and (..., ny+1, nx))."""
    return 0.5 * (u[..., :-1] + u[..., 1:]), 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def sample(eta, u, v, sampler):
    """(R, 3) = [eta, u, v] at every receiver."""
    uc, vc = centre_velocities(u, v)
    fields = np.stack([eta.ravel(), uc.ravel(), vc.ravel()])          # (3, ny*nx)
    vals = (fields[:, sampler.idx] * sampler.w[None]).sum(axis=-1)    # (3, R)
    return vals.T


def run_reference(grid, eta0, sampler, cfg, t_end_s=None, sample_interval_s=None):
    """Run one scenario. Returns rec30 (T, R, 3), sample k at t = interval * (k+1).

    t_end_s and sample_interval_s override config (tests only).
    """
    if sample_interval_s is not None:
        cfg = {**cfg, "solver": {**cfg["solver"], "sample_interval_s": sample_interval_s}}
    t_end = cfg["solver"]["t_end_s"] if t_end_s is None else t_end_s
    g = np.float32(cfg["physics"]["g_m_s2"])
    dt, n_sub = time_step(grid, cfg)
    n_samples = int(round(t_end / cfg["solver"]["sample_interval_s"]))
    ops = grid_ops(grid, cfg)
    ny, nx = grid.depth.shape
    eta = np.asarray(eta0, np.float32) * ops.wet
    u = np.zeros((ny, nx + 1), np.float32)
    v = np.zeros((ny + 1, nx), np.float32)
    dt = np.float32(dt)
    out = np.empty((n_samples, sampler.idx.shape[0], 3), np.float32)
    for k in range(n_samples):
        for _ in range(n_sub):
            eta, u, v = step(eta, u, v, ops, dt, g)
        out[k] = sample(eta, u, v, sampler)
    return out


def run_fields(grid, eta0, cfg, n_steps):
    """Run n_steps and return the final (eta, u, v) (for volume and snapshot tests)."""
    g = np.float32(cfg["physics"]["g_m_s2"])
    dt, _ = time_step(grid, cfg)
    ops = grid_ops(grid, cfg)
    ny, nx = grid.depth.shape
    eta = np.asarray(eta0, np.float32) * ops.wet
    u = np.zeros((ny, nx + 1), np.float32)
    v = np.zeros((ny + 1, nx), np.float32)
    for _ in range(n_steps):
        eta, u, v = step(eta, u, v, ops, np.float32(dt), g)
    return eta, u, v


def volume(eta, grid):
    """Displaced water volume, m^3 (cell area = dx * dy)."""
    return float((eta.astype(np.float64) * grid.dx[:, None] * grid.dy).sum())
