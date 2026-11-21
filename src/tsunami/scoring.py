"""Matched-filter network scoring (T15). PLAN.md Phase 4 "Scoring"; docs/interfaces.md section 10.

One traffic draw at a time. K = library candidates, M = cell groups, S = active ships,
E = events x noise draws, T = minutes.

  1. gather the active ships from the library on the CPU and project onto each ship's OWN normal
  2. sign-align to the group normal (25 km cell x 45 deg folded-normal bin): only the sign comes
     from the bin centre, never the projection
  3. average aligned observations and templates identically per group; weight by
     w = 1 / (sigma_shared^2 + sum sigma_i^2 / N^2), empty groups masked
  4. gd_t = cumsum_t sum_m w G D ; gg_t = cumsum_t sum_m w G^2   (batched matmul + prefix sum)
  5. S = max(0, gd)^2 / (gg + eps); Aegean candidates also scored flipped (-gd, same gg)
  6. magnitude and location from L = 2 gd - gg; top three by L
  7. score_loop_reference: the same with explicit loops (equivalence test, benchmark)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend
from .geometry import cell_index, fold_axis, heading_normal, project_normal
from .units import deg2rad


# ------------------------------------------------------------------ groups

@dataclass
class Groups:
    group: np.ndarray      # (S,) int64 group id per ship
    sign: np.ndarray       # (S,) float32 +1 / -1
    n_groups: int          # M
    normal: np.ndarray     # (M,) float64 group normal (compass bearing, rad); 0 for GPS
    key: np.ndarray        # (M, 3) int64 (cell_j, cell_i, bin)


def build_groups(draw, pool, cfg, operator):
    """Cell groups for one traffic draw. AIS: (25 km cell, 45 deg folded-normal bin); GPS: cell only."""
    idx = np.asarray(draw.pool_idx)
    lon, lat = np.asarray(pool.lon)[idx], np.asarray(pool.lat)[idx]
    sc, d = cfg["scoring"], cfg["domain"]
    cj, ci = cell_index(lon, lat, sc["cell_m"], sc["cell_ref_lat_deg"], d["lon_min_deg"], d["lat_min_deg"],
                        cfg["physics"]["earth_radius_m"])
    S = len(idx)
    if operator == "ais":
        width = deg2rad(sc["normal_bin_deg"])
        n_bins = int(round(np.pi / width))
        theta = np.asarray(draw.heading, np.float64)
        axis = fold_axis(theta + np.pi / 2)                    # ship normal as an undirected axis
        b = np.minimum((axis // width).astype(np.int64), n_bins - 1)
    elif operator == "gps":
        b = np.zeros(S, np.int64)
    else:
        raise ValueError(operator)
    keys = np.stack([cj, ci, b], axis=1).astype(np.int64)
    ukeys, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()
    if operator == "ais":
        normal = (ukeys[:, 2] + 0.5) * width
        ne, nn = heading_normal(theta)                          # each ship's own normal vector
        ge, gn = np.sin(normal[group]), np.cos(normal[group])   # group normal vector (compass bearing)
        sign = np.where(ne * ge + nn * gn >= 0, 1.0, -1.0).astype(np.float32)
    else:
        normal = np.zeros(len(ukeys))
        sign = np.ones(S, np.float32)
    return Groups(group=group.astype(np.int64), sign=sign, n_groups=len(ukeys), normal=normal, key=ukeys)


def _membership(groups, xp):
    """(M, S) matrix with sign_i in row group_i: one matmul does sign alignment and group sums."""
    A = np.zeros((groups.n_groups, len(groups.group)), np.float32)
    A[groups.group, np.arange(len(groups.group))] = groups.sign
    return xp.asarray(A)


def _counts(groups, valid, xp):
    """N (M, T): valid ships per group and minute."""
    A = xp.abs(_membership(groups, xp))
    return A @ xp.asarray(valid, np.float32)


def aggregate(x, groups, valid=None, xp=None):
    """(..., S, T) -> (..., M, T): mean over valid ships of sign_i * x_i; 0 where a group is empty.

    Applied identically to observations and templates.
    """
    xp = xp or backend.array_module(x)
    x = xp.asarray(x, np.float32)
    T = x.shape[-1]
    if valid is None:
        valid = np.ones((len(groups.group), T), bool)
    v = xp.asarray(valid, np.float32)
    A = _membership(groups, xp)
    sums = xp.matmul(A, x * v)                                  # broadcasts over leading dims
    n = _counts(groups, valid, xp)
    return xp.where(n > 0, sums / xp.maximum(n, 1.0), 0.0).astype(np.float32)


def cell_weights(groups, sigma_i, sigma_shared, valid=None, xp=np):
    """w (M, T) = 1 / (sigma_shared^2 + sum_valid sigma_i^2 / N^2), 0 where N = 0."""
    S = len(groups.group)
    if valid is None:
        valid = np.ones((S, 1), bool)
    A = xp.abs(_membership(groups, xp))
    v = xp.asarray(valid, np.float32)
    n = A @ v
    s2 = A @ (xp.asarray(sigma_i, np.float32)[:, None] ** 2 * v)
    var = sigma_shared**2 + s2 / xp.maximum(n, 1.0) ** 2
    return xp.where(n > 0, 1.0 / var, 0.0).astype(np.float32)


# ------------------------------------------------------------------ library access

class Library:
    """Read-only view of a library-format directory (docs/interfaces.md section 6).

    lib[ids] -> (k, 120, R, 3) float32 numpy, reading only the chunk files that hold `ids`.
    """

    def __init__(self, directory):
        self.dir = Path(directory)
        self.index = pd.read_csv(self.dir / "index.csv").sort_values("run_id").reset_index(drop=True)
        self._mm = {}

    def _chunk(self, c):
        if c not in self._mm:
            self._mm[c] = np.load(self.dir / f"rec_{c:04d}.npy", mmap_mode="r")
        return self._mm[c]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ids):
        ids = np.atleast_1d(ids)
        rows = self.index.set_index("run_id").loc[ids]
        out = None
        for n, (c, r) in enumerate(zip(rows["chunk"].to_numpy(), rows["row"].to_numpy())):
            a = self._chunk(int(c))[int(r)]
            if out is None:
                out = np.empty((len(ids),) + a.shape, np.float32)
            out[n] = a
        return out


def project_templates(block, draw, operator):
    """(k, 120, R or S, 3) -> (k, S, 120) on each ship's own normal (AIS) or eta (GPS)."""
    idx = np.asarray(draw.pool_idx)
    sub = block[:, :, idx, :] if block.shape[2] != len(idx) else block
    if operator == "ais":
        theta = np.asarray(draw.heading)[None, None, :]
        g = project_normal(sub[..., 1], sub[..., 2], theta)
    else:
        g = sub[..., 0]
    return np.ascontiguousarray(np.swapaxes(g, 1, 2), dtype=np.float32)


def aggregate_templates(lib_recf, cand_idx, draw, groups, valid, operator, chunk=32):
    """G (K, M, 120) numpy: CPU gather + project + aggregate, chunked over candidates.

    lib_recf: anything indexable as lib[ids] -> (k, 120, R, 3) (a Library, a memmap or an array).
    """
    cand_idx = np.asarray(cand_idx)
    out = None
    for s in range(0, len(cand_idx), chunk):
        ids = cand_idx[s:s + chunk]
        block = np.asarray(lib_recf[ids], np.float32)
        Gc = aggregate(project_templates(block, draw, operator), groups, valid, xp=np)
        if out is None:
            out = np.empty((len(cand_idx),) + Gc.shape[1:], np.float32)
        out[s:s + len(ids)] = Gc
    return out


# ------------------------------------------------------------------ statistics

def cumulative_stats(G, D, w, xp=None):
    """gd (E, K, T) = cumsum_t sum_m w G D ; gg (K, T) = cumsum_t sum_m w G^2.

    The sum over cells is one batched matmul per minute; the sum over minutes is a prefix scan.
    """
    xp = xp or backend.xp
    G, D, w = (xp.asarray(a, np.float32) for a in (G, D, w))
    Gw = G * w                                                  # (K, M, T)
    inc = xp.matmul(Gw.transpose(2, 0, 1), D.transpose(2, 1, 0))  # (T, K, E)
    gd = xp.cumsum(inc.transpose(2, 1, 0).astype(np.float64), axis=-1)
    gg = xp.cumsum((Gw * G).sum(axis=1).astype(np.float64), axis=-1)
    return gd, gg


def expand_flips(gd, gg, flip_allowed):
    """Append flipped (subsidence-first) versions of flip_allowed candidates: -gd, same gg."""
    xp = backend.array_module(gd)
    fl = np.flatnonzero(np.asarray(flip_allowed, bool))
    K = gg.shape[0]
    gd2 = xp.concatenate([gd, -gd[:, fl]], axis=1)
    gg2 = xp.concatenate([gg, gg[fl]], axis=0)
    parent = np.concatenate([np.arange(K), fl])
    polarity = np.concatenate([np.ones(K, np.int8), -np.ones(len(fl), np.int8)])
    return gd2, gg2, parent, polarity


def detection_scores(gd2, gg2, eps):
    """S (E, K2, T) = max(0, gd)^2 / (gg + eps): anti-correlated fits score 0, no 0/0 early on."""
    xp = backend.array_module(gd2)
    return xp.maximum(gd2, 0.0) ** 2 / (gg2[None] + eps)


def detection_statistic(S):
    """stat (E, T) = max over candidates; argk (E, T)."""
    return S.max(axis=1), S.argmax(axis=1)


def log_likelihood(gd2, gg2):
    """Fixed-amplitude log-likelihood L (E, K2, T) = 2 gd - gg (up to a constant)."""
    return 2.0 * gd2 - gg2[None]


def calibrate_threshold(stat_quiet, far):
    """(1 - far) quantile of the window maximum on quiet days."""
    xp = backend.array_module(stat_quiet)
    return float(np.quantile(backend.to_numpy(stat_quiet.max(axis=-1)), 1.0 - far))


def confirmation_minute(stat, thr):
    """(E,) first minute index with stat > thr, -1 if never. Confirmation time = 60 (m+1) s."""
    st = backend.to_numpy(stat)
    above = st > thr
    return np.where(above.any(axis=-1), above.argmax(axis=-1), -1)


def estimate_source(L, t, cand_table, parent, polarity, top_n):
    """Best and top_n candidates by L at minute t[e] (last minute if t[e] < 0).

    Returns one row per (event, rank): e, rank, k2, cand_id, polarity, lon_deg, lat_deg, mw, L.
    """
    L = backend.to_numpy(L)
    E, K2, T = L.shape
    rows = []
    for e in range(E):
        te = int(t[e]) if t[e] >= 0 else T - 1
        order = np.argsort(-L[e, :, te])[:top_n]
        for rank, k2 in enumerate(order):
            c = cand_table.iloc[int(parent[k2])]
            rows.append(dict(e=e, rank=rank, k2=int(k2), cand_id=int(c["cand_id"]), polarity=int(polarity[k2]),
                             lon_deg=float(c["lon_deg"]), lat_deg=float(c["lat_deg"]), mw=float(c["mw"]),
                             minute=te, L=float(L[e, k2, te])))
    return pd.DataFrame(rows)


def score_tensor(d, g, groups, sigma_i, sigma_shared, flip_allowed, eps, valid=None, xp=None):
    """Whole tensor path from per-ship arrays: d (E, S, T), g (K, S, T) -> (gd2, gg2, S, stat, parent, polarity)."""
    xp = xp or backend.xp
    D = aggregate(xp.asarray(d), groups, valid, xp=xp)
    G = aggregate(xp.asarray(g), groups, valid, xp=xp)
    w = cell_weights(groups, sigma_i, sigma_shared, valid, xp=xp)
    gd, gg = cumulative_stats(G, D, w, xp=xp)
    gd2, gg2, parent, polarity = expand_flips(gd, gg, flip_allowed)
    S = detection_scores(gd2, gg2, eps)
    stat, _ = detection_statistic(S)
    return gd2, gg2, S, stat, parent, polarity


# ------------------------------------------------------------------ loop reference

def score_loop_reference(d, g, groups, sigma_i, sigma_shared, flip_allowed, eps, valid=None):
    """Same outputs as score_tensor, written with explicit Python loops (slow; tests and benchmark)."""
    d, g = np.asarray(d, np.float64), np.asarray(g, np.float64)
    E, S, T = d.shape
    K = g.shape[0]
    M = groups.n_groups
    if valid is None:
        valid = np.ones((S, T), bool)
    members = [[] for _ in range(M)]
    for i in range(S):
        members[groups.group[i]].append(i)
    flips = [k for k in range(K) if flip_allowed[k]]
    K2 = K + len(flips)
    parent = np.array(list(range(K)) + flips)
    polarity = np.array([1] * K + [-1] * len(flips), np.int8)
    gd2 = np.zeros((E, K2, T))
    gg2 = np.zeros((K2, T))
    for t in range(T):
        for m in range(M):
            live = [i for i in members[m] if valid[i, t]]
            if not live:
                continue
            n = len(live)
            var = sigma_shared**2 + sum(float(sigma_i[i]) ** 2 for i in live) / n**2
            w = 1.0 / var
            for k2 in range(K2):
                k = parent[k2]
                gm = polarity[k2] * sum(groups.sign[i] * g[k, i, t] for i in live) / n
                gg2[k2, t] += w * gm * gm
                for e in range(E):
                    dm = sum(groups.sign[i] * d[e, i, t] for i in live) / n
                    gd2[e, k2, t] += w * gm * dm
        if t > 0:
            gd2[:, :, t] += gd2[:, :, t - 1]
            gg2[:, t] += gg2[:, t - 1]
    S_ = np.zeros_like(gd2)
    stat = np.zeros((E, T))
    for e in range(E):
        for t in range(T):
            best = 0.0
            for k2 in range(K2):
                s = max(0.0, gd2[e, k2, t]) ** 2 / (gg2[k2, t] + eps)
                S_[e, k2, t] = s
                best = max(best, s)
            stat[e, t] = best
    return gd2, gg2, S_, stat, parent, polarity
