"""Earthquake sources as Gaussian dipoles (T03). docs/interfaces.md section 7.

Arc candidates: every 25 km along the shallow (trench) edge of the EFSM20 subduction-interface
meshes (Hellenic GRSS001, Calabrian ITSS001, Cyprus CYSS001), with strike from the edge and dip
from the shallow part of the mesh. Aegean candidates: a coarse crustal grid (about 50 km) kept away
from the Samos 2020 and Amorgos 1956 epicentres.

Source shape: Gaussian uplift elongated along strike, plus a smaller Gaussian subsidence lobe
down-dip (landward), so horizontal currents see a realistic dipole. Rupture length and width from a
magnitude scaling relation (interface: Strasser et al. 2010; crustal: Wells & Coppersmith 1994),
slip from M0 = mu L W D, peak uplift = slip x sin(mean dip) (arcs: EFSM20 dipavg), subsidence = ratio x
uplift (PLACEHOLDER). The source SHAPE uses dip_deg (unchanged since the T14 library build); only the
amplitude uses uplift_dip_deg, so library templates rescale exactly by uplift_scale().

Angles in the tables are degrees (strike_deg, dip_deg) and converted with units.deg2rad.
Strike follows Aki-Richards: dip direction = strike + 90 deg.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import magnitudes, repo_path
from .units import deg2rad, km_to_m, rad2deg

REGIONS = ("hellenic", "calabrian", "cyprus", "aegean")


# ------------------------------------------------------------------ local geometry

def _enu(lon, lat, lon0, lat0, R):
    """East and north offsets (m) of (lon, lat) from (lon0, lat0), local equirectangular."""
    east = R * np.cos(deg2rad(lat0)) * deg2rad(np.asarray(lon) - lon0)
    north = R * deg2rad(np.asarray(lat) - lat0)
    return east, north


def _move(lon0, lat0, east, north, R):
    """Point displaced by (east, north) metres from (lon0, lat0)."""
    return lon0 + rad2deg(east / (R * np.cos(deg2rad(lat0)))), lat0 + rad2deg(north / R)


def _bearing_vec(bearing_deg):
    b = deg2rad(bearing_deg)
    return np.sin(b), np.cos(b)


# ------------------------------------------------------------------ scaling

def rupture_size(mw, region, cfg):
    """(length_m, width_m, slip_m) from the scaling relation for the region."""
    s = cfg["sources"]
    sc = s["crustal_scaling"] if region == "aegean" else s["interface_scaling"]
    L = km_to_m(10 ** (sc["length_a"] + sc["length_b"] * mw))
    W = km_to_m(10 ** (sc["width_a"] + sc["width_b"] * mw))
    if region == "aegean":
        W = min(W, s["crustal_max_width_m"])
    m0 = 10 ** (s["moment_b"] * mw + s["moment_a"])
    slip = m0 / (s["rigidity_pa"] * L * W)
    return float(L), float(W), float(slip)


# ------------------------------------------------------------------ EFSM20 arcs

@lru_cache(maxsize=1)
def _load_meshes(path):
    feats = json.load(open(path))["features"]
    by = {}
    for f in feats:
        p = f["properties"]
        by.setdefault(p["idfs"], []).append(p)
    return by


def arc_mesh(idfs, cfg):
    """Vertices (n, 3) [lon, lat, depth_m positive down] and triangle table for one EFSM20 source."""
    tris = _load_meshes(str(repo_path(cfg["sources"]["arc_geometry"])))[idfs]
    verts = {}
    for p in tris:
        for k in (1, 2, 3):
            verts[p[f"idvertex{k}"]] = (p[f"lon{k}"], p[f"lat{k}"], -p[f"depth{k}"])   # depths to positive down
    v = np.array(list(verts.values()), dtype=float)
    t = pd.DataFrame([{"lon_c": p["lon_c"], "lat_c": p["lat_c"], "depth_c_m": -p["depth_c"],
                       "strike": p["strike"], "dip": p["dip"], "area": p["area"]} for p in tris])
    return v, t


def trench_edge(idfs, cfg):
    """Shallowest-vertex polyline of the mesh, ordered along the arc: (n, 2) lon, lat."""
    R = cfg["physics"]["earth_radius_m"]
    v, _ = arc_mesh(idfs, cfg)
    top = v[v[:, 2] <= v[:, 2].min() + 1.0][:, :2]
    lon0, lat0 = top.mean(axis=0)
    e, n = _enu(top[:, 0], top[:, 1], lon0, lat0, R)
    xy = np.stack([e, n], 1)
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    start = int(np.unravel_index(d.argmax(), d.shape)[0])        # one end of the arc
    order, used = [start], {start}
    for _ in range(len(top) - 1):                                 # greedy nearest-neighbour chain
        dd = d[order[-1]].copy()
        dd[list(used)] = np.inf
        nxt = int(dd.argmin())
        order.append(nxt)
        used.add(nxt)
    return top[order]


def arc_positions(region, cfg):
    """Trench points every arc_spacing_m along the edge, with strike (deg) and local dip (deg)."""
    s = cfg["sources"]
    R = cfg["physics"]["earth_radius_m"]
    idfs = s["arcs"][region]
    edge = trench_edge(idfs, cfg)
    lon0, lat0 = edge.mean(axis=0)
    e, n = _enu(edge[:, 0], edge[:, 1], lon0, lat0, R)
    k = 3                                                         # light smoothing of the vertex chain
    pad = lambda a: np.r_[np.full(k, a[0]), a, np.full(k, a[-1])]
    es = np.convolve(pad(e), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    ns = np.convolve(pad(n), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    seg = np.hypot(np.diff(es), np.diff(ns))
    cum = np.r_[0.0, np.cumsum(seg)]
    spacing = s["arc_spacing_m"]
    targets = np.arange(spacing / 2, cum[-1], spacing)
    pe, pn = np.interp(targets, cum, es), np.interp(targets, cum, ns)
    te, tn = np.gradient(pe), np.gradient(pn)
    _, tris = arc_mesh(idfs, cfg)
    ce, cn = _enu(tris.lon_c.to_numpy(), tris.lat_c.to_numpy(), lon0, lat0, R)
    shallow = tris.depth_c_m.to_numpy() <= s["arc_max_depth_for_dip_m"]
    tri_strike = deg2rad(tris.strike.to_numpy())
    rows = []
    for x, y, tx, ty in zip(pe, pn, te, tn):
        strike = np.rad2deg(np.arctan2(tx, ty)) % 360.0
        near = shallow & (np.hypot(ce - x, cn - y) < s["arc_local_radius_m"])
        if near.any():
            # orient the edge tangent like EFSM20's own Aki-Richards strikes (dip to the right = landward)
            ref = np.arctan2(np.sin(tri_strike[near]).mean(), np.cos(tri_strike[near]).mean())
            if np.cos(deg2rad(strike) - ref) < 0:
                strike = (strike + 180.0) % 360.0
        local = shallow & (np.hypot(ce - x, cn - y) < s["arc_local_radius_m"])
        dip = float(tris.dip.to_numpy()[local].mean()) if local.any() else float(tris.dip.median())
        lon, lat = _move(lon0, lat0, x, y, R)
        rows.append(dict(trench_lon_deg=lon, trench_lat_deg=lat, strike_deg=strike, dip_deg=dip))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Aegean grid

def aegean_positions(cfg, grid):
    s = cfg["sources"]
    R = cfg["physics"]["earth_radius_m"]
    box = s["aegean_box"]
    lon0, lat0 = box["lon_deg"][0], box["lat_deg"][0]
    ew, nw = _enu(box["lon_deg"][1], box["lat_deg"][1], lon0, lat0, R)
    sp = s["aegean_spacing_m"]
    rows = []
    for north in np.arange(sp / 2, nw, sp):
        for east in np.arange(sp / 2, ew, sp):
            lon, lat = _move(lon0, lat0, east, north, R)
            i = int(np.argmin(np.abs(grid.lon - lon)))
            j = int(np.argmin(np.abs(grid.lat - lat)))
            if not grid.wet[j, i] or grid.depth[j, i] < s["aegean_min_depth_m"]:
                continue
            far = all(np.hypot(*_enu(lon, lat, ep["lon_deg"], ep["lat_deg"], R)) >= s["aegean_exclusion_radius_m"]
                      for ep in s["exclusion_epicentres"].values())
            if far:
                rows.append(dict(trench_lon_deg=lon, trench_lat_deg=lat,
                                 strike_deg=s["aegean_strike_deg"], dip_deg=s["aegean_dip_deg"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ table

@lru_cache(maxsize=1)
def _arc_dipavg(path):
    feats = json.load(open(path))["features"]
    return {f["properties"]["idfs"]: float(f["properties"]["dipavg"]) for f in feats}


def rupture_mean_dip(region, dip, cfg):
    """Mean dip (deg) for the uplift rule: arcs = EFSM20 published mean interface dip (dipavg) of the source;
    crustal and historic sources = their fault dip."""
    if region in cfg["sources"]["arcs"]:
        return _arc_dipavg(str(repo_path(cfg["sources"]["arc_parameters"])))[cfg["sources"]["arcs"][region]]
    return float(dip)


def uplift_factor(row):
    """Peak uplift / slip = sin(mean dip)."""
    return float(np.sin(deg2rad(row.get("uplift_dip_deg", row["dip_deg"]))))


def uplift_scale(row, cfg):
    """Multiplier from the legacy fixed fraction (T14 library) to the current rule; exact because eta0 is linear in uplift."""
    return uplift_factor(row) / cfg["sources"]["legacy_uplift_fraction_of_slip"]


def _source_row(region, pos_id, trench_lon, trench_lat, strike, dip, mw, cfg):
    """One source: rupture size from Mw; uplift centre down-dip of the trench point (arcs only)."""
    R = cfg["physics"]["earth_radius_m"]
    L, W, slip = rupture_size(mw, region, cfg)
    if region == "aegean":
        lon, lat = trench_lon, trench_lat
    else:
        off = cfg["sources"]["uplift_centre_downdip_frac"] * W * np.cos(deg2rad(dip))
        dx, dy = _bearing_vec(strike + 90.0)
        lon, lat = _move(trench_lon, trench_lat, off * dx, off * dy, R)
    udip = rupture_mean_dip(region, dip, cfg)
    return dict(region=region, pos_id=pos_id, lon_deg=float(lon), lat_deg=float(lat),
                trench_lon_deg=float(trench_lon), trench_lat_deg=float(trench_lat),
                strike_deg=float(strike), dip_deg=float(dip), mw=float(mw),
                length_m=L, width_m=W, slip_m=slip, polarity=1, flip_allowed=region == "aegean",
                uplift_dip_deg=udip)


def _in_aoi(lon, lat, cfg):
    a = cfg["domain"]["area_of_interest"]
    return a["lon_deg"][0] <= lon <= a["lon_deg"][1] and a["lat_deg"][0] <= lat <= a["lat_deg"][1]


def candidate_table(cfg, grid=None, save=True):
    """Library candidates in library order (cand_id = row). Saved to paths.candidates."""
    if grid is None:
        from .grid import load_grid
        grid = load_grid(cfg)
    mags = magnitudes(cfg)
    rows, pos_id = [], 0
    for region in ("hellenic", "calabrian", "cyprus", "aegean"):
        pos = aegean_positions(cfg, grid) if region == "aegean" else arc_positions(region, cfg)
        region_mags = mags[mags <= cfg["sources"]["aegean_mw_max"] + 1e-9] if region == "aegean" else mags
        for p in pos.itertuples():
            if not _in_aoi(p.trench_lon_deg, p.trench_lat_deg, cfg):
                continue
            for mw in region_mags:
                rows.append(_source_row(region, pos_id, p.trench_lon_deg, p.trench_lat_deg,
                                        p.strike_deg, p.dip_deg, mw, cfg))
            pos_id += 1
    table = pd.DataFrame(rows)
    table.insert(0, "cand_id", np.arange(len(table)))
    if save:
        out = repo_path(cfg["paths"]["candidates"])
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
    return table


def perturb(row, along_strike_m, dmw, cfg):
    """A new source moved along strike (trench point and centre) with magnitude mw + dmw."""
    R = cfg["physics"]["earth_radius_m"]
    dx, dy = _bearing_vec(row["strike_deg"])
    lon, lat = _move(row["trench_lon_deg"], row["trench_lat_deg"], along_strike_m * dx, along_strike_m * dy, R)
    out = _source_row(row["region"], row["pos_id"], lon, lat, row["strike_deg"], row["dip_deg"],
                      row["mw"] + dmw, cfg)
    out["polarity"] = int(row.get("polarity", 1))
    return out


def custom_source(lon_deg, lat_deg, strike_deg, dip_deg, mw, fault, cfg, length_m=None, width_m=None,
                  slip_m=None, region="aegean", name=""):
    """Source from published parameters (T26). fault: 'thrust' (polarity +1) or 'normal' (-1, subsidence-first).

    (lon_deg, lat_deg) is the rupture centre. Missing dimensions come from the scaling relation for
    `region` ('aegean' = crustal); slip is then set from M0 = mu L W D unless given.
    """
    if fault not in ("thrust", "normal"):
        raise ValueError(fault)
    L, W, slip = rupture_size(mw, region, cfg)
    L = length_m or L
    W = width_m or W
    if slip_m is None:
        slip_m = 10 ** (cfg["sources"]["moment_b"] * mw + cfg["sources"]["moment_a"]) / (cfg["sources"]["rigidity_pa"] * L * W)
    return dict(region=region, pos_id=-1, lon_deg=float(lon_deg), lat_deg=float(lat_deg),
                trench_lon_deg=float(lon_deg), trench_lat_deg=float(lat_deg), strike_deg=float(strike_deg),
                dip_deg=float(dip_deg), mw=float(mw), length_m=float(L), width_m=float(W), slip_m=float(slip_m),
                polarity=1 if fault == "thrust" else -1, flip_allowed=False, name=name, uplift_dip_deg=float(dip_deg))


# ------------------------------------------------------------------ initial surface

def source_eta0(grid, row, cfg, kind=None, polarity=None):
    """Initial sea-surface displacement (ny, nx) float32, m (= seafloor uplift, instantaneous)."""
    s = cfg["sources"]
    R = cfg["physics"]["earth_radius_m"]
    kind = kind or s["kind"]
    polarity = row.get("polarity", 1) if polarity is None else polarity
    lon0, lat0 = float(row["lon_deg"]), float(row["lat_deg"])
    e, n = _enu(grid.lon[None, :], grid.lat[:, None], lon0, lat0, R)
    sx, sy = _bearing_vec(row["strike_deg"])
    along = e * sx + n * sy
    across = e * sy - n * sx                                       # positive down-dip (strike + 90 deg)
    w_h = row["width_m"] * np.cos(deg2rad(row["dip_deg"]))
    floor = s["min_sigma_m"]
    sig_a = max(s["gaussian_sigma_frac"] * row["length_m"], floor)
    sig_c = max(s["gaussian_sigma_frac"] * w_h, floor)
    up = uplift_factor(row) * row["slip_m"]
    eta = up * np.exp(-0.5 * ((along / sig_a) ** 2 + (across / sig_c) ** 2))
    if kind == "dipole":
        off = s["landward_offset_widths"] * max(w_h, floor)
        eta -= s["subsidence_ratio"] * up * np.exp(-0.5 * ((along / sig_a) ** 2 + ((across - off) / sig_c) ** 2))
    elif kind != "monopole":
        raise ValueError(kind)
    return (polarity * eta).astype(np.float32)


def dipole_volume(eta0, grid):
    """(uplift_m3, subsidence_m3, net_m3) over wet cells (cell area = dx * dy)."""
    area = grid.dx[:, None] * grid.dy
    e = np.where(grid.wet, eta0.astype(np.float64), 0.0)
    up = float((np.clip(e, 0, None) * area).sum())
    down = float((np.clip(e, None, 0) * area).sum())
    return up, down, up + down
