"""Full sweep (T17): every detector on every trial. Contract: docs/interfaces.md sections 10, 11 and 14.

Trial = (truth event, traffic draw, noise draw) in a stratum = (operator, range cap, weather regime).

Efficiency (the reshaping in PLAN.md section 0):
  * the library is read once per pass for ALL traffic draws at once (gather_pass), because reading it is the slow part;
  * the observation is signal + noise and scoring is linear in it, so the aggregated truth signal Ds (events, M, T) and the
    aggregated noise Dn (N, M, T) are built separately and added per batch (every event meets every noise draw);
  * thresholds come from quiet days (no solver) and are applied unchanged to the event trials of the same stratum.
"""
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend, gauges, noise, observe, scoring, ships
from .baseline_detector import alarm_minute, single_ship_stat
from .geometry import haversine_m

DETECTORS = ("network", "single_literal", "single_calibrated")
PURPOSE_TRAFFIC, PURPOSE_AIS, PURPOSE_GPS, PURPOSE_QTRAFFIC, PURPOSE_QNOISE = 1, 2, 3, 5, 6


@dataclass
class Fleet:
    """One traffic draw with its range-cap subsets and scoring groups."""
    purpose: int
    idx: int
    draw100: object
    draws: dict        # cap_m -> TrafficDraw
    masks: dict        # cap_m -> bool (S100,) positions of the capped fleet inside the largest fleet
    groups: dict       # (cap_m, operator) -> Groups


def make_fleet(pool, cfg, purpose, idx):
    caps = sorted(cfg["traffic"]["range_caps_m"])
    d_all = ships.traffic_draw(pool, cfg, idx, purpose)
    d_big = ships.apply_range_cap(d_all, pool, caps[-1])
    draws = {c: ships.apply_range_cap(d_big, pool, c) for c in caps}
    masks = {c: pool.coast_dist[d_big.pool_idx] <= c for c in caps}
    groups = {(c, op): scoring.build_groups(draws[c], pool, cfg, op) for c in caps for op in cfg["operators"]}
    return Fleet(purpose, idx, d_big, draws, masks, groups)


def sigmas(draw, regime, operator, cfg):
    """(sigma_i independent per ship (S,), sigma_shared, sigma_total per ship (S,))."""
    si = noise.ship_sigma(draw, regime, operator, cfg)
    ss = noise.shared_sigma(regime, operator, cfg)
    return si, ss, np.sqrt(si.astype(np.float64) ** 2 + ss ** 2).astype(np.float32)


# ------------------------------------------------------------------ uplift rescale

def load_scale(csv_path, id_col):
    """Per-run amplitude multipliers written by scripts/write_uplift_scales.py (uplift = slip x sin(mean dip)), indexed by run id.

    The stored runs were solved with the legacy fixed uplift fraction. The model is linear, so multiplying the saved arrays
    (block average, baseline filter and the gauge series are all linear) is exactly a re-run with the current rule.
    """
    s = pd.read_csv(csv_path)
    return pd.Series(s["uplift_scale"].values.astype(np.float64), index=s[id_col].values)


class ScaledLibrary:
    """scoring.Library whose runs are multiplied by their uplift scale at load time: lib[ids] -> scale[ids] * runs."""

    def __init__(self, lib, scale):
        self.lib, self.scale = lib, scale
        missing = set(lib.index["run_id"]) - set(scale.index)
        assert not missing, f"no uplift scale for runs {sorted(missing)[:5]}"

    @property
    def index(self):
        return self.lib.index

    def __len__(self):
        return len(self.lib)

    def __getitem__(self, ids):
        ids = np.atleast_1d(ids)
        k = self.scale.loc[ids].values.astype(np.float32)
        return self.lib[ids] * k[:, None, None, None]


# ------------------------------------------------------------------ library pass

def _key(f, cap, op):
    return f"{f.purpose}_{f.idx}_{int(cap)}_{op}"


def gather_pass(lib, ids, fleets, cfg, out_dir, tag, per_ship=False, chunk=None, valid_fn=None):
    """Aggregate library (or truth) runs for every fleet, cap and operator in ONE pass over the files.

    Writes out_dir/{tag}_{purpose}_{idx}_{cap}_{op}.npy with shape (len(ids), M, 120): the same operations as
    scoring.aggregate_templates (own-normal projection, sign alignment, group mean). If per_ship, also writes
    {tag}_ship_{purpose}_{idx}_{op}.npy (len(ids), S_big, 120): the projected signal at every ship of the largest fleet.
    Finished passes leave {tag}.done and are skipped on re-run. valid_fn(fleet, cap, op) -> (S, T) bool masks the reports that
    did not arrive (data-gap sensitivity); templates are then aggregated with the same mask as the observations.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"{tag}.done"
    if done.exists():
        return
    chunk = chunk or cfg["sweep"]["gather_chunk"]
    ids = np.asarray(ids)
    T = int(cfg["library"]["n_minutes"])
    ops = cfg["operators"]
    caps = sorted(cfg["traffic"]["range_caps_m"])
    mm = {}
    for f in fleets:
        for c, op in itertools.product(caps, ops):
            mm[("G", f.purpose, f.idx, c, op)] = np.lib.format.open_memmap(
                out_dir / f"{tag}_{_key(f, c, op)}.npy", mode="w+", dtype=np.float32, shape=(len(ids), f.groups[(c, op)].n_groups, T))
        if per_ship:
            for op in ops:
                mm[("S", f.purpose, f.idx, op)] = np.lib.format.open_memmap(
                    out_dir / f"{tag}_ship_{f.purpose}_{f.idx}_{op}.npy", mode="w+", dtype=np.float32, shape=(len(ids), len(f.draw100.pool_idx), T))
    for s in range(0, len(ids), chunk):
        block = np.asarray(lib[ids[s:s + chunk]], np.float32)                      # (k, 120, R, 3)
        e = s + block.shape[0]
        for f in fleets:
            sub = block[:, :, f.draw100.pool_idx, :]                                # ships of the largest fleet
            for op in ops:
                if per_ship:
                    mm[("S", f.purpose, f.idx, op)][s:e] = scoring.project_templates(sub, f.draw100, op)
                for c in caps:
                    sc = sub[:, :, f.masks[c], :]
                    g = scoring.project_templates(sc, f.draws[c], op)
                    v = valid_fn(f, c, op) if valid_fn else None
                    mm[("G", f.purpose, f.idx, c, op)][s:e] = scoring.aggregate(g, f.groups[(c, op)], v, xp=np)
        del block
    for m in mm.values():
        m.flush()
    done.write_text("ok")


def load_G(out_dir, tag, f, cap, op):
    return np.load(Path(out_dir) / f"{tag}_{_key(f, cap, op)}.npy", mmap_mode="r")


# ------------------------------------------------------------------ scoring pieces

def make_noise(pool, fleet, cap, op, regime, cfg, rng, n):
    """(n, S, 120) filtered noise for the capped fleet."""
    d = fleet.draws[cap]
    if op == "ais":
        return noise.ais_noise(d, pool, regime, cfg, rng, n)
    return noise.gps_noise(d, regime, cfg, rng, n)


def network_trials(G, D, w, flip_allowed, eps):
    """gd2 (E,K2,T), gg2 (K2,T), stat (E,T) for aggregated observations D (E, M, T) against templates G (K, M, T)."""
    gd, gg = scoring.cumulative_stats(G, D, w)
    gd2, gg2, parent, polarity = scoring.expand_flips(gd, gg, flip_allowed)
    del gd
    stat, _ = scoring.detection_statistic(scoring.detection_scores(gd2, gg2, eps))
    return gd2, gg2, stat, parent, polarity


def top3_at(gd2, gg2, t_idx, k=3):
    """Top-k candidates (E, k) by fixed-amplitude log-likelihood 2 gd - gg at minute t_idx[e]."""
    xp = backend.array_module(gd2)
    t = xp.asarray(t_idx)
    E = gd2.shape[0]
    L = 2.0 * gd2[xp.arange(E), :, t] - gg2[:, t].T                               # (E, K2)
    order = xp.argsort(-L, axis=1)[:, :k]
    return backend.to_numpy(order)


# ------------------------------------------------------------------ quiet days

def quiet_stats(lib_G_dir, tag, fleets_by_idx, quiet, pool, cfg, cand, rows=None, strata=None, valid_fn=None, G_fn=None):
    """Statistic arrays on quiet days per stratum: {(op, cap, regime): {split: (net (N,T), single (N,T))}}.

    rows: optional positions into `cand` (and into the cached templates) for a smaller catalogue, e.g. arcs only. The
    threshold then reflects that catalogue: fewer candidates means a lower window maximum on quiet days.
    """
    if rows is not None:
        cand = cand.iloc[rows].reset_index(drop=True)
    # strata: optional list of (operator, cap, regime) to restrict the work; valid_fn(fleet, cap, op) -> (S, T) bool report mask and
    # G_fn(fleet, cap, op) -> templates aggregated with that mask replace the defaults (used by the data-gap sensitivity)
    sc = cfg["scoring"]
    eps = sc["eps"]
    flip = cand["flip_allowed"].values.astype(bool)
    out = {}
    caps = sorted(cfg["traffic"]["range_caps_m"])
    for op, cap, regime in (strata if strata is not None else itertools.product(cfg["operators"], caps, cfg["weather_regimes"])):
        per = {"cal": ([], []), "hold": ([], [])}
        for tseed, qrows in quiet.groupby("traffic_seed"):
            f = fleets_by_idx[int(tseed)]
            d = f.draws[cap]
            groups = f.groups[(cap, op)]
            si, ss, st = sigmas(d, regime, op, cfg)
            valid = valid_fn(f, cap, op) if valid_fn else None
            w = scoring.cell_weights(groups, si, ss, valid)
            G = np.asarray(G_fn(f, cap, op) if G_fn else load_G(lib_G_dir, tag, f, cap, op))
            if rows is not None:
                G = G[rows]
            ns = []
            for q in qrows.itertuples():
                rng = np.random.default_rng([cfg["meta"]["seed"], PURPOSE_QNOISE, int(q.noise_seed)])
                ns.append(make_noise(pool, f, cap, op, regime, cfg, rng, 1))
            nz = np.concatenate(ns)                                                  # (n, S, 120)
            Dn = scoring.aggregate(nz, groups, valid, xp=np)
            _, _, stat, _, _ = network_trials(G, Dn, w, flip, eps)
            single = single_ship_stat(backend.to_xp(nz), st, valid)
            split = qrows["split"].iloc[0]
            per[split][0].append(backend.to_numpy(stat))
            per[split][1].append(single)
        out[(op, cap, regime)] = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in per.items()}
    return out


def thresholds(qs, cfg):
    """DataFrame in the docs/interfaces.md section 10 format, plus the per-stratum thresholds dict."""
    far = cfg["scoring"]["false_alarm_rate"]
    k_lit = cfg["single_ship"]["literal_k_sigma"]
    rows, thr = [], {}
    for (op, cap, regime), d in qs.items():
        net_cal, sing_cal = d["cal"]
        net_hold, sing_hold = d["hold"]
        t_net = scoring.calibrate_threshold(net_cal, far)
        t_cal = float(np.quantile(sing_cal.max(axis=-1), 1.0 - far))
        thr[(op, cap, regime)] = {"network": t_net, "single_literal": k_lit, "single_calibrated": t_cal}
        for det, t, c, h in [("network", t_net, net_cal, net_hold), ("single_literal", k_lit, sing_cal, sing_hold),
                             ("single_calibrated", t_cal, sing_cal, sing_hold)]:
            rows.append(dict(operator=op, range_cap_m=cap, regime=regime, detector=det, threshold=t, n_cal=len(c),
                             far_cal=float((c.max(axis=-1) > t).mean()), n_hold=len(h), far_hold=float((h.max(axis=-1) > t).mean())))
    return pd.DataFrame(rows), thr


# ------------------------------------------------------------------ gauges

def gauge_table(truth, cfg, gb_lib_dir, gtable, scale=None):
    """Per truth event: trigger time of every gauge and buoy (s, inf if never), plus the sample-0 eta at each receiver."""
    lib_index = pd.read_csv(Path(gb_lib_dir) / "index.csv").sort_values("run_id")
    trig, first = [], []
    for c, rows in lib_index.groupby("chunk"):
        gb = np.load(Path(gb_lib_dir) / f"gb_{int(c):04d}.npy", mmap_mode="r")
        part = np.asarray(gb[rows["row"].values])
        if scale is not None:                                                         # raw eta at gauges rescales like the library
            part = part * scale.loc[rows["run_id"].values].values.astype(np.float32)[:, None, None]
        trig.append(gauges.trigger_times(part, gtable, cfg["solver"]["sample_interval_s"]))
        first.append(part[:, 0, :])
    return np.concatenate(trig), np.concatenate(first)


def gauge_columns(truth, trig, cfg, gtable):
    R = cfg["physics"]["earth_radius_m"]
    is_g = (gtable["rtype"].values == 1)
    first_gauge = np.where(is_g[None], trig, np.inf).min(axis=1)
    first_inst = trig.min(axis=1)
    dist = haversine_m(gtable["lon_deg"].values[None, :], gtable["lat_deg"].values[None, :], truth["lon_deg"].values[:, None],
                       truth["lat_deg"].values[:, None], R)
    dist = np.where(is_g[None], dist, np.inf)
    near = dist.argmin(axis=1)
    return pd.DataFrame(dict(run_id=truth["run_id"].values, first_gauge_s=first_gauge,
                             nearest_gauge_code=gtable["code"].values[near], nearest_gauge_s=trig[np.arange(len(truth)), near],
                             first_instrument_s=first_inst))


# ------------------------------------------------------------------ event trials

def event_shard(pool, f, cap, op, regime, truth, Ds, sig_ship, G, cand, thr, gcols, cfg, combo_index, valid=None):
    """All trials of one (traffic draw, cap, operator, regime): events x noise draws x 3 detectors -> DataFrame."""
    sc, sw = cfg["scoring"], cfg["sweep"]
    N, nb = sw["n_noise_draws"], sw["event_batch"]
    d = f.draws[cap]
    groups = f.groups[(cap, op)]
    si, ss, st = sigmas(d, regime, op, cfg)
    w = scoring.cell_weights(groups, si, ss, valid)
    purpose = PURPOSE_AIS if op == "ais" else PURPOSE_GPS
    rng = np.random.default_rng([cfg["meta"]["seed"], purpose, 1_000_000 + combo_index])
    nz = make_noise(pool, f, cap, op, regime, cfg, rng, N)                           # (N, S, 120)
    Dn = scoring.aggregate(nz, groups, valid, xp=np)                                 # (N, M, 120)
    flip = cand["flip_allowed"].values.astype(bool)
    n_ev = len(truth)
    tcap = f.masks[cap]
    parts = []
    for s in range(0, n_ev, nb):
        e = slice(s, min(s + nb, n_ev))
        ne = e.stop - e.start
        D = (np.asarray(Ds[e])[:, None] + Dn[None]).reshape(ne * N, groups.n_groups, -1)
        gd2, gg2, stat, parent, polarity = network_trials(G, D, w, flip, sc["eps"])
        stat = backend.to_numpy(stat)
        c_net = scoring.confirmation_minute(stat, thr["network"])
        t_idx = np.where(c_net >= 0, c_net, stat.shape[1] - 1)
        top = top3_at(gd2, gg2, t_idx)
        del gd2, gg2
        best_k2 = top[:, 0]
        bp = parent[best_k2]
        sig = np.asarray(sig_ship[e])[:, tcap, :]                                    # (ne, S, 120)
        dd = backend.to_xp(sig)[:, None] + backend.to_xp(nz)[None]                  # (ne, N, S, 120)
        sstat = single_ship_stat(dd.reshape(ne * N, dd.shape[2], dd.shape[3]), st, valid)
        c_lit = alarm_minute(sstat, thr["single_literal"])
        c_cal = alarm_minute(sstat, thr["single_calibrated"])
        tr = truth.iloc[e]
        run_id = np.repeat(tr["run_id"].values, N)
        base = dict(run_id=run_id, traffic_seed=f.idx, noise_seed=combo_index * 1000 + np.tile(np.arange(N), ne), regime=regime,
                    range_cap_m=cap, operator=op, n_ships=len(d.pool_idx), run_kind=np.repeat(tr["run_kind"].values, N),
                    truth_mw=np.repeat(tr["mw"].values, N))
        g = gcols.set_index("run_id").loc[run_id]
        gc = {k: g[k].values for k in ("first_gauge_s", "nearest_gauge_code", "nearest_gauge_s", "first_instrument_s")}
        R = cfg["physics"]["earth_radius_m"]
        loc_err = haversine_m(cand["lon_deg"].values[bp], cand["lat_deg"].values[bp], np.repeat(tr["lon_deg"].values, N),
                              np.repeat(tr["lat_deg"].values, N), R)
        netd = dict(base, detector="network", detected=c_net >= 0, confirm_s=np.where(c_net >= 0, 60.0 * (c_net + 1), np.inf),
                    best_cand_id=cand["cand_id"].values[bp], best_polarity=polarity[best_k2], best_lon_deg=cand["lon_deg"].values[bp],
                    best_lat_deg=cand["lat_deg"].values[bp], best_mw=cand["mw"].values[bp], loc_err_m=loc_err,
                    mw_err=cand["mw"].values[bp] - np.repeat(tr["mw"].values, N),
                    top3_cand_ids=[";".join(str(int(x)) for x in cand["cand_id"].values[parent[row]]) for row in top], **gc)
        parts.append(pd.DataFrame(netd))
        for det, c in (("single_literal", c_lit), ("single_calibrated", c_cal)):
            parts.append(pd.DataFrame(dict(base, detector=det, detected=c >= 0, confirm_s=np.where(c >= 0, 60.0 * (c + 1), np.inf),
                                           best_cand_id=-1, best_polarity=0, best_lon_deg=np.nan, best_lat_deg=np.nan, best_mw=np.nan,
                                           loc_err_m=np.nan, mw_err=np.nan, top3_cand_ids="", **gc)))
    return pd.concat(parts, ignore_index=True)


def twin_summary(sweep_dir):
    """Twin check: detection fraction and localisation / magnitude errors of the network detector on twin events."""
    frames = []
    for p in sorted(Path(sweep_dir).glob("part_*.parquet")):
        df = pd.read_parquet(p)
        frames.append(df[(df.detector == "network") & (df.run_kind == "truth_twin")])
    df = pd.concat(frames)
    g = df.groupby(["operator", "range_cap_m", "regime"])
    return g.apply(lambda x: pd.Series(dict(
        n_trials=len(x), detected_frac=x.detected.mean(),
        median_loc_err_km=x.loc[x.detected, "loc_err_m"].median() / 1000.0, p90_loc_err_km=x.loc[x.detected, "loc_err_m"].quantile(0.9) / 1000.0,
        median_abs_mw_err=x.loc[x.detected, "mw_err"].abs().median(), median_confirm_min=x.loc[x.detected, "confirm_s"].median() / 60.0)),
        include_groups=False).reset_index()


class Scaled:
    """Lazy k * array (rows on demand): truth signals under a source-amplitude change (the model is linear)."""

    def __init__(self, arr, k):
        self.arr, self.k = arr, np.float32(k)

    def __getitem__(self, i):
        return self.k * np.asarray(self.arr[i])

    def __len__(self):
        return len(self.arr)


def map2_levels(df, truth, level=0.9):
    """Map 2 definition from network rows: per source position (pos_id), the smallest truth magnitude detected in at least
    `level` of trials. Returns a DataFrame pos_id, region, lon, lat, mw_min (NaN if never)."""
    net = df[df.detector == "network"]
    frac = net.groupby("run_id").detected.mean().rename("frac")
    t = truth.set_index("run_id").join(frac)
    rows = []
    for pos, g in t.groupby("pos_id"):
        ok = g[g.frac >= level]
        rows.append(dict(pos_id=pos, region=g.region.iloc[0], lon_deg=g.lon_deg.mean(), lat_deg=g.lat_deg.mean(),
                         mw_min=ok.mw.min() if len(ok) else np.nan))
    return pd.DataFrame(rows)
