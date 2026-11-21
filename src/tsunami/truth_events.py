"""Truth-event and quiet-day lists (T16). Contract: docs/interfaces.md sections 7, 10 and 13.

Truth events are separate solver runs, never library entries: the detector must find sources that are not on its grid
(avoids the inverse crime). Quiet days are only lists of seeds; they need no solver run.
"""
import json

import numpy as np
import pandas as pd

from .config import repo_path
from .geometry import haversine_m

PURPOSE_TRUTH = 4          # docs/interfaces.md seed purposes: 4 = truth perturbation


def grid_spacing_m(row, cfg):
    return cfg["sources"]["aegean_spacing_m"] if row["region"] == "aegean" else cfg["sources"]["arc_spacing_m"]


def _tag(d, kind, parent, name):
    d = dict(d)
    d.update(run_kind=kind, parent_cand_id=int(parent), name=name)
    return d


def map2_events(cand, sources, cfg):
    """One truth event per library candidate: half a grid spacing along strike, magnitude + map2_dmw."""
    t = cfg["truth"]
    out = []
    for row in cand.to_dict("records"):
        d = sources.perturb(row, t["map2_along_strike_frac"] * grid_spacing_m(row, cfg), t["map2_dmw"], cfg)
        if row["region"] == "aegean":
            d["polarity"] = int(t["aegean_truth_polarity"])
        out.append(_tag(d, "truth_map2", row["cand_id"], f"map2_{row['cand_id']}"))
    return out


def twin_events(cand, sources, cfg):
    """Random off-grid events: random library position and magnitude, random along-strike offset and magnitude nudge."""
    t = cfg["truth"]
    rng = np.random.default_rng([cfg["meta"]["seed"], PURPOSE_TRUTH, 1])
    rows = cand.to_dict("records")
    out = []
    for k in range(t["n_twin"]):
        row = rows[int(rng.integers(len(rows)))]
        lo, hi = t["twin_offset_frac_min"], t["twin_offset_frac_max"]
        off = rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]) * grid_spacing_m(row, cfg)
        mw_lo, mw_hi = cand.loc[cand["region"] == row["region"], "mw"].agg(["min", "max"])
        dmw = rng.uniform(-t["twin_dmw_max"], t["twin_dmw_max"])
        dmw = float(np.clip(row["mw"] + dmw, mw_lo, mw_hi) - row["mw"])
        d = sources.perturb(row, off, dmw, cfg)
        if row["region"] == "aegean":
            d["polarity"] = int(rng.choice([-1, 1]))
        out.append(_tag(d, "truth_twin", row["cand_id"], f"twin_{k}"))
    return out


def truth_table(cand, sources, cfg):
    rows = map2_events(cand, sources, cfg) + twin_events(cand, sources, cfg)
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", np.arange(len(df)))
    return df


def coincidences(truth, cand, cfg):
    """Truth sources that coincide with a library source: same magnitude, same strike, and within min_separation_m.

    Returns a DataFrame of offending (run_id, cand_id, distance_m); empty means every truth source is off-grid.
    """
    t, R = cfg["truth"], cfg["physics"]["earth_radius_m"]
    bad = []
    for r in truth.itertuples():
        same_mw = np.abs(cand["mw"].values - r.mw) < t["mw_tolerance"]
        if not same_mw.any():
            continue
        c = cand[same_mw]
        d = haversine_m(c["lon_deg"].values, c["lat_deg"].values, r.lon_deg, r.lat_deg, R)
        close = (d < t["min_separation_m"]) & (np.abs(c["strike_deg"].values - r.strike_deg) < 1e-6)
        for cid, dist in zip(c["cand_id"].values[close], d[close]):
            bad.append((r.run_id, cid, dist))
    return pd.DataFrame(bad, columns=["run_id", "cand_id", "distance_m"])


def quiet_days(cfg):
    """Quiet-day seed list: calibration and held-out splits with disjoint traffic seeds and disjoint noise seeds.

    traffic_seed indexes PURPOSE 5 draws, noise_seed indexes PURPOSE 6 draws. Calibration uses traffic draws 0..n-1 and
    noise draws 0..; held-out uses the next traffic draws and the next noise draws, so nothing is shared.
    """
    s = cfg["scoring"]
    n_tr = s["n_quiet_traffic_draws"]
    rows, noise0 = [], 0
    for split, n_total, tr0 in (("cal", s["n_quiet_calibration"], 0), ("hold", s["n_quiet_holdout"], n_tr)):
        per = -(-n_total // n_tr)                                  # noise draws per traffic draw, rounded up so the total reaches n_total
        for k in range(n_tr):
            for m in range(per):
                rows.append((split, tr0 + k, noise0))
                noise0 += 1
    df = pd.DataFrame(rows, columns=["split", "traffic_seed", "noise_seed"])
    df.insert(0, "quiet_id", np.arange(len(df)))
    return df


def historic_table(sources, cfg):
    """Historic truth sources (Samos 2020, Amorgos 1956, ...) from the JSON files of T26a (`recommended_for_replay`).

    The fault type in the file sets the polarity (normal fault = subsidence-first = -1) through sources.custom_source.
    """
    folder = repo_path(cfg["paths"]["historic_dir"])
    rows = []
    for fname in cfg["paths"].get("historic_source_files", []):
        h = json.loads((folder / fname).read_text())["recommended_for_replay"]
        d = sources.custom_source(h["lon_deg"], h["lat_deg"], h["strike_deg"], h["dip_deg"], h["mw"], h["fault"], cfg,
                                  length_m=h.get("length_m"), width_m=h.get("width_m"), slip_m=h.get("slip_m"))
        rows.append(_tag(d, "historic", -1, fname.removesuffix("_sources.json")))
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "run_id", np.arange(len(df)))
    return df
