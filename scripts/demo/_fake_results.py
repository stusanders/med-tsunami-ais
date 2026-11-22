"""Synthetic, contract-shaped stand-ins for the pending T17 results, for testing scenes 10, 12 and 13 ONLY.

Writes (never under outputs/):
  /tmp/t27_fake/outputs/sweep/part_{traffic}_{cap}_{op}_{regime}.parquet   docs/interfaces.md section 14 columns
                                                                            (+ run_kind, truth_mw as in sweep.event_shard)
  /tmp/t27_fake/outputs/sweep/gauge_triggers.parquet                       (different schema; scenes must ignore it)
  /tmp/t27_fake/outputs/sweep/twin_summary.csv
  /tmp/t27_fake/outputs/thresholds.csv                                     docs/interfaces.md section 10 columns

The numbers are made up (a logistic detection curve per source position that gets worse with depth and weather). They exist
only to exercise the plotting path, and every frame rendered from them carries a "SYNTHETIC TEST DATA" watermark.

    python scripts/demo/_fake_results.py                 # (re)write the fake files
    python scripts/demo/_fake_results.py --render 13 --at 12 [--at 20 ...]   # stills -> /tmp/t27_fake/stills/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAKE_ROOT = Path("/tmp/t27_fake/outputs")
STILLS = Path("/tmp/t27_fake/stills")


def write_fake(n_traffic=2, n_noise=4, seed=7):
    import common as cm
    from tsunami import gauges
    from tsunami.geometry import haversine_m

    cfg = cm.load_cfg()
    rng = np.random.default_rng(seed)
    grid = cm.load_grid_cached()
    truth = pd.read_csv(cm.REPO / "data/truth/sources.csv").sort_values("run_id").reset_index(drop=True)
    cand = pd.read_csv(cm.REPO / "data/processed/candidates.csv")
    gt = gauges.load_gauges(grid, cfg)
    R = cfg["physics"]["earth_radius_m"]
    caps = sorted(cfg["traffic"]["range_caps_m"])
    regimes = list(cfg["weather_regimes"])

    # depth at each truth source -> a made-up 50% detection magnitude per position (shallow = easier)
    j = np.clip(np.searchsorted(grid.lat, truth.lat_deg), 0, len(grid.lat) - 1)
    i = np.clip(np.searchsorted(grid.lon, truth.lon_deg), 0, len(grid.lon) - 1)
    dep = np.where(grid.wet[j, i], grid.depth[j, i], 50.0)
    pos_key = truth.region + "_" + truth.pos_id.astype(str)
    pos_mw50 = {k: 6.9 + 0.55 * np.log10(np.clip(d, 50, 5000) / 50) + rng.normal(0, 0.25)
                for k, d in zip(pos_key, dep)}
    base50 = pos_key.map(pos_mw50).to_numpy()

    # nearest gauge and plausible trigger times
    isg = gt.rtype.values == 1
    dist = haversine_m(gt.lon_deg.values[None], gt.lat_deg.values[None], truth.lon_deg.values[:, None],
                       truth.lat_deg.values[:, None], R)
    dg = np.where(isg[None], dist, np.inf)
    near = dg.argmin(1)
    speed = np.sqrt(9.81 * 1500.0)
    nearest_s = dg[np.arange(len(truth)), near] / speed + 180 + rng.uniform(0, 600, len(truth))
    first_gauge_s = nearest_s * rng.uniform(0.6, 1.0, len(truth))
    db = np.where(~isg[None], dist, np.inf).min(1)
    first_inst = np.minimum(first_gauge_s, db / speed + 300)

    # candidate lookup: same region & pos, nearest magnitude
    cand_key = cand.region + "_" + cand.pos_id.astype(str)
    by_key = {k: (g.cand_id.values, g.mw.values) for k, g in cand.groupby(cand_key)}
    all_c = (cand.cand_id.values, cand.mw.values)
    out = FAKE_ROOT / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    n = 0
    for ti in range(n_traffic):
        for cap in caps:
            for op in cfg["operators"]:
                for reg in regimes:
                    shift = {"calm": 0.0, "moderate": 0.25, "rough": 0.5}[reg] + (-0.55 if op == "gps" else 0.0) \
                        + (0.2 if cap < caps[-1] else 0.0)
                    E = len(truth)
                    mw = np.repeat(truth.mw.values, n_noise)
                    m50 = np.repeat(base50, n_noise) + shift
                    p = 1 / (1 + np.exp(-(mw - m50) / 0.12))
                    det = rng.random(E * n_noise) < p
                    confirm = np.where(det, 60.0 * rng.integers(6, 50, E * n_noise), np.inf)
                    # best candidate: same position, magnitude within a bin, sometimes the neighbour position
                    bc = np.empty(E * n_noise, int)
                    rep_key = np.repeat(pos_key.values, n_noise)
                    jit = rng.normal(0, 0.15, E * n_noise)
                    for k in range(E * n_noise):
                        ids, mws = by_key.get(rep_key[k], all_c)
                        bc[k] = ids[np.abs(mws - (mw[k] + jit[k])).argmin()]
                    cl = cand.set_index("cand_id").loc[bc]
                    tl, tb = np.repeat(truth.lon_deg.values, n_noise), np.repeat(truth.lat_deg.values, n_noise)
                    row = dict(
                        run_id=np.repeat(truth.run_id.values, n_noise), traffic_seed=ti,
                        noise_seed=n * 1000 + np.tile(np.arange(n_noise), E), regime=reg, range_cap_m=cap, operator=op,
                        n_ships=600 + 10 * ti, run_kind=np.repeat(truth.run_kind.values, n_noise), truth_mw=mw,
                        first_gauge_s=np.repeat(first_gauge_s, n_noise),
                        nearest_gauge_code=np.repeat(gt.code.values[near], n_noise),
                        nearest_gauge_s=np.repeat(nearest_s, n_noise), first_instrument_s=np.repeat(first_inst, n_noise))
                    net = dict(row, detector="network", detected=det, confirm_s=confirm, best_cand_id=bc,
                               best_polarity=1, best_lon_deg=cl.lon_deg.values, best_lat_deg=cl.lat_deg.values,
                               best_mw=cl.mw.values, loc_err_m=haversine_m(cl.lon_deg.values, cl.lat_deg.values, tl, tb, R),
                               mw_err=cl.mw.values - mw, top3_cand_ids=[str(b) for b in bc])
                    parts = [pd.DataFrame(net)]
                    for dname, extra in (("single_literal", -0.3), ("single_calibrated", 0.8)):
                        pd_ = 1 / (1 + np.exp(-(mw - m50 - extra) / 0.12))
                        d2 = rng.random(E * n_noise) < pd_
                        parts.append(pd.DataFrame(dict(row, detector=dname, detected=d2,
                                                       confirm_s=np.where(d2, 60.0 * rng.integers(3, 50, E * n_noise), np.inf),
                                                       best_cand_id=-1, best_polarity=0, best_lon_deg=np.nan,
                                                       best_lat_deg=np.nan, best_mw=np.nan, loc_err_m=np.nan, mw_err=np.nan,
                                                       top3_cand_ids="")))
                    pd.concat(parts, ignore_index=True).to_parquet(out / f"part_{ti}_{int(cap)}_{op}_{reg}.parquet")
                    n += 1
    pd.DataFrame(dict(run_id=[0], code=["x"], rtype=[1], trigger_s=[1.0], eta_first_sample_m=[0.0])).to_parquet(
        out / "gauge_triggers.parquet")
    # thresholds
    rows = []
    for op in cfg["operators"]:
        for cap in caps:
            for reg in regimes:
                for det_ in ("network", "single_literal", "single_calibrated"):
                    fh = {"network": rng.uniform(0.006, 0.016), "single_literal": rng.uniform(0.98, 1.0),
                          "single_calibrated": rng.uniform(0.006, 0.016)}[det_]
                    rows.append(dict(operator=op, range_cap_m=cap, regime=reg, detector=det_, threshold=rng.uniform(5, 30),
                                     n_cal=500, far_cal=0.01, n_hold=500, far_hold=round(fh * 500) / 500))
    pd.DataFrame(rows).to_csv(FAKE_ROOT / "thresholds.csv", index=False)
    pd.DataFrame([dict(operator="ais", range_cap_m=caps[-1], regime=r, n_trials=80 * n_traffic * n_noise, detected_frac=0.5,
                       median_loc_err_km=30.0, p90_loc_err_km=80.0, median_abs_mw_err=0.15, median_confirm_min=20.0)
                  for r in regimes]).to_csv(out / "twin_summary.csv", index=False)
    print("fake results written to", FAKE_ROOT, f"({n} shards)")


def render_stills(num, times):
    os.environ["T27_RESULTS_DIR"] = str(FAKE_ROOT)
    import common as cm
    mod = cm.load_scene_module(num)
    ctx = cm.Ctx(cfg=cm.load_cfg(), cut="full")
    sc = mod.build("full", ctx)
    STILLS.mkdir(parents=True, exist_ok=True)
    dur = cm.duration(num, "full")
    for t in times:
        sc.update(min(t, dur - 1.0 / cm.FPS))
        p = STILLS / f"scene_{num:02d}_t{t:05.1f}.png"
        sc.fig.savefig(p, dpi=cm.DPI, facecolor=cm.BG)
        print(p)
    if sc.meta:
        print(sc.meta)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=int, default=None)
    ap.add_argument("--at", type=float, action="append")
    a = ap.parse_args()
    if a.render is None:
        write_fake()
    else:
        render_stills(a.render, a.at or [None])
