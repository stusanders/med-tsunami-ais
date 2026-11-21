"""T20: sensitivity runs, one stratum (AIS, 100 km, moderate weather) and the first n traffic draws of the sweep.

Needs the caches and thresholds written by 05_sweep.py (data/sweep_cache, outputs/thresholds.csv, outputs/sweep/part_*.parquet) on the
runtime that holds the library. Each variant is scored on all truth events (Map 2 level per source position) unless it needs new
solver runs or event-specific masks, in which case it uses the informative events (closest to 50% baseline detection).

Variants
  baseline            the sweep's own numbers for this stratum
  corr_L km           shared-noise correlation length changed; thresholds recalibrated on quiet days
  response_0.5        ships feel 50% of the current (signal x 0.5)
  uplift_x0.5 / x2    truth amplitude scaled, library fixed (source-size mismatch; the model is linear so no new solver run)
  monopole            truth sources rebuilt as single Gaussian uplift (informative events; new solver runs on this runtime)
  gaps_10pct          10% of reports dropped at random; templates and weights use the same mask; thresholds recalibrated
  near_source_off     ships within off_radius of the epicentre silent for that event (informative events; thresholds unchanged)
Outputs: outputs/sensitivity.csv, outputs/sensitivity.png, outputs/sensitivity.md
"""
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from tsunami import backend, gauges, observe, scoring, ships, sweep as SW
from tsunami import library_build as LB
from tsunami.config import load_config, repo_path
from tsunami.geometry import haversine_m
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami.sources import source_eta0

cfg = load_config()
S = cfg["sensitivity"]
op, cap, regime = S["operator"], S["range_cap_m"], S["regime"]
out = repo_path("outputs")
sweep_dir, cache = repo_path(cfg["sweep"]["out_dir"]), repo_path(cfg["sweep"]["cache_dir"])
grid, pool = load_grid(cfg), ships.load_pool(cfg)
cand = pd.read_csv(repo_path(cfg["paths"]["candidates"])).sort_values("cand_id").reset_index(drop=True)
truth = pd.read_csv(repo_path(cfg["paths"]["truth_dir"]) / "sources.csv").sort_values("run_id").reset_index(drop=True)
quiet = pd.read_csv(repo_path(cfg["paths"]["truth_dir"]) / "quiet_days.csv")
lib = SW.ScaledLibrary(scoring.Library(repo_path(cfg["paths"]["library_dir"])), SW.load_scale(repo_path("data/processed/uplift_scale.csv"), "cand_id"))
tscale = SW.load_scale(repo_path(cfg["paths"]["truth_dir"]) / "uplift_scale.csv", "run_id")
tlib_dir = repo_path(cfg["paths"]["truth_dir"])
tdf = pd.read_csv(out / "thresholds.csv")
tsel = tdf[(tdf.operator == op) & (tdf.range_cap_m == cap) & (tdf.regime == regime)].set_index("detector")["threshold"].to_dict()
efleets = [SW.make_fleet(pool, cfg, SW.PURPOSE_TRAFFIC, i) for i in range(S["n_traffic"])]
gtable = gauges.load_gauges(grid, cfg)
trig, _ = SW.gauge_table(truth, cfg, tlib_dir, gtable, scale=tscale)
gcols = SW.gauge_columns(truth, trig[:len(truth)], cfg, gtable)
rows_out, log = [], lambda *a: print(*a, flush=True)
m2 = truth[truth.run_kind == "truth_map2"].reset_index(drop=True)


def shard(f, i, truth_t, Ds, sig, G, thr, cfg_v, valid=None, combo=0):
    return SW.event_shard(pool, f, cap, op, regime, truth_t, Ds, sig, G, cand, thr, gcols, cfg_v, 5000 + 10 * i + combo, valid=valid)


def baseline_frames():
    fr = []
    for i in range(S["n_traffic"]):
        fr.append(pd.read_parquet(sweep_dir / f"part_{i}_{int(cap)}_{op}_{regime}.parquet"))
    return pd.concat(fr, ignore_index=True)


def record(name, df, truth_t, note=""):
    net = df[df.detector == "network"]
    lv = SW.map2_levels(df, truth_t[truth_t.run_kind == "truth_map2"])
    per_reg = lv.groupby("region").mw_min.agg(["median", lambda x: x.notna().mean()])
    for reg, r in per_reg.iterrows():
        rows_out.append(dict(variant=name, region=reg, median_mw_min90=r["median"], frac_positions_detectable=r.iloc[1],
                             detected_frac=net[net.run_kind == "truth_map2"].detected.mean(), n_events=net.run_id.nunique(), note=note))
    for det in SW.DETECTORS:
        pass
    log(name, "detected_frac(map2)=%.3f" % net[net.run_kind == "truth_map2"].detected.mean())
    return lv


base = baseline_frames()
lv_base = record("baseline", base, truth)
lv_base.to_csv(out / "sensitivity_map2_baseline.csv", index=False)

# informative events: closest to 50% detection per region (map2 events only)
bf = base[(base.detector == "network") & (base.run_kind == "truth_map2")].groupby("run_id").detected.mean()
t2 = m2.set_index("run_id").join(bf.rename("frac"))
info = []
for reg, g in t2.groupby("region"):
    info += list((g.frac - 0.5).abs().sort_values().index[:S["n_info_per_region"]])
truth_info = truth[truth.run_id.isin(info)].reset_index(drop=True)
log("informative events:", info)

# ---- noise-only and linear variants (all truth events)
Gs = {i: np.asarray(SW.load_G(cache, "lib", f, cap, op)) for i, f in enumerate(efleets)}
Ds = {i: SW.load_G(cache, "truth", f, cap, op) for i, f in enumerate(efleets)}
sig = {i: np.load(cache / f"truth_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r") for i, f in enumerate(efleets)}
qfleets = {int(k): SW.make_fleet(pool, cfg, SW.PURPOSE_QTRAFFIC, int(k)) for k in sorted(quiet.traffic_seed.unique())}


def run_all(name, cfg_v, thr, scale=1.0, note=""):
    fr = []
    for i, f in enumerate(efleets):
        fr.append(shard(f, i, truth, SW.Scaled(Ds[i], scale), SW.Scaled(sig[i], scale), Gs[i], thr, cfg_v))
    df = pd.concat(fr, ignore_index=True)
    df.to_parquet(out / f"sensitivity_{name}.parquet")
    return record(name, df, truth, note)


for L in S["corr_lengths_m"]:
    cv = copy.deepcopy(cfg)
    cv["ais_noise"]["shared_corr_length_m"] = L
    qs = SW.quiet_stats(cache, "quiet", qfleets, quiet, pool, cv, cand, strata=[(op, cap, regime)])
    _, thr_all = SW.thresholds(qs, cv)
    run_all(f"corr_length_{int(L/1000)}km", cv, thr_all[(op, cap, regime)], note="thresholds recalibrated")
run_all("response_0.5", cfg, tsel, scale=S["response"], note="signal x response; thresholds unchanged (quiet days have no signal)")
for k in S["uplift_scales"]:
    run_all(f"uplift_x{k}", cfg, tsel, scale=k, note="truth amplitude scaled; library fixed")

# ---- gaps (random drops): templates, weights and noise aggregation share the mask; thresholds recalibrated on quiet days
def valid_for(f, c, o):
    rng = np.random.default_rng([cfg["meta"]["seed"], 7, 1000 * f.purpose + f.idx])
    return observe.gaps(f.draws[c], cfg, rng, drop_frac=S["gap_frac"])


cal_idx = sorted(quiet[quiet.split == "cal"].traffic_seed.unique())[:S["quiet_cal_draws"]]
hold_idx = sorted(quiet[quiet.split == "hold"].traffic_seed.unique())[:1]
qsub = quiet[quiet.traffic_seed.isin(cal_idx + hold_idx)]
qf_g = {k: qfleets[k] for k in cal_idx + hold_idx}
SW.gather_pass(lib, cand.cand_id.values, list(qf_g.values()), cfg, cache, "quiet_gaps", valid_fn=valid_for)
qs = SW.quiet_stats(cache, "quiet_gaps", qf_g, qsub, pool, cfg, cand, strata=[(op, cap, regime)], valid_fn=valid_for)
_, thr_g = SW.thresholds(qs, cfg)
SW.gather_pass(lib, cand.cand_id.values, efleets, cfg, cache, "lib_gaps", valid_fn=valid_for)
fr = []
for i, f in enumerate(efleets):
    v = valid_for(f, cap, op)
    groups = f.groups[(cap, op)]
    sg = np.asarray(sig[i])[:, f.masks[cap], :]
    Dsv = scoring.aggregate(sg, groups, v, xp=np)
    Gg = np.asarray(SW.load_G(cache, "lib_gaps", f, cap, op))
    fr.append(shard(f, i, truth, Dsv, sig[i], Gg, thr_g[(op, cap, regime)], cfg, valid=v))
dfg = pd.concat(fr, ignore_index=True)
record("gaps_10pct", dfg, truth, note=f"{S['gap_frac']:.0%} reports dropped; thresholds recalibrated on {len(cal_idx)} cal draws")

# ---- monopole truth (informative events): new solver runs on this runtime
mono = truth_info.copy()
mono["kind"] = "monopole"
mono["run_id"] = np.arange(len(mono))
mono_dir = repo_path("data/truth_monopole")
solver = Solver(grid, make_sampler(grid, pool), cfg)
gb_idx = np.nonzero(pool.rtype > 0)[0]
LB.build_stage("monopole", mono, mono_dir, solver, grid, cfg, gb_idx, source_eta0, batch=len(mono), gpu_batch=4)
mlib = scoring.Library(mono_dir)
fm, fb = [], []
for i, f in enumerate(efleets):
    SW.gather_pass(mlib, mono.run_id.values, [f], cfg, cache, f"mono{i}", per_ship=True)
    Dm = SW.load_G(cache, f"mono{i}", f, cap, op)
    sm = np.load(cache / f"mono{i}_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r")
    fm.append(shard(f, i, mono, Dm, sm, Gs[i], tsel, cfg))
    rows_i = np.searchsorted(truth.run_id.values, truth_info.run_id.values)
    fb.append(shard(f, i, truth_info, np.asarray(Ds[i][rows_i]), np.asarray(sig[i][rows_i]), Gs[i], tsel, cfg))
net_m = pd.concat(fm)
net_b = pd.concat(fb)
for name, d in (("dipole_informative", net_b), ("monopole_informative", net_m)):
    n = d[d.detector == "network"]
    rows_out.append(dict(variant=name, region="informative_events", median_mw_min90=np.nan, frac_positions_detectable=np.nan,
                         detected_frac=n.detected.mean(), n_events=n.run_id.nunique(), note="same events, dipole versus monopole truth"))
    log(name, "detected_frac=%.3f" % n.detected.mean())

# ---- near-source-off (informative events, one mask per event)
off_rows = []
for i, f in enumerate(efleets):
    d = f.draws[cap]
    lon, lat = pool.lon[d.pool_idx], pool.lat[d.pool_idx]
    valids = []
    for r in truth_info.itertuples():
        dist = haversine_m(lon, lat, r.lon_deg, r.lat_deg, cfg["physics"]["earth_radius_m"])
        valids.append(observe.gaps(d, cfg, None, 0.0, off_mask=dist < S["off_radius_m"]))
    # templates for each event's mask in one pass over the library
    groups = f.groups[(cap, op)]
    Gv = [np.empty((len(cand), groups.n_groups, int(cfg["library"]["n_minutes"])), np.float32) for _ in valids]
    for s0 in range(0, len(cand), cfg["sweep"]["gather_chunk"]):
        blk = np.asarray(lib[cand.cand_id.values[s0:s0 + cfg["sweep"]["gather_chunk"]]], np.float32)
        sub = blk[:, :, f.draw100.pool_idx, :][:, :, f.masks[cap], :]
        g = scoring.project_templates(sub, d, op)
        for k, v in enumerate(valids):
            Gv[k][s0:s0 + len(g)] = scoring.aggregate(g, groups, v, xp=np)
    for k, r in enumerate(truth_info.itertuples()):
        one = truth_info.iloc[[k]]
        row = int(np.searchsorted(truth.run_id.values, r.run_id))
        sg = np.asarray(sig[i][row:row + 1])[:, f.masks[cap], :]
        Dsv = scoring.aggregate(sg, groups, valids[k], xp=np)
        off_rows.append(shard(f, i, one, Dsv, sig[i][row:row + 1], Gv[k], tsel, cfg, valid=valids[k], combo=k))
dfo = pd.concat(off_rows)
n = dfo[dfo.detector == "network"]
rows_out.append(dict(variant="near_source_off", region="informative_events", median_mw_min90=np.nan, frac_positions_detectable=np.nan,
                     detected_frac=n.detected.mean(), n_events=n.run_id.nunique(),
                     note=f"ships within {S['off_radius_m']/1000:.0f} km of the epicentre silent; thresholds unchanged"))
log("near_source_off detected_frac=%.3f" % n.detected.mean())

res = pd.DataFrame(rows_out)
res.to_csv(out / "sensitivity.csv", index=False)
print(res.to_string())

# ---- figure and short report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tsunami.viz import save_figure
v = res[res.region.isin(["hellenic", "calabrian", "cyprus", "aegean"])]
fig, ax = plt.subplots(figsize=(10, 4.8))
variants = list(dict.fromkeys(v.variant))
regions = ["hellenic", "calabrian", "cyprus", "aegean"]
w = 0.8 / len(variants)
for k, name in enumerate(variants):
    y = [v[(v.variant == name) & (v.region == r)].median_mw_min90.mean() for r in regions]
    ax.bar(np.arange(len(regions)) + k * w, y, w, label=name)
ax.set_xticks(np.arange(len(regions)) + 0.4 - w / 2)
ax.set_xticklabels(regions)
ax.set_ylabel("median smallest magnitude detected in >= 90% of trials (Mw)")
ax.set_ylim(6.0, None)
ax.legend(fontsize=7, ncol=2)
save_figure(fig, out / "sensitivity.png", "Sensitivity of the Map 2 level to the main assumptions (AIS, 100 km, moderate weather)",
            "Caption: median over source positions of the smallest truth magnitude the network detects in at least 90% of trials, by region; missing bars: never detected.")
md = ["# Sensitivity runs (T20)", "", f"Stratum: {op}, {int(cap/1000)} km range cap, {regime} weather; {S['n_traffic']} traffic draws x {cfg['sweep']['n_noise_draws']} noise draws.",
      "Median smallest magnitude detected in >= 90% of trials, by region (blank = never):", "", res.to_markdown(index=False)]
(out / "sensitivity.md").write_text("\n".join(md))
