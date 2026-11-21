"""T17: full sweep on the runtime that holds the library (data/library, data/truth).

    PYTHONPATH=src python -u scripts/05_sweep.py [--stages quiet,events,summary] [--n-traffic N] [--n-noise N]

Stages (each resumable; finished shards and passes are skipped):
  quiet   quiet-day statistics for all 12 strata -> outputs/thresholds.csv  (library pass over 10 quiet traffic draws)
  events  every truth event x traffic draw x noise draw x stratum x detector -> outputs/sweep/part_*.parquet,
          gauge_triggers.parquet  (library pass over the event traffic draws, truth pass for the signals)
  summary twin summary -> outputs/sweep/twin_summary.csv
Only results (parquet, csv) need to be downloaded; the caches under data/sweep_cache stay on the runtime.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsunami import backend, gauges, scoring, ships, sweep as SW
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="quiet,events,summary")
ap.add_argument("--n-traffic", type=int, default=None)
ap.add_argument("--n-noise", type=int, default=None)
ap.add_argument("--max-events", type=int, default=None, help="smoke tests only")
ap.add_argument("--cache-dir", default=None, help="smoke tests only: keep their caches apart from the real ones")
ap.add_argument("--out-dir", default=None, help="smoke tests only")
ap.add_argument("--thresholds", default=None, help="smoke tests only")
ap.add_argument("--library", default=None)
ap.add_argument("--truth", default=None)
args = ap.parse_args()

cfg = load_config()
if args.n_traffic:
    cfg["sweep"]["n_traffic_draws"] = args.n_traffic
if args.n_noise:
    cfg["sweep"]["n_noise_draws"] = args.n_noise
grid, pool = load_grid(cfg), ships.load_pool(cfg)
cand = pd.read_csv(repo_path(cfg["paths"]["candidates"])).sort_values("cand_id").reset_index(drop=True)
lib_dir = Path(args.library) if args.library else repo_path(cfg["paths"]["library_dir"])
truth_dir = Path(args.truth) if args.truth else repo_path(cfg["paths"]["truth_dir"])
cache = Path(args.cache_dir) if args.cache_dir else repo_path(cfg["sweep"]["cache_dir"])
out = Path(args.out_dir) if args.out_dir else repo_path(cfg["sweep"]["out_dir"])
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter()-t0:7.0f}s]", *a, flush=True)
log("backend gpu:", backend.GPU, "candidates", len(cand), "receivers", len(pool))

lib = SW.ScaledLibrary(scoring.Library(lib_dir), SW.load_scale(repo_path("data/processed/uplift_scale.csv"), "cand_id"))
assert len(lib) == len(cand), (len(lib), len(cand))
assert (lib.index["run_id"].values == cand["cand_id"].values).all()
thr_path = Path(args.thresholds) if args.thresholds else repo_path("outputs") / "thresholds.csv"

# ---------------------------------------------------------------- quiet days -> thresholds
if "quiet" in args.stages:
    quiet = pd.read_csv(truth_dir / "quiet_days.csv")
    qfleets = {int(i): SW.make_fleet(pool, cfg, SW.PURPOSE_QTRAFFIC, int(i)) for i in sorted(quiet.traffic_seed.unique())}
    log("quiet traffic draws:", {i: len(f.draw100.pool_idx) for i, f in qfleets.items()})
    SW.gather_pass(lib, cand["cand_id"].values, list(qfleets.values()), cfg, cache, SW.quiet_tag(quiet), per_ship=False)
    log("quiet library pass done")
    qs = SW.quiet_stats(cache, SW.quiet_tag(quiet), qfleets, quiet, pool, cfg, cand)
    tdf, thr = SW.thresholds(qs, cfg)
    tdf.to_csv(thr_path, index=False)
    log("thresholds written", thr_path)
    print(tdf.to_string())

# ---------------------------------------------------------------- events
if "events" in args.stages:
    tdf = pd.read_csv(thr_path)
    thr = {(r.operator, r.range_cap_m, r.regime): {} for r in tdf.itertuples()}
    for r in tdf.itertuples():
        thr[(r.operator, r.range_cap_m, r.regime)][r.detector] = r.threshold
    truth = pd.read_csv(truth_dir / "sources.csv").sort_values("run_id").reset_index(drop=True)
    if args.max_events:
        truth = truth.iloc[:args.max_events]
    tscale = SW.load_scale(truth_dir / "uplift_scale.csv", "run_id")
    tlib = SW.ScaledLibrary(scoring.Library(truth_dir), tscale)
    gtable = gauges.table_from_pool(pool, cfg)
    from tsunami.sources import source_eta0 as _eta0
    e0_gb = SW.run_eta0_at_gauges(truth, pd.read_csv(truth_dir / "uplift_scale.csv"), grid, cfg, gtable, _eta0)
    trig, first_eta = SW.gauge_table(truth, cfg, truth_dir, gtable, scale=tscale, eta0_gb=e0_gb)
    SW.check_eta0_consistency(first_eta[:len(truth)], e0_gb.values, log)   # first_eta is already rescaled by gauge_table
    trig, first_eta = trig[:len(truth)], first_eta[:len(truth)]
    gcols = SW.gauge_columns(truth, trig, cfg, gtable)
    gl = pd.DataFrame({"run_id": np.repeat(truth.run_id.values, len(gtable)), "code": np.tile(gtable.code.values, len(truth)),
                       "rtype": np.tile(gtable.rtype.values, len(truth)), "trigger_s": trig.ravel(),
                       "eta_first_sample_m": first_eta.ravel()})
    gl.to_parquet(out / "gauge_triggers.parquet")
    log("gauge triggers written")
    efleets = [SW.make_fleet(pool, cfg, SW.PURPOSE_TRAFFIC, i) for i in range(cfg["sweep"]["n_traffic_draws"])]
    log("event traffic draws:", [len(f.draw100.pool_idx) for f in efleets])
    SW.gather_pass(lib, cand["cand_id"].values, efleets, cfg, cache, "lib", per_ship=False)
    log("library pass done")
    SW.gather_pass(tlib, truth["run_id"].values, efleets, cfg, cache, "truth" + (f"_{args.max_events}" if args.max_events else ""), per_ship=True)
    log("truth pass done")
    ttag = "truth" + (f"_{args.max_events}" if args.max_events else "")
    caps = sorted(cfg["traffic"]["range_caps_m"])
    regimes = list(cfg["weather_regimes"])
    combos = list(itertools.product(range(len(efleets)), caps, cfg["operators"], regimes))
    for n, (i, cap, op, regime) in enumerate(combos):
        f = efleets[i]
        part = out / f"part_{i}_{int(cap)}_{op}_{regime}.parquet"
        if part.exists():
            continue
        G = np.asarray(SW.load_G(cache, "lib", f, cap, op))
        Ds = SW.load_G(cache, ttag, f, cap, op)
        sig = np.load(cache / f"{ttag}_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r")
        df = SW.event_shard(pool, f, cap, op, regime, truth, Ds, sig, G, cand, thr[(op, cap, regime)], gcols, cfg, n)
        df.to_parquet(part.with_suffix(".tmp"))
        part.with_suffix(".tmp").rename(part)
        log(f"shard {n+1}/{len(combos)} {part.name} rows={len(df)} network detected={df[df.detector=='network'].detected.mean():.3f}")

if "summary" in args.stages:
    ts = SW.twin_summary(out)
    ts.to_csv(out / "twin_summary.csv", index=False)
    print(ts.to_string())
    log("done")
