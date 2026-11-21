"""T26: replay Samos 2020 and Amorgos 1956 with today's (merchant) traffic. Outputs outputs/historic_replays.md (+ csv/parquet).

Stages
  gauges  (any machine, CPU is fine) Samos: simulated versus recorded gauge amplitudes at the IOC stations, with and without
          the Green's-law correction (h_grid / h_gauge)^(1/4). Runs the two historic sources once with receivers at the
          recorded stations (they are mostly not in the receiver pool).
  replay  (on the runtime that holds data/library and data/historic_runs) every detector x weather regime x range cap x
          operator on the historic observations, scored twice: against the arc-only catalogue and against the catalogue
          with the Aegean grid. Thresholds are recalibrated for each catalogue on the same quiet days.
  report  writes outputs/historic_replays.md from the two result files.
The historic runs only supply observations; they are never library entries (no inverse crime).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsunami import backend, gauges, scoring, ships, sweep as SW
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami.sources import source_eta0

ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="gauges,replay,report")
ap.add_argument("--n-traffic", type=int, default=None)
ap.add_argument("--n-noise", type=int, default=None)
args = ap.parse_args()
cfg = load_config()
if args.n_traffic:
    cfg["sweep"]["n_traffic_draws"] = args.n_traffic
if args.n_noise:
    cfg["sweep"]["n_noise_draws"] = args.n_noise
grid = load_grid(cfg)
out = repo_path(cfg["paths"]["outputs"])
hist_dir = repo_path(cfg["paths"]["historic_runs_dir"])
hist = pd.read_csv(hist_dir / "sources.csv").sort_values("run_id").reset_index(drop=True)
rep = cfg["replay"]
R = cfg["physics"]["earth_radius_m"]


# --------------------------------------------------------------------------------------- gauges
def stage_gauges():
    summ = pd.read_csv(repo_path(rep["summary_csv"]))
    summ = summ.dropna(subset=["post_max_crest_m"])
    summ = summ[~summ["station_code"].isin(rep["exclude_stations"])].reset_index(drop=True)
    cj, ci, dist = gauges._snap(grid, summ["lon_deg"].values, summ["lat_deg"].values, R)
    n = len(summ)
    pool = ships.ReceiverPool(lon=grid.lon[ci], lat=grid.lat[cj], depth=grid.depth[cj, ci], lane_dir=np.full(n, np.nan, np.float32),
                              rtype=np.ones(n, np.int8), code=summ["station_code"].values.astype("<U24"), cell_j=cj, cell_i=ci,
                              coast_dist=grid.coast_dist[cj, ci], weight=np.zeros(n, np.float32), area_id=np.full(n, -1, np.int32))
    solver = Solver(grid, make_sampler(grid, pool), cfg)
    eta0 = np.stack([source_eta0(grid, r, cfg) for r in hist.to_dict("records")])
    rec30 = backend.to_numpy(solver.run(backend.to_xp(eta0)))                     # (B, 240, n, 3)
    eta = rec30[..., 0] - eta0[:, cj, ci][:, None, :]                              # gauges on moving land do not see the static offset
    rows = []
    sim_samos = eta[hist.index[hist["name"] == "samos_2020"][0]]
    hg_lo, hg_hi = rep["gauge_depth_range_m"]
    for k, r in enumerate(summ.itertuples()):
        h_grid = float(grid.depth[cj[k], ci[k]])
        rec_amp = max(abs(r.post_max_crest_m), abs(r.post_min_trough_m))
        sim_amp = float(np.abs(sim_samos[:, k]).max())
        f = lambda hg: (h_grid / hg) ** 0.25
        rows.append(dict(station=r.station_code, name=r.name, snap_km=dist[k] / 1000, h_grid_m=h_grid, recorded_max_abs_m=rec_amp,
                         recorded_crest_m=r.post_max_crest_m, simulated_max_abs_m=sim_amp,
                         ratio_uncorrected=sim_amp / rec_amp, green_factor=f(rep["gauge_depth_m"]),
                         ratio_corrected=sim_amp * f(rep["gauge_depth_m"]) / rec_amp,
                         ratio_corrected_low=sim_amp * f(hg_hi) / rec_amp, ratio_corrected_high=sim_amp * f(hg_lo) / rec_amp))
    df = pd.DataFrame(rows)
    fac = rep["agreement_factor"]
    df["comparable"] = df.snap_km <= rep["max_comparable_snap_km"]
    df["within_factor_2"] = (df.ratio_corrected >= 1 / fac) & (df.ratio_corrected <= fac)
    df.to_csv(out / "historic_samos_gauges.csv", index=False)
    # Amorgos has no recorded gauge series; report the simulated amplitudes at the same stations for reference
    j_am = hist.index[hist["name"] == "amorgos_1956"][0]
    pd.DataFrame({"station": summ["station_code"].values, "simulated_max_abs_m_amorgos": np.abs(eta[j_am]).max(axis=0)}).to_csv(
        out / "historic_amorgos_gauges.csv", index=False)
    print(df.round(3).to_string())
    return df


# --------------------------------------------------------------------------------------- replay
def stage_replay():
    pool = ships.load_pool(cfg)
    cand = pd.read_csv(repo_path(cfg["paths"]["candidates"])).sort_values("cand_id").reset_index(drop=True)
    lib = SW.ScaledLibrary(scoring.Library(repo_path(cfg["paths"]["library_dir"])), SW.load_scale(repo_path("data/processed/uplift_scale.csv"), "cand_id"))
    hscale = SW.load_scale(hist_dir / "uplift_scale.csv", "run_id")
    hlib = SW.ScaledLibrary(scoring.Library(hist_dir), hscale)
    cache = repo_path(cfg["sweep"]["cache_dir"])
    quiet = pd.read_csv(repo_path(cfg["paths"]["truth_dir"]) / "quiet_days.csv")
    qfleets = {int(i): SW.make_fleet(pool, cfg, SW.PURPOSE_QTRAFFIC, int(i)) for i in sorted(quiet.traffic_seed.unique())}
    SW.gather_pass(lib, cand["cand_id"].values, list(qfleets.values()), cfg, cache, SW.quiet_tag(quiet))
    efleets = [SW.make_fleet(pool, cfg, SW.PURPOSE_TRAFFIC, i) for i in range(cfg["sweep"]["n_traffic_draws"])]
    SW.gather_pass(lib, cand["cand_id"].values, efleets, cfg, cache, "lib")
    SW.gather_pass(hlib, hist["run_id"].values, efleets, cfg, cache, "hist", per_ship=True)
    gtable = gauges.table_from_pool(pool, cfg)
    e0_gb = SW.run_eta0_at_gauges(hist, pd.read_csv(hist_dir / "uplift_scale.csv"), grid, cfg, gtable, source_eta0)
    trig, _ = SW.gauge_table(hist, cfg, hist_dir, gtable, scale=hscale, eta0_gb=e0_gb)
    gcols = SW.gauge_columns(hist, trig[:len(hist)], cfg, gtable)
    catalogues = {"arcs_only": np.nonzero((cand["region"] != "aegean").values)[0], "arcs_plus_aegean": np.arange(len(cand))}
    frames = []
    for name, rows in catalogues.items():
        sub = cand.iloc[rows].reset_index(drop=True)
        tdf, thr = SW.thresholds(SW.quiet_stats(cache, SW.quiet_tag(quiet), qfleets, quiet, pool, cfg, cand, rows=rows), cfg)
        tdf.insert(0, "catalogue", name)
        tdf.to_csv(out / f"historic_thresholds_{name}.csv", index=False)
        combos = list(itertools.product(range(len(efleets)), sorted(cfg["traffic"]["range_caps_m"]), cfg["operators"], cfg["weather_regimes"]))
        for n, (i, cap, op, regime) in enumerate(combos):
            f = efleets[i]
            G = np.asarray(SW.load_G(cache, "lib", f, cap, op))[rows]
            Ds = SW.load_G(cache, "hist", f, cap, op)
            sig = np.load(cache / f"hist_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r")
            df = SW.event_shard(pool, f, cap, op, regime, hist, Ds, sig, G, sub, thr[(op, cap, regime)], gcols, cfg, n)
            df.insert(0, "catalogue", name)
            frames.append(df)
        print("catalogue done", name, flush=True)
    res = pd.concat(frames, ignore_index=True)
    res = res.merge(hist[["run_id", "name"]], on="run_id")
    res.to_parquet(out / "historic_replays.parquet")
    return res


def summarise(res):
    g = res.groupby(["name", "catalogue", "operator", "range_cap_m", "regime", "detector"])
    t = g.apply(lambda x: pd.Series(dict(
        trials=len(x), detected_frac=x.detected.mean(),
        median_confirm_min=x.loc[x.detected, "confirm_s"].median() / 60.0 if x.detected.any() else np.nan,
        median_loc_err_km=x.loc[x.detected, "loc_err_m"].median() / 1000.0 if x.detected.any() else np.nan,
        median_best_mw=x.loc[x.detected, "best_mw"].median() if x.detected.any() else np.nan)), include_groups=False).reset_index()
    t.to_csv(out / "historic_replays_summary.csv", index=False)
    return t


def stage_report(res=None, gdf=None):
    res = res if res is not None else pd.read_parquet(out / "historic_replays.parquet")
    gdf = gdf if gdf is not None else pd.read_csv(out / "historic_samos_gauges.csv")
    t = summarise(res)
    ev = res.drop_duplicates("name").set_index("name")
    lines = ["# Historic event replays (T26)", "",
             "Today's merchant traffic (EMODnet 2024 density), the same detectors as the sweep, three weather regimes, two range caps, two operators. "
             f"{cfg['sweep']['n_traffic_draws']} traffic draws x {cfg['sweep']['n_noise_draws']} noise draws per cell of the tables. "
             "The historic runs are separate simulations; the detector's catalogue never contains them. Each event is scored against two catalogues: "
             "arcs only (Hellenic, Calabrian, Cyprus: the blind-spot case) and arcs plus the coarse Aegean grid. Thresholds were recalibrated on the "
             "same quiet days for each catalogue (1% false-alarm rate per 2 h). **A null result is a result.**", ""]
    for name in hist["name"]:
        h = hist[hist["name"] == name].iloc[0]
        lines += [f"## {name}", "", f"Source: Mw {h['mw']:.2f} at {h['lon_deg']:.2f} E, {h['lat_deg']:.2f} N, strike {h['strike_deg']:.0f} deg, polarity {int(h['polarity'])} "
                  f"({'subsidence-first' if h['polarity'] < 0 else 'uplift-first'}); L x W = {h['length_m']/1000:.0f} x {h['width_m']/1000:.0f} km, slip {h['slip_m']:.2f} m.", "",
                  "Fraction of trials detected (and median confirmation time in minutes) by detector, catalogue, operator and range cap, regime = calm / moderate / rough:", "",
                  "| catalogue | operator | cap km | detector | calm | moderate | rough |", "|---|---|---|---|---|---|---|"]
        s = t[t["name"] == name]
        for (cat, op, cap, det), x in s.groupby(["catalogue", "operator", "range_cap_m", "detector"]):
            cells = []
            for reg in cfg["weather_regimes"]:
                r = x[x.regime == reg].iloc[0]
                cells.append(f"{100*r.detected_frac:.0f}%" + (f" ({r.median_confirm_min:.0f} min)" if r.detected_frac > 0 else ""))
            lines.append(f"| {cat} | {op} | {cap/1000:.0f} | {det} | " + " | ".join(cells) + " |")
        e = ev.loc[name]
        lines += ["", f"Tide gauges (simulated, threshold {cfg['gauges']['threshold_m']*100:.0f} cm + {cfg['gauges']['telemetry_delay_s']:.0f} s delay): first gauge {e.first_gauge_s/60:.1f} min, "
                  f"nearest gauge ({e.nearest_gauge_code}) {e.nearest_gauge_s/60:.1f} min, first instrument incl. buoys {e.first_instrument_s/60:.1f} min.", ""]
    lines += ["## Samos 2020: simulated versus recorded gauge amplitudes", "",
              f"Recorded: IOC series 30 to 31 Oct 2020, post-event de-tided max |amplitude| (data/README.md). Simulated: maximum |eta| over 2 h at the grid cell nearest each station "
              f"(2 arc-min grid, min depth 10 m, no harbour). Green's-law factor (h_grid / h_gauge)^(1/4) with h_gauge = {rep['gauge_depth_m']} m (PLACEHOLDER; range "
              f"{rep['gauge_depth_range_m'][0]} to {rep['gauge_depth_range_m'][1]} m shown). Agreement criterion: within a factor of {rep['agreement_factor']}.", "",
              "| station | snap km | h_grid m | recorded cm | simulated cm | ratio raw | Green factor | ratio corrected | corrected range (h_gauge 6..2 m) | within x2 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in gdf.itertuples():
        lines.append(f"| {r.station} | {r.snap_km:.1f} | {r.h_grid_m:.0f} | {100*r.recorded_max_abs_m:.1f} | {100*r.simulated_max_abs_m:.1f} | {r.ratio_uncorrected:.2f} | "
                     f"{r.green_factor:.2f} | {r.ratio_corrected:.2f} | {r.ratio_corrected_low:.2f} to {r.ratio_corrected_high:.2f} | {('yes' if r.within_factor_2 else 'no') if r.comparable else 'not comparable at this resolution'} |")
    comp = gdf[gdf.comparable]
    n_ok = int(comp.within_factor_2.sum())
    lines += ["", f"{n_ok} of {len(comp)} comparable stations agree within a factor of {rep['agreement_factor']} after the correction "
              f"(uplift = slip x sin(dip) = {float(np.sin(np.deg2rad(hist[hist['name']=='samos_2020'].iloc[0]['dip_deg']))):.2f} x slip). The grid cannot resolve harbour resonance, "
              "so large misses at sheltered stations are expected and are a limitation, not a test of the source.", "",
              "Stations marked not comparable (snapped more than "
              f"{rep['max_comparable_snap_km']} km): **Marmaris** (`mrms`) sits inside the enclosed Marmaris Bay; at 2 arc-min its own cell is land, so it was snapped "
              "onto the open coast outside the bay, where the wave is not the one the gauge recorded. This is a resolution limit, not a model failure.", "",
              "**Heraklion** (`hrak`) is snapped correctly (0.3 km, 24 m deep) yet still comes out several times too small. The simulated wave reaches the north coast of Crete "
              "through the Cyclades, whose narrow passages are partly closed on a 2 arc-min grid, and the Heraklion record also contains harbour resonance; "
              "Ierapetra on the south-east coast behaves the same way. These are listed as misses, with that likely cause, not as tuned away.", ""]
    (out / "historic_replays.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


gdf = res = None
if "gauges" in args.stages:
    gdf = stage_gauges()
if "replay" in args.stages:
    res = stage_replay()
if "report" in args.stages:
    stage_report(res, gdf)
