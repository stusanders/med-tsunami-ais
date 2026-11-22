"""T27 demo data: replay the 2020 Samos earthquake with today's traffic (scenes 14 and 15).

Runs on the Mac with NumPy only (never on Colab):

    cd ~/projects/mts-demo && TSUNAMI_BACKEND=numpy PYTHONPATH=src \
        ../med-tsunami-ais/.venv/bin/python scripts/demo/make_samos_data.py

Source: row run_id 0 of data/historic_runs/sources.csv (built by Stream B's make_truth_events.py with
sources.custom_source from data/raw/historic/samos_2020_sources.json, key recommended_for_replay).
The row is cross-checked against custom_source(...) rebuilt from the JSON here.

Uplift (docs/interfaces.md section 7): the historic LIBRARY runs were built with uplift = 0.3 x slip and are
rescaled by data/historic_runs/uplift_scale.csv. This script does NOT use those runs: it calls the current
sources.source_eta0, which already applies peak uplift = slip x sin(dip). So no uplift_scale factor is applied
here (that would double-scale). The simulated maxima are compared with the simulated column of
outputs/historic_samos_gauges.csv (Stream B): a ratio of 1 means B's table already uses slip x sin(dip), a ratio
of uplift_scale (2.48) would mean it still used the legacy 0.3 x slip. The check is written to samos_meta.json.
(First run, Sat 21:28: ratio 1.000 at all 9 stations, so the CSV is on the new rule; the 'Kos 0.22' figure in
status/B.md predates it.)

Writes outputs/demo_data/:
  samos_snapshots.npz   eta (121, ny_c, nx_c) float16 every minute 0..120 min, cropped to the Aegean; lon, lat, wet
  samos_gauges.npz      simulated eta at the IOC stations of historic_samos_gauges.csv (30 s samples) and the
                        recorded, de-tided series (1 min, 60-min centred rolling mean removed, as data/README.md)
  samos_ships.npz       one seeded traffic draw (traffic_draw index 0, purpose 1, 100 km cap): positions, depth,
                        heading-normal signal (S, 120), per-ship calm sigma, SNR(t) = |signal| / sigma (float16),
                        one seeded calm noise draw and the observed series d = signal + noise
  samos_meta.json       provenance, source row, uplift check, peak amplitudes, gauge trigger times, run times
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TSUNAMI_BACKEND", "numpy")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tsunami import backend, gauges, noise, observe, ships, sources  # noqa: E402
from tsunami.config import load_config, repo_path  # noqa: E402
from tsunami.grid import load_grid  # noqa: E402
from tsunami.solver import Solver, make_sampler, to_library  # noqa: E402

assert not backend.GPU, "make_samos_data.py is a Mac/NumPy script (set TSUNAMI_BACKEND=numpy)"

RUN_ID = 0
CROP = dict(lon_deg=(22.0, 30.5), lat_deg=(34.6, 41.0))      # Aegean, for the snapshots
SNAP_STEP_MIN = 1
REGIME, OPERATOR = "calm", "ais"
TRAFFIC_INDEX, TRAFFIC_PURPOSE = 0, 1                        # same draw as scene 1
NOISE_PURPOSE = 27                                            # demo-only seed stream ([seed, 27, 14])
DETIDE_WINDOW_MIN = 60                                        # data/README.md section 3: 60-min centred rolling mean
RECORD_WINDOW_MIN = (-30, 370)                                # recorded series saved relative to origin (IOC summary window ends 18:00 UTC)

cfg = load_config()
OUT = repo_path(cfg["paths"]["outputs"]) / "demo_data"
OUT.mkdir(parents=True, exist_ok=True)
R_EARTH = cfg["physics"]["earth_radius_m"]
timings = {}

# ---------------------------------------------------------------------------- source
hist_dir = repo_path(cfg["paths"]["historic_runs_dir"])
hist = pd.read_csv(hist_dir / "sources.csv")
row = hist[hist["run_id"] == RUN_ID].iloc[0].to_dict()
assert row["name"] == "samos_2020", row["name"]

src_json = json.loads((repo_path(cfg["paths"]["historic_dir"]) / "samos_2020_sources.json").read_text())
rec = src_json["recommended_for_replay"]
rebuilt = sources.custom_source(rec["lon_deg"], rec["lat_deg"], rec["strike_deg"], rec["dip_deg"], rec["mw"], rec["fault"], cfg,
                                length_m=rec["length_m"], width_m=rec["width_m"], slip_m=rec["slip_m"], region="aegean",
                                name="samos_2020")
for k in ("lon_deg", "lat_deg", "strike_deg", "dip_deg", "mw", "length_m", "width_m", "slip_m", "polarity"):
    assert np.isclose(float(rebuilt[k]), float(row[k])), (k, rebuilt[k], row[k])

grid = load_grid(cfg)
eta0 = sources.source_eta0(grid, row, cfg).astype(np.float32)
uplift_factor = sources.uplift_factor(row)                      # sin(dip): no uplift_dip_deg column -> dip_deg
legacy = cfg["sources"]["legacy_uplift_fraction_of_slip"]
scale_csv = pd.read_csv(hist_dir / "uplift_scale.csv").set_index("run_id").loc[RUN_ID, "uplift_scale"]
assert np.isclose(uplift_factor / legacy, scale_csv), (uplift_factor / legacy, scale_csv)
peak_lobe = float(uplift_factor * row["slip_m"])
print(f"Samos source: Mw {row['mw']}, slip {row['slip_m']} m, dip {row['dip_deg']} deg -> lobe peak slip*sin(dip) = "
      f"{peak_lobe:.3f} m (legacy 0.3*slip = {legacy*row['slip_m']:.3f}); eta0 min {eta0.min():.3f} max {eta0.max():.3f} m")

# ---------------------------------------------------------------------------- receivers: project pool + IOC Samos stations
pool = ships.load_pool(cfg)
R0 = len(pool)
cmp_csv = pd.read_csv(repo_path(cfg["paths"]["outputs"]) / "historic_samos_gauges.csv")
summ = pd.read_csv(repo_path(cfg["paths"]["historic_dir"]) / "ioc_samos_2020" / "summary_samos_2020_ioc.csv")
st = cmp_csv[["station", "name"]].merge(summ[["station_code", "lon_deg", "lat_deg", "post_max_crest_m", "post_min_trough_m"]],
                                       left_on="station", right_on="station_code", how="left")
cj, ci, dist = gauges._snap(grid, st["lon_deg"].values, st["lat_deg"].values, R_EARTH)   # same snap as Stream B
n_st = len(st)
ext = ships.ReceiverPool(
    lon=np.concatenate([pool.lon, grid.lon[ci]]), lat=np.concatenate([pool.lat, grid.lat[cj]]),
    depth=np.concatenate([pool.depth, grid.depth[cj, ci].astype(np.float32)]),
    lane_dir=np.concatenate([pool.lane_dir, np.full(n_st, np.nan, np.float32)]),
    rtype=np.concatenate([pool.rtype, np.ones(n_st, np.int8)]),
    code=np.concatenate([pool.code, st["station"].values.astype(pool.code.dtype)]),
    cell_j=np.concatenate([pool.cell_j, cj]), cell_i=np.concatenate([pool.cell_i, ci]),
    coast_dist=np.concatenate([pool.coast_dist, grid.coast_dist[cj, ci].astype(np.float32)]),
    weight=np.concatenate([pool.weight, np.zeros(n_st, np.float32)]),
    area_id=np.concatenate([pool.area_id, np.full(n_st, -1, np.int32)]))
solver = Solver(grid, make_sampler(grid, ext), cfg, xp=np)

# ---------------------------------------------------------------------------- run 1: receivers
t0 = time.time()
rec30 = solver.run(eta0[None])                                                 # (1, 240, R0 + n_st, 3)
timings["receivers_run_s"] = time.time() - t0
gb_idx = np.nonzero(pool.rtype > 0)[0]
recf, gb30 = to_library(rec30[:, :, :R0], gb_idx, cfg)                         # (1, 120, R0, 3), (1, 240, R_gb)
ioc_eta = rec30[0, :, R0:, 0]                                                  # (240, n_st), raw eta at 30 s
t30_s = solver.sample_interval_s * (np.arange(rec30.shape[1]) + 1)
del rec30
print(f"receiver run {timings['receivers_run_s']:.0f} s")

# ---------------------------------------------------------------------------- run 2: snapshots every minute
times_s = np.arange(0, cfg["solver"]["t_end_s"] + 1, 60.0 * SNAP_STEP_MIN)
t0 = time.time()
snaps = solver.snapshots(eta0[None], times_s)[0]                               # (121, ny, nx) float32
timings["snapshots_run_s"] = time.time() - t0
jc = np.nonzero((grid.lat >= CROP["lat_deg"][0]) & (grid.lat <= CROP["lat_deg"][1]))[0]
ic = np.nonzero((grid.lon >= CROP["lon_deg"][0]) & (grid.lon <= CROP["lon_deg"][1]))[0]
sl = (slice(jc[0], jc[-1] + 1), slice(ic[0], ic[-1] + 1))
snap_c = snaps[:, sl[0], sl[1]]
crop_max_abs = np.abs(snap_c).max(axis=(1, 2))
np.savez_compressed(OUT / "samos_snapshots.npz", eta=snap_c.astype(np.float16), t_min=(times_s / 60.0).astype(np.float32),
                    lon=grid.lon[sl[1]], lat=grid.lat[sl[0]], wet=grid.wet[sl], depth=grid.depth[sl].astype(np.float32))
del snaps
print(f"snapshot run {timings['snapshots_run_s']:.0f} s; crop {snap_c.shape}")

# ---------------------------------------------------------------------------- recorded IOC series (de-tided)
origin = pd.Timestamp(src_json["origin_time_utc"])
rec_t, rec_eta = {}, {}
for code in st["station"]:
    p = repo_path(cfg["paths"]["historic_dir"]) / "ioc_samos_2020" / f"{code}_20201030_20201031.csv"
    if not p.exists():
        continue
    d = pd.read_csv(p, parse_dates=["stime"]).dropna(subset=["slevel"]).set_index("stime").sort_index()
    s = d["slevel"].astype(float)
    tide = s.rolling(f"{DETIDE_WINDOW_MIN}min", center=True, min_periods=10).mean()
    res = s - tide
    tm = (res.index - origin).total_seconds() / 60.0
    keep = (tm >= RECORD_WINDOW_MIN[0]) & (tm <= RECORD_WINDOW_MIN[1])
    rec_t[code], rec_eta[code] = tm[keep].values.astype(np.float32), res.values[keep].astype(np.float32)

gz = dict(codes=st["station"].values.astype("<U8"), names=st["name"].values.astype("<U40"),
          lon=st["lon_deg"].values, lat=st["lat_deg"].values, snap_lon=grid.lon[ci], snap_lat=grid.lat[cj],
          snap_km=dist / 1000.0, h_grid_m=grid.depth[cj, ci].astype(np.float32),
          sim_t_min=(t30_s / 60.0).astype(np.float32), sim_eta=ioc_eta.astype(np.float32))
for code in rec_t:
    gz[f"rec_t_min_{code}"], gz[f"rec_eta_{code}"] = rec_t[code], rec_eta[code]
np.savez_compressed(OUT / "samos_gauges.npz", **gz)

# ---------------------------------------------------------------------------- traffic draw, signal, SNR, one noise draw
cap = cfg["traffic"]["range_caps_m"][-1]
draw = ships.apply_range_cap(ships.traffic_draw(pool, cfg, TRAFFIC_INDEX, purpose=TRAFFIC_PURPOSE), pool, cap)
sig = observe.ais_signal(observe.gather_truth(recf[0], draw), draw)                    # (S, 120) m/s
sigma = np.sqrt(noise.ship_sigma(draw, REGIME, OPERATOR, cfg).astype(np.float64) ** 2
                + noise.shared_sigma(REGIME, OPERATOR, cfg) ** 2)                      # (S,) total filtered sd
snr_t = np.abs(sig) / sigma[:, None]
rng = np.random.default_rng([cfg["meta"]["seed"], NOISE_PURPOSE, 14])
nz = noise.ais_noise(draw, pool, REGIME, cfg, rng, n_draws=1)[0]                       # (S, 120)
dobs = observe.observe(sig, nz[None])[0]
peak = snr_t.max(axis=1)
k_sig = cfg["single_ship"]["literal_k_sigma"]
np.savez_compressed(OUT / "samos_ships.npz", lon=pool.lon[draw.pool_idx], lat=pool.lat[draw.pool_idx],
                    depth=pool.depth[draw.pool_idx], pool_idx=draw.pool_idx, heading=draw.heading, sog=draw.sog,
                    signal=sig.astype(np.float32), sigma=sigma.astype(np.float32), snr_t=snr_t.astype(np.float16),
                    peak_snr=peak.astype(np.float32), noise=nz.astype(np.float32), observed=dobs.astype(np.float32),
                    t_min=np.arange(1, sig.shape[1] + 1, dtype=np.float32))

# ---------------------------------------------------------------------------- project gauges and buoys (5 cm rule)
gt = gauges.load_gauges(grid, cfg)
if len(gt) != len(gb_idx):                                  # pool rebuilt before/after buoys: keep the matching part
    gt = gt[gt["rtype"] == 1].reset_index(drop=True) if (gt["rtype"] == 1).sum() == len(gb_idx) else gt.iloc[:len(gb_idx)]
trig = gauges.trigger_times(gb30, gt, cfg["solver"]["sample_interval_s"])[0]
gfin = np.isfinite(trig) & (gt["rtype"].values == 1)
first_gauge = dict(code=str(gt["code"].values[gfin][np.argmin(trig[gfin])]), name=str(gt["name"].values[gfin][np.argmin(trig[gfin])]),
                   t_s=float(trig[gfin].min())) if gfin.any() else None

# ---------------------------------------------------------------------------- meta + consistency with Stream B's table
st_rows = []
for k, r in st.iterrows():
    sim_amp = float(np.abs(ioc_eta[:, k]).max())
    b = cmp_csv.iloc[k]
    rec_series_amp = rec_full_amp = rec_full_t = None
    if r.station in rec_t:
        rt_, re_ = rec_t[r.station], rec_eta[r.station]
        w2 = (rt_ >= 0) & (rt_ <= cfg["solver"]["t_end_s"] / 60.0)
        rec_series_amp = float(np.abs(re_[w2]).max())
        wa = rt_ >= 0
        rec_full_amp = float(np.abs(re_[wa]).max())
        rec_full_t = float(rt_[wa][np.argmax(np.abs(re_[wa]))])
    st_rows.append(dict(
        station=r.station, name=r["name"], snap_km=float(dist[k] / 1000), h_grid_m=float(grid.depth[cj[k], ci[k]]),
        sim_max_abs_m=sim_amp, sim_peak_t_min=float(t30_s[np.argmax(np.abs(ioc_eta[:, k]))] / 60.0),
        recorded_max_abs_m=float(b.recorded_max_abs_m), recorded_series_max_abs_0_120min_m=rec_series_amp,
        recorded_series_max_abs_full_m=rec_full_amp, recorded_series_max_t_min=rec_full_t,
        ratio_sim_over_recorded_0_120min=(sim_amp / rec_series_amp) if rec_series_amp else None,
        ratio_sim_over_recorded=sim_amp / float(b.recorded_max_abs_m),
        streamB_sim_max_abs_legacy_m=float(b.simulated_max_abs_m), ratio_to_streamB=sim_amp / float(b.simulated_max_abs_m),
        green_factor=float(b.green_factor), ratio_green_corrected=sim_amp * float(b.green_factor) / float(b.recorded_max_abs_m)))
rat = np.array([s["ratio_to_streamB"] for s in st_rows])
kos = next(s for s in st_rows if s["station"] == "kos1")
meta = dict(
    script="scripts/demo/make_samos_data.py", backend="numpy", created_unix=time.time(),
    source=dict(file="data/historic_runs/sources.csv", run_id=RUN_ID, row={k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()},
                crosscheck="matches sources.custom_source(recommended_for_replay of data/raw/historic/samos_2020_sources.json)",
                origin_time_utc=src_json["origin_time_utc"]),
    uplift=dict(rule=cfg["sources"]["uplift_rule"], uplift_factor_sin_dip=uplift_factor, peak_lobe_m=peak_lobe,
                eta0_min_m=float(eta0.min()), eta0_max_m=float(eta0.max()),
                legacy_fraction=legacy, uplift_scale_csv=float(scale_csv), applied_extra_scale=1.0,
                note="source_eta0 on this branch already uses slip*sin(dip); data/historic_runs/uplift_scale.csv is NOT applied "
                     "(it converts the legacy 0.3*slip library runs). ratio_to_streamB = 1 means outputs/historic_samos_gauges.csv "
                     "is on the same slip*sin(dip) rule; = uplift_scale_csv would mean it used the legacy 0.3*slip."),
    consistency_with_streamB=dict(ratio_to_streamB_median=float(np.median(rat)), ratio_to_streamB_min=float(rat.min()),
                                  ratio_to_streamB_max=float(rat.max()),
                                  streamB_rule=("slip*sin(dip) (same as ours)" if np.allclose(rat, 1.0, rtol=1e-3) else
                                                "legacy 0.3*slip" if np.allclose(rat, scale_csv, rtol=1e-3) else "unknown")),
    stations=st_rows, kos=kos,
    recorded=dict(dir="data/raw/historic/ioc_samos_2020", detide=f"{DETIDE_WINDOW_MIN}-min centred rolling mean removed (data/README.md section 3)",
                  stations_with_series=sorted(rec_t)),
    snapshots=dict(file="samos_snapshots.npz", step_min=SNAP_STEP_MIN, n=int(len(times_s)), crop=CROP, dtype="float16",
                   crop_max_abs_m=crop_max_abs.astype(float).tolist()),
    ships=dict(file="samos_ships.npz", traffic_index=TRAFFIC_INDEX, traffic_purpose=TRAFFIC_PURPOSE, range_cap_m=cap,
               regime=REGIME, operator=OPERATOR, noise_seed=[cfg["meta"]["seed"], NOISE_PURPOSE, 14], n_ships=int(len(draw.pool_idx)),
               snr_definition="|heading-normal filtered current| / sqrt(ship_sigma^2 + shared_sigma^2), calm AIS (as scripts/03_one_scenario.py)",
               max_peak_snr=float(peak.max()), n_peak_snr_gt1=int((peak > 1).sum()), n_peak_snr_gt3=int((peak > k_sig).sum()),
               k_sigma=float(k_sig), sigma_median_m_s=float(np.median(sigma))),
    project_gauges=dict(threshold_m=float(cfg["gauges"]["threshold_m"]), delay_s=float(cfg["gauges"]["telemetry_delay_s"]),
                        first_gauge=first_gauge, n_triggered=int(gfin.sum()), n_gauges=int((gt["rtype"] == 1).sum())),
    timings_s=timings)
(OUT / "samos_meta.json").write_text(json.dumps(meta, indent=1, default=float))
print(json.dumps(dict(kos=kos, consistency=meta["consistency_with_streamB"], ships=meta["ships"], first_gauge=first_gauge), indent=1, default=float))
