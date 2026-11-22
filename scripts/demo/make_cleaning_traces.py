"""T27 demo data: AIS cleaning traces for scene 6 (step 3), regenerated with the real project code.

Reimplements the T24 cleaning illustration (Stream C, `scripts/fig_cleaning_steps.py`) for the
demo animation, without the synthetic turn artefact, and at the per-report level so the video can
show every stage of the cleaning:

  1. raw drift      every AIS report (report_interval_s), sideways component
                    = true tsunami current (real solver, interpolated to the report time)
                    + sog x heading error (wobble + integer-degree rounding, per report)
                    + sog x constant heading misalignment (per ship)
                    + shared 2D current field (noise.shared_field, AR(1) per minute)
  2. 1-min average  timeseries-style mean of the reports in each minute
  3. baseline       the ship's normal drift: causal 45-min trailing mean (noise.filter_with_history,
                    i.e. timeseries.baseline_filter with W minutes of pre-origin history)
  4. cleaned        1-min average minus baseline, compared with the true current
                    (unfiltered and after the same causal filter)

The minute means of the per-report errors have exactly the distribution that `noise.ais_noise`
draws directly (sd = noise.heading_sd_per_minute), so the cleaned trace is a realisation of the
noise the detector sees.

Source: showcase cand_id 186 (outputs/first_detection.json), real grid and receiver pool,
`solver.Solver` on NumPy. Ships: pool 3179 (shelf near Crete) and pool 2557 (deep water),
the two ships T24 selects; the script re-runs T24's selection rule and records whether it agrees.

Outputs: outputs/demo_data/cleaning_traces.npz and cleaning_traces.json (metadata sidecar).

Usage (from the repo root):
  TSUNAMI_BACKEND=numpy PYTHONPATH=src python scripts/demo/make_cleaning_traces.py
"""
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

from tsunami import backend, geometry, noise, ships, timeseries  # noqa: E402
from tsunami.config import load_config, repo_path  # noqa: E402
from tsunami.grid import load_grid  # noqa: E402
from tsunami.solver import Solver, make_sampler  # noqa: E402
from tsunami.sources import source_eta0, uplift_factor  # noqa: E402
from tsunami.units import deg2rad, knots_to_ms  # noqa: E402

SHOWCASE_CAND = 186
SHIPS = {"shelf": 3179, "deep": 2557}
REGIME = "calm"
SEED_TAG = 27  # rng = default_rng([cfg seed, 27, 6])


def t24_selection(pool, recf):
    """T24's rule: strongest filtered sideways current on the Crete shelf / deep path boxes."""
    idx = np.where(pool.rtype == 0)[0]
    uf = geometry.project_normal(recf[:, idx, 1], recf[:, idx, 2], pool.lane_dir[idx][None, :]).T
    peaks = np.max(np.abs(uf), axis=1)
    lon, lat, dep = pool.lon[idx], pool.lat[idx], pool.depth[idx]
    shelf = (dep <= 200) & (lat >= 34.0) & (lat <= 36.5) & (lon >= 22.5) & (lon <= 27.0)
    deep = (dep >= 2000) & (lat >= 33.5) & (lat <= 36.5) & (lon >= 20.0) & (lon <= 25.0)
    return (int(idx[np.where(shelf)[0][np.argmax(peaks[shelf])]]),
            int(idx[np.where(deep)[0][np.argmax(peaks[deep])]]))


def main():
    t_start = time.time()
    cfg = load_config(REPO / "config.yaml")
    grid = load_grid(cfg)
    pool = ships.load_pool(cfg)
    cand = pd.read_csv(repo_path(cfg["paths"]["candidates"]))
    fd = json.loads((REPO / "outputs" / "first_detection.json").read_text())
    cand_id = int(fd["source"]["cand_id"]) if fd else SHOWCASE_CAND
    row = cand[cand["cand_id"] == cand_id].iloc[0].to_dict()
    eta0 = source_eta0(grid, row, cfg)

    print(f"solver: cand {cand_id} (Mw {row['mw']}), backend {backend.xp.__name__}")
    solver = Solver(grid, make_sampler(grid, pool), cfg)
    rec30 = backend.to_numpy(solver.run(backend.to_xp(eta0[None])))[0]     # (240, R, 3)
    t_solve = time.time() - t_start
    per_min = int(round(cfg["library"]["minute_s"] / cfg["solver"]["sample_interval_s"]))
    W, T = int(cfg["library"]["baseline_window_min"]), int(cfg["library"]["n_minutes"])
    rec1 = timeseries.block_average(rec30, per_min, axis=0)                 # (120, R, 3)
    recf = timeseries.baseline_filter(rec1, W, axis=0)                      # (120, R, 3)

    sel = t24_selection(pool, recf)
    p = np.array([SHIPS["shelf"], SHIPS["deep"]])
    sog_kn = float(np.mean(cfg["traffic"]["speed_kn"]))
    sog = float(knots_to_ms(sog_kn))
    heading = pool.lane_dir[p].astype(np.float64)
    draw = ships.TrafficDraw(pool_idx=p.astype(np.int32), heading=heading.astype(np.float32),
                             sog=np.full(2, sog, np.float32), seed=0)

    # true sideways current (starboard normal of each ship's heading)
    proj = lambda rec: geometry.project_normal(rec[:, p, 1].astype(np.float64),
                                               rec[:, p, 2].astype(np.float64), heading[None, :]).T
    u30, u1, uf = proj(rec30), proj(rec1), proj(recf)                       # (2, 240), (2, 120), (2, 120)
    eta30 = rec30[:, p, 0].T.astype(np.float64)

    # per-report noise, W minutes of pre-origin history + T minutes
    a = cfg["ais_noise"]
    dt_rep = float(a["report_interval_s"])
    n_rep = int(round(cfg["library"]["minute_s"] / dt_rep))
    n_all = (W + T) * n_rep
    rng = np.random.default_rng([int(cfg["meta"]["seed"]), SEED_TAG, 6])
    wob = deg2rad(a["heading_wobble_sd_deg"]) * noise.regime_multiplier(REGIME, cfg)
    res = deg2rad(a["heading_resolution_deg"])
    phi = rng.uniform(0, res, size=(2, 1))                                  # sub-degree true heading
    head_err = res * np.round((phi + wob * rng.standard_normal((2, n_all))) / res) - phi
    bias = deg2rad(a["heading_bias_sd_deg"]) * rng.standard_normal((2, 1))
    sd_sh = a["shared_sd_m_s"] * noise.regime_multiplier(REGIME, cfg)
    f = noise.shared_field(noise.ship_xy_m(pool, draw, cfg), sd_sh, cfg, rng, 1, W + T)[0]  # (2c, 2s, W+T)
    shared_min = geometry.project_normal(f[0].astype(np.float64), f[1].astype(np.float64), heading[:, None])
    noise_rep = sog * head_err + sog * bias + np.repeat(shared_min, n_rep, axis=1)       # (2, n_all)

    # true current at each report time (0 before origin; linear between 30 s samples)
    t_rep_all = dt_rep * (np.arange(n_all) + 1) - W * 60.0                  # s after origin
    t30 = np.concatenate([[0.0], cfg["solver"]["sample_interval_s"] * (np.arange(u30.shape[1]) + 1)])
    true_rep = np.stack([np.interp(t_rep_all, t30, np.concatenate([[0.0], u30[s]]), left=0.0)
                         for s in range(2)])
    raw_rep = true_rep + noise_rep                                          # (2, n_all)
    raw_min_all = raw_rep.reshape(2, W + T, n_rep).mean(axis=2)             # (2, W+T)
    clean = noise.filter_with_history(raw_min_all.astype(np.float64), cfg)  # (2, T)
    raw_min = raw_min_all[:, W:]
    baseline = raw_min - clean

    sig_i = noise.ship_sigma(draw, REGIME, "ais", cfg).astype(np.float64)
    sig_tot = np.sqrt(sig_i ** 2 + noise.shared_sigma(REGIME, "ais", cfg) ** 2)
    sig_rep = sog * np.sqrt(wob ** 2 + res ** 2 / 12.0)

    out_dir = REPO / "outputs" / "demo_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "cleaning_traces.npz",
        ship_names=np.array(["shelf", "deep"]), pool_idx=p,
        lon=pool.lon[p], lat=pool.lat[p], depth_m=pool.depth[p], heading_rad=heading,
        t_report_s=t_rep_all[W * n_rep:], raw_report=raw_rep[:, W * n_rep:],
        t_report_hist_s=t_rep_all[:W * n_rep], raw_report_hist=raw_rep[:, :W * n_rep],
        t_min_end=np.arange(1, T + 1, dtype=np.float64), raw_min=raw_min, baseline=baseline,
        clean=clean, raw_min_hist=raw_min_all[:, :W],
        t30_s=t30[1:], true_30s=u30, eta_30s=eta30, true_1min=u1, true_filtered=uf,
        sigma_total=sig_tot, sigma_report=np.full(2, sig_rep), bias_m_s=(sog * bias)[:, 0],
        eta0_max_m=float(eta0.max()), eta0_min_m=float(eta0.min()),
    )
    meta = {
        "script": "scripts/demo/make_cleaning_traces.py",
        "based_on": "stream-c:scripts/fig_cleaning_steps.py (T24), reimplemented; no synthetic turn",
        "cand_id": cand_id, "mw": float(row["mw"]), "region": row["region"],
        "slip_m": float(row["slip_m"]), "uplift_dip_deg": float(row["uplift_dip_deg"]),
        "uplift_factor": uplift_factor(row), "peak_uplift_m": float(eta0.max()),
        "regime": REGIME, "sog_kn": sog_kn, "baseline_window_min": W,
        "report_interval_s": dt_rep, "seed": [int(cfg["meta"]["seed"]), SEED_TAG, 6],
        "backend": backend.xp.__name__, "solver_seconds": round(t_solve, 1),
        "ships": {k: {"pool_idx": int(i), "depth_m": float(pool.depth[i]), "lon": float(pool.lon[i]),
                      "lat": float(pool.lat[i]), "sigma_total_m_s": float(sig_tot[s]),
                      "peak_true_filtered_m_s": float(np.abs(uf[s]).max()),
                      "peak_true_1min_m_s": float(np.abs(u1[s]).max()),
                      "peak_snr": float(np.abs(uf[s]).max() / sig_tot[s])}
                  for s, (k, i) in enumerate(SHIPS.items())},
        "t24_selection_rule_gives": {"shelf": sel[0], "deep": sel[1]},
        "t24_selection_agrees": bool(sel == (SHIPS["shelf"], SHIPS["deep"])),
    }
    (out_dir / "cleaning_traces.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {out_dir / 'cleaning_traces.npz'} in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
