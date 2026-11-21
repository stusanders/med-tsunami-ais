"""T24: Cleaning-steps illustration on the showcase Hellenic Mw 8 run (cand 186).

Uses the real solver (`solver.Solver`) on the real grid and receiver pool with the same
Hellenic Arc Mw 8.0 source as T13 (`outputs/first_detection.json`, cand_id 186), comparing
two receiver-pool ships side-by-side:
  - Left column:  strongest shelf ship near Crete (depth ~61 m, strong shoaled current)
  - Right column: deep-water ship along the wave path (depth ~2,766 m, weak current)

Noise is generated strictly by `noise.ais_noise(draw, pool, "calm", cfg, rng, 1)` (heading
rounding + wobble + shared 2D current field + calm multiplier + 45-min causal baseline
history, all read from `config.yaml`). A synthetic 5-minute turn manoeuvre (`v_yaw = -L * ROT`)
is added only in Panel 1 to illustrate why yaw correction is required before baseline filtering.

Outputs:
  - outputs/cleaning_steps.png

Usage:
  PYTHONPATH=src python scripts/fig_cleaning_steps.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import backend, geometry, noise, ships, timeseries, viz
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami.sources import source_eta0
from tsunami.units import deg2rad, knots_to_ms


def select_showcase_source(cfg):
    """Select the exact Hellenic Mw 8.0 source used in T13 (outputs/first_detection.json, cand 186)."""
    cand = pd.read_csv(repo_path(cfg["paths"]["candidates"]))
    fd_path = repo_path(cfg["paths"]["outputs"]) / "first_detection.json"
    cand_id = None
    if fd_path.exists():
        try:
            cand_id = int(json.loads(fd_path.read_text(encoding="utf-8"))["source"]["cand_id"])
        except Exception:
            cand_id = None
    if cand_id is not None and (cand["cand_id"] == cand_id).any():
        return cand[cand["cand_id"] == cand_id].iloc[0].to_dict()
    hel = cand[(cand["region"] == "hellenic") & np.isclose(cand["mw"], 8.0)]
    d2 = (hel["lon_deg"] - 23.35) ** 2 + (hel["lat_deg"] - 34.08) ** 2
    return hel.loc[d2.idxmin()].to_dict()


def simulate_two_ships(cfg, regime="calm"):
    """Run cand 186 with Solver, select shelf (Crete) and deep-water ships, generate simulation noise."""
    grid = load_grid(cfg)
    pool = ships.load_pool(cfg)
    row = select_showcase_source(cfg)
    eta0 = source_eta0(grid, row, cfg)

    solver = Solver(grid, make_sampler(grid, pool), cfg)
    rec30 = backend.to_numpy(solver.run(backend.to_xp(eta0[None])))[0]  # (240, R, 3)
    per_min = int(round(cfg["library"]["minute_s"] / cfg["solver"]["sample_interval_s"]))
    rec1 = timeseries.block_average(rec30, per_min, axis=0)              # (120, R, 3) unfiltered 1-min
    W = int(cfg["library"]["baseline_window_min"])
    recf = timeseries.baseline_filter(rec1, W, axis=0)                   # (120, R, 3) causally filtered

    # Project all ship pool points onto each point's own lane normal to find shelf vs deep examples
    ship_idx = np.where(pool.rtype == 0)[0]
    hdg_all = pool.lane_dir[ship_idx]
    uf_all = geometry.project_normal(recf[:, ship_idx, 1], recf[:, ship_idx, 2], hdg_all[None, :]).T
    peaks = np.max(np.abs(uf_all), axis=1)

    crete_shelf = (
        (pool.depth[ship_idx] <= 200.0)
        & (pool.lat[ship_idx] >= 34.0)
        & (pool.lat[ship_idx] <= 36.5)
        & (pool.lon[ship_idx] >= 22.5)
        & (pool.lon[ship_idx] <= 27.0)
    )
    deep_path = (
        (pool.depth[ship_idx] >= 2000.0)
        & (pool.lat[ship_idx] >= 33.5)
        & (pool.lat[ship_idx] <= 36.5)
        & (pool.lon[ship_idx] >= 20.0)
        & (pool.lon[ship_idx] <= 25.0)
    )
    p_shelf = int(ship_idx[np.where(crete_shelf)[0][np.argmax(peaks[crete_shelf])]])
    p_deep = int(ship_idx[np.where(deep_path)[0][np.argmax(peaks[deep_path])]])

    # Build 2-ship TrafficDraw at 14 kn (midpoint of config traffic.speed_kn)
    sog_kn = float(np.mean(cfg["traffic"]["speed_kn"]))
    sog_ms = float(knots_to_ms(sog_kn))
    draw = ships.TrafficDraw(
        pool_idx=np.array([p_shelf, p_deep], dtype=np.int32),
        heading=np.array([pool.lane_dir[p_shelf], pool.lane_dir[p_deep]], dtype=np.float32),
        sog=np.full(2, sog_ms, dtype=np.float32),
        seed=0,
    )

    # Project unfiltered and causally filtered true currents onto each ship's own heading normal
    u_unfilt = geometry.project_normal(
        rec1[:, draw.pool_idx, 1], rec1[:, draw.pool_idx, 2], draw.heading[None, :]
    ).T.astype(np.float64)                                               # (2, 120)
    u_filt = geometry.project_normal(
        recf[:, draw.pool_idx, 1], recf[:, draw.pool_idx, 2], draw.heading[None, :]
    ).T.astype(np.float64)                                               # (2, 120)

    # Simulation noise directly from noise.ais_noise (reads calm multiplier, wobble, shared field from config)
    rng = np.random.default_rng([cfg["meta"]["seed"], 24, 1])
    nz_filt = noise.ais_noise(draw, pool, regime, cfg, rng, 1)[0].astype(np.float64)  # (2, 120)
    sig_i = noise.ship_sigma(draw, regime, "ais", cfg).astype(np.float64)
    sig_sh = float(noise.shared_sigma(regime, "ais", cfg))
    sig_tot = np.sqrt(sig_i ** 2 + sig_sh ** 2)

    # Constant heading-misalignment offset (removed exactly by the causal 45-min baseline filter)
    bias_rad = deg2rad(float(cfg["ais_noise"]["heading_bias_sd_deg"]))
    v_bias = np.array([sog_ms * bias_rad, -sog_ms * bias_rad], dtype=np.float64)  # (2,)

    # Synthetic 5-min course-change yaw artefact: v_yaw(t) = -L_antenna * ROT(t)
    t_min = np.arange(1, int(cfg["library"]["n_minutes"]) + 1, dtype=np.float64)
    turn_start, turn_end = 18.0, 23.0
    in_turn = (t_min >= turn_start) & (t_min <= turn_end)
    rot_rad_s = np.zeros_like(t_min)
    # Smooth half-sine course change peaking at 18 deg/min (~0.0052 rad/s); L = 120 m merchant ship
    rot_peak_deg_min = 18.0
    rot_rad_s[in_turn] = deg2rad(rot_peak_deg_min / 60.0) * np.sin(
        np.pi * (t_min[in_turn] - turn_start) / (turn_end - turn_start)
    )
    L_illust_m = 120.0
    v_yaw = -L_illust_m * rot_rad_s  # clockwise turn swings aft GNSS antenna to port (negative)

    return {
        "row": row,
        "pool": pool,
        "draw": draw,
        "t_min": t_min,
        "u_unfilt": u_unfilt,
        "u_filt": u_filt,
        "nz_filt": nz_filt,
        "sig_tot": sig_tot,
        "v_bias": v_bias,
        "v_yaw": v_yaw,
        "turn_window": (turn_start, turn_end),
        "sog_kn": sog_kn,
        "regime": regime,
        "W": W,
    }


def make_cleaning_figure(cfg, out_path=None):
    viz.apply_style()
    data = simulate_two_ships(cfg, regime="calm")
    t_min = data["t_min"]
    pool = data["pool"]
    draw = data["draw"]
    t0, t1 = data["turn_window"]
    v_yaw = data["v_yaw"]

    col_titles = [
        f"Shelf near Crete (depth {pool.depth[draw.pool_idx[0]]:.0f} m, {pool.lat[draw.pool_idx[0]]:.2f}\u00b0N {pool.lon[draw.pool_idx[0]]:.2f}\u00b0E)",
        f"Deep water (depth {pool.depth[draw.pool_idx[1]]:.0f} m, {pool.lat[draw.pool_idx[1]]:.2f}\u00b0N {pool.lon[draw.pool_idx[1]]:.2f}\u00b0E)",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 9.2), sharex=True)

    for col in range(2):
        u_un = data["u_unfilt"][col]
        u_fi = data["u_filt"][col]
        nz = data["nz_filt"][col]
        vb = data["v_bias"][col]
        sig = data["sig_tot"][col]

        # Stage 1: raw 1-min drift = true current + calm noise + steady compass bias + synthetic yaw spike
        v_step2 = u_un + nz + vb
        v_step1 = v_step2 + v_yaw
        # Stage 3: main simulation observation = causally baseline-filtered current + ais_noise
        v_step3 = u_fi + nz

        # Panel 1: Raw drift + separate v_yaw curve
        ax1 = axes[0, col]
        ax1.plot(t_min, v_step1, color="#555555", lw=1.0, label="Raw 1-min AIS drift (incl. bias & turn)")
        ax1.plot(t_min, v_yaw, color=viz.COLORS["negative"], lw=1.6, ls="--", label="Synthetic yaw artefact $v_{\\mathrm{yaw}} = -L \\cdot \\mathrm{ROT}$")
        ax1.axhline(0.0, color="#888888", lw=0.6, ls=":")
        ax1.set_title(f"{col_titles[col]}\n1. Raw drift with synthetic turn artefact", fontsize=9.8)
        ax1.grid(True, linestyle=":", alpha=0.55)
        y1_lim = max(0.85, float(np.max(np.abs(v_step1))) * 1.28)
        ax1.set_ylim(-y1_lim, y1_lim)
        if col == 0:
            ax1.set_ylabel("Sideways drift (m/s)")
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.92)

        # Panel 2: After yaw subtraction (turn window shaded; legend kept clear of trace)
        ax2 = axes[1, col]
        ax2.axvspan(t0, t1, color=viz.COLORS["negative"], alpha=0.14, label="Turn window (yaw subtracted)")
        ax2.plot(t_min, v_step2, color=viz.COLORS["moderate"], lw=1.15, label="After yaw correction (still has steady bias)")
        ax2.axhline(vb, color="#333333", lw=1.0, ls="--", label=f"Steady compass bias ({vb:+.2f} m/s)")
        ax2.axhline(0.0, color="#888888", lw=0.6, ls=":")
        ax2.set_title("2. After yaw subtraction (before baseline filter)", fontsize=9.5)
        ax2.grid(True, linestyle=":", alpha=0.55)
        # Expand upper margin so legend sits cleanly above the trace
        y2_lo = min(-0.55, float(np.min(v_step2)) - 0.15)
        y2_hi = max(0.75, float(np.max(v_step2)) + 0.42)
        ax2.set_ylim(y2_lo, y2_hi)
        if col == 0:
            ax2.set_ylabel("Sideways drift (m/s)")
        ax2.legend(loc="upper right", fontsize=7.8, ncol=1, framealpha=0.95)

        # Panel 3: After causal 45-min baseline subtraction (exact simulation noise + true currents)
        ax3 = axes[2, col]
        ax3.plot(t_min, v_step3, color=viz.COLORS["ships"], lw=1.05, alpha=0.85,
                 label=f"Cleaned drift (calm $\\sigma={sig:.3f}$ m/s, $3\\sigma={3*sig:.2f}$ m/s)")
        ax3.plot(t_min, u_un, color="#111111", lw=2.2, ls="-", label="True tsunami current (unfiltered)")
        ax3.plot(t_min, u_fi, color=viz.COLORS["positive"], lw=1.6, ls="--",
                 label=f"True current after {data['W']}-min causal baseline")
        ax3.axhline(3 * sig, color="#666666", lw=0.8, ls=":")
        ax3.axhline(-3 * sig, color="#666666", lw=0.8, ls=":")
        ax3.axhline(0.0, color="#888888", lw=0.6, ls=":")
        pk = float(np.max(np.abs(u_fi)))
        ax3.set_title(
            f"3. After {data['W']}-min causal baseline (peak |u\u22a5| = {pk:.3f} m/s = {pk/sig:.1f}\u03c3)",
            fontsize=9.5,
        )
        ax3.set_xlabel("Time after earthquake origin (minutes)")
        ax3.grid(True, linestyle=":", alpha=0.55)
        y3_lim = max(0.45, float(np.max(np.abs(v_step3))) * 1.25, pk * 1.25)
        ax3.set_ylim(-y3_lim, y3_lim)
        if col == 0:
            ax3.set_ylabel("Cleaned drift (m/s)")
        ax3.legend(loc="lower right", fontsize=7.6, framealpha=0.94)

    out_path = (repo_path(cfg["paths"]["outputs"]) / "cleaning_steps.png") if out_path is None else Path(out_path)
    row = data["row"]
    viz.save_figure(
        fig,
        out_path,
        f"AIS Drift Cleaning Pipeline — Hellenic Arc Mw {row['mw']:.1f} (cand {int(row['cand_id'])}, {data['sog_kn']:.0f} kn, calm)",
        f"True currents from Solver (cand {int(row['cand_id'])}) with noise.ais_noise calm noise (\u03c3 = {data['sig_tot'][0]:.3f} m/s, 3\u03c3 = {3*data['sig_tot'][0]:.2f} m/s) and a synthetic turn artefact in Panel 1; in Panel 3 the {data['W']}-min causal baseline barely distorts a tsunami wave of this period because the filtered and unfiltered true currents overlap.",
    )
    print(f"Saved {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="T24 AIS cleaning-steps figure")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args(argv)
    cfg = load_config()
    make_cleaning_figure(cfg, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
