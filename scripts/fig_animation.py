"""T22 (stretch): Animation of a Hellenic Mw 8 tsunami crossing the sea with ships lighting up.

Simulates one Hellenic Arc Mw 8.0 candidate scenario on the 390x795 grid, renders:
  - outputs/tsunami_animation.mp4        (H.264 video, 61 frames at 2-min intervals, 0-120 min)
  - outputs/tsunami_animation_frames.png (4-panel static storyboard at t = 15, 35, 60, 90 min)

Read-only with respect to T03 sources, T04 receiver pool, and T06 gauges.

Usage:
  PYTHONPATH=src python scripts/fig_animation.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tsunami import gauges, noise, observe, scoring, ships, sources, viz
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler


def simulate_demo_event(cfg, step_min=2, regime="calm"):
    grid = load_grid(cfg)
    pool = ships.load_pool(cfg)
    table = sources.candidate_table(cfg)
    hel = table[(table["region"] == "hellenic") & np.isclose(table["mw"], 8.0)]
    row = hel.iloc[len(hel) // 2]
    eta0 = sources.source_eta0(grid, row, cfg)

    sampler = make_sampler(grid, pool)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    solver = Solver(grid, sampler, cfg, xp=np)

    times_min = np.arange(0, int(cfg["library"]["n_minutes"]) + 1, int(step_min))
    times_s = times_min.astype(np.float64) * 60.0

    # Run snapshots + library-format receiver time series
    snaps = solver.snapshots(eta0[None, :, :], times_s)[0]          # (F, ny, nx)
    recf, gb30 = solver.run_library(eta0[None, :, :], gb_idx)        # (1, 120, R, 3), (1, 240, R_gb)
    recf0 = recf[0]
    gb30_0 = gb30[0]

    # Traffic draw & AIS signals/noise
    draw = ships.traffic_draw(pool, cfg, index=0, purpose=1)
    truth_s = observe.gather_truth(recf0, draw)                      # (120, S, 3)
    sig = observe.ais_signal(truth_s, draw, response=1.0)            # (S, 120)
    rng = np.random.default_rng([cfg["meta"]["seed"], 22, 0])
    nz = noise.ais_noise(draw, pool, regime, cfg, rng, n_draws=1)    # (1, S, 120)
    d = observe.observe(sig, nz)                                     # (1, S, 120)

    sigma_i = noise.ship_sigma(draw, regime, "ais", cfg)
    sigma_shared = noise.shared_sigma(regime, "ais", cfg)
    sigma_tot = np.sqrt(sigma_i.astype(np.float64) ** 2 + sigma_shared ** 2)
    # Per-ship noiseless signal SNR |u_perp| / sigma_tot for clean spatial visualisation
    ship_snr = np.abs(sig) / sigma_tot[:, None]                      # (S, 120)

    # Single-template matched-filter cumulative score S(t) on the noisy observation d
    groups = scoring.build_groups(draw, pool, cfg, "ais")
    D = scoring.aggregate(d, groups, valid=None, xp=np)
    G = scoring.aggregate(sig[None, :, :], groups, valid=None, xp=np)
    w = scoring.cell_weights(groups, sigma_i, sigma_shared, valid=None, xp=np)
    gd, gg = scoring.cumulative_stats(G, D, w, xp=np)
    eps = float(cfg["scoring"]["eps"])
    score_t = scoring.detection_scores(gd, gg, eps)[0, 0]            # (120,)
    net_snr_t = np.sqrt(np.maximum(score_t, 0.0))

    # Gauges & buoys trigger times (handle pool before/after buoy rebuild gracefully)
    g_table = gauges.load_gauges(grid, cfg)
    if len(g_table) > len(gb_idx):
        g_table = g_table[g_table["rtype"] == 1].reset_index(drop=True)
    trig_s = gauges.trigger_times(gb30_0[None, :, :], g_table)[0]    # (R_gb,)
    trig_min = trig_s / 60.0

    ship_lon = pool.lon[draw.pool_idx]
    ship_lat = pool.lat[draw.pool_idx]

    return {
        "grid": grid,
        "row": row,
        "times_min": times_min,
        "snaps": snaps,
        "ship_lon": ship_lon,
        "ship_lat": ship_lat,
        "ship_snr": ship_snr,
        "net_snr_t": net_snr_t,
        "g_table": g_table,
        "trig_min": trig_min,
        "regime": regime,
    }


def _eta_rgba(eta_2d, wet_mask, vmax=0.25):
    """Convert 2D eta to RGBA with transparent near-zero wave height so map ocean/land stay crisp."""
    norm = np.clip((eta_2d / vmax) * 0.5 + 0.5, 0.0, 1.0)
    rgba = plt.get_cmap("RdBu_r")(norm)
    alpha = np.clip((np.abs(eta_2d) - 0.006) / (0.06 - 0.006), 0.0, 0.85)
    alpha = np.where(wet_mask, alpha, 0.0)
    rgba[..., 3] = alpha
    return rgba


def render_storyboard(data, cfg, out_path):
    """Render a 4-panel static storyboard (t = 15, 35, 60, 90 min) for report/slides."""
    viz.apply_style()
    grid = data["grid"]
    row = data["row"]
    times_min = data["times_min"]
    snaps = data["snaps"]
    g_table = data["g_table"]
    trig_min = data["trig_min"]

    lon_ext = (float(grid.lon[0]), float(grid.lon[-1]), float(grid.lat[0]), float(grid.lat[-1]))
    pick_mins = [15, 35, 60, 90]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), constrained_layout=True)

    snr_cmap = plt.get_cmap("inferno_r")
    snr_norm = mcolors.Normalize(vmin=0.0, vmax=3.0)
    last_sc = None

    for idx, (ax, t_target) in enumerate(zip(axes.flat, pick_mins)):
        f_idx = int(np.argmin(np.abs(times_min - t_target)))
        t_cur = int(times_min[f_idx])
        m_idx = min(max(t_cur - 1, 0), data["ship_snr"].shape[1] - 1)
        viz.map_axes(cfg, ax=ax)

        rgba = _eta_rgba(snaps[f_idx], grid.wet, vmax=0.25)
        ax.imshow(rgba, origin="lower", extent=lon_ext, aspect="auto", zorder=1.2)

        snr_now = data["ship_snr"][:, m_idx]
        quiet = snr_now < 0.4
        active = ~quiet
        ax.scatter(data["ship_lon"][quiet], data["ship_lat"][quiet], s=5, c="#4A6FA5", alpha=0.35, zorder=2.8)
        if np.any(active):
            last_sc = ax.scatter(
                data["ship_lon"][active],
                data["ship_lat"][active],
                c=np.clip(snr_now[active], 0.0, 3.0),
                cmap=snr_cmap,
                norm=snr_norm,
                s=14 + 14 * np.clip(snr_now[active], 0.0, 3.0),
                edgecolors="#111111",
                linewidths=0.3,
                zorder=3.8,
            )

        fired = np.isfinite(trig_min) & (trig_min <= t_cur)
        unfired = ~fired
        if np.any(unfired):
            ax.scatter(g_table.loc[unfired, "lon_deg"], g_table.loc[unfired, "lat_deg"],
                       marker="^", s=24, c="#888888", alpha=0.55, zorder=4.0)
        if np.any(fired):
            ax.scatter(g_table.loc[fired, "lon_deg"], g_table.loc[fired, "lat_deg"],
                       marker="^", s=46, c=viz.COLORS["positive"], edgecolors="k", lw=0.5, zorder=4.8)

        ax.scatter([row["lon_deg"]], [row["lat_deg"]], marker="*", s=160, c="#FFD700", edgecolors="k", lw=0.8, zorder=5.5)
        n_lit = int(np.sum(snr_now >= 1.0))
        n_g = int(np.sum(fired))
        net_val = float(data["net_snr_t"][m_idx])
        ax.set_title(
            f"t = {t_cur} min | ships >1\u03c3: {n_lit} | net \u221aS: {net_val:.1f} | gauges triggered: {n_g}",
            fontsize=9.5,
        )
        if idx < 2:
            ax.set_xlabel("")
        if idx % 2 == 1:
            ax.set_ylabel("")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, orientation="horizontal", fraction=0.035, pad=0.04, aspect=40)
        cbar.set_label("Per-ship tsunami-current signal SNR |u\u22a5| / \u03c3_total (calm AIS)")

    viz.save_figure(
        fig,
        out_path,
        f"Hellenic Arc Mw 8.0 Tsunami Crossing the Eastern Mediterranean ({row['lat_deg']:.2f}\u00b0N, {row['lon_deg']:.2f}\u00b0E)",
        "Shading shows sea-surface elevation (\u00b10.25 m, RdBu_r); ships light up by per-ship sideways-drift SNR; green triangles mark triggered IOC tide gauges.",
    )


def render_mp4(data, cfg, out_path, fps=8):
    """Render H.264 MP4 animation (wave propagation + ships lighting up + live network score)."""
    viz.apply_style()
    grid = data["grid"]
    row = data["row"]
    times_min = data["times_min"]
    snaps = data["snaps"]
    g_table = data["g_table"]
    trig_min = data["trig_min"]
    lon_ext = (float(grid.lon[0]), float(grid.lon[-1]), float(grid.lat[0]), float(grid.lat[-1]))

    fig = plt.figure(figsize=(12.0, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.15], hspace=0.28, top=0.90, bottom=0.11, left=0.07, right=0.95)
    ax = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])

    viz.map_axes(cfg, ax=ax)
    im = ax.imshow(_eta_rgba(snaps[0], grid.wet, vmax=0.25), origin="lower", extent=lon_ext, aspect="auto", zorder=1.2)
    ax.scatter([row["lon_deg"]], [row["lat_deg"]], marker="*", s=190, c="#FFD700", edgecolors="k", lw=0.8, zorder=5.5,
               label=f"Epicentre (Mw {row['mw']:.1f})")

    snr_cmap = plt.get_cmap("inferno_r")
    snr_norm = mcolors.Normalize(vmin=0.0, vmax=3.0)
    sc_ships = ax.scatter(data["ship_lon"], data["ship_lat"], s=10, c="#4A6FA5", alpha=0.4, zorder=3.5)
    sc_gauges = ax.scatter(g_table["lon_deg"], g_table["lat_deg"], marker="^", s=34, c="#888888", edgecolors="#222222", lw=0.4, zorder=4.6)

    sm = plt.cm.ScalarMappable(cmap=snr_cmap, norm=snr_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Ship drift SNR |u\u22a5|/\u03c3")

    time_txt = ax.text(
        0.015, 0.955, "t = 0 min", transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#333333", "alpha": 0.92},
        zorder=6,
    )

    mins = np.arange(1, len(data["net_snr_t"]) + 1)
    ax_t.plot(mins, data["net_snr_t"], color=viz.COLORS["ships"], lw=2.0, label="Matched-filter network \u221aS(t) (noisy calm AIS)")
    cursor = ax_t.axvline(0, color=viz.COLORS["negative"], lw=1.6, ls="--")
    ax_t.set_xlim(0, 120)
    ax_t.set_ylim(0, max(5.0, float(np.nanmax(data["net_snr_t"])) * 1.12))
    ax_t.set_xlabel("Time after earthquake origin (minutes)")
    ax_t.set_ylabel("Network \u221aS(t)")
    ax_t.grid(True, linestyle=":", alpha=0.6)
    ax_t.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        f"Ships as a Tsunami Network — Hellenic Arc Mw {row['mw']:.1f} Wave & AIS Fleet Response",
        fontsize=12.5, fontweight="bold", y=0.965,
    )
    fig.text(
        0.015, 0.015,
        "Linear shallow-water wave height (\u00b10.25 m, RdBu_r); underway AIS ships brighten as sideways-drift SNR rises; green triangles mark triggered IOC gauges.",
        fontsize=8.5, style="italic", color="#333333",
    )

    base_rgba = np.array([0.29, 0.435, 0.647, 0.35])

    def _update(f_idx):
        t_cur = int(times_min[f_idx])
        im.set_data(_eta_rgba(snaps[f_idx], grid.wet, vmax=0.25))

        if t_cur == 0:
            snr_now = np.zeros(len(data["ship_lon"]), dtype=np.float32)
            net_now = 0.0
        else:
            m_idx = min(t_cur - 1, data["ship_snr"].shape[1] - 1)
            snr_now = data["ship_snr"][:, m_idx]
            net_now = float(data["net_snr_t"][m_idx])

        colors = np.tile(base_rgba, (len(snr_now), 1))
        sizes = np.full(len(snr_now), 6.0)
        active = snr_now >= 0.4
        if np.any(active):
            colors[active] = snr_cmap(snr_norm(np.clip(snr_now[active], 0.0, 3.0)))
            colors[active, 3] = 0.95
            sizes[active] = 14.0 + 18.0 * np.clip(snr_now[active], 0.0, 3.0)
        sc_ships.set_facecolors(colors)
        sc_ships.set_sizes(sizes)

        fired = np.isfinite(trig_min) & (trig_min <= t_cur)
        g_cols = np.where(fired, viz.COLORS["positive"], "#888888")
        g_sizes = np.where(fired, 56.0, 24.0)
        sc_gauges.set_facecolors(g_cols)
        sc_gauges.set_sizes(g_sizes)

        cursor.set_xdata([t_cur, t_cur])
        n_lit = int(np.sum(snr_now >= 1.0))
        n_g = int(np.sum(fired))
        time_txt.set_text(f"t = {t_cur:3d} min | ships >1\u03c3: {n_lit:3d} | net \u221aS: {net_now:5.1f} | gauges: {n_g:2d}")
        return im, sc_ships, sc_gauges, cursor, time_txt

    anim = animation.FuncAnimation(fig, _update, frames=len(times_min), interval=int(1000 / fps), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "22"],
    )
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="T22 Tsunami & Ship Network Animation")
    parser.add_argument("--step-min", type=int, default=2, help="Frame spacing in minutes (default: 2)")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS (default: 8)")
    parser.add_argument("--regime", default="calm", help="Weather regime for noise (default: calm)")
    args = parser.parse_args(argv)

    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["outputs"])
    data = simulate_demo_event(cfg, step_min=args.step_min, regime=args.regime)
    render_storyboard(data, cfg, out_dir / "tsunami_animation_frames.png")
    render_mp4(data, cfg, out_dir / "tsunami_animation.mp4", fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
