"""T24: Standalone cleaning-steps illustration figure -> outputs/cleaning_steps.png.

Synthesises a single ship's AIS heading-normal drift locally (not from the main
simulation), including a slow background current, a tsunami wave packet, AIS
heading rounding + wobble noise, and a course-change turn artefact
    v_yaw(t) = rate_of_turn(t) * ship_length
then shows three stacked panels:
  1. Raw 10-second AIS heading-normal drift
  2. After 1-minute averaging and yaw-rate correction
  3. After causal trailing baseline subtraction, with the true tsunami current overlaid

Run: PYTHONPATH=src ../med-tsunami-ais/.venv/bin/python scripts/fig_cleaning_steps.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami import timeseries, viz
from tsunami.config import load_config, repo_path
from tsunami.units import deg2rad, knots_to_ms


def synthesise_ship_drift(cfg):
    """Create a reproducible synthetic 1-ship AIS drift series with a turn artefact."""
    rng = np.random.default_rng([cfg["meta"]["seed"], 24, 0])

    dt_s = float(cfg["ais_noise"]["report_interval_s"])
    w_min = int(cfg["library"]["baseline_window_min"])
    n_min = int(cfg["library"]["n_minutes"])
    hdg_res_deg = float(cfg["ais_noise"]["heading_resolution_deg"])
    wobble_sd_deg = float(cfg["ais_noise"]["heading_wobble_sd_deg"])
    sog_kn = float(np.mean(cfg["traffic"]["speed_kn"]))  # 14 kn mid-range
    sog_ms = float(knots_to_ms(sog_kn))
    samples_per_min = int(round(float(cfg["library"]["minute_s"]) / dt_s))

    # Include W minutes of pre-origin history so the causal baseline filter is warmed up at t = 0
    total_min = w_min + n_min
    n_rep = total_min * samples_per_min
    t_s = (np.arange(n_rep, dtype=np.float64) - w_min * samples_per_min + 0.5) * dt_s
    t_min = t_s / 60.0

    # 1. Slow background current / leeway drift (~0.32 to 0.38 m/s over 2.75 h)
    u_bg = 0.34 + 0.04 * np.sin(2.0 * np.pi * (t_min + 20.0) / 360.0)

    # 2. Synthetic tsunami current packet arriving around t = 38 min (period ~18 min, peak ~0.26 m/s)
    tau = t_min - 38.0
    env = np.where(tau > 0.0, np.exp(-((tau - 14.0) / 18.0) ** 2) * (1.0 - np.exp(-tau / 3.0)), 0.0)
    u_tsunami = 0.26 * env * np.sin(2.0 * np.pi * tau / 18.0)

    # 3. Course-change turn artefact around t = 18 min (smooth 16-deg turn over ~3 min, lever arm L = 150 m)
    # Apparent sway at stern GPS antenna: v_yaw(t) = rate_of_turn(t) [rad/s] * ship_length [m]
    ship_length_m = 150.0
    turn_total_rad = float(deg2rad(16.0))
    turn_sigma_s = 45.0
    turn_center_s = 18.0 * 60.0
    rate_of_turn_rad_s = (
        turn_total_rad
        / (np.sqrt(2.0 * np.pi) * turn_sigma_s)
        * np.exp(-0.5 * ((t_s - turn_center_s) / turn_sigma_s) ** 2)
    )
    v_yaw = rate_of_turn_rad_s * ship_length_m

    # 4. AIS heading error: 1-deg rounding + Gaussian wobble -> velocity error SOG * delta_theta
    true_hdg_deg = 90.0 + 0.3 * np.sin(2.0 * np.pi * t_min / 25.0)
    wobble_deg = rng.normal(0.0, wobble_sd_deg, size=n_rep)
    measured_hdg_deg = np.round((true_hdg_deg + wobble_deg) / hdg_res_deg) * hdg_res_deg
    delta_theta_rad = deg2rad(measured_hdg_deg - true_hdg_deg)
    v_noise = sog_ms * delta_theta_rad

    v_raw = u_bg + u_tsunami + v_yaw + v_noise

    # Step A: subtract yaw artefact and block-average to 1 minute
    v_noyaw_10s = v_raw - v_yaw
    v_avg_1m = timeseries.block_average(v_noyaw_10s, samples_per_min, axis=0)
    u_tsu_1m = timeseries.block_average(u_tsunami, samples_per_min, axis=0)
    u_bg_1m = timeseries.block_average(u_bg, samples_per_min, axis=0)

    # Step B: causal trailing W-minute baseline subtraction (never uses future samples)
    v_clean_1m = timeseries.baseline_filter(v_avg_1m[w_min:], w_min, axis=0, history=v_avg_1m[:w_min])
    u_tsu_filt_1m = timeseries.baseline_filter(u_tsu_1m[w_min:], w_min, axis=0, history=np.zeros(w_min))

    t_1m = np.arange(1, n_min + 1, dtype=np.float64)
    post = t_min >= 0.0
    return {
        "t_raw_min": t_min[post],
        "v_raw": v_raw[post],
        "v_yaw": v_yaw[post],
        "t_1m": t_1m,
        "v_avg_1m": v_avg_1m[w_min:],
        "u_bg_1m": u_bg_1m[w_min:],
        "v_clean_1m": v_clean_1m,
        "u_tsu_1m": u_tsu_1m[w_min:],
        "u_tsu_filt_1m": u_tsu_filt_1m,
        "w_min": w_min,
        "sog_kn": sog_kn,
        "ship_length_m": ship_length_m,
    }


def make_figure(cfg=None):
    cfg = cfg or load_config()
    d = synthesise_ship_drift(cfg)
    viz.apply_style()

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.6), sharex=True)
    xlim = (0.0, float(cfg["library"]["n_minutes"]))

    # Panel 1: Raw 10-s AIS drift
    ax = axes[0]
    ax.plot(d["t_raw_min"], d["v_raw"], color="#6E7F8F", lw=0.7, alpha=0.85, label="Raw 10 s AIS drift: SOG·sin(COG − HDG)")
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
    peak_i = int(np.argmax(d["v_yaw"]))
    ax.annotate(
        f"Turn artefact (rate of turn × {d['ship_length_m']:.0f} m)\npeak +{d['v_yaw'][peak_i]:.2f} m/s",
        xy=(d["t_raw_min"][peak_i], d["v_raw"][peak_i]),
        xytext=(d["t_raw_min"][peak_i] + 10.0, d["v_raw"][peak_i] - 0.12),
        arrowprops={"arrowstyle": "->", "color": viz.COLORS["negative"], "lw": 1.1},
        fontsize=8.5,
        color=viz.COLORS["negative"],
        fontweight="bold",
    )
    ax.set_ylabel("Drift (m/s)")
    ax.set_title("1. Raw 10-second AIS heading-normal drift (turn spike + heading jitter + background current)")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: After 1-minute averaging and yaw-rate correction
    ax = axes[1]
    ax.plot(d["t_1m"], d["v_avg_1m"], color=viz.COLORS["ships"], lw=1.4, label="1-min averaged & yaw-corrected drift")
    ax.plot(d["t_1m"], d["u_bg_1m"], color=viz.COLORS["moderate"], lw=1.1, ls="--", label="Slow background ocean/leeway drift (~0.35 m/s)")
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
    ax.set_ylabel("Drift (m/s)")
    ax.set_title("2. After 1-minute block averaging and subtracting v_yaw = rate_of_turn × ship_length")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: After causal 45-min trailing baseline subtraction, with true tsunami current overlaid
    ax = axes[2]
    ax.plot(d["t_1m"], d["v_clean_1m"], color=viz.COLORS["ships"], lw=1.5, label=f"Cleaned drift (causal {d['w_min']}-min baseline subtracted)")
    ax.plot(d["t_1m"], d["u_tsu_filt_1m"], color=viz.COLORS["negative"], lw=1.8, label="True tsunami current (same causal baseline filter)")
    ax.plot(d["t_1m"], d["u_tsu_1m"], color=viz.COLORS["truth"], lw=1.0, ls="--", alpha=0.7, label="True unfiltered tsunami current")
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time after earthquake origin (min)")
    ax.set_ylabel("Current (m/s)")
    ax.set_title(f"3. After causal {d['w_min']}-minute trailing baseline subtraction (tsunami current recovered)")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.6)

    out_path = repo_path(cfg["paths"]["outputs"]) / "cleaning_steps.png"
    viz.save_figure(
        fig,
        out_path,
        "AIS Heading-Normal Drift Cleaning Pipeline (Standalone Synthetic Illustration)",
        "Synthetic 1-ship illustration (14 kn, 1° rounding + 1° wobble, 150 m ship turning 16°); the main simulation does not model turning and uses Inazu et al. (2020) post-correction noise levels.",
    )
    return out_path


if __name__ == "__main__":
    p = make_figure()
    print(f"Saved {p}")
