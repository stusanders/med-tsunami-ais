"""T18: Maps 2 and 3 from T17 sweep parquet results.

Reads outputs/sweep/*.parquet (contract: docs/interfaces.md section 14) joined with
data/truth/index.csv and gauges.load_gauges, and produces:
  - outputs/map2.png            (smallest Mw detected in >=90% of draws, with 50% marker; AIS & GPS x 3 weather regimes)
  - outputs/map2_40km.png       (same at 40 km AIS range cap)
  - outputs/map2_uncertainty.png (detection probability curves vs Mw with 95% CI across draws, network vs single-ship)
  - outputs/map2_table.csv      (per-position 90% and 50% detection floor + CI)
  - outputs/map3.png            (warning time ships add by coast: 40 km & 100 km x 3 instrument benchmarks)
  - outputs/map3_table.csv      (median & IQR warning time added by coast/gauge and benchmark)

Usage:
  PYTHONPATH=src python scripts/06_maps.py
  PYTHONPATH=src python scripts/06_maps.py --sweep-dir outputs/sweep --truth-index data/truth/index.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import gauges, viz
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid


REGIMES = ("calm", "moderate", "rough")
OPERATORS = ("ais", "gps")
BENCHMARKS = (
    ("nearest_gauge_s", "vs. coast's nearest gauge"),
    ("first_gauge_s", "vs. first gauge anywhere"),
    ("first_instrument_s", "vs. first instrument (incl. Ionian buoys)"),
)


def load_sweep(sweep_dir):
    sweep_dir = Path(sweep_dir)
    files = sorted(sweep_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {sweep_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _min_mw_at_threshold(df_pos, prob_threshold=0.90):
    """Smallest magnitude at a position with detection rate >= prob_threshold (NaN if none)."""
    by_mw = df_pos.groupby("mw", sort=True)["detected"].agg(["mean", "count"])
    hits = by_mw[by_mw["mean"] >= prob_threshold]
    if hits.empty:
        return np.nan
    return float(hits.index.min())


def compute_map2_summary(sweep_df, truth_df, detector="network"):
    """Compute per-(operator, range_cap_m, regime, pos_id) 90% and 50% detection-floor Mw."""
    t_cols = [c for c in ("run_id", "run_kind", "region", "pos_id", "lon_deg", "lat_deg", "mw") if c in truth_df.columns]
    truth_use = truth_df[t_cols].copy()
    if "run_kind" in truth_use.columns:
        map2_truth = truth_use[truth_use["run_kind"].isin(["truth_map2", "map2"])]
        if not map2_truth.empty:
            truth_use = map2_truth

    df = sweep_df[sweep_df["detector"] == detector].merge(truth_use, on="run_id", how="inner")
    if df.empty:
        raise ValueError(f"No rows after joining sweep (detector={detector}) with truth index.")

    rows = []
    group_cols = ["operator", "range_cap_m", "regime", "pos_id", "region", "lon_deg", "lat_deg"]
    for keys, grp in df.groupby(group_cols, sort=False):
        op, rcap, reg, pos_id, region, lon, lat = keys
        mw90 = _min_mw_at_threshold(grp, 0.90)
        mw50 = _min_mw_at_threshold(grp, 0.50)
        ref_mw = mw90 if np.isfinite(mw90) else float(grp["mw"].max())
        at_ref = grp[np.isclose(grp["mw"], ref_mw)]["detected"].astype(float)
        p_hat = float(at_ref.mean()) if len(at_ref) else np.nan
        n_draws = int(len(at_ref))
        ci95 = float(1.96 * np.sqrt(p_hat * (1.0 - p_hat) / n_draws)) if n_draws > 0 else np.nan
        rows.append(
            dict(
                operator=op,
                range_cap_m=float(rcap),
                regime=reg,
                pos_id=int(pos_id),
                region=region,
                lon_deg=float(lon),
                lat_deg=float(lat),
                mw_90=mw90,
                mw_50=mw50,
                p_at_ref=p_hat,
                ci95_at_ref=ci95,
                n_draws=n_draws,
            )
        )
    return pd.DataFrame(rows)


def plot_map2(map2_df, cfg, out_path, range_cap_m=100000.0):
    """Plot 2x3 Map 2: operators (AIS, GPS) x weather regimes (calm, moderate, rough)."""
    viz.apply_style()
    sub_rc = map2_df[np.isclose(map2_df["range_cap_m"], range_cap_m)]
    if sub_rc.empty:
        range_cap_m = float(map2_df["range_cap_m"].max())
        sub_rc = map2_df[np.isclose(map2_df["range_cap_m"], range_cap_m)]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=6.5, vmax=8.5)
    last_sc = None

    for r_idx, op in enumerate(OPERATORS):
        for c_idx, reg in enumerate(REGIMES):
            ax = axes[r_idx, c_idx]
            viz.map_axes(cfg, ax=ax)
            panel = sub_rc[(sub_rc["operator"] == op) & (sub_rc["regime"] == reg)]

            undet = panel[panel["mw_90"].isna()]
            det90 = panel[panel["mw_90"].notna()]
            det50_only = undet[undet["mw_50"].notna()]

            if not undet.empty:
                ax.scatter(
                    undet["lon_deg"], undet["lat_deg"], s=24, c="#CCCCCC",
                    edgecolors="#666666", linewidths=0.4, marker="o", zorder=3,
                    label="Not detected at 90%" if (r_idx == 0 and c_idx == 0) else None,
                )
            if not det50_only.empty:
                ax.scatter(
                    det50_only["lon_deg"], det50_only["lat_deg"], s=34,
                    facecolors="none", edgecolors=cmap(norm(det50_only["mw_50"])),
                    linewidths=1.2, marker="o", zorder=4,
                    label="50% level only (ring)" if (r_idx == 0 and c_idx == 0) else None,
                )
            if not det90.empty:
                last_sc = ax.scatter(
                    det90["lon_deg"], det90["lat_deg"], c=det90["mw_90"],
                    cmap=cmap, norm=norm, s=38, edgecolors="#222222", linewidths=0.45, zorder=5,
                )

            op_label = "AIS drift" if op == "ais" else "Ship GPS height"
            med90 = float(det90["mw_90"].median()) if not det90.empty else np.nan
            med_txt = f"median Mw {med90:.2f}" if np.isfinite(med90) else "no 90% detections"
            ax.set_title(f"{op_label} — {reg.capitalize()} ({len(det90)}/{len(panel)} pos; {med_txt})", fontsize=9.5)
            if r_idx == 0:
                ax.set_xlabel("")
            if c_idx > 0:
                ax.set_ylabel("")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
        cbar.set_label("Smallest magnitude confirmed within 2 h in \u2265 90% of traffic \u00d7 noise draws (Mw)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    rc_km = int(round(range_cap_m / 1000.0))
    viz.save_figure(
        fig,
        out_path,
        f"Map 2: Smallest Earthquake Magnitude Ships Confirm (\u226590% of Draws, {rc_km} km Range Cap)",
        "Filled circles show 90% detection floor (Mw) at each candidate source position; hollow rings mark positions reaching 50% but not 90%; grey circles remain below 50%.",
    )


def plot_map2_uncertainty(sweep_df, truth_df, cfg, out_path, range_cap_m=100000.0):
    """Detection probability vs Mw with 95% binomial CI bands across weather regimes and detectors."""
    viz.apply_style()
    t_cols = [c for c in ("run_id", "run_kind", "region", "mw") if c in truth_df.columns]
    truth_use = truth_df[t_cols].copy()
    if "run_kind" in truth_use.columns:
        m2 = truth_use[truth_use["run_kind"].isin(["truth_map2", "map2"])]
        if not m2.empty:
            truth_use = m2

    df = sweep_df.merge(truth_use, on="run_id", how="inner")
    if not df[np.isclose(df["range_cap_m"], range_cap_m)].empty:
        df = df[np.isclose(df["range_cap_m"], range_cap_m)]

    regions = [r for r in ("hellenic", "calabrian", "cyprus", "aegean") if r in df["region"].unique()]
    if not regions:
        regions = sorted(df["region"].unique())

    fig, axes = plt.subplots(1, len(regions), figsize=(4.2 * len(regions), 4.6), sharey=True)
    if len(regions) == 1:
        axes = [axes]

    reg_styles = {
        ("ais", "network", "calm"): (viz.COLORS["calm"], "-", "AIS network (calm)"),
        ("ais", "network", "moderate"): (viz.COLORS["moderate"], "-", "AIS network (moderate)"),
        ("ais", "network", "rough"): (viz.COLORS["rough"], "-", "AIS network (rough)"),
        ("gps", "network", "moderate"): (viz.COLORS["gauges"], "--", "GPS network (moderate)"),
        ("ais", "single_calibrated", "moderate"): (viz.COLORS["neutral"], ":", "Single-ship cal. (AIS mod.)"),
    }

    for ax, region in zip(axes, regions):
        rsub = df[df["region"] == region]
        for (op, det, reg), (col, ls, label) in reg_styles.items():
            sub = rsub[(rsub["operator"] == op) & (rsub["detector"] == det) & (rsub["regime"] == reg)]
            if sub.empty:
                continue
            g = sub.groupby("mw", sort=True)["detected"].agg(["mean", "count"]).reset_index()
            p = g["mean"].to_numpy(float)
            n = g["count"].to_numpy(float)
            se = 1.96 * np.sqrt(np.clip(p * (1.0 - p) / np.maximum(n, 1.0), 0.0, 1.0))
            ax.plot(g["mw"], 100.0 * p, color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=label)
            ax.fill_between(g["mw"], 100.0 * np.clip(p - se, 0, 1), 100.0 * np.clip(p + se, 0, 1), color=col, alpha=0.14)
        ax.axhline(90.0, color="#444444", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(50.0, color="#888888", ls=":", lw=0.9, alpha=0.7)
        ax.set_title(region.capitalize())
        ax.set_xlabel("Magnitude (Mw)")
        ax.set_ylim(-3, 105)
        ax.grid(True, linestyle=":", alpha=0.55)

    axes[0].set_ylabel("Detection probability within 2 h (%)")
    if axes[-1].get_legend_handles_labels()[0]:
        axes[-1].legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    viz.save_figure(
        fig,
        out_path,
        "Detection Probability vs. Magnitude by Source Region (95% Uncertainty Bands)",
        "Shaded bands show 95% binomial confidence intervals across traffic and noise draws; horizontal lines mark the 90% and 50% Map 2 detection thresholds.",
    )


def compute_map3_summary(sweep_df, gauge_table, regime="moderate", operator="ais", detector="network"):
    """Compute warning time added (minutes) per coast (nearest_gauge_code) for each range cap & benchmark."""
    df = sweep_df[
        (sweep_df["operator"] == operator)
        & (sweep_df["detector"] == detector)
        & (sweep_df["regime"] == regime)
    ].copy()
    if df.empty:
        raise ValueError(f"No rows for operator={operator}, detector={detector}, regime={regime}")

    rows = []
    g_lookup = gauge_table.set_index("code") if "code" in gauge_table.columns else pd.DataFrame()
    for (rcap, code), grp in df.groupby(["range_cap_m", "nearest_gauge_code"], sort=False):
        if pd.isna(code) or str(code) == "":
            continue
        code_s = str(code)
        if code_s not in g_lookup.index:
            continue
        glon = float(g_lookup.loc[code_s, "lon_deg"])
        glat = float(g_lookup.loc[code_s, "lat_deg"])
        gname = str(g_lookup.loc[code_s, "name"]) if "name" in g_lookup.columns else code_s
        det_grp = grp[grp["detected"].astype(bool) & np.isfinite(grp["confirm_s"])]

        rec = {
            "range_cap_m": float(rcap),
            "regime": regime,
            "operator": operator,
            "gauge_code": code_s,
            "gauge_name": gname,
            "lon_deg": glon,
            "lat_deg": glat,
            "n_trials": int(len(grp)),
            "n_detected": int(len(det_grp)),
            "det_rate": float(len(det_grp) / max(len(grp), 1)),
        }
        for bcol, _ in BENCHMARKS:
            if bcol not in det_grp.columns:
                rec[f"gain_med_min_{bcol}"] = np.nan
                rec[f"gain_q25_min_{bcol}"] = np.nan
                rec[f"gain_q75_min_{bcol}"] = np.nan
                continue
            valid = det_grp[np.isfinite(det_grp[bcol])]
            if valid.empty:
                rec[f"gain_med_min_{bcol}"] = np.nan
                rec[f"gain_q25_min_{bcol}"] = np.nan
                rec[f"gain_q75_min_{bcol}"] = np.nan
            else:
                gain_min = (valid[bcol].to_numpy(float) - valid["confirm_s"].to_numpy(float)) / 60.0
                rec[f"gain_med_min_{bcol}"] = float(np.median(gain_min))
                rec[f"gain_q25_min_{bcol}"] = float(np.percentile(gain_min, 25))
                rec[f"gain_q75_min_{bcol}"] = float(np.percentile(gain_min, 75))
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_map3(map3_df, gauge_table, cfg, out_path):
    """Plot 2x3 Map 3: 40 km & 100 km AIS range caps x 3 instrument benchmarks."""
    viz.apply_style()
    range_caps = sorted(map3_df["range_cap_m"].unique())
    if not range_caps:
        raise ValueError("Empty map3 summary table.")

    fig, axes = plt.subplots(len(range_caps), len(BENCHMARKS), figsize=(15.5, 4.4 * len(range_caps)), constrained_layout=True)
    if len(range_caps) == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("RdBu")
    norm = mcolors.TwoSlopeNorm(vmin=-20.0, vcenter=0.0, vmax=20.0)
    last_sc = None
    buoys = gauge_table[gauge_table["rtype"] == 2] if "rtype" in gauge_table.columns else pd.DataFrame()

    for r_idx, rcap in enumerate(range_caps):
        sub = map3_df[np.isclose(map3_df["range_cap_m"], rcap)]
        rc_km = int(round(rcap / 1000.0))
        for c_idx, (bcol, blabel) in enumerate(BENCHMARKS):
            ax = axes[r_idx, c_idx]
            viz.map_axes(cfg, ax=ax)
            col_med = f"gain_med_min_{bcol}"
            valid = sub[sub[col_med].notna()]
            nodata = sub[sub[col_med].isna()]

            if not nodata.empty:
                ax.scatter(nodata["lon_deg"], nodata["lat_deg"], s=28, c="#CCCCCC", edgecolors="#666666", lw=0.4, zorder=3)
            if not valid.empty:
                last_sc = ax.scatter(
                    valid["lon_deg"], valid["lat_deg"], c=valid[col_med],
                    cmap=cmap, norm=norm, s=54, edgecolors="#222222", lw=0.55, zorder=5,
                )
            if not buoys.empty:
                ax.scatter(
                    buoys["lon_deg"], buoys["lat_deg"], marker="D", s=62,
                    c=viz.COLORS["buoys"], edgecolors="k", lw=0.7, zorder=6,
                    label="2025 Ionian buoys" if (r_idx == 0 and c_idx == 2) else None,
                )

            overall = float(valid[col_med].median()) if not valid.empty else np.nan
            med_txt = f"median {overall:+.1f} min" if np.isfinite(overall) else "n/a"
            ax.set_title(f"AIS {rc_km} km — {blabel} ({med_txt})", fontsize=9.5)
            if r_idx < len(range_caps) - 1:
                ax.set_xlabel("")
            if c_idx > 0:
                ax.set_ylabel("")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
        cbar.set_label("Median warning time ships add over benchmark (minutes; positive / blue = ships faster)")
    if axes[0, 2].get_legend_handles_labels()[0]:
        axes[0, 2].legend(loc="lower left", fontsize=8, framealpha=0.92)

    viz.save_figure(
        fig,
        out_path,
        "Map 3: Warning Time Added by AIS Ship Network by Coast (Moderate Weather)",
        "Positive values (blue) indicate ship network confirmation earlier than the benchmark instrument; diamonds mark the two 2025 CAT-INGV Ionian deep-sea buoys.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="T18 Maps 2 and 3")
    parser.add_argument("--sweep-dir", default=None, help="Directory with sweep *.parquet files")
    parser.add_argument("--truth-index", default=None, help="Path to data/truth/index.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--map3-regime", default="moderate", help="Weather regime for Map 3 (default: moderate)")
    args = parser.parse_args(argv)

    cfg = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(cfg["paths"]["outputs"])
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else (out_dir / "sweep")
    truth_path = Path(args.truth_index) if args.truth_index else (repo_path(cfg["paths"]["truth_dir"]) / "index.csv")

    sweep_df = load_sweep(sweep_dir)
    truth_df = pd.read_csv(truth_path)
    grid = load_grid(cfg)
    gauge_table = gauges.load_gauges(grid, cfg)

    map2_df = compute_map2_summary(sweep_df, truth_df, detector="network")
    map2_df.to_csv(out_dir / "map2_table.csv", index=False)
    plot_map2(map2_df, cfg, out_dir / "map2.png", range_cap_m=100000.0)
    if np.any(np.isclose(map2_df["range_cap_m"], 40000.0)):
        plot_map2(map2_df, cfg, out_dir / "map2_40km.png", range_cap_m=40000.0)
    plot_map2_uncertainty(sweep_df, truth_df, cfg, out_dir / "map2_uncertainty.png", range_cap_m=100000.0)

    map3_df = compute_map3_summary(sweep_df, gauge_table, regime=args.map3_regime, operator="ais", detector="network")
    map3_df.to_csv(out_dir / "map3_table.csv", index=False)
    plot_map3(map3_df, gauge_table, cfg, out_dir / "map3.png")

    print(f"Wrote Map 2 and Map 3 outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
