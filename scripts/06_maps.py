"""T18: Maps 2 and 3 + Detector Comparison from T17 sweep parquet results.

Reads outputs/sweep/part_*.parquet (contract: docs/interfaces.md section 14) joined with
data/truth/sources.csv, outputs/thresholds.csv, and gauges.load_gauges, and produces:
  - outputs/map2.png                  (smallest Mw confirmed in >=90% of draws, 100 km; AIS & GPS x 3 regimes)
  - outputs/map2_40km.png             (same at 40 km range cap)
  - outputs/map2_uncertainty.png      (2x4 grid split by operator AND region: P(det) vs Mw + 95% CI + FAR floor, 100 km)
  - outputs/map2_uncertainty_40km.png (same at 40 km range cap)
  - outputs/map2_table.csv            (per-position 90% and 50% detection floor + CI + held-out FAR)
  - outputs/map3.png                  (warning time AIS adds by coast: 40 km & 100 km x 3 instrument benchmarks)
  - outputs/map3_gps.png              (warning time Ship GPS height adds by coast, with explicit GPS caveats)
  - outputs/map3_table.csv            (median & IQR warning time added by coast across all operators, caps & regimes)
  - outputs/detector_comparison.csv   (strictly split by operator x cap x regime x region x detector, w/ held-out FAR & % within 100 km)
  - outputs/twin_location_summary.csv (off-grid twin detections-only location error p50/p90 & % within 100 km + held-out FAR)

Usage:
  PYTHONPATH=src python scripts/06_maps.py
  PYTHONPATH=src python scripts/06_maps.py --sweep-dir outputs/sweep --truth-index data/truth/sources.csv
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
REGIONS = ("hellenic", "calabrian", "cyprus", "aegean")
DETECTORS = ("network", "single_calibrated", "single_literal")
BENCHMARKS = (
    ("nearest_gauge_s", "vs. nearest gauge"),
    ("first_gauge_s", "vs. first gauge anywhere"),
    ("first_instrument_s", "vs. first instrument (+ buoys)"),
)

GPS_CAVEAT = (
    "GPS caveat: Ship GPS speed comes from floating ships directly above the source rising with the sea surface "
    "at t = 0 (unlike land-mounted gauges), assuming an instantaneous rupture, 5 cm/min GPS noise with no heave "
    "or weather effect (config gps), and ships inside the source footprint."
)


def load_sweep(sweep_dir):
    """Load all sweep trial shards (part_*.parquet), excluding gauge_triggers.parquet."""
    sweep_dir = Path(sweep_dir)
    files = sorted(sweep_dir.glob("part_*.parquet"))
    if not files:
        files = sorted(f for f in sweep_dir.glob("*.parquet") if f.name != "gauge_triggers.parquet")
    if not files:
        raise FileNotFoundError(f"No sweep parquet files found in {sweep_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _far_lookup(thresholds_df):
    """Map (operator, range_cap_m, regime, detector) -> held-out FAR."""
    if thresholds_df is None or thresholds_df.empty:
        return {}
    out = {}
    for r in thresholds_df.itertuples(index=False):
        out[(str(r.operator), float(r.range_cap_m), str(r.regime), str(r.detector))] = float(r.far_hold)
    return out


def _min_mw_at_threshold(df_pos, prob_threshold=0.90):
    """Smallest magnitude at a position with detection rate >= prob_threshold (NaN if none)."""
    by_mw = df_pos.groupby("mw", sort=True)["detected"].agg(["mean", "count"])
    hits = by_mw[by_mw["mean"] >= prob_threshold]
    if hits.empty:
        return np.nan
    return float(hits.index.min())


def _prepare_truth(truth_df, map2_only=True):
    t_cols = [c for c in ("run_id", "run_kind", "region", "pos_id", "lon_deg", "lat_deg", "mw") if c in truth_df.columns]
    truth_use = truth_df[t_cols].copy()
    if map2_only and "run_kind" in truth_use.columns:
        m2 = truth_use[truth_use["run_kind"].isin(["truth_map2", "map2"])]
        if not m2.empty:
            truth_use = m2
    return truth_use


def _merge_truth(sweep_df, truth_use):
    drop_overlap = [c for c in ("run_kind",) if c in sweep_df.columns and c in truth_use.columns]
    right = truth_use.drop(columns=drop_overlap) if drop_overlap else truth_use
    df = sweep_df.merge(right, on="run_id", how="inner")
    if "mw" not in df.columns and "truth_mw" in df.columns:
        df["mw"] = df["truth_mw"]
    return df


def compute_map2_summary(sweep_df, truth_df, detector="network", thresholds_df=None):
    """Compute per-(operator, range_cap_m, regime, pos_id) 90% and 50% detection-floor Mw."""
    truth_use = _prepare_truth(truth_df, map2_only=True)
    sub = sweep_df[sweep_df["detector"] == detector]
    if "run_kind" in sub.columns:
        m2_sub = sub[sub["run_kind"].isin(["truth_map2", "map2"])]
        if not m2_sub.empty:
            sub = m2_sub
    df = _merge_truth(sub, truth_use)
    if df.empty:
        raise ValueError(f"No rows after joining sweep (detector={detector}) with truth table.")

    far_map = _far_lookup(thresholds_df)
    rows = []
    group_cols = ["operator", "range_cap_m", "regime", "region", "pos_id"]
    for keys, grp in df.groupby(group_cols, sort=False):
        op, rcap, reg, region, pos_id = keys
        lon = float(grp["lon_deg"].median())
        lat = float(grp["lat_deg"].median())
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
                far_hold=far_map.get((str(op), float(rcap), str(reg), str(detector)), np.nan),
            )
        )
    return pd.DataFrame(rows)


def compute_detector_comparison(sweep_df, truth_df, thresholds_df=None):
    """Stratified detector comparison: strictly split by operator x range_cap x regime x region x detector.

    Includes held-out quiet-day FAR alongside detection rates, confirmation & first-gauge timing,
    and detections-only location errors (median, p90, % within 100 km; NaN for single-ship rules).
    """
    truth_use = _prepare_truth(truth_df, map2_only=True)
    sub = sweep_df
    if "run_kind" in sub.columns:
        m2_sub = sub[sub["run_kind"].isin(["truth_map2", "map2"])]
        if not m2_sub.empty:
            sub = m2_sub
    df = _merge_truth(sub, truth_use)
    far_map = _far_lookup(thresholds_df)

    # Region-specific reference magnitudes (truth Mw = candidate bin + 0.1)
    ref_mws = (7.10, 7.60, 7.85, 8.10, 8.35, 8.60)
    rows = []
    group_cols = ["operator", "range_cap_m", "regime", "region", "detector"]
    for (op, rcap, reg, region, det), grp in df.groupby(group_cols, sort=True):
        det_mask = grp["detected"].astype(bool)
        det_grp = grp[det_mask]
        n_trials = int(len(grp))
        n_det = int(len(det_grp))
        far_h = far_map.get((str(op), float(rcap), str(reg), str(det)), np.nan)

        # Timing (detections only for confirm_s; finite triggers for first_gauge_s)
        conf_fin = det_grp["confirm_s"][np.isfinite(det_grp["confirm_s"])] if n_det else pd.Series(dtype=float)
        med_conf_min = float(conf_fin.median() / 60.0) if len(conf_fin) else np.nan
        fg_all = grp.drop_duplicates("run_id")["first_gauge_s"]
        fg_fin = fg_all[np.isfinite(fg_all)]
        med_fg_min = float(fg_fin.median() / 60.0) if len(fg_fin) else np.nan
        if n_det:
            before_fg = np.isfinite(det_grp["confirm_s"]) & (det_grp["confirm_s"] < det_grp["first_gauge_s"])
            pct_before_fg = float(100.0 * before_fg.mean())
        else:
            pct_before_fg = np.nan

        # Location errors (detections only; single-ship rules have no location -> NaN)
        if det == "network" and n_det and "loc_err_m" in det_grp.columns:
            loc = det_grp["loc_err_m"].dropna()
            med_loc_km = float(loc.median() / 1000.0) if len(loc) else np.nan
            p90_loc_km = float(loc.quantile(0.90) / 1000.0) if len(loc) else np.nan
            pct_lt100km = float(100.0 * (loc <= 100_000.0).mean()) if len(loc) else np.nan
        else:
            med_loc_km = np.nan
            p90_loc_km = np.nan
            pct_lt100km = np.nan

        rec = dict(
            operator=op,
            range_cap_m=float(rcap),
            regime=reg,
            region=region,
            detector=det,
            far_hold=far_h,
            n_trials=n_trials,
            n_detected=n_det,
            det_rate_all=float(n_det / max(n_trials, 1)),
            med_confirm_min=med_conf_min,
            med_first_gauge_min=med_fg_min,
            pct_before_first_gauge=pct_before_fg,
            med_loc_err_km=med_loc_km,
            p90_loc_err_km=p90_loc_km,
            pct_within_100km=pct_lt100km,
        )
        by_mw = grp.groupby("mw")["detected"].mean()
        for mw_val in ref_mws:
            matches = [v for k, v in by_mw.items() if np.isclose(k, mw_val, atol=0.03)]
            key = f"det_rate_mw{int(round(mw_val * 100)):03d}"
            rec[key] = float(matches[0]) if matches else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_twin_location_summary(sweep_df, thresholds_df=None):
    """Detections-only location error summary on off-grid twin events, including share within 100 km and held-out FAR."""
    if "run_kind" not in sweep_df.columns:
        return pd.DataFrame()
    tw = sweep_df[(sweep_df["run_kind"] == "truth_twin") & (sweep_df["detector"] == "network")]
    if tw.empty:
        return pd.DataFrame()
    far_map = _far_lookup(thresholds_df)
    rows = []
    for (op, rcap, reg), grp in tw.groupby(["operator", "range_cap_m", "regime"], sort=True):
        det = grp[grp["detected"].astype(bool)]
        loc = det["loc_err_m"].dropna() if len(det) else pd.Series(dtype=float)
        mwe = det["mw_err"].dropna().abs() if len(det) else pd.Series(dtype=float)
        conf = det["confirm_s"][np.isfinite(det["confirm_s"])] if len(det) else pd.Series(dtype=float)
        rows.append(
            dict(
                operator=op,
                range_cap_m=float(rcap),
                regime=reg,
                n_trials=int(len(grp)),
                n_detected=int(len(det)),
                detected_frac=float(len(det) / max(len(grp), 1)),
                far_hold=far_map.get((str(op), float(rcap), str(reg), "network"), np.nan),
                median_loc_err_km=float(loc.median() / 1000.0) if len(loc) else np.nan,
                p90_loc_err_km=float(loc.quantile(0.90) / 1000.0) if len(loc) else np.nan,
                pct_within_100km=float(100.0 * (loc <= 100_000.0).mean()) if len(loc) else np.nan,
                median_abs_mw_err=float(mwe.median()) if len(mwe) else np.nan,
                median_confirm_min=float(conf.median() / 60.0) if len(conf) else np.nan,
            )
        )
    return pd.DataFrame(rows)


def _reserve_caption_space(fig, bottom=0.095, top=0.935):
    """Reserve top/bottom figure margins for suptitle and wrapped caption with constrained layout."""
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(0.0, bottom, 1.0, top - bottom))


def plot_map2(map2_df, cfg, out_path, range_cap_m=100000.0):
    """Plot 2x3 Map 2: operators (AIS, GPS) x weather regimes (calm, moderate, rough)."""
    viz.apply_style()
    sub_rc = map2_df[np.isclose(map2_df["range_cap_m"], range_cap_m)]
    if sub_rc.empty:
        range_cap_m = float(map2_df["range_cap_m"].max())
        sub_rc = map2_df[np.isclose(map2_df["range_cap_m"], range_cap_m)]

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), constrained_layout=True)
    _reserve_caption_space(fig, bottom=0.105, top=0.94)
    cmap = plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=6.6, vmax=8.6)
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
                    label="Below 50% at all Mw" if (r_idx == 0 and c_idx == 0) else None,
                )
            if not det50_only.empty:
                ax.scatter(
                    det50_only["lon_deg"], det50_only["lat_deg"], s=38,
                    facecolors="none", edgecolors=cmap(norm(det50_only["mw_50"])),
                    linewidths=1.35, marker="o", zorder=4,
                    label="50% level only (ring)" if (r_idx == 0 and c_idx == 0) else None,
                )
            if not det90.empty:
                last_sc = ax.scatter(
                    det90["lon_deg"], det90["lat_deg"], c=det90["mw_90"],
                    cmap=cmap, norm=norm, s=38, edgecolors="#222222", linewidths=0.45, zorder=5,
                )

            # Never pool Aegean (caps at 7.85) with arcs (to 8.60) into one pooled median
            aeg = panel[panel["region"] == "aegean"]
            arcs = panel[panel["region"] != "aegean"]
            aeg90 = int(aeg["mw_90"].notna().sum())
            arc90 = int(arcs["mw_90"].notna().sum())
            op_label = "AIS drift" if op == "ais" else "Ship GPS height*"
            ax.set_title(
                f"{op_label} — {reg.capitalize()}\nPositions ≥90%: Aegean {aeg90}/{len(aeg)}, Arcs {arc90}/{len(arcs)}",
                fontsize=9.0,
            )
            if r_idx == 0:
                ax.set_xlabel("")
            if c_idx > 0:
                ax.set_ylabel("")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
        cbar.set_label("Smallest off-grid truth magnitude confirmed within 2 h in ≥90% of traffic × noise draws (Mw)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    rc_km = int(round(range_cap_m / 1000.0))
    viz.save_figure(
        fig,
        out_path,
        f"Map 2: Smallest Earthquake Magnitude Ships Confirm (≥90% of Draws, {rc_km} km Range Cap)",
        (
            "Filled circles show the 90% detection floor per source position (62 Aegean to Mw 7.85, 79 arc positions to Mw 8.60; "
            "Aegean sweep rates are for earthquakes shaped like the catalogue — replayed Amorgos/Samos faults were not detected by AIS). "
            "Hollow rings reach 50% but not 90%; grey circles stay below 50%. "
            f"*{GPS_CAVEAT}"
        ),
    )


def plot_map2_uncertainty(sweep_df, truth_df, cfg, out_path, range_cap_m=100000.0, thresholds_df=None):
    """2x4 detection probability vs Mw split strictly by operator (AIS, GPS) AND region (Hellenic, Calabrian, Cyprus, Aegean)."""
    viz.apply_style()
    truth_use = _prepare_truth(truth_df, map2_only=True)
    sub_all = sweep_df
    if "run_kind" in sub_all.columns:
        m2_sub = sub_all[sub_all["run_kind"].isin(["truth_map2", "map2"])]
        if not m2_sub.empty:
            sub_all = m2_sub
    df = _merge_truth(sub_all, truth_use)
    if not df[np.isclose(df["range_cap_m"], range_cap_m)].empty:
        df = df[np.isclose(df["range_cap_m"], range_cap_m)]
    else:
        range_cap_m = float(df["range_cap_m"].max())
        df = df[np.isclose(df["range_cap_m"], range_cap_m)]

    regions = [r for r in REGIONS if r in df["region"].unique()]
    if not regions:
        regions = sorted(df["region"].unique())
    far_map = _far_lookup(thresholds_df)

    fig, axes = plt.subplots(
        len(OPERATORS),
        len(regions),
        figsize=(4.0 * len(regions), 8.2),
        sharey=True,
        constrained_layout=True,
    )
    _reserve_caption_space(fig, bottom=0.125, top=0.93)
    if len(regions) == 1:
        axes = np.asarray(axes).reshape(len(OPERATORS), 1)

    styles_by_op = {
        "ais": [
            ("network", "calm", viz.COLORS["calm"], "-", "AIS network (calm)"),
            ("network", "moderate", viz.COLORS["moderate"], "-", "AIS network (moderate)"),
            ("network", "rough", viz.COLORS["rough"], "-", "AIS network (rough)"),
            ("single_calibrated", "calm", viz.COLORS["neutral"], "--", "Single-ship cal. (calm)"),
        ],
        "gps": [
            ("network", "calm", viz.COLORS["calm"], "-", "GPS network* (calm/mod/rough)"),
            ("single_calibrated", "calm", viz.COLORS["neutral"], "--", "GPS single-ship cal.* (calm)"),
        ],
    }

    for r_idx, op in enumerate(OPERATORS):
        op_df = df[df["operator"] == op]
        far_net = far_map.get((op, float(range_cap_m), "calm", "network"), np.nan)
        for c_idx, region in enumerate(regions):
            ax = axes[r_idx, c_idx]
            rsub = op_df[op_df["region"] == region]
            for det, reg, col, ls, label in styles_by_op[op]:
                sub = rsub[(rsub["detector"] == det) & (rsub["regime"] == reg)]
                if sub.empty:
                    continue
                g = sub.groupby("mw", sort=True)["detected"].agg(["mean", "count"]).reset_index()
                p = g["mean"].to_numpy(float)
                n = g["count"].to_numpy(float)
                se = 1.96 * np.sqrt(np.clip(p * (1.0 - p) / np.maximum(n, 1.0), 0.0, 1.0))
                ax.plot(g["mw"], 100.0 * p, color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=label)
                ax.fill_between(g["mw"], 100.0 * np.clip(p - se, 0, 1), 100.0 * np.clip(p + se, 0, 1), color=col, alpha=0.14)

            ax.axhline(90.0, color="#444444", ls="--", lw=0.8, alpha=0.65)
            ax.axhline(50.0, color="#888888", ls=":", lw=0.8, alpha=0.65)
            if np.isfinite(far_net):
                ax.axhline(
                    100.0 * far_net,
                    color=viz.COLORS["negative"],
                    ls="-.",
                    lw=0.9,
                    alpha=0.75,
                    label=f"Held-out quiet FAR ({100.0 * far_net:.1f}%)" if c_idx == len(regions) - 1 else None,
                )

            if r_idx == 0:
                max_mw_note = " (to Mw 7.85)" if region == "aegean" else " (to Mw 8.60)"
                ax.set_title(f"{region.capitalize()}{max_mw_note}", fontsize=10, fontweight="bold")
            if r_idx == len(OPERATORS) - 1:
                ax.set_xlabel("Truth magnitude (Mw)")
            ax.set_xlim(6.50, 7.95 if region == "aegean" else 8.70)
            ax.set_ylim(-3, 105)
            ax.grid(True, linestyle=":", alpha=0.55)

        op_ylabel = "AIS drift — P(confirm ≤ 2 h) (%)" if op == "ais" else "Ship GPS height* — P(confirm ≤ 2 h) (%)"
        axes[r_idx, 0].set_ylabel(op_ylabel, fontsize=9.5)
        if axes[r_idx, -1].get_legend_handles_labels()[0]:
            loc = "upper left" if op == "ais" else "lower right"
            axes[r_idx, -1].legend(loc=loc, fontsize=7.3, framealpha=0.92)

    rc_km = int(round(range_cap_m / 1000.0))
    viz.save_figure(
        fig,
        out_path,
        f"Detection Probability vs. Magnitude Split by Operator and Region ({rc_km} km Range Cap)",
        (
            "Shaded bands show 95% binomial CI across draws; lines mark 90%, 50%, and held-out quiet-day FAR "
            "(low-Mw AIS ~1–4% equals FAR; Aegean sweep rates are for earthquakes shaped like the catalogue — replayed Amorgos/Samos were not detected by AIS). "
            "GPS curves are identical across calm/moderate/rough because config gps noise has no weather multiplier. "
            f"*{GPS_CAVEAT}"
        ),
    )


def compute_map3_summary(sweep_df, gauge_table, regime="calm", operator="ais", detector="network"):
    """Compute warning time added (minutes) per coast (nearest_gauge_code) for each range cap & benchmark.

    Uses first_gauge_s, nearest_gauge_s, and first_instrument_s directly from the sweep parquet
    (already computed on the eta(t) - eta(0) gauge trigger).
    """
    sub = sweep_df[
        (sweep_df["operator"] == operator)
        & (sweep_df["detector"] == detector)
        & (sweep_df["regime"] == regime)
    ].copy()
    if "run_kind" in sub.columns:
        m2_sub = sub[sub["run_kind"].isin(["truth_map2", "map2"])]
        if not m2_sub.empty:
            sub = m2_sub
    if sub.empty:
        raise ValueError(f"No rows for operator={operator}, detector={detector}, regime={regime}")

    rows = []
    g_lookup = gauge_table.set_index("code") if "code" in gauge_table.columns else pd.DataFrame()
    for (rcap, code), grp in sub.groupby(["range_cap_m", "nearest_gauge_code"], sort=False):
        if pd.isna(code) or str(code) == "":
            continue
        code_s = str(code)
        if code_s not in g_lookup.index:
            continue
        glon = float(g_lookup.loc[code_s, "lon_deg"])
        glat = float(g_lookup.loc[code_s, "lat_deg"])
        gname = str(g_lookup.loc[code_s, "name"]) if "name" in g_lookup.columns else code_s
        det_grp = grp[grp["detected"].astype(bool) & np.isfinite(grp["confirm_s"])]
        loc = det_grp["loc_err_m"].dropna() if ("loc_err_m" in det_grp.columns and len(det_grp)) else pd.Series(dtype=float)

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
            "pct_within_100km": float(100.0 * (loc <= 100_000.0).mean()) if len(loc) else np.nan,
        }
        for bcol, _ in BENCHMARKS:
            if bcol not in det_grp.columns:
                rec[f"gain_med_min_{bcol}"] = np.nan
                rec[f"gain_q25_min_{bcol}"] = np.nan
                rec[f"gain_q75_min_{bcol}"] = np.nan
                rec[f"pct_faster_{bcol}"] = np.nan
                continue
            valid = det_grp[np.isfinite(det_grp[bcol])]
            if valid.empty:
                rec[f"gain_med_min_{bcol}"] = np.nan
                rec[f"gain_q25_min_{bcol}"] = np.nan
                rec[f"gain_q75_min_{bcol}"] = np.nan
                rec[f"pct_faster_{bcol}"] = np.nan
            else:
                gain_min = (valid[bcol].to_numpy(float) - valid["confirm_s"].to_numpy(float)) / 60.0
                rec[f"gain_med_min_{bcol}"] = float(np.median(gain_min))
                rec[f"gain_q25_min_{bcol}"] = float(np.percentile(gain_min, 25))
                rec[f"gain_q75_min_{bcol}"] = float(np.percentile(gain_min, 75))
                rec[f"pct_faster_{bcol}"] = float(100.0 * (gain_min > 0.0).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_map3(map3_df, gauge_table, cfg, out_path, operator=None, regime=None):
    """Plot 2x3 Map 3: 40 km & 100 km range caps x 3 instrument benchmarks for a single operator/regime."""
    viz.apply_style()
    sub_all = map3_df.copy()
    if operator is not None and "operator" in sub_all.columns:
        sub_all = sub_all[sub_all["operator"] == operator]
    if regime is not None and "regime" in sub_all.columns:
        sub_all = sub_all[sub_all["regime"] == regime]
    range_caps = sorted(sub_all["range_cap_m"].unique())
    if not range_caps:
        raise ValueError("Empty map3 summary table.")

    op_str = operator if operator is not None else str(sub_all["operator"].iloc[0])
    reg_str = regime if regime is not None else str(sub_all["regime"].iloc[0])
    op_label = "AIS drift" if op_str == "ais" else "Ship GPS height*"

    fig, axes = plt.subplots(len(range_caps), len(BENCHMARKS), figsize=(15.8, 4.9 * len(range_caps)), constrained_layout=True)
    _reserve_caption_space(fig, bottom=0.105, top=0.94)
    if len(range_caps) == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("RdBu")
    norm = mcolors.TwoSlopeNorm(vmin=-80.0, vcenter=0.0, vmax=40.0) if op_str == "ais" else mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=30.0)
    last_sc = None
    buoys = gauge_table[gauge_table["rtype"] == 2] if "rtype" in gauge_table.columns else pd.DataFrame()

    for r_idx, rcap in enumerate(range_caps):
        sub = sub_all[np.isclose(sub_all["range_cap_m"], rcap)]
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
                    cmap=cmap, norm=norm, s=58, edgecolors="#222222", lw=0.55, zorder=5,
                )
            if not buoys.empty:
                ax.scatter(
                    buoys["lon_deg"], buoys["lat_deg"], marker="D", s=64,
                    c=viz.COLORS["buoys"], edgecolors="k", lw=0.7, zorder=6,
                    label="2025 Ionian buoys" if (r_idx == 0 and c_idx == 2) else None,
                )

            overall = float(valid[col_med].median()) if not valid.empty else np.nan
            n_pos = int((valid[col_med] > 0).sum()) if not valid.empty else 0
            med_txt = f"median {overall:+.1f} min | faster: {n_pos}/{len(valid)} coasts" if np.isfinite(overall) else "n/a"
            ax.set_title(f"{op_label} {rc_km} km — {blabel}\n{med_txt}", fontsize=8.8)
            if r_idx < len(range_caps) - 1:
                ax.set_xlabel("")
            if c_idx > 0:
                ax.set_ylabel("")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
        cbar.set_label("Median warning time ships add over benchmark (minutes; blue / positive = ships faster, red / negative = gauges faster)")
    if axes[0, 2].get_legend_handles_labels()[0]:
        axes[0, 2].legend(loc="lower left", fontsize=8, framealpha=0.92)

    title = (
        f"Map 3: Warning Time Added by AIS Ship Network by Coast ({reg_str.capitalize()} Weather)"
        if op_str == "ais"
        else "Map 3 (GPS Upgrade*): Warning Time Added by Ship GPS Height Network by Coast"
    )
    cap_extra = f" *{GPS_CAVEAT}" if op_str == "gps" else ""
    viz.save_figure(
        fig,
        out_path,
        title,
        (
            "Benchmarks use first_gauge_s, nearest_gauge_s, and first_instrument_s from the sweep parquet on the η(t) − η(0) "
            "gauge trigger (negative/red = gauges confirm earlier than ships; positive/blue = ships confirm earlier). "
            f"Diamonds mark the two 2025 CAT-INGV Ionian buoys.{cap_extra}"
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="T18 Maps 2 and 3 + Detector Comparison")
    parser.add_argument("--sweep-dir", default=None, help="Directory with sweep part_*.parquet files")
    parser.add_argument("--truth-index", default=None, help="Path to data/truth/sources.csv")
    parser.add_argument("--thresholds", default=None, help="Path to outputs/thresholds.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--map3-regime", default="calm", help="Weather regime for AIS Map 3 figure (default: calm)")
    args = parser.parse_args(argv)

    cfg = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(cfg["paths"]["outputs"])
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else (out_dir / "sweep")
    if args.truth_index:
        truth_path = Path(args.truth_index)
    else:
        cand_src = repo_path(cfg["paths"]["truth_dir"]) / "sources.csv"
        truth_path = cand_src if cand_src.exists() else (repo_path(cfg["paths"]["truth_dir"]) / "index.csv")
    thr_path = Path(args.thresholds) if args.thresholds else (out_dir / "thresholds.csv")
    thresholds_df = pd.read_csv(thr_path) if thr_path.exists() else None

    sweep_df = load_sweep(sweep_dir)
    truth_df = pd.read_csv(truth_path)
    grid = load_grid(cfg)
    gauge_table = gauges.load_gauges(grid, cfg)

    # Map 2 (100 km + 40 km) and 2x4 operator x region uncertainty curves
    map2_df = compute_map2_summary(sweep_df, truth_df, detector="network", thresholds_df=thresholds_df)
    map2_df.to_csv(out_dir / "map2_table.csv", index=False)
    plot_map2(map2_df, cfg, out_dir / "map2.png", range_cap_m=100000.0)
    if np.any(np.isclose(map2_df["range_cap_m"], 40000.0)):
        plot_map2(map2_df, cfg, out_dir / "map2_40km.png", range_cap_m=40000.0)
    plot_map2_uncertainty(sweep_df, truth_df, cfg, out_dir / "map2_uncertainty.png", range_cap_m=100000.0, thresholds_df=thresholds_df)
    if np.any(np.isclose(map2_df["range_cap_m"], 40000.0)):
        plot_map2_uncertainty(sweep_df, truth_df, cfg, out_dir / "map2_uncertainty_40km.png", range_cap_m=40000.0, thresholds_df=thresholds_df)

    # Stratified detector comparison & twin location accuracy table
    det_cmp = compute_detector_comparison(sweep_df, truth_df, thresholds_df=thresholds_df)
    det_cmp.to_csv(out_dir / "detector_comparison.csv", index=False)
    twin_loc = compute_twin_location_summary(sweep_df, thresholds_df=thresholds_df)
    if not twin_loc.empty:
        twin_loc.to_csv(out_dir / "twin_location_summary.csv", index=False)

    # Map 3 table across ALL (operator, regime) combinations + AIS and GPS figures
    map3_parts = []
    for op in OPERATORS:
        for reg in REGIMES:
            sub_check = sweep_df[
                (sweep_df["operator"] == op)
                & (sweep_df["detector"] == "network")
                & (sweep_df["regime"] == reg)
            ]
            if not sub_check.empty:
                map3_parts.append(compute_map3_summary(sweep_df, gauge_table, regime=reg, operator=op, detector="network"))
    map3_df = pd.concat(map3_parts, ignore_index=True)
    map3_df.to_csv(out_dir / "map3_table.csv", index=False)
    plot_map3(map3_df, gauge_table, cfg, out_dir / "map3.png", operator="ais", regime=args.map3_regime)
    if (map3_df["operator"] == "gps").any():
        plot_map3(map3_df, gauge_table, cfg, out_dir / "map3_gps.png", operator="gps", regime="calm")

    print(f"Wrote Map 2, Map 3, detector comparison, and twin location outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
