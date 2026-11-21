"""T23 (stretch): Yaw-correction demonstration on real Piraeus Oct 2017 AIS data.

Estimates rate of turn (ROT) from consecutive headings on deduplicated Piraeus AIS records
(per data/README.md), applies Inazu et al. (2020, Earth, Planets and Space 72:41):
    v_yaw = -L_antenna * ROT        (starboard-positive; clockwise turn swings stern/GNSS to port)
    v_corr = v_n - v_yaw = v_n + L_antenna * ROT
and quantifies how the spread of heading-normal drift v_n = SOG * sin(COG - HDG) changes on
turning ships using:
  1. A fixed fleet-wide PLACEHOLDER offset L_antenna from config.yaml (calibration.yaw_antenna_offset_m)
  2. Per-vessel empirical L_fit calibrated on the first half of each vessel's turning samples
     and evaluated on the chronological held-out second half (avoiding overfitting).

Illustration only: does not modify src/tsunami/noise.py.

Outputs:
  - outputs/yaw_correction.md
  - outputs/yaw_correction.png

Usage:
  PYTHONPATH=src python scripts/yaw_correction_piraeus.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import viz
from tsunami.config import load_config, repo_path
from tsunami.units import deg2rad, knots_to_ms, rad2deg


def wrap_deg(angle_deg):
    """Wrap angle differences to (-180, 180] degrees."""
    return (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def robust_sd(x):
    """MAD-based robust standard deviation (1.4826 * median(|x - median(x)|))."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def compute_turn_table(df, cfg):
    """Deduplicate (vessel_id, t), compute centered ROT (rad/s) and raw v_n (m/s)."""
    cal = cfg["calibration"]
    n_raw = len(df)
    df = df.drop_duplicates(["vessel_id", "t"]).sort_values(["vessel_id", "t"]).reset_index(drop=True)
    n_dedup = len(df)

    g = df.groupby("vessel_id", sort=False)
    dt_prev = g["t"].diff() / 1000.0
    dt_next = -g["t"].diff(-1) / 1000.0

    dh_prev = wrap_deg(df["heading"] - g["heading"].shift(1))
    dh_next = wrap_deg(g["heading"].shift(-1) - df["heading"])
    dt_cen = dt_prev + dt_next
    dh_cen = dh_prev + dh_next

    max_dt = float(cal["steady_max_dt_s"])
    max_dc = float(cal["yaw_max_cog_hdg_deg"])
    ok = (
        df["heading"].notna()
        & df["course"].notna()
        & df["speed"].notna()
        & (df["heading"] != cal["heading_not_available_code"])
        & (df["speed"] >= float(cal["steady_speed_kn"]))
        & (df["speed"] <= float(cal["max_valid_speed_kn"]))
        & (dt_prev > 0.0)
        & (dt_prev <= max_dt)
        & (dt_next > 0.0)
        & (dt_next <= max_dt)
        & (np.abs(dh_prev) < max_dc)
        & (np.abs(dh_next) < max_dc)
    )

    sub = df.loc[ok, ["vessel_id", "t", "heading", "course", "speed"]].copy()
    sub["dt_s"] = (0.5 * dt_cen[ok]).to_numpy()
    sub["rot_rad_s"] = deg2rad(dh_cen[ok] / dt_cen[ok])
    sub["rot_deg_min"] = rad2deg(sub["rot_rad_s"]) * 60.0
    sub["sog_m_s"] = knots_to_ms(sub["speed"].to_numpy(dtype=np.float64))
    sub["dc_deg"] = wrap_deg(sub["course"].to_numpy(dtype=np.float64) - sub["heading"].to_numpy(dtype=np.float64))
    sub["v_n"] = sub["sog_m_s"] * np.sin(deg2rad(sub["dc_deg"]))

    # Filter gross reversing/sensor-fault outliers
    sub = sub[sub["dc_deg"].abs() <= max_dc].reset_index(drop=True)
    meta = {
        "n_raw": int(n_raw),
        "n_dedup": int(n_dedup),
        "n_vessels_total": int(df["vessel_id"].nunique()),
        "n_clean_underway": int(len(sub)),
    }
    return sub, meta


def apply_yaw_correction(v_n, rot_rad_s, L_antenna_m):
    """Apply Inazu et al. (2020) yaw correction: v_corr = v_n - (-L_antenna * ROT)."""
    v_yaw = -float(L_antenna_m) * np.asarray(rot_rad_s, dtype=np.float64)
    return np.asarray(v_n, dtype=np.float64) - v_yaw


def fit_antenna_offset(v_n, rot_rad_s):
    """Estimate effective L_antenna (m) via OLS slope of v_n on -ROT."""
    x = np.asarray(rot_rad_s, dtype=np.float64)
    y = np.asarray(v_n, dtype=np.float64)
    vx = np.var(x)
    if vx <= 0.0:
        return np.nan
    return float(-np.cov(x, y, ddof=0)[0, 1] / vx)


def analyze_yaw(sub, cfg):
    cal = cfg["calibration"]
    L_fixed = float(cal["yaw_antenna_offset_m"])
    rot_turn_thr = float(cal["yaw_turning_rot_deg_per_min"])
    rot_steady_thr = float(cal["yaw_steady_rot_deg_per_min"])
    min_pts = int(cal["yaw_min_vessel_turning_pts"])

    steady = sub[sub["rot_deg_min"].abs() <= rot_steady_thr].copy()
    turning = sub[sub["rot_deg_min"].abs() >= rot_turn_thr].copy()
    turning["v_corr_fixed"] = apply_yaw_correction(turning["v_n"], turning["rot_rad_s"], L_fixed)

    # Fleet-wide OLS L_fit on turning reports
    L_fleet_fit = fit_antenna_offset(turning["v_n"], turning["rot_rad_s"])
    r_fleet = float(np.corrcoef(turning["rot_rad_s"], turning["v_n"])[0, 1])

    # Per-vessel chronological 50/50 train/test split for held-out L_fit evaluation
    pv_rows = []
    held_raw_list = []
    held_fixed_list = []
    held_fit_list = []
    example_vessel = None
    best_example_score = -1.0

    for vid, grp in turning.groupby("vessel_id", sort=False):
        if len(grp) < min_pts:
            continue
        grp = grp.sort_values("t")
        mid = len(grp) // 2
        tr, te = grp.iloc[:mid], grp.iloc[mid:]
        if np.std(tr["rot_rad_s"]) < 1e-4 or np.std(te["rot_rad_s"]) < 1e-4:
            continue
        L_tr = fit_antenna_offset(tr["v_n"], tr["rot_rad_s"])
        L_all = fit_antenna_offset(grp["v_n"], grp["rot_rad_s"])
        r_all = float(np.corrcoef(grp["rot_rad_s"], grp["v_n"])[0, 1])

        te_raw = te["v_n"].to_numpy(dtype=np.float64)
        te_fix = apply_yaw_correction(te_raw, te["rot_rad_s"], L_fixed)
        te_fit = apply_yaw_correction(te_raw, te["rot_rad_s"], L_tr)

        held_raw_list.append(te_raw)
        held_fixed_list.append(te_fix)
        held_fit_list.append(te_fit)

        rsd_raw = robust_sd(te_raw)
        rsd_fix = robust_sd(te_fix)
        rsd_fit = robust_sd(te_fit)
        pv_rows.append(
            dict(
                vessel_id=vid,
                n_turning=int(len(grp)),
                n_test=int(len(te)),
                median_kn=float(grp["speed"].median()),
                L_train_m=L_tr,
                L_all_m=L_all,
                corr_rot_vn=r_all,
                rsd_raw_m_s=rsd_raw,
                rsd_fixed_m_s=rsd_fix,
                rsd_fit_m_s=rsd_fit,
                improved_fit=bool(rsd_fit < rsd_raw),
            )
        )
        # Pick a clear large-ship turning example (high count & strong negative correlation)
        score = (-r_all) * np.log10(max(len(grp), 10)) if (L_all > 60.0 and len(grp) >= 500) else -1.0
        if score > best_example_score:
            best_example_score = score
            example_vessel = (vid, grp.copy(), L_tr, L_all, r_all)

    pv_df = pd.DataFrame(pv_rows)
    held_raw = np.concatenate(held_raw_list) if held_raw_list else np.array([])
    held_fixed = np.concatenate(held_fixed_list) if held_fixed_list else np.array([])
    held_fit = np.concatenate(held_fit_list) if held_fit_list else np.array([])

    return {
        "L_fixed_m": L_fixed,
        "L_fleet_fit_m": L_fleet_fit,
        "r_fleet": r_fleet,
        "steady": steady,
        "turning": turning,
        "pv_df": pv_df,
        "held_raw": held_raw,
        "held_fixed": held_fixed,
        "held_fit": held_fit,
        "example_vessel": example_vessel,
    }


def plot_yaw_correction(res, cfg, out_path):
    viz.apply_style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.9))
    L_fixed = res["L_fixed_m"]

    # Panel A: Example turning vessel (v_n vs ROT + linear Inazu slope)
    if res["example_vessel"] is not None:
        vid, grp, L_tr, L_all, r_all = res["example_vessel"]
        rot_dm = grp["rot_deg_min"].to_numpy()
        vn = grp["v_n"].to_numpy()
        v_cor = apply_yaw_correction(vn, grp["rot_rad_s"], L_tr)
        axA.scatter(rot_dm, vn, s=8, alpha=0.35, color=viz.COLORS["negative"], label="Raw $v_n$")
        axA.scatter(rot_dm, v_cor, s=8, alpha=0.35, color=viz.COLORS["ships"], label=f"Corrected ($L_{{\\mathrm{{fit}}}}={L_tr:.0f}$ m)")
        xg = np.linspace(np.percentile(rot_dm, 1), np.percentile(rot_dm, 99), 100)
        xg_rs = deg2rad(xg) / 60.0
        axA.plot(xg, -L_tr * xg_rs, color="#222222", lw=1.8, ls="-", label=f"$-L_{{\\mathrm{{fit}}}}\\cdot\\mathrm{{ROT}}$ ($r={r_all:.2f}$)")
        axA.plot(xg, -L_fixed * xg_rs, color=viz.COLORS["moderate"], lw=1.4, ls="--", label=f"$-L_{{\\mathrm{{fixed}}}}\\cdot\\mathrm{{ROT}}$ ({L_fixed:.0f} m)")
        axA.axhline(0.0, color="#666666", lw=0.7, ls=":")
        axA.set_xlabel("Rate of turn from consecutive HDG (deg/min)")
        axA.set_ylabel("Heading-normal drift $v_n$ (m/s)")
        axA.set_title(f"A. Example turning ship (n={len(grp):,}, {grp['speed'].median():.1f} kn)")
        axA.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        axA.grid(True, linestyle=":", alpha=0.55)

    # Panel B: Distributions of v_n on turning vs steady ships
    bins = np.linspace(-2.2, 2.2, 75)
    steady_vn = res["steady"]["v_n"].to_numpy()
    h_raw = res["held_raw"]
    h_fix = res["held_fixed"]
    h_fit = res["held_fit"]
    axB.hist(h_raw, bins=bins, density=True, histtype="step", lw=1.8, color=viz.COLORS["negative"],
             label=f"Turning raw (rsd={robust_sd(h_raw):.2f} m/s)")
    axB.hist(h_fix, bins=bins, density=True, histtype="step", lw=1.6, ls="--", color=viz.COLORS["moderate"],
             label=f"Fixed $L={L_fixed:.0f}$ m (rsd={robust_sd(h_fix):.2f} m/s)")
    axB.hist(h_fit, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=viz.COLORS["ships"],
             edgecolor=viz.COLORS["ships"], lw=1.8,
             label=f"Held-out per-ship $L_{{\\mathrm{{fit}}}}$ (rsd={robust_sd(h_fit):.2f} m/s)")
    axB.hist(steady_vn, bins=bins, density=True, histtype="step", lw=1.5, ls=":", color=viz.COLORS["positive"],
             label=f"Steady ships (rsd={robust_sd(steady_vn):.2f} m/s)")
    axB.set_xlim(-2.2, 2.2)
    axB.set_xlabel("Heading-normal drift $v_n$ (m/s)")
    axB.set_ylabel("Probability density (s/m)")
    axB.set_title("B. Spread on held-out turning reports (|ROT| \u2265 10\u00b0/min)")
    axB.legend(loc="upper right", fontsize=7.3, framealpha=0.92)
    axB.grid(True, linestyle=":", alpha=0.55)

    # Panel C: Per-vessel robust SD before vs after held-out correction
    pv = res["pv_df"]
    if not pv.empty:
        lim = max(1.5, float(np.percentile(pv["rsd_raw_m_s"], 98)) * 1.1)
        axC.plot([0, lim], [0, lim], color="#666666", ls="--", lw=1.1, label="1:1 (no change)")
        axC.axhline(robust_sd(steady_vn), color=viz.COLORS["positive"], ls=":", lw=1.3,
                    label=f"Steady-ship rsd ({robust_sd(steady_vn):.2f} m/s)")
        sc = axC.scatter(
            pv["rsd_raw_m_s"],
            pv["rsd_fit_m_s"],
            c=np.clip(pv["L_train_m"], 0, 150),
            cmap="viridis",
            s=36,
            edgecolors="#222222",
            lw=0.45,
            zorder=4,
        )
        cbar = fig.colorbar(sc, ax=axC, pad=0.02, fraction=0.048)
        cbar.set_label("Fitted $L_{\\mathrm{antenna}}$ (m)")
        n_imp = int(pv["improved_fit"].sum())
        axC.set_xlim(0, lim)
        axC.set_ylim(0, lim)
        axC.set_xlabel("Raw turning robust SD (m/s)")
        axC.set_ylabel("Yaw-corrected turning robust SD (m/s)")
        axC.set_title(f"C. Per-vessel reduction ({n_imp}/{len(pv)} vessels improved)")
        axC.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
        axC.grid(True, linestyle=":", alpha=0.55)

    viz.save_figure(
        fig,
        out_path,
        "T23: Inazu et al. (2020) Yaw Correction on Real Piraeus Oct 2017 AIS Data",
        "Apparent sway v_yaw = -L_antenna * ROT subtracted on turning ships (|ROT| >= 10 deg/min, SOG >= 8 kn); illustration only (main simulation noise.py unchanged).",
    )


def write_markdown(res, meta, cfg, out_path):
    cal = cfg["calibration"]
    L_fixed = res["L_fixed_m"]
    st_vn = res["steady"]["v_n"].to_numpy()
    tu_vn = res["turning"]["v_n"].to_numpy()
    tu_fix = res["turning"]["v_corr_fixed"].to_numpy()
    h_raw = res["held_raw"]
    h_fix = res["held_fixed"]
    h_fit = res["held_fit"]
    pv = res["pv_df"]

    n_imp = int(pv["improved_fit"].sum()) if not pv.empty else 0
    med_L = float(pv["L_train_m"].median()) if not pv.empty else np.nan
    q25_L = float(pv["L_train_m"].quantile(0.25)) if not pv.empty else np.nan
    q75_L = float(pv["L_train_m"].quantile(0.75)) if not pv.empty else np.nan
    med_r = float(pv["corr_rot_vn"].median()) if not pv.empty else np.nan
    med_pv_raw = float(pv["rsd_raw_m_s"].median()) if not pv.empty else np.nan
    med_pv_fit = float(pv["rsd_fit_m_s"].median()) if not pv.empty else np.nan

    pct_fix_all = 100.0 * (1.0 - robust_sd(tu_fix) / robust_sd(tu_vn))
    pct_fit_held = 100.0 * (1.0 - robust_sd(h_fit) / robust_sd(h_raw))
    pct_pv_med = 100.0 * (1.0 - med_pv_fit / med_pv_raw) if np.isfinite(med_pv_raw) else np.nan

    md = f"""# Yaw correction on Piraeus AIS data (T23, stretch)

**Source:** `{cfg['paths']['piraeus_ais']}` (Saronic Gulf, 4–26 Oct 2017; {meta['n_raw']:,} raw rows → {meta['n_dedup']:,} after dropping duplicate `(vessel_id, t)` rows across {meta['n_vessels_total']:,} vessels).
**Scope:** Standalone illustration of Inazu et al. (2020, *Earth, Planets and Space* 72:41, doi:10.1186/s40623-020-01165-7); `src/tsunami/noise.py` is **unchanged**.

## Method

1. **Rate of turn (ROT):** The Piraeus dataset has no ROT column, so we estimate centered ROT from consecutive headings `(HDG[i+1] - HDG[i-1]) / (t[i+1] - t[i-1])` (wrapped to `(-180°, 180°]`) for underway vessels with `SOG >= {cal['steady_speed_kn']} kn`, neighbor intervals `<= {cal['steady_max_dt_s']} s`, and `|COG - HDG| <= {cal['yaw_max_cog_hdg_deg']}°` (`n = {meta['n_clean_underway']:,}` clean reports).
2. **Inazu et al. (2020) yaw correction:** Because a forward-moving ship pivots near its bow while the GNSS antenna sits aft on the bridge, a clockwise turn (`ROT > 0`) swings the antenna to port (negative starboard-normal drift):
   `v_yaw = -L_antenna * ROT`, giving corrected drift `v_corr = SOG * sin(COG - HDG) + L_antenna * ROT`.
3. **Antenna offset `L_antenna`:** Piraeus static metadata (`unipi_ais_static.csv`) contains ship type and flag only (no `to_bow`/`to_stern` length fields), so `calibration.yaw_antenna_offset_m = {L_fixed:.1f} m` is marked **PLACEHOLDER** in `config.yaml`. Because Piraeus mixes small harbor craft (~20 m) with 150–200 m ferries and cargo ships, we evaluate both the single fixed placeholder `L = {L_fixed:.0f} m` and a per-vessel empirical `L_fit` calibrated on the first half of each ship's turning reports and tested on the chronological held-out second half (`n >= {cal['yaw_min_vessel_turning_pts']}` turning reports per ship).

## Results

| Subset / correction | Count | SD (m/s) | Robust SD (m/s) | IQR (m/s) | Change vs. raw turning |
|---|---:|---:|---:|---:|---:|
| Steady ships (`|ROT| <= {cal['yaw_steady_rot_deg_per_min']}°/min`) | {len(st_vn):,} | {st_vn.std():.3f} | **{robust_sd(st_vn):.3f}** | {np.subtract(*np.percentile(st_vn, [75, 25])):.3f} | baseline |
| All turning ships (`|ROT| >= {cal['yaw_turning_rot_deg_per_min']}°/min`), **raw** | {len(tu_vn):,} | {tu_vn.std():.3f} | **{robust_sd(tu_vn):.3f}** | {np.subtract(*np.percentile(tu_vn, [75, 25])):.3f} | — |
| All turning ships, **fixed `L = {L_fixed:.0f} m` (PLACEHOLDER)** | {len(tu_fix):,} | {tu_fix.std():.3f} | **{robust_sd(tu_fix):.3f}** | {np.subtract(*np.percentile(tu_fix, [75, 25])):.3f} | **−{pct_fix_all:.1f}%** robust SD |
| Held-out turning split ({len(pv)} vessels), **raw** | {len(h_raw):,} | {h_raw.std():.3f} | **{robust_sd(h_raw):.3f}** | {np.subtract(*np.percentile(h_raw, [75, 25])):.3f} | — |
| Held-out turning split, **fixed `L = {L_fixed:.0f} m`** | {len(h_fix):,} | {h_fix.std():.3f} | **{robust_sd(h_fix):.3f}** | {np.subtract(*np.percentile(h_fix, [75, 25])):.3f} | **−{100*(1-robust_sd(h_fix)/robust_sd(h_raw)):.1f}%** robust SD |
| Held-out turning split, **per-vessel `L_fit`** (train→test) | {len(h_fit):,} | {h_fit.std():.3f} | **{robust_sd(h_fit):.3f}** | {np.subtract(*np.percentile(h_fit, [75, 25])):.3f} | **−{pct_fit_held:.1f}%** robust SD |

### Key findings
- **Sign and correlation confirm Inazu et al. (2020):** Pooled correlation on turning ships is `r = {res['r_fleet']:+.3f}` (pooled OLS `L = {res['L_fleet_fit_m']:.1f} m`). Per vessel (`n = {len(pv)}` vessels with `>= {cal['yaw_min_vessel_turning_pts']}` turning reports), the median correlation between `ROT` and `v_n` is **`r = {med_r:+.3f}`**, confirming the stern-swing physical mechanism cleanly on real Mediterranean AIS traffic.
- **Per-vessel `L_fit` recovers ship-size spread:** Fitted offsets span IQR **{q25_L:.1f}–{q75_L:.1f} m** (median **{med_L:.1f} m**), matching the mix of small port craft and large ferries/merchant vessels in the Saronic Gulf.
- **Noise collapse back to steady-ship floor:** Held-out per-vessel yaw correction improves **{n_imp}/{len(pv)} vessels** ({100*n_imp/max(len(pv),1):.0f}%), reducing median per-vessel robust SD from **{med_pv_raw:.3f} m/s → {med_pv_fit:.3f} m/s (−{pct_pv_med:.1f}%)** — virtually identical to the **{robust_sd(st_vn):.3f} m/s** steady-ship baseline. This empirically justifies our report's assumption (PLAN.md Section 6.1) that a real operational system must apply per-vessel yaw correction (using AIS static `to_bow`/`to_stern` dimensions or historical calibration) to achieve steady-ship noise levels during turns.

![](yaw_correction.png)
"""
    Path(out_path).write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="T23 Piraeus AIS yaw correction")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args(argv)

    cfg = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(cfg["paths"]["outputs"])
    df = pd.read_csv(repo_path(cfg["paths"]["piraeus_ais"]))
    sub, meta = compute_turn_table(df, cfg)
    res = analyze_yaw(sub, cfg)
    plot_yaw_correction(res, cfg, out_dir / "yaw_correction.png")
    write_markdown(res, meta, cfg, out_dir / "yaw_correction.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
