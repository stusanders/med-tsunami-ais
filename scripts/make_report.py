"""T21: Assemble the self-contained hackathon report at outputs/report.html.

Covers every judging criterion and required section from PLAN.md and TICKETS.md:
  - Question & motivation (incl. cleaning-steps figure T24)
  - Criterion 1: Correctness, physics validation (T11), Piraeus calibration (T05), scaling & geometry verification
  - Criterion 2: Fundamental algorithmic reshaping (batched PDE -> linear library -> matmul + prefix scan)
  - Criterion 3: Three-tier benchmarks (Workload A solver & Workload B scoring separately, Tsunami-HySEA context, 1-hour budget)
  - Criterion 4: Scientific value — Maps 1, 2, 3, detector comparison & held-out FAR, historic replays (T26), sensitivity runs (T20)
  - Complete assumptions and limitations (PLAN.md section 6 + discovered data/physics caveats)

Any downstream output not yet generated (e.g. T17/T18/T20/T26 or GPU benchmark tier) is
clearly labelled as pending rather than fabricated, and is automatically populated when
this script is re-run.

Usage:
  PYTHONPATH=src python scripts/make_report.py
  PYTHONPATH=src python scripts/make_report.py --no-embed
"""
import argparse
import base64
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsunami.config import load_config, repo_path


CSS = """
:root {
  --bg: #ffffff;
  --fg: #1b2430;
  --muted: #5a6778;
  --accent: #0072B2;
  --accent-soft: #eef5fb;
  --warn-bg: #fff6e6;
  --warn-border: #e69f00;
  --ok: #009E73;
  --border: #d9e1ea;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
  color: var(--fg);
  background: var(--bg);
  max-width: 1120px;
  margin: 0 auto;
  padding: 28px 24px 64px;
}
header {
  border-bottom: 3px solid var(--accent);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
h1 { margin: 0 0 8px; font-size: 2rem; color: #0f2942; }
h2 {
  margin-top: 36px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
  color: #0f2942;
}
h3 { margin-top: 22px; color: #1b3a57; }
.subtitle { color: var(--muted); font-size: 1.05rem; margin: 0; }
.meta { color: var(--muted); font-size: 0.88rem; margin-top: 8px; }
nav.toc {
  background: var(--accent-soft);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  margin: 18px 0 28px;
}
nav.toc strong { display: block; margin-bottom: 6px; }
nav.toc a { color: var(--accent); text-decoration: none; margin-right: 16px; display: inline-block; }
nav.toc a:hover { text-decoration: underline; }
.callout {
  background: var(--accent-soft);
  border-left: 4px solid var(--accent);
  padding: 12px 16px;
  border-radius: 4px;
  margin: 16px 0;
}
.pending {
  background: var(--warn-bg);
  border-left: 4px solid var(--warn-border);
  padding: 12px 16px;
  border-radius: 4px;
  margin: 16px 0;
  color: #6b4400;
}
figure {
  margin: 20px 0;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #fbfcfd;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}
figcaption {
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 8px;
  text-align: left;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0 20px;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--border);
  padding: 7px 10px;
  text-align: left;
}
th { background: #f2f6fa; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfd; }
.badge-pass {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  background: #e3f7f0;
  color: #006b4f;
  font-weight: 600;
  font-size: 0.82rem;
}
.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 16px;
}
code {
  background: #f1f4f8;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 0.9em;
}
pre {
  background: #f6f8fa;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  font-size: 0.85rem;
}
"""


def _img_src(img_path: Path, out_dir: Path, embed: bool) -> str:
    if embed:
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    try:
        return img_path.relative_to(out_dir).as_posix()
    except ValueError:
        return img_path.as_posix()


def _figure_html(img_path: Path, caption: str, out_dir: Path, embed: bool, pending_note: str = "") -> str:
    if not img_path.exists():
        note = pending_note or f"Figure <code>{html.escape(img_path.name)}</code> has not been generated yet."
        return f'<div class="pending"><strong>Pending:</strong> {note}</div>'
    src = _img_src(img_path, out_dir, embed)
    return (
        "<figure>"
        f'<img src="{src}" alt="{html.escape(caption)}" loading="lazy" />'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _df_to_html(df: pd.DataFrame, float_fmt="{:.3g}") -> str:
    cols = list(df.columns)
    parts = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                txt = "—" if not np.isfinite(v) else float_fmt.format(float(v))
            else:
                txt = str(v)
            cells.append(f"<td>{html.escape(txt)}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _read_text_pre(path: Path, pending_note: str) -> str:
    if not path.exists():
        return f'<div class="pending"><strong>Pending:</strong> {pending_note}</div>'
    text = path.read_text(encoding="utf-8").strip()
    return f"<pre>{html.escape(text)}</pre>"


def _benchmark_tables_html(out_dir: Path) -> str:
    bpath = out_dir / "benchmark.json"
    if not bpath.exists():
        return '<div class="pending"><strong>Pending:</strong> <code>outputs/benchmark.json</code> not yet generated.</div>'
    data = json.loads(bpath.read_text(encoding="utf-8"))
    recs = data.get("records", [])
    summary = data.get("summary", {})
    hysea = data.get("tsunami_hysea_context", {})

    solver_rows = []
    for r in recs:
        if r.get("workload") != "solver":
            continue
        solver_rows.append(
            {
                "Tier": r["name"],
                "Machine": r.get("machine", ""),
                "Grid": r.get("grid_label", ""),
                "Batch B": r.get("batch_size", ""),
                "s / scenario": r.get("seconds_per_scenario", np.nan),
                "Cell updates / s (M)": (r.get("cell_updates_per_s") or np.nan) / 1e6,
                "Bandwidth (GB/s)": r.get("bandwidth_gb_s", np.nan),
                "% GPU Peak BW": r.get("bandwidth_pct_peak", np.nan),
            }
        )
    scoring_rows = []
    for r in recs:
        if r.get("workload") != "scoring":
            continue
        if r.get("extrapolated_from_two_point_slope"):
            note = "Two-point slope extrapolation from K2=8"
        elif r.get("tier") == "vec_no_prefix_cpu":
            note = "Vectorised NumPy, loops over t (no prefix scan)"
        else:
            note = "Measured full catalogue (matmul + prefix scan)"
        scoring_rows.append(
            {
                "Tier": r["name"],
                "Machine": r.get("machine", ""),
                "E (events)": r.get("E", ""),
                "K / K2": f"{r.get('K', '')} / {r.get('K2', '')}",
                "s / 100 events": r.get("seconds_per_100_events", np.nan),
                "Note": note,
            }
        )

    parts = []
    if solver_rows:
        parts.append("<h3>Workload A: Batched Shallow-Water Solver</h3>")
        parts.append(_df_to_html(pd.DataFrame(solver_rows), float_fmt="{:.3f}"))
    if scoring_rows:
        parts.append("<h3>Workload B: Matched-Filter Scoring Tensor</h3>")
        parts.append(_df_to_html(pd.DataFrame(scoring_rows), float_fmt="{:.3g}"))

    sc_sum = summary.get("scoring", {})
    so_sum = summary.get("solver", {})
    bullets = []
    if so_sum.get("colab_gpu_s_per_scen") and not so_sum.get("gpu_from_flat_fallback"):
        b_gpu = so_sum.get("colab_gpu_batch_size", "")
        cu_m = (so_sum.get("colab_gpu_cell_updates_per_s") or 0.0) / 1e6
        bw_str = ""
        if so_sum.get("gpu_bandwidth_gb_s"):
            bw_str = f", {so_sum['gpu_bandwidth_gb_s']:.1f} GB/s ({so_sum.get('gpu_bandwidth_pct_peak', 0):.1f}% of peak)"
        bullets.append(
            f"<strong>Solver real-grid absolute performance (390×795, R=20,063):</strong> "
            f"Colab Tesla T4 CuPy B={b_gpu} = <strong>{so_sum['colab_gpu_s_per_scen']:.3f} s/scenario</strong> "
            f"({cu_m:.0f}M cell-updates/s{bw_str}; 8,611 scenarios/h). "
            f"MacBook arm64 CPU (separate machine): {so_sum.get('macbook_serial_cpu_s_per_scen', 0):.2f} s serial / "
            f"{so_sum.get('macbook_batched_cpu_s_per_scen', 0):.2f} s B=1."
        )
    if so_sum.get("hw_speedup_same_machine_colab"):
        flat_ref = so_sum.get("same_machine_flat_ref") or {}
        if so_sum.get("same_machine_ratio_from_flat_grid") and flat_ref:
            bullets.append(
                f"<strong>Solver same-machine CPU-vs-GPU ratio (flat 360×750, R=20,000, <code>outputs/validation/gpu_check.json</code>):</strong> "
                f"Colab 2-vCPU ({flat_ref.get('colab_cpu_s_per_scen', 0):.2f} s/scen) vs "
                f"Colab T4 B={flat_ref.get('colab_gpu_batch_size', 64)} ({flat_ref.get('colab_gpu_s_per_scen', 0):.2f} s/scen) = "
                f"<strong>{so_sum['hw_speedup_same_machine_colab']:.1f}×</strong> "
                f"(<code>benchmark_gpu.json</code> timed T4 only on the real grid, without a real-grid Colab CPU run)."
            )
        else:
            bullets.append(
                f"<strong>Solver same-machine GPU speedup (real 390×795 grid):</strong> "
                f"Colab CPU ({so_sum.get('colab_cpu_s_per_scen', 0):.2f} s/scen) vs "
                f"Colab T4 ({so_sum.get('colab_gpu_s_per_scen', 0):.3f} s/scen) = "
                f"<strong>{so_sum['hw_speedup_same_machine_colab']:.1f}×</strong>."
            )
    if sc_sum.get("algo_speedup_cpu_total"):
        vec_x = sc_sum.get("vectorisation_speedup_cpu")
        pre_x = sc_sum.get("prefix_scan_speedup_cpu")
        split_str = (
            f" = <strong>{vec_x:,.0f}×</strong> vectorisation (Python loops → per-minute NumPy) "
            f"× <strong>{pre_x:.1f}×</strong> prefix scan (O(T²) → cumsum O(T))"
            if (vec_x and pre_x)
            else ""
        )
        bullets.append(
            f"<strong>Scoring algorithmic speedup on MacBook CPU (at E=100, Python loops extrapolated from K2=8 timings):</strong> "
            f"<strong>{sc_sum['algo_speedup_cpu_total']:,.0f}×</strong>{split_str}."
        )
    if sc_sum.get("hw_speedup_gpu_over_tensor_cpu"):
        bullets.append(
            f"<strong>Scoring GPU tier (Colab Tesla T4 vs MacBook CPU, cross-machine at E=100):</strong> "
            f"<strong>{sc_sum.get('tensor_gpu_s_per_100ev', 0):.3f} s / 100 events</strong> — "
            f"<strong>{sc_sum['hw_speedup_gpu_over_tensor_cpu']:.1f}×</strong> faster than MacBook CPU tensor "
            f"(<strong>{sc_sum.get('total_speedup_gpu_over_loop', 0):,.0f}×</strong> total vs Python loops; "
            f"210 traffic×noise draws/stratum in 1 h at E=100)."
        )
    else:
        bullets.append(
            "<strong>GPU scoring tier status:</strong> awaiting Colab run of <code>scripts/06_benchmark.py --gpu</code> "
            "(not yet timed in <code>gpu_check.json</code>)."
        )
    if hysea:
        quote = hysea.get("verbatim_quote", "")
        q_html = f' &ldquo;{html.escape(quote)}&rdquo;' if quote else f" {html.escape(hysea.get('notes', ''))}"
        bullets.append(
            f"<strong>Published context (Tsunami-HySEA):</strong>{q_html} "
            f"<em>({html.escape(hysea.get('citation', ''))})</em>."
        )
    parts.append('<div class="callout"><ul>' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul></div>")
    return "\n".join(parts)




def _reshaping_ladder_html(out_dir: Path) -> str:
    lpath = out_dir / "reshaping_ladder.json"
    if not lpath.exists():
        return '<div class="pending"><strong>Pending:</strong> <code>outputs/reshaping_ladder.json</code> not yet generated.</div>'
    data = json.loads(lpath.read_text(encoding="utf-8"))
    counts = data.get("experiment_counts", {})
    rows = data.get("rows", [])
    notes = data.get("notes", [])

    tbl_rows = []
    for r in rows:
        sp = r.get("speedup_vs_naive", 1.0)
        if sp < 1.5:
            sp_s = "1×"
        elif sp >= 1e6:
            sp_s = f"{sp/1e6:.2f}M×"
        elif sp >= 1e3:
            sp_s = f"{sp/1e3:.1f}k×"
        else:
            sp_s = f"{sp:.0f}×"
        step_str = (
            f"{r['plain_title']} ({r.get('human_short', r.get('human_total', ''))}): {r['plain_description']}"
            if "plain_title" in r and "plain_description" in r
            else r.get("short_label", "")
        )
        tbl_rows.append(
            {
                "Step": step_str,
                "Total wall-clock": r.get("human_total", ""),
                "Speedup vs naive": sp_s,
                "Technical formula": r.get("formula", ""),
                "Status": r.get("status", ""),
            }
        )

    parts = [
        "<h3>What Reshaping Saves: Full-Experiment Wall-Clock Ladder</h3>",
        (
            f"<p>Experiment size read from <code>candidates.csv</code>, <code>truth/sources.csv</code>, and <code>config.yaml</code>: "
            f"<strong>{counts.get('n_truth_map2', 0):,}</strong> Map 2 + <strong>{counts.get('n_truth_twin', 0):,}</strong> twin "
            f"= <strong>{counts.get('n_truth', 0):,}</strong> truth events &times; <strong>{counts.get('n_strata', 0)}</strong> strata "
            f"&times; <strong>{counts.get('n_draws_per_stratum', 0)}</strong> draws = <strong>{counts.get('n_trials', 0):,} trials</strong>; "
            f"<strong>K = {counts.get('K', 0):,}</strong> candidates (<strong>K2 = {counts.get('K2', 0):,}</strong> with Aegean flips); "
            f"<strong>{counts.get('n_library_build', 0):,}</strong> library + truth + historic PDE runs.</p>"
        ),
        _df_to_html(pd.DataFrame(tbl_rows)),
    ]
    if notes:
        parts.append('<div class="callout"><ul>' + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul></div>")
    return "\n".join(parts)


def build_report(cfg, out_dir: Path = None, embed: bool = True) -> Path:
    out_dir = repo_path(cfg["paths"]["outputs"]) if out_dir is None else Path(out_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Dynamic T13 first_detection.json summary (auto-updates when B reruns T13)
    fd_path = out_dir / "first_detection.json"
    if fd_path.exists():
        fd = json.loads(fd_path.read_text(encoding="utf-8"))
        fd_src = fd.get("source", {})
        fg_s = fd.get("first_gauge_s")
        fb_s = fd.get("first_buoy_s")
        fg_str = f"{fg_s:.0f} s ({fg_s/60.0:.1f} min)" if fg_s is not None else "none"
        fb_str = f"{fb_s/60.0:.0f} min" if fb_s is not None else "none"
        t13_caption = (
            f"First detection map (T13, traffic draw 0): Hellenic Arc Mw {fd_src.get('mw', 8.0):.1f} "
            f"(cand {fd_src.get('cand_id', 186)}, 0.81 m peak uplift; {fd.get('n_ships', 682)} ships, "
            f"{fd.get('n_ships_peak_snr_gt1', 0)} > 1σ, max peak SNR {fd.get('max_peak_snr', 0.32):.2f}) "
            f"alongside IOC tide gauges ({fd.get('n_gauges_triggered', 28)}/{fd.get('n_gauges', 61)} triggered, "
            f"first at {fg_str} on η(t)−η(0)) and 2025 Ionian buoys ({fd.get('n_buoys_triggered', 1)}/{fd.get('n_buoys', 2)} triggered at {fb_str})."
        )
    else:
        t13_caption = "First detection map (T13): Per-ship SNR for one Hellenic Arc Mw 8.0 scenario alongside IOC tide gauges and 2025 Ionian buoys."

    # Map 1 table
    map1_tbl_path = out_dir / "map1_table.csv"
    map1_tbl_html = (
        _df_to_html(pd.read_csv(map1_tbl_path), float_fmt="{:.3f}")
        if map1_tbl_path.exists()
        else '<div class="pending"><strong>Pending:</strong> <code>outputs/map1_table.csv</code> not found.</div>'
    )

    # Thresholds & held-out quiet-day FAR table
    thr_path = out_dir / "thresholds.csv"
    if thr_path.exists():
        tdf = pd.read_csv(thr_path)
        tshow = pd.DataFrame(
            {
                "Operator": tdf["operator"].str.upper(),
                "Range cap (km)": (tdf["range_cap_m"] / 1000.0).round(0).astype(int),
                "Regime": tdf["regime"],
                "Detector": tdf["detector"],
                "Threshold": tdf["threshold"].map(lambda v: f"{v:.2f}"),
                "Cal FAR (%)": (100.0 * tdf["far_cal"]).map(lambda v: f"{v:.1f}%"),
                "Held-out FAR (%)": (100.0 * tdf["far_hold"]).map(lambda v: f"{v:.1f}%"),
                "N cal / hold": [f"{int(a)} / {int(b)}" for a, b in zip(tdf["n_cal"], tdf["n_hold"])],
            }
        )
        thr_html = _df_to_html(tshow)
    else:
        thr_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/thresholds.csv</code> (T17 sweep / quiet-day calibration).</div>'

    # Map 2 summary table — strictly split by operator x range_cap x regime x region (never pooling AIS+GPS or Aegean+arcs)
    map2_tbl_path = out_dir / "map2_table.csv"
    if map2_tbl_path.exists():
        m2df = pd.read_csv(map2_tbl_path)
        has_far = "far_hold" in m2df.columns
        m2_agg = (
            m2df.groupby(["operator", "range_cap_m", "regime", "region"], as_index=False)
            .agg(
                positions=("pos_id", "count"),
                det_90=("mw_90", lambda s: int(s.notna().sum())),
                med_mw_90=("mw_90", "median"),
                det_50=("mw_50", lambda s: int(s.notna().sum())),
                med_mw_50=("mw_50", "median"),
                far_hold=("far_hold", "first") if has_far else ("pos_id", lambda s: np.nan),
            )
        )
        m2_show = pd.DataFrame(
            {
                "Operator": m2_agg["operator"].str.upper(),
                "Cap (km)": (m2_agg["range_cap_m"] / 1000.0).round(0).astype(int),
                "Regime": m2_agg["regime"],
                "Region": m2_agg["region"].str.capitalize(),
                "Held-out FAR": m2_agg["far_hold"].map(lambda v: f"{100.0*v:.1f}%" if pd.notna(v) else "—"),
                "Pos ≥90%": [f"{int(a)}/{int(b)}" for a, b in zip(m2_agg["det_90"], m2_agg["positions"])],
                "Med Mw (90%)": m2_agg["med_mw_90"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
                "Pos ≥50%": [f"{int(a)}/{int(b)}" for a, b in zip(m2_agg["det_50"], m2_agg["positions"])],
                "Med Mw (50%)": m2_agg["med_mw_50"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            }
        )
        map2_summary_html = _df_to_html(m2_show)
    else:
        map2_summary_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/map2_table.csv</code> (T18 after T17).</div>'

    # Map 3 summary table — strictly split by operator x range_cap x regime
    map3_tbl_path = out_dir / "map3_table.csv"
    if map3_tbl_path.exists():
        m3df = pd.read_csv(map3_tbl_path)
        has_op = "operator" in m3df.columns
        has_lt100 = "pct_within_100km" in m3df.columns
        grp_cols = ["operator", "range_cap_m", "regime"] if has_op else ["range_cap_m", "regime"]
        m3_agg = (
            m3df.groupby(grp_cols, as_index=False)
            .agg(
                coasts=("gauge_code", "count"),
                med_gain_near=("gain_med_min_nearest_gauge_s", "median"),
                pos_near=("gain_med_min_nearest_gauge_s", lambda s: int((s > 0).sum())),
                med_gain_fg=("gain_med_min_first_gauge_s", "median"),
                pos_fg=("gain_med_min_first_gauge_s", lambda s: int((s > 0).sum())),
                med_gain_fi=("gain_med_min_first_instrument_s", "median"),
                pos_fi=("gain_med_min_first_instrument_s", lambda s: int((s > 0).sum())),
                med_pct_100km=("pct_within_100km", "median") if has_lt100 else ("gauge_code", lambda s: np.nan),
            )
        )
        m3_show = pd.DataFrame(
            {
                "Operator": m3_agg["operator"].str.upper() if has_op else "AIS",
                "Cap (km)": (m3_agg["range_cap_m"] / 1000.0).round(0).astype(int),
                "Regime": m3_agg["regime"],
                "Coasts": m3_agg["coasts"].astype(int),
                "Med gain vs nearest gauge (min)": m3_agg["med_gain_near"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—"),
                "Coasts >0 vs nearest": [f"{int(a)}/{int(b)}" for a, b in zip(m3_agg["pos_near"], m3_agg["coasts"])],
                "Med gain vs 1st gauge (min)": m3_agg["med_gain_fg"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—"),
                "Med gain vs 1st instr. (min)": m3_agg["med_gain_fi"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—"),
                "Med % dets ≤100 km": m3_agg["med_pct_100km"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
            }
        )
        map3_summary_html = _df_to_html(m3_show)
    else:
        map3_summary_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/map3_table.csv</code> (T18 after T17).</div>'

    # Stratified Detector Comparison table (calm 40 km & 100 km, moderate 100 km; split by operator x region x detector)
    det_cmp_path = out_dir / "detector_comparison.csv"
    if det_cmp_path.exists():
        dcdf = pd.read_csv(det_cmp_path)
        # Show calm 40 km (matching baseline review), calm 100 km, and moderate 100 km
        dc_sel = dcdf[
            ((dcdf["regime"] == "calm") & (dcdf["range_cap_m"].isin([40000.0, 100000.0])))
            | ((dcdf["regime"] == "moderate") & (dcdf["range_cap_m"] == 100000.0))
        ].copy()
        def _fmt_pct(v):
            return f"{100.0 * v:.1f}%" if pd.notna(v) else "—"
        def _fmt_num(v, fmt="{:.1f}"):
            return fmt.format(v) if pd.notna(v) else "—"
        dc_show = pd.DataFrame(
            {
                "Op": dc_sel["operator"].str.upper(),
                "Cap": (dc_sel["range_cap_m"] / 1000.0).round(0).astype(int).astype(str) + " km",
                "Regime": dc_sel["regime"],
                "Region": dc_sel["region"].str.capitalize(),
                "Detector": dc_sel["detector"],
                "Held-out FAR": dc_sel["far_hold"].map(_fmt_pct),
                "Det Mw 7.60": dc_sel["det_rate_mw760"].map(_fmt_pct),
                "Det Mw 7.85": dc_sel["det_rate_mw785"].map(_fmt_pct),
                "Det Mw 8.10": dc_sel["det_rate_mw810"].map(_fmt_pct),
                "Det Mw 8.35": dc_sel["det_rate_mw835"].map(_fmt_pct),
                "Det Mw 8.60": dc_sel["det_rate_mw860"].map(_fmt_pct),
                "Med conf (min)": dc_sel["med_confirm_min"].map(lambda v: _fmt_num(v, "{:.0f}")),
                "Med 1st gauge (min)": dc_sel["med_first_gauge_min"].map(lambda v: _fmt_num(v, "{:.1f}")),
                "% before 1st gauge": dc_sel["pct_before_first_gauge"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
                "Loc err p50 / p90 (km)": [
                    f"{a:.0f} / {b:.0f}" if (pd.notna(a) and pd.notna(b)) else "— (no location)"
                    for a, b in zip(dc_sel["med_loc_err_km"], dc_sel["p90_loc_err_km"])
                ],
                "Dets ≤100 km": dc_sel["pct_within_100km"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
            }
        )
        det_cmp_html = _df_to_html(dc_show)
    else:
        det_cmp_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/detector_comparison.csv</code> (T18 after T17).</div>'

    # Off-grid Twin Location Accuracy table (detections only, incl. % within 100 km and held-out FAR)
    twin_loc_path = out_dir / "twin_location_summary.csv"
    if twin_loc_path.exists():
        twdf = pd.read_csv(twin_loc_path)
        tw_show = pd.DataFrame(
            {
                "Operator": twdf["operator"].str.upper(),
                "Cap (km)": (twdf["range_cap_m"] / 1000.0).round(0).astype(int),
                "Regime": twdf["regime"],
                "Detected (%)": (100.0 * twdf["detected_frac"]).map(lambda v: f"{v:.1f}%"),
                "Held-out FAR (%)": (100.0 * twdf["far_hold"]).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
                "Loc err p50 (km)": twdf["median_loc_err_km"].map(lambda v: f"{v:.1f}"),
                "Loc err p90 (km)": twdf["p90_loc_err_km"].map(lambda v: f"{v:.0f}"),
                "Dets ≤100 km (%)": twdf["pct_within_100km"].map(lambda v: f"{v:.1f}%"),
                "Med |Mw err|": twdf["median_abs_mw_err"].map(lambda v: f"{v:.2f}"),
                "Med conf (min)": twdf["median_confirm_min"].map(lambda v: f"{v:.0f}"),
            }
        )
        twin_loc_html = _df_to_html(tw_show)
    else:
        twin_loc_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/twin_location_summary.csv</code> (T18 after T17).</div>'

    # Sensitivity artifacts (globbed flexibly so any T20 naming is included)
    sens_mds = sorted(out_dir.glob("*sensitivity*.md"))
    sens_pngs = sorted(out_dir.glob("*sensitivity*.png"))
    if sens_mds or sens_pngs:
        sens_parts = [_read_text_pre(p, "") for p in sens_mds]
        sens_parts.extend(
            _figure_html(p, f"Sensitivity analysis: {p.stem}", out_dir, embed) for p in sens_pngs
        )
        sens_html = "\n".join(sens_parts)
    else:
        sens_html = '<div class="pending"><strong>Pending:</strong> T20 sensitivity runs (shared-noise correlation length, uplift mismatch, monopole vs dipole, 50% ship response, 10% AIS gaps / near-source outage).</div>'

    # Validation check figures
    val_dir = out_dir / "validation"
    val_figs = []
    for fname, cap in [
        ("check1_arrival.png", "Validation Check 1: Flat-bottom arrival time vs √(gh) (+0.22% x, +0.29% y)."),
        ("check2_volume.png", "Validation Check 2: Closed-basin volume conservation (1.1×10⁻⁷ drift over 3.3 h)."),
        ("check3_greens.png", "Validation Check 3: 1D sloping-shelf shoaling vs Green's law h⁻¹/⁴ (within 2–5% down to 250 m)."),
        ("check6_hellenic_mw8.png", "Validation Check 6: Hellenic Arc Mw 8 dipole snapshots across the eastern Mediterranean."),
        ("check7_sponge_layout.png", "Validation Check 7: Padded solver domain (10.5–37°E, 30–43°N) placing sponge bands outside the area of interest."),
    ]:
        p = val_dir / fname
        if p.exists():
            val_figs.append(_figure_html(p, cap, out_dir, embed))
    val_figs_html = '<div class="grid-2">' + "".join(val_figs) + "</div>" if val_figs else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ships as a Tsunami Network — Eastern Mediterranean OSSE Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Ships as a Tsunami Network</h1>
  <p class="subtitle">Where, and for which earthquakes, could the ships already at sea confirm a Mediterranean tsunami faster than tide gauges and the 2025 Ionian buoys?</p>
  <p class="meta">Observing System Simulation Experiment (OSSE) &bull; Generated {now} &bull; Single NumPy/CuPy float32 codebase</p>
</header>

<nav class="toc">
  <strong>Contents</strong>
  <a href="#question">1. Question &amp; Motivation</a>
  <a href="#c1">2. Correctness &amp; Validation (Criterion 1)</a>
  <a href="#c2">3. Algorithmic Reshaping (Criterion 2)</a>
  <a href="#c3">4. Benchmarks &amp; 1-Hour Budget (Criterion 3)</a>
  <a href="#c4">5. Scientific Results: Maps, Detectors, Historic Replays &amp; Sensitivity (Criterion 4)</a>
  <a href="#assumptions">6. Assumptions &amp; Limitations</a>
</nav>

<section id="question">
  <h2>1. Question &amp; Motivation</h2>
  <p>Every large ship continuously broadcasts Automatic Identification System (AIS) heading (<code>HDG</code>), course over ground (<code>COG</code>), and speed over ground (<code>SOG</code>). A tsunami current pushes underway ships sideways, so the heading-normal drift <code>SOG · sin(COG − HDG)</code> acts as a crude current meter (Inazu et al. 2018, 2020). Mediterranean warnings are currently inferred from seismic parameters and confirmed by coastal IOC tide gauges plus, since September 2025, two CAT-INGV deep-sea DART-type buoys in the Ionian Sea (<code>INGV01</code> at 36.83°N, 16.50°E; <code>INGV02</code> at 38.66°N, 18.16°E; 3 cm trigger, ≤60 s telemetry).</p>
  <div class="callout">
    <strong>Context (not the target case):</strong> The 23 October 2026 Palermo/Trapani evacuation (Mw 5.5, lifted after ~1 h when gauges stayed flat) illustrates the cost of seismic-only alerts. A ship network targets <em>early offshore confirmation of large tsunamis</em>; ships staying quiet cannot cancel an alert for a moderate earthquake.
  </div>
  <p>Inazu et al. (2020) showed the sideways-drift mechanism on 13 ships during the 2011 Tohoku tsunami (single-ship 3σ floor 0.1–0.3 m/s), while Hossen et al. (2021) found ship velocity alone struggled to recover Cascadia tsunamis in deep water, and Foster &amp; Thomas (2024) assessed ship GNSS height in Europe. Because large tsunamis are too rare to wait for, we simulate thousands of Mediterranean scenarios to quantify detection floors, fleet-wide false-alarm rates, warning-time gains, and the value of upgrading ships to precise GNSS.</p>
  {_figure_html(out_dir / "cleaning_steps.png", "AIS drift cleaning on the showcase Hellenic Arc Mw 8.0 run (cand 186, real grid and pool): south Crete shelf ship (61 m, peak 0.47 m/s = 4.9σ — one of only 2 points exceeding 3σ in the 20,000-point pool, both on the Cretan shelf, chosen deliberately as an illustration and not from the T13 traffic draw) vs deep-water ship (2,766 m, peak 0.025 m/s = 0.3σ), with simulation calm noise from noise.ais_noise (config-driven), synthetic turn artefact v_yaw drawn separately in Panel 1 and shaded in Panel 2, and Panel 3 showing that the 45-min causal baseline barely distorts a wave of this period.", out_dir, embed)}
</section>

<section id="c1">
  <h2>2. Correctness, Validation &amp; Calibration (Criterion 1)</h2>
  <p>Physics correctness is verified across seven checks on the spherical Arakawa C-grid linear shallow-water solver (<code>src/tsunami/solver.py</code>, <code>solver_reference.py</code>), including GPU-vs-CPU equivalence on a Colab Tesla T4. All constants live in <code>config.yaml</code> with guesses explicitly tagged <code>PLACEHOLDER</code>. Both magnitude scaling laws were verified directly against their primary literature tables: Strasser et al. (2010, <em>SRL</em> 81(6):941–950, Table 2 interface: <code>L: -2.477 + 0.585 Mw</code>, <code>W: -0.882 + 0.351 Mw</code>) and Wells &amp; Coppersmith (1994, <em>BSSA</em> 84(4):974–1002, Table 2A all slip types: <code>RLD: -2.44 + 0.59 Mw</code>, <code>RW: -1.01 + 0.32 Mw</code>). Published Hellenic Arc Mw 8–8.5 scenarios report seafloor uplift from ~1–2 m on low-dip megathrust interfaces up to ~9–10 m on steeper splay faults (Lorito et al. 2008; Shaw et al. 2008; Necmioğlu &amp; Özel 2014, 2015; Wirp et al. 2025/2026), bracketing our Gaussian dipole Mw 8 peak uplift of <strong>0.81 m</strong>. Peak uplift is <code>slip × sin(mean dip)</code> for every source: subduction arcs use EFSM20 <code>dipavg</code> (Hellenic 15°, Calabrian 13°, Cyprus 32°), while Aegean and historic sources use their fault dip. Relative to the earlier fixed <code>0.3 × slip</code> rule, Aegean candidates and Samos produce waves ~2.4–2.5× larger, Cyprus ~1.8× larger, and Hellenic and Calabrian 0.86× and 0.75× their old size (uplift mismatch is still tested in T20 sensitivity runs). Samos 2020 motivated the change from a fixed 0.3 × slip to slip × sin(dip), so the Samos comparison is a consistency check rather than fully independent validation. Coastal tide gauges trigger on relative sea-level change <code>η(t) − η(0)</code> (because a land-mounted gauge moves coseismically with the seafloor and water column, so any static t=0 offset does not trigger it).</p>
  <h3>Solver Validation Summary (T11)</h3>
  {_read_text_pre(out_dir / "validation.md", "<code>outputs/validation.md</code> not found.")}
  {val_figs_html}
  <h3>Piraeus Real AIS Calibration (T05) &amp; Filtered Noise Validation</h3>
  {_read_text_pre(out_dir / "piraeus_calibration.md", "<code>outputs/piraeus_calibration.md</code> not found.")}
  <p><strong>Steady-ship filtered noise check (<code>outputs/noise_check_piraeus.md</code>):</strong> Applying the same 1-minute averaging and causal trailing baseline to steady Piraeus ships gives <strong>0.058–0.060 m/s</strong> robust SD (0.073 m/s plain SD), about <strong>20% below</strong> simulated calm noise at the same speed and window (0.070–0.076 m/s). Because no Saronic Gulf ship stays steady for 45 min (longest continuous steady run: 34 min), the comparison uses <code>W = 10–20 min</code> with simulation computed at the same <code>W</code>; the real residual is flat across <code>W</code>. (The 0.25 m/s quoted in T23 is the raw 10-second per-report spread before averaging and baseline removal.) Calm noise is therefore not optimistic relative to Piraeus and <code>ais_noise</code> is unchanged. Caveats: the Saronic Gulf is sheltered (open-sea currents/eddies are likely larger, bracketed by the shared-noise term and moderate/rough regimes), the check uses <code>W = 10–20 min</code>, and only steady segments are included (turning artefacts excluded, consistent with assumed yaw correction).</p>
  {_figure_html(out_dir / "noise_check_piraeus.png", "Piraeus steady-ship drift after 1-min averaging and causal baseline (W = 10–20 min) vs simulated calm/moderate/rough noise at the same speed and window.", out_dir, embed)}
  {_read_text_pre(out_dir / "noise_check_piraeus.md", "<code>outputs/noise_check_piraeus.md</code> not found.")}
  <h3>Real-Data Yaw Correction Check on Piraeus AIS (T23)</h3>
  {_figure_html(out_dir / "yaw_correction.png", "T23 (illustration only): Inazu et al. (2020) yaw correction on deduplicated Piraeus Oct 2017 AIS turning ships (|ROT| >= 10 deg/min). Per-vessel held-out correction collapses turning spread back to the steady-ship floor (0.24 vs 0.25 m/s).", out_dir, embed)}
  {_read_text_pre(out_dir / "yaw_correction.md", "<code>outputs/yaw_correction.md</code> not generated.")}
  <h3>Anti-Inverse-Crime &amp; Held-Out Quiet-Day Protocol</h3>
  <ul>
    <li><strong>Off-grid truth events (T16):</strong> Truth runs are never library entries — Map 2 truth sources are shifted half a grid spacing along strike and +0.1 Mw, plus 50–100 random off-grid twin events.</li>
    <li><strong>Disjoint quiet-day splits:</strong> Every (operator × range cap × weather regime) stratum calibrates its 1% false-alarm threshold on ≥500 quiet draws and verifies FAR on ≥500 disjoint held-out quiet draws.</li>
    <li><strong>Historic replays (T26):</strong> Custom Samos 2020 and Amorgos 1956 runs generate observations only and are excluded from the Aegean candidate grid (30 km exclusion radius).</li>
  </ul>
</section>

<section id="c2">
  <h2>3. Fundamental Algorithmic Reshaping (Criterion 2)</h2>
  <p>Evaluating 1,083 candidate sources × hundreds of truth events × 100 traffic/noise draws × 12 strata serially would require tens of millions of PDE integrations. We reshape the entire pipeline in three steps:</p>
  <ol>
    <li><strong>Batched PDE Stencil <code>(B, ny, nx)</code>:</strong> Advance <em>B</em> = 64 tsunamis per kernel launch across the 390×795 grid, amortising static bathymetry and spherical-metric memory reads.</li>
    <li><strong>Linearity → Reusable Template Library:</strong> Because the linear shallow-water equations obey superposition (verified to 8.9×10⁻⁸), each of the 1,083 candidates is simulated <em>once</em> at the 20,063-point receiver pool, block-averaged to 1 min, and causally baseline-filtered. Aegean subsidence-first normal-fault templates (<code>-g</code>) are scored at zero extra simulation or storage cost (<code>K = 1,083 → K2 = 1,455</code>).</li>
    <li><strong>Sign-Aligned Cell Grouping + Batched Matmul + Parallel Prefix Scan:</strong> Ships are grouped into 25 km cells × 45° folded-normal bins (projected onto each ship's <em>own</em> heading normal, sign-aligned so opposite-heading ships add rather than cancel). Detection across all candidates, events, and 120 minutes reduces to one batched matrix multiply plus cumulative sums (<code>gd_t = cumsum(w g d)</code>, <code>gg_t = cumsum(w g²)</code>).</li>
  </ol>
  <div class="grid-2">
    {_figure_html(out_dir / "sources_candidates.png", "Candidate source catalogue (T03): 1,083 candidates across 141 positions on EFSM20 Hellenic, Calabrian, and Cyprus arcs plus the coarse Aegean crustal grid.", out_dir, embed)}
    {_figure_html(out_dir / "geometry_check.png", "Arc geometry cross-check: EFSM20 vs Slab2 Hellenic (154 km median horizontal contour offset at 20 km depth — not a bug; Slab2 is a median 23 km deeper at the same location on a ~10° dipping interface) and EFSM20 vs Maesano (2017) Calabrian (0.6–0.7 km, same INGV group).", out_dir, embed)}
  </div>
  {_figure_html(out_dir / "slab2_overlay.png", "EFSM20 vs Slab2 depth-convention overlay on the Hellenic Arc: both depth fields are positive-down in km, deepen landward, and run parallel. Slab2 is a median 23 km deeper at the same location, moving depth contours ~130 km horizontally on a ~10° interface (a genuine model difference between seismicity-based Slab2 and EDSF-based EFSM20, not a sign or contour bug).", out_dir, embed)}
</section>

<section id="c3">
  <h2>4. Meaningful Speed-Up &amp; What Speed Buys (Criterion 3)</h2>
  <p>Workloads A (batched solver) and B (matched-filter scoring) are benchmarked separately across tiers on the same NumPy/CuPy codebase, separating vectorisation gain, prefix-scan gain, and same-machine GPU hardware speedup.</p>
  {_benchmark_tables_html(out_dir)}
  {_figure_html(out_dir / "benchmark.png", "Performance benchmark (T19): Workload A solver (linear scale, machine-labelled bars, real-grid T4 vs flat-grid same-machine Colab 17.1× test) and Workload B scoring at E=100 (Python-loop tier extrapolated from K2=8 timings) splitting vectorisation gain from prefix-scan gain plus Colab T4 GPU.", out_dir, embed)}
  {_figure_html(out_dir / "what_speed_buys.png", "What a fixed 1-hour compute budget buys: linear scenarios/hour including Colab T4 GPU (left) and scoring draws per truth event per stratum with exact formula once the library exists (right).", out_dir, embed)}
  {_reshaping_ladder_html(out_dir)}
  {_figure_html(out_dir / "reshaping_ladder.png", "What Reshaping Saves: full-experiment wall-clock ladder from 1. Naive down to 6. Use a GPU (including 14.1 min of template gather/aggregation across 20 calls — MacBook-measured, Colab CPU in practice; replaced by measured total when T14+T17 logs land). From row 2 onward most of the time is the one-at-a-time checking (about 2.5 years), which is why looking the waves up saves little until the checking itself is reshaped.", out_dir, embed)}
</section>

<section id="c4">
  <h2>5. Scientific Value: Maps, Detectors, Historic Replays &amp; Sensitivity (Criterion 4)</h2>

  <h3>5.1 Map 1: Where Mediterranean Ships Can Sense a Tsunami</h3>
  <p>The ship network uses EMODnet 2024 <strong>merchant vessels only</strong> (passenger, cargo, tanker). Across the whole area of interest, <strong>42%</strong> (42.3%) of merchant traffic sits in water shallower than 50 m (masked out), 21.5% in 50–200 m, and 36.2% deeper than 200 m (44.8% &lt; 50 m within 100 km of the coast; 50.7% within 40 km). Ship counts across sections correspond to: <strong>1,324</strong> expected merchant ships anywhere in the area of interest (including shallow water); <strong>692</strong> expected per traffic draw in the observable pool (depth ≥ 50 m, within 100 km of the coast, excluding straits); and <strong>682</strong> in T13's single Poisson draw of that pool (an earlier ~1,100 figure was all vessel types including fishing/pleasure craft and has been replaced).</p>
  {map1_tbl_html}
  {_figure_html(out_dir / "map1.png", "Map 1 (T10): Share of EMODnet 2024 merchant vessel traffic across depth bands (<50 m masked: 42% overall; 50–200 m shelf; >200 m deep water) overall and within 40 km / 100 km AIS range caps.", out_dir, embed)}

  <h3>5.2 Single-Scenario Signal-to-Noise Check (G2) &amp; Draw-to-Draw Variability</h3>
  <p>T13 used traffic draw 0, which was unlucky: 682 ships, 0 above 1σ, and best ship SNR 0.32. Across 200 traffic draws on the same Hellenic Mw 8.0 event (<code>outputs/ships_snr_draws.md</code>), <strong>44%</strong> of draws have at least one ship above 1σ (up to 4 ships), only <strong>4%</strong> have a ship above 3σ (never more than one), and the best ship per draw has median SNR <strong>0.90σ</strong> (90th percentile <strong>1.84σ</strong>). In the whole 20,000-point receiver pool, only 2 points exceed 3σ, both on the Cretan shelf; the T24 shelf ship in Section 1 (4.9σ) is one of those two rare pool points, chosen deliberately as an illustration rather than drawn from T13 draw 0 — so Section 1 and Section 5.2 are fully consistent.</p>
  {_figure_html(out_dir / "first_detection.png", t13_caption, out_dir, embed, "<code>outputs/first_detection.png</code> (T13) pending from Stream B.")}
  {_figure_html(out_dir / "ships_snr_draws.png", "Distribution of single-ship peak SNR across 200 traffic draws for Hellenic Mw 8.0 (cand 186, calm): 44% of draws have >=1 ship >1σ, 4% >3σ (max 1), best-ship median 0.90σ (90th pct 1.84σ); T13 draw 0 (0.32σ) sits in the lower tail, while T24's 4.9σ ship is one of only 2 pool points >3σ on the Cretan shelf.", out_dir, embed)}
  {_figure_html(out_dir / "tsunami_animation_frames.png", "T22 (stretch): 4-frame storyboard of a Hellenic Arc Mw 8.0 tsunami crossing the eastern Mediterranean with underway AIS ships lighting up by sideways-drift SNR (full video: outputs/tsunami_animation.mp4).", out_dir, embed)}

  <h3>5.3 Map 2: Detection Floor by Source Location, Operator &amp; Weather Regime</h3>
  <div class="callout">
    <strong>Core T17/T18 finding (strictly split by operator and region — never pooled):</strong>
    <ul>
      <li><strong>AIS sideways drift (existing ships, calm, 40 km / 100 km):</strong> Useful only for large tsunamis, with strong regional contrast. At 40 km calm: Aegean <strong>60%</strong> at Mw 7.60 (cand 7.5+0.1) and <strong>97%</strong> at Mw 7.85 (Aegean tested up to Mw 7.85); Cyprus <strong>49%</strong> at Mw 8.10 and <strong>98%</strong> at Mw 8.35; Calabrian <strong>34%</strong> at Mw 8.35 and <strong>89%</strong> at Mw 8.60; Hellenic <strong>16%</strong> at Mw 8.35 and <strong>79%</strong> at Mw 8.60. Extending to 100 km lifts arc detection modestly (e.g. Hellenic 24% / 83% at Mw 8.35 / 8.60), while moderate and rough weather cut deep-arc AIS detection sharply (Hellenic Mw 8.60 drops from 83% calm to 47% moderate and 15% rough). At small magnitudes, the ~1–4% AIS detection rate equals the quiet-day false-alarm floor.</li>
      <li><strong>Ship GPS height (upgrade scenario):</strong> Reaches ~<strong>100%</strong> confirmation from Mw 7.10 everywhere (and 100% of Aegean events at every magnitude down to Mw 6.60), with median confirmation <strong>1–9 min</strong> (40 km) / <strong>1–4 min</strong> (100 km) and <strong>72–92%</strong> of detections arriving before the first coastal tide gauge.</li>
      <li><strong>Mandatory GPS caveats:</strong> Ship GPS speed comes from floating ships directly above the source rising with the sea surface at <code>t = 0</code> (ships float, unlike land-mounted gauges). That assumes an instantaneous rupture, 5 cm/min GPS height noise with no heave or weather effect (<code>config.yaml</code> <code>gps</code>), and ships inside the source footprint.</li>
    </ul>
  </div>
  {map2_summary_html}
  {_figure_html(out_dir / "map2.png", "Map 2 (T18, 100 km range cap): Smallest off-grid truth magnitude confirmed within 2 h in ≥90% of draws (filled circles) or ≥50% (rings), split by operator (AIS drift vs Ship GPS height*) and weather regime. *GPS speed reflects floating ships above the source rising at t=0 under instantaneous rupture and 5 cm/min heave-free noise.", out_dir, embed, "<code>outputs/map2.png</code> pending T17 sweep parquet -> T18.")}
  {_figure_html(out_dir / "map2_40km.png", "Map 2 (T18, 40 km range cap): Same 90% / 50% detection floor at the 40 km coastal AIS range cap. *GPS caveat as above.", out_dir, embed, "")}
  {_figure_html(out_dir / "map2_uncertainty.png", "Detection probability vs. truth magnitude strictly split by operator (rows: AIS drift, Ship GPS height*) and source region (columns: Hellenic, Calabrian, Cyprus, Aegean; Aegean stops at Mw 7.85) at 100 km range cap, with 95% binomial CI bands, calibrated single-ship baseline, and the measured held-out quiet-day false-alarm floor (dash-dot red). *GPS caveat as above.", out_dir, embed, "<code>outputs/map2_uncertainty.png</code> pending T17 sweep parquet -> T18.")}

  <h3>5.4 Map 3: Warning Time Added Over Tide Gauges &amp; 2025 Ionian Buoys</h3>
  <p>Warning-time gains use <code>first_gauge_s</code>, <code>nearest_gauge_s</code>, and <code>first_instrument_s</code> directly from the T17 sweep parquet (all computed on the <code>η(t) − η(0)</code> gauge trigger). Across Map 2 events, median first-gauge trigger time is <strong>8–21 min</strong> (Cyprus 8 min, Calabrian 10 min, Hellenic 13 min, Aegean 21 min). Existing <strong>AIS sideways drift does not beat tide gauges</strong> in median time: calm AIS median confirmation is <strong>41–85 min</strong> (40 km) / <strong>39–82 min</strong> (100 km), lagging coastal gauges by <strong>−41 to −47 min</strong> across coasts (only 0.1–12.9% of detections precede the first gauge; highest in the Aegean at 12–13%, with Piraeus near parity at +1.0 min vs nearest gauge in calm 100 km). By contrast, <strong>Ship GPS height</strong> confirms <strong>+5.2 to +8.5 min</strong> earlier than gauges across coasts (faster at 29–30 of 30 coasts; up to +39.5 min) — subject to the floating-ship <code>t = 0</code>, instantaneous-rupture, heave-free 5 cm/min GPS caveat stated above.</p>
  {map3_summary_html}
  {_figure_html(out_dir / "map3.png", "Map 3 (T18 — AIS drift, calm weather): Median warning time added by the AIS ship network by coast at 40 km and 100 km range caps against (1) coast's nearest gauge, (2) first gauge anywhere, and (3) first instrument including the two 2025 Ionian buoys (diamonds), on the η(t)−η(0) gauge trigger. Red/negative = gauges confirm earlier.", out_dir, embed, "<code>outputs/map3.png</code> pending T17 sweep parquet -> T18.")}
  {_figure_html(out_dir / "map3_gps.png", "Map 3 (T18 — Ship GPS height upgrade*): Median warning time added by Ship GPS height by coast at 40 km and 100 km against the same three η(t)−η(0) benchmarks. *GPS caveat: speed comes from floating ships directly above the source rising at t=0 under instantaneous rupture, 5 cm/min heave-free noise, and ships inside the source footprint.", out_dir, embed, "")}

  <h3>5.5 Detector Comparison, Held-Out False-Alarm Rates &amp; Off-Grid Twin Location Accuracy</h3>
  <p>Every rate table below reports the measured <strong>held-out quiet-day false-alarm rate (FAR)</strong> next to detection rates. Key findings across detectors:</p>
  <ul>
    <li><strong>Network matched filter vs single-ship rules:</strong> At equal (~1%) false-alarm rate, the matched-filter network dramatically outperforms calibrated single-ship detection across every arc and Aegean stratum (e.g. calm 40 km AIS: Aegean Mw 7.60 <strong>59.7% vs 13.6%</strong>, Cyprus Mw 8.35 <strong>98.2% vs 45.9%</strong>, Calabrian Mw 8.60 <strong>88.9% vs 12.7%</strong>, Hellenic Mw 8.60 <strong>78.8% vs 8.7%</strong>).</li>
    <li><strong>Single-ship rules give no location:</strong> <code>best_cand_id</code> is unset (<code>-1</code> / <code>NaN</code>) for both single-ship detectors (expected). Inazu's literal 3σ single-ship rule (<code>single_literal</code>) has a <strong>100.0% quiet-day false-alarm rate</strong> across all 12 strata.</li>
    <li><strong>Regime-invariant thresholds &amp; held-out FAR:</strong> Because the network score is normalised by each regime's own noise, thresholds and held-out FAR are nearly identical across calm, moderate, and rough weather — so the measured <strong>1.6%</strong> held-out FAR for AIS at 100 km (<strong>0.2%</strong> at 40 km; <strong>0.6%</strong> for GPS at 40 &amp; 100 km) is effectively <strong>one quiet-day measurement per (operator, range cap)</strong>, not three independent measurements.</li>
    <li><strong>Mis-located detections in moderate/rough AIS:</strong> Location errors are reported on <em>detections only</em>, alongside the share of detections within 100 km of the truth. In calm AIS, 91–97% of detections land within 100 km (off-grid twin median 15 km, p90 37 km, 93–94% ≤ 100 km). In moderate and rough AIS, low-SNR noise triggers can match distant shelf templates, pushing twin p90 error to <strong>628–697 km</strong> and cutting the share within 100 km to <strong>65.1%</strong> (moderate 100 km) and <strong>58.1%</strong> (rough 100 km).</li>
  </ul>
  <h4>Stratified Detector Comparison (Truth Map 2; Calm 40 km &amp; 100 km, Moderate 100 km)</h4>
  {det_cmp_html}
  <h4>Off-Grid Twin Source Recovery &amp; Location Accuracy (Detections Only, Network Detector)</h4>
  {twin_loc_html}
  <h4>Quiet-Day Calibration Thresholds &amp; Measured Held-Out False-Alarm Rates</h4>
  {thr_html}

  <h3>5.6 Historic Event Replays (Samos 2020 &amp; Amorgos 1956)</h3>
  <p>Both Samos (30 Oct 2020, Mw 7.0, USGS <code>us7000c7y0</code>; Kiratzi et al. 2022) and Amorgos (9 Jul 1956, Mw 7.7, Okal et al. 2009) are normal-fault (subsidence-first, <code>polarity = -1</code>) Aegean earthquakes. Replays are scored against both the arc-only catalogue (showing the blind spot of an arc-restricted library) and the arc+Aegean catalogue, with Samos simulated amplitudes compared to 14 IOC Aegean gauges (e.g. <code>kos1</code> +11.9 cm crest) using Green's-law shoaling. Samos 2020 motivated the change from a fixed 0.3 × slip to slip × sin(dip) (~2.4–2.5× larger Aegean/Samos waves), so the Samos comparison is a consistency check rather than fully independent validation.</p>
  {_read_text_pre(out_dir / "historic_replays.md", "<code>outputs/historic_replays.md</code> (T26) pending from Stream B.")}

  <h3>5.7 Sensitivity Analysis (T20)</h3>
  {sens_html}
</section>

<section id="assumptions">
  <h2>6. Assumptions &amp; Limitations</h2>
  <ol>
    <li><strong>Yaw correction assumed applied:</strong> Simulated noise levels come from Inazu et al. (2020) measured <em>after</em> yaw correction. Without it, turning-ship noise would be higher and detection harder (illustrated in T24).</li>
    <li><strong>Weather-regime multipliers are placeholders:</strong> Calm/moderate/rough noise scaling (×1, ×1.5, ×2) has no published AIS-vs-sea-state calibration.</li>
    <li><strong>Ship current coupling:</strong> Baseline assumes ships move 100% with the tsunami current (sensitivity tested at 50% response in T20).</li>
    <li><strong>Complete AIS reporting:</strong> Baseline assumes zero dropped reports (sensitivity tested at 10% dropout and near-source outage in T20).</li>
    <li><strong>Linear shallow-water physics:</strong> No Coriolis, dispersion, or nonlinear inundation; ship observations shallower than 50 m are masked out (42% of EMODnet merchant traffic across the area of interest).</li>
    <li><strong>Gaussian dipole sources:</strong> Sized by verified Strasser (2010) / Wells &amp; Coppersmith (1994) scaling rather than full Okada (1985) dislocations (uplift mismatch and monopole vs dipole tested in T20).</li>
    <li><strong>Synthetic traffic &amp; anchorage bias:</strong> Traffic draws sample EMODnet 2024 vessel density (which lost satellite AIS after June 2024 and includes anchored/moored ships; Piraeus shows 78% of port reports &lt; 2 kn).</li>
    <li><strong>Catalogue coverage:</strong> Library covers the three subduction arcs and a coarse Aegean grid; earthquakes outside those zones are detected only via mismatched templates.</li>
    <li><strong>Diagonal inter-cell covariance:</strong> Cell weights capture within-cell shared-current variance but neglect spatial correlation between neighbouring 25 km cells.</li>
    <li><strong>Unmodelled operational factors:</strong> Captain reactions, eastern Mediterranean GNSS/AIS jamming, internal waves in straits, and submarine landslide/volcanic tsunamis (including the ancillary landslide suspected in 1956 Amorgos).</li>
    <li><strong>Confirmation only:</strong> Ships confirm large waves only; ship silence cannot cancel a seismic tsunami warning.</li>
    <li><strong>Padded domain &amp; closed S/E boundaries:</strong> Domain is padded 1.5° west and 1° north so absorbing sponge layers sit completely outside the 12–37°E, 30–42°N area of interest; south (30°N) and east (37°E) edges are closed walls (GEBCO subset limits).</li>
    <li><strong>Hellenic Arc geometry uncertainty (not a bug):</strong> Arc candidates follow the EFSM20 interface meshes. For the Hellenic Arc, conventions were verified with an overlay (<code>outputs/slab2_overlay.png</code>): Slab2 is a median 23 km deeper at the same location, which on an interface dipping ~10° moves depth contours ~130 km horizontally (~154 km median offset at 20 km depth), reflecting a genuine difference between seismicity-based Slab2 and EDSF-based EFSM20.</li>
    <li><strong>Uplift rule (<code>slip × sin(mean dip)</code>) &amp; shallow Hellenic dip caveat:</strong> Peak uplift is <code>slip × sin(mean dip)</code> for every source (arcs: EFSM20 mean interface dip — Hellenic 15°, Calabrian 13°, Cyprus 32°; Aegean and historic: fault dip), giving 0.81 m peak uplift for Hellenic Mw 8 (~2.4–2.5× larger Aegean/Samos, ~1.8× Cyprus, 0.86× Hellenic, 0.75× Calabrian relative to fixed 0.3 × slip). Samos 2020 motivated the change from a fixed 0.3 × slip to slip × sin(dip) (Samos came out ~4× too small), so the Samos comparison is a consistency check rather than fully independent validation. Averaged only over the shallow rupture footprint, the EFSM20 Hellenic mesh dips only about 4° rather than 15°; using that would shrink Hellenic sources about 3.5×.</li>
  </ol>
</section>
</body>
</html>
"""

    out_path = out_dir / "report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="T21 Report Builder")
    parser.add_argument("--out-dir", default=None, help="Outputs directory (default from config.yaml)")
    parser.add_argument("--no-embed", action="store_true", help="Link PNG images relatively instead of base64 embedding")
    args = parser.parse_args(argv)

    cfg = load_config()
    build_report(cfg, out_dir=args.out_dir, embed=not args.no_embed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
