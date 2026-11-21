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
        scoring_rows.append(
            {
                "Tier": r["name"],
                "E (events)": r.get("E", ""),
                "K / K2": f"{r.get('K', '')} / {r.get('K2', '')}",
                "s / 100 events": r.get("seconds_per_100_events", np.nan),
                "Note": "Two-point slope extrapolation from K2=8" if r.get("extrapolated_from_two_point_slope") else "Measured full catalogue",
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
    if sc_sum.get("algo_speedup_cpu"):
        bullets.append(
            f"<strong>Scoring algorithmic speedup (CPU loops → matmul + prefix scan):</strong> "
            f"{sc_sum['algo_speedup_cpu']:,.0f}× (direct slice measured: {sc_sum.get('direct_slice_algo_speedup', 0):.0f}×)."
        )
    if sc_sum.get("hw_speedup_gpu_over_tensor_cpu"):
        bullets.append(
            f"<strong>Scoring hardware speedup (CPU tensor → GPU tensor):</strong> "
            f"{sc_sum['hw_speedup_gpu_over_tensor_cpu']:.1f}× (total {sc_sum.get('total_speedup_gpu_over_loop', 0):,.0f}×)."
        )
    if so_sum.get("hw_speedup_gpu_over_batched_cpu"):
        bullets.append(
            f"<strong>Solver hardware speedup (batched CPU → batched GPU):</strong> "
            f"{so_sum['hw_speedup_gpu_over_batched_cpu']:.1f}× "
            f"(achieved {so_sum.get('gpu_bandwidth_gb_s', 0):.1f} GB/s, {so_sum.get('gpu_bandwidth_pct_peak', 0):.1f}% of peak)."
        )
    else:
        bullets.append(
            "<strong>GPU tier status:</strong> awaiting Colab T4/L4 run of <code>scripts/06_benchmark.py --gpu</code> "
            "(preliminary T4 smoke runs achieved 0.31–0.46 s/scenario vs 2.55–2.80 s/scenario on MacBook CPU)."
        )
    if hysea:
        bullets.append(
            f"<strong>Published context (Tsunami-HySEA):</strong> {html.escape(hysea.get('notes', ''))} "
            f"<em>({html.escape(hysea.get('citation', ''))})</em>."
        )
    parts.append('<div class="callout"><ul>' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul></div>")
    return "\n".join(parts)


def build_report(cfg, out_dir: Path = None, embed: bool = True) -> Path:
    out_dir = repo_path(cfg["paths"]["outputs"]) if out_dir is None else Path(out_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Map 1 table
    map1_tbl_path = out_dir / "map1_table.csv"
    map1_tbl_html = (
        _df_to_html(pd.read_csv(map1_tbl_path), float_fmt="{:.3f}")
        if map1_tbl_path.exists()
        else '<div class="pending"><strong>Pending:</strong> <code>outputs/map1_table.csv</code> not found.</div>'
    )

    # Thresholds / detector comparison table
    thr_path = out_dir / "thresholds.csv"
    thr_html = (
        _df_to_html(pd.read_csv(thr_path), float_fmt="{:.4g}")
        if thr_path.exists()
        else '<div class="pending"><strong>Pending:</strong> <code>outputs/thresholds.csv</code> (T17 sweep / quiet-day calibration).</div>'
    )

    # Map 2 / Map 3 summary tables
    map2_tbl_path = out_dir / "map2_table.csv"
    if map2_tbl_path.exists():
        m2df = pd.read_csv(map2_tbl_path)
        m2_agg = (
            m2df.groupby(["operator", "range_cap_m", "regime"], as_index=False)
            .agg(
                positions=("pos_id", "count"),
                detected_90=("mw_90", lambda s: int(s.notna().sum())),
                median_mw_90=("mw_90", "median"),
                detected_50=("mw_50", lambda s: int(s.notna().sum())),
                median_mw_50=("mw_50", "median"),
            )
        )
        map2_summary_html = _df_to_html(m2_agg, float_fmt="{:.2f}")
    else:
        map2_summary_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/map2_table.csv</code> (T18 after T17).</div>'

    map3_tbl_path = out_dir / "map3_table.csv"
    if map3_tbl_path.exists():
        m3df = pd.read_csv(map3_tbl_path)
        m3_agg = (
            m3df.groupby(["range_cap_m", "regime"], as_index=False)
            .agg(
                coasts=("gauge_code", "count"),
                med_gain_nearest_min=("gain_med_min_nearest_gauge_s", "median"),
                med_gain_first_gauge_min=("gain_med_min_first_gauge_s", "median"),
                med_gain_first_inst_min=("gain_med_min_first_instrument_s", "median"),
            )
        )
        map3_summary_html = _df_to_html(m3_agg, float_fmt="{:+.1f}")
    else:
        map3_summary_html = '<div class="pending"><strong>Pending:</strong> <code>outputs/map3_table.csv</code> (T18 after T17).</div>'

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
  <p>Inazu et al. (2020) proved the sideways-drift mechanism on 13 ships during the 2011 Tohoku tsunami (single-ship 3σ floor 0.1–0.3 m/s), while Hossen et al. (2021) found ship velocity alone struggled to recover Cascadia tsunamis in deep water, and Foster &amp; Thomas (2024) assessed ship GNSS height in Europe. Because large tsunamis are too rare to wait for, we simulate thousands of Mediterranean scenarios to quantify detection floors, fleet-wide false-alarm rates, warning-time gains, and the value of upgrading ships to precise GNSS.</p>
  {_figure_html(out_dir / "cleaning_steps.png", "Standalone illustration (T24) of AIS drift cleaning: raw 10-s drift with a synthetic turn artefact (rate_of_turn × 150 m), after 1-min averaging and yaw subtraction, and after causal 45-min baseline filtering. The main simulation does not model turning and assumes post-correction noise levels.", out_dir, embed)}
</section>

<section id="c1">
  <h2>2. Correctness, Validation &amp; Calibration (Criterion 1)</h2>
  <p>Physics correctness is verified across seven checks on the spherical Arakawa C-grid linear shallow-water solver (<code>src/tsunami/solver.py</code>, <code>solver_reference.py</code>), including GPU-vs-CPU equivalence on a Colab Tesla T4. All constants live in <code>config.yaml</code> with guesses explicitly tagged <code>PLACEHOLDER</code>. Both magnitude scaling laws were verified directly against their primary literature tables: Strasser et al. (2010, <em>SRL</em> 81(6):941–950, Table 2 interface: <code>L: -2.477 + 0.585 Mw</code>, <code>W: -0.882 + 0.351 Mw</code>) and Wells &amp; Coppersmith (1994, <em>BSSA</em> 84(4):974–1002, Table 2A all slip types: <code>RLD: -2.44 + 0.59 Mw</code>, <code>RW: -1.01 + 0.32 Mw</code>). Published Hellenic Arc Mw 8–8.5 scenarios report seafloor uplift from ~1–2 m on low-dip megathrust interfaces up to ~9–10 m on steeper splay faults (Lorito et al. 2008; Shaw et al. 2008; Necmioğlu &amp; Özel 2014, 2015; Wirp et al. 2025/2026), bracketing our Gaussian dipole Mw 8 uplift of 0.93 m (PLACEHOLDER <code>uplift_fraction_of_slip = 0.3</code>, tested in sensitivity runs).</p>
  <h3>Solver Validation Summary (T11)</h3>
  {_read_text_pre(out_dir / "validation.md", "<code>outputs/validation.md</code> not found.")}
  {val_figs_html}
  <h3>Piraeus Real AIS Calibration (T05)</h3>
  {_read_text_pre(out_dir / "piraeus_calibration.md", "<code>outputs/piraeus_calibration.md</code> not found.")}
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
    {_figure_html(out_dir / "geometry_check.png", "Arc geometry cross-check: EFSM20 vs Slab2 Hellenic (~154 km median offset at 20 km depth, flagged as a limitation) and EFSM20 vs Maesano (2017) Calabrian (0.6–0.7 km, same INGV group).", out_dir, embed)}
  </div>
</section>

<section id="c3">
  <h2>4. Meaningful Speed-Up &amp; What Speed Buys (Criterion 3)</h2>
  <p>Workloads A (batched solver) and B (matched-filter scoring) are benchmarked separately across three tiers (naive reference → reshaped CPU → reshaped GPU) on the same NumPy/CuPy codebase.</p>
  {_benchmark_tables_html(out_dir)}
  {_figure_html(out_dir / "benchmark.png", "Three-tier performance benchmark (T19) separating algorithmic reshaping from GPU hardware speedup for Solver (A) and Scoring (B).", out_dir, embed)}
  {_figure_html(out_dir / "what_speed_buys.png", "What a fixed 1-hour compute budget buys: affordable scenario count (left) and resulting Map 2 binomial 95% confidence interval half-width at p = 0.90 (right).", out_dir, embed)}
</section>

<section id="c4">
  <h2>5. Scientific Value: Maps, Detectors, Historic Replays &amp; Sensitivity (Criterion 4)</h2>

  <h3>5.1 Map 1: Where Mediterranean Ships Can Sense a Tsunami</h3>
  {map1_tbl_html}
  {_figure_html(out_dir / "map1.png", "Map 1 (T10): Share of EMODnet 2024 vessel traffic across depth bands (<50 m masked, 50–200 m shelf, >200 m deep water) overall and within 40 km / 100 km AIS range caps.", out_dir, embed)}

  <h3>5.2 Single-Scenario Signal-to-Noise Check (G2)</h3>
  {_figure_html(out_dir / "first_detection.png", "First detection map (T13): Per-ship SNR for one Hellenic Arc Mw 8 scenario alongside IOC tide gauge and 2025 Ionian buoy trigger times.", out_dir, embed, "<code>outputs/first_detection.png</code> (T13) pending from Stream B.")}

  <h3>5.3 Map 2: Detection Floor by Source Location, Operator &amp; Weather Regime</h3>
  {map2_summary_html}
  {_figure_html(out_dir / "map2.png", "Map 2 (T18): Smallest magnitude confirmed within 2 h in ≥90% of traffic × noise draws (with 50% rings), comparing AIS drift and ship GPS height across calm, moderate, and rough regimes.", out_dir, embed, "<code>outputs/map2.png</code> pending T17 sweep parquet -> T18.")}
  {_figure_html(out_dir / "map2_uncertainty.png", "Detection probability vs. magnitude with 95% binomial uncertainty bands across draws, comparing the matched-filter network against the calibrated single-ship baseline.", out_dir, embed, "<code>outputs/map2_uncertainty.png</code> pending T17 sweep parquet -> T18.")}

  <h3>5.4 Map 3: Warning Time Added Over Tide Gauges &amp; 2025 Ionian Buoys</h3>
  {map3_summary_html}
  {_figure_html(out_dir / "map3.png", "Map 3 (T18): Median warning time added by the AIS ship network by coast at 40 km and 100 km range caps against (1) coast's nearest gauge, (2) first gauge anywhere, and (3) first instrument including the two Ionian buoys.", out_dir, embed, "<code>outputs/map3.png</code> pending T17 sweep parquet -> T18.")}

  <h3>5.5 Detector Comparison &amp; Held-Out False-Alarm Rates</h3>
  <p>Calibrated thresholds and held-out quiet-day false-alarm rates per stratum (comparing the matched-filter network against Inazu's literal 3σ single-ship rule — which triggers on nearly 100% of quiet 2-hour windows across ~1,000 ships — and the 1%-FAR calibrated single-ship rule):</p>
  {thr_html}

  <h3>5.6 Historic Event Replays (Samos 2020 &amp; Amorgos 1956)</h3>
  <p>Both Samos (30 Oct 2020, Mw 7.0, USGS <code>us7000c7y0</code>; Kiratzi et al. 2022) and Amorgos (9 Jul 1956, Mw 7.7, Okal et al. 2009) are normal-fault (subsidence-first, <code>polarity = -1</code>) Aegean earthquakes. Replays are scored against both the arc-only catalogue (showing the blind spot of an arc-restricted library) and the arc+Aegean catalogue, with Samos simulated amplitudes compared to 14 IOC Aegean gauges (e.g. <code>kos1</code> +11.9 cm crest) using Green's-law shoaling.</p>
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
    <li><strong>Linear shallow-water physics:</strong> No Coriolis, dispersion, or nonlinear inundation; ship observations shallower than 50 m are masked out (~55% of EMODnet traffic).</li>
    <li><strong>Gaussian dipole sources:</strong> Sized by verified Strasser (2010) / Wells &amp; Coppersmith (1994) scaling rather than full Okada (1985) dislocations (uplift mismatch and monopole vs dipole tested in T20).</li>
    <li><strong>Synthetic traffic &amp; anchorage bias:</strong> Traffic draws sample EMODnet 2024 vessel density (which lost satellite AIS after June 2024 and includes anchored/moored ships; Piraeus shows 78% of port reports &lt; 2 kn).</li>
    <li><strong>Catalogue coverage:</strong> Library covers the three subduction arcs and a coarse Aegean grid; earthquakes outside those zones are detected only via mismatched templates.</li>
    <li><strong>Diagonal inter-cell covariance:</strong> Cell weights capture within-cell shared-current variance but neglect spatial correlation between neighbouring 25 km cells.</li>
    <li><strong>Unmodelled operational factors:</strong> Captain reactions, eastern Mediterranean GNSS/AIS jamming, internal waves in straits, and submarine landslide/volcanic tsunamis (including the ancillary landslide suspected in 1956 Amorgos).</li>
    <li><strong>Confirmation only:</strong> Ships confirm large waves only; ship silence cannot cancel a seismic tsunami warning.</li>
    <li><strong>Padded domain &amp; closed S/E boundaries:</strong> Domain is padded 1.5° west and 1° north so absorbing sponge layers sit completely outside the 12–37°E, 30–42°N area of interest; south (30°N) and east (37°E) edges are closed walls (GEBCO subset limits).</li>
    <li><strong>Hellenic Arc geometry uncertainty:</strong> EFSM20 (<code>GRSS001</code>) and Slab2 differ by ~154 km median at 20 km depth along the Hellenic Arc, shifting candidate positions accordingly.</li>
    <li><strong>Static initial-field gauge trigger caveat (from T13):</strong> Coastal gauges sitting directly inside a source's instantaneous t=0 subsidence lobe (e.g. Paleochora for a nearby Hellenic Mw 8 event) can trigger immediately at origin time from static coseismic sea-level drop rather than wave propagation; this inflates raw gauge speed for near-field coastal stations.</li>
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
