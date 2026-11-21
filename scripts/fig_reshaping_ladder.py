"""What Reshaping Saves: full-experiment wall-clock ladder (T19 follow-up).

Computes total wall-clock time across the entire eastern Mediterranean OSSE from
measured per-unit timings and actual experiment sizes read dynamically from:
  - data/processed/candidates.csv  (K, K2)
  - data/truth/sources.csv         (Map 2 + twin truth events)
  - data/historic_runs/sources.csv (historic replays)
  - config.yaml                    (strata, traffic draws, n_twin)
  - outputs/benchmark_cpu.json     (MacBook arm64 CPU solver & 3 scoring tiers)
  - outputs/benchmark_gpu.json     (Colab Tesla T4 real-grid solver & scoring)
  - data/{library,truth,historic_runs}/run_log.csv & outputs/sweep/*log* (when present)

Also measures the per-traffic-draw template gather + projection + cell-aggregation
cost (`scoring.aggregate_templates`) once per `(traffic draw, range cap, operator)`
on the MacBook and includes it in Rows 3-6 so those rows are not understated.

Outputs:
  - outputs/reshaping_ladder.json
  - outputs/reshaping_ladder.png

Usage:
  PYTHONPATH=src python scripts/fig_reshaping_ladder.py
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import bench, scoring, ships, viz
from tsunami.config import load_config, repo_path

SEC_PER_MIN = 60.0
SEC_PER_HOUR = 3600.0
SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 365.25 * SEC_PER_DAY


def format_duration(seconds: float) -> str:
    """Format a duration in human-readable units (s, min, h, days, years)."""
    s = float(seconds)
    if s < 90.0:
        return f"{s:.1f} s"
    if s < 5400.0:
        return f"{s / SEC_PER_MIN:.1f} min"
    if s < 48.0 * SEC_PER_HOUR:
        return f"{s / SEC_PER_HOUR:.1f} h"
    if s < 365.25 * SEC_PER_DAY:
        return f"{s / SEC_PER_DAY:.0f} days"
    yrs = s / SEC_PER_YEAR
    return f"{yrs:.1f} years" if yrs < 10.0 else f"{yrs:.0f} years"


def load_experiment_counts(cfg) -> dict:
    """Read actual experiment dimensions from data files and config (never hard-coded)."""
    cand_path = repo_path(cfg["paths"]["candidates"])
    truth_path = repo_path(cfg["paths"]["truth_dir"]) / "sources.csv"
    hist_path = repo_path(cfg["paths"]["historic_runs_dir"]) / "sources.csv"

    cand = pd.read_csv(cand_path)
    truth = pd.read_csv(truth_path)
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()

    K = int(len(cand))
    K2 = int(K + cand["flip_allowed"].astype(bool).sum())
    n_truth_map2 = int((truth["run_kind"] == "truth_map2").sum()) if "run_kind" in truth.columns else int(len(truth))
    n_truth_twin_csv = int((truth["run_kind"] == "truth_twin").sum()) if "run_kind" in truth.columns else 0
    n_twin_cfg = int(cfg["truth"]["n_twin"])
    n_truth_twin = n_truth_twin_csv if n_truth_twin_csv > 0 else n_twin_cfg
    n_truth = int(len(truth))
    n_historic = int(len(hist)) if len(hist) > 0 else int(len(cfg["paths"].get("historic_source_files", [])))
    n_library_build = K + n_truth + n_historic

    n_operators = int(len(cfg["operators"]))
    n_range_caps = int(len(cfg["traffic"]["range_caps_m"]))
    n_regimes = int(len(cfg["weather_regimes"]))
    n_strata = n_operators * n_range_caps * n_regimes

    n_traffic = int(cfg["sweep"]["n_traffic_draws"])
    n_noise_per_traffic = int(cfg["sweep"]["n_noise_draws"])
    n_draws_per_stratum = n_traffic * n_noise_per_traffic
    n_trials = n_truth * n_strata * n_draws_per_stratum
    n_agg_calls = n_traffic * n_range_caps * n_operators

    # Quiet-day calibration & holdout (separate small scoring-only cost)
    n_quiet_splits = 2
    n_quiet_traffic_draws = int(cfg["scoring"]["n_quiet_traffic_draws"])
    n_quiet_cal = int(cfg["scoring"]["n_quiet_calibration"])
    n_quiet_hold = int(cfg["scoring"]["n_quiet_holdout"])
    n_quiet_noise_per_traffic = int(round(n_quiet_cal / max(n_quiet_traffic_draws, 1)))
    n_quiet_trials_per_stratum = n_quiet_cal + n_quiet_hold
    n_quiet_trials = n_quiet_trials_per_stratum * n_strata
    n_quiet_extra_fleets = n_quiet_splits * n_quiet_traffic_draws

    return {
        "K": K,
        "K2": K2,
        "n_truth_map2": n_truth_map2,
        "n_truth_twin": n_truth_twin,
        "n_truth_twin_config": n_twin_cfg,
        "n_truth": n_truth,
        "n_historic": n_historic,
        "n_library_build": n_library_build,
        "n_operators": n_operators,
        "n_range_caps": n_range_caps,
        "n_regimes": n_regimes,
        "n_strata": n_strata,
        "n_traffic_draws": n_traffic,
        "n_noise_per_traffic": n_noise_per_traffic,
        "n_draws_per_stratum": n_draws_per_stratum,
        "n_trials": n_trials,
        "n_agg_calls": n_agg_calls,
        "n_quiet_splits": n_quiet_splits,
        "n_quiet_traffic_draws": n_quiet_traffic_draws,
        "n_quiet_noise_per_traffic": n_quiet_noise_per_traffic,
        "n_quiet_trials_per_stratum": n_quiet_trials_per_stratum,
        "n_quiet_trials": n_quiet_trials,
        "n_quiet_extra_fleets": n_quiet_extra_fleets,
        "sources": {
            "candidates_csv": str(cfg["paths"]["candidates"]),
            "truth_sources_csv": f"{cfg['paths']['truth_dir']}/sources.csv",
            "historic_sources_csv": f"{cfg['paths']['historic_runs_dir']}/sources.csv",
            "config_yaml": "config.yaml",
        },
    }


def measure_aggregate_templates(cfg, K: int, chunk: int = 8) -> dict:
    """Measure one real-shaped `scoring.Library` + `scoring.aggregate_templates` call per (range cap, operator) on CPU.

    Creates a temporary real-format Library directory (`index.csv` + `rec_0000.npy` of shape
    `(chunk, T, R, 3)` float32 matching the real R=20,063 receiver pool), times
    `scoring.aggregate_templates`, and scales linearly by `K / chunk` to obtain the full
    K-candidate gather + projection + group aggregation cost.
    """
    pool = ships.load_pool(cfg)
    R = len(pool.lon)
    T = int(cfg["library"]["n_minutes"])
    draw_full = ships.traffic_draw(pool, cfg, index=0, purpose=1)
    range_caps = sorted(float(x) for x in cfg["traffic"]["range_caps_m"])
    operators = list(cfg["operators"])

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        arr = np.full((chunk, T, R, 3), 0.01, dtype=np.float32)
        np.save(tdp / "rec_0000.npy", arr)
        del arr
        pd.DataFrame({"run_id": np.arange(chunk), "chunk": 0, "row": np.arange(chunk)}).to_csv(
            tdp / "index.csv", index=False
        )

        lib = scoring.Library(tdp)
        ids = np.arange(chunk, dtype=np.int64)
        # Single warmup pass to fault the temp file into OS cache before timing each (cap, op)
        warm_groups = scoring.build_groups(draw_full, pool, cfg, operators[0])
        _ = scoring.aggregate_templates(lib, ids, draw_full, warm_groups, None, operators[0], chunk=chunk)

        breakdown = {}
        for rc in range_caps:
            mask = pool.coast_dist[draw_full.pool_idx] <= rc
            dr = ships.TrafficDraw(
                pool_idx=draw_full.pool_idx[mask],
                heading=draw_full.heading[mask],
                sog=draw_full.sog[mask],
                seed=draw_full.seed,
            )
            for op in operators:
                groups = scoring.build_groups(dr, pool, cfg, op)
                times = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    _ = scoring.aggregate_templates(lib, ids, dr, groups, None, op, chunk=chunk)
                    times.append(time.perf_counter() - t0)
                dt_chunk = float(np.median(times))
                dt_full = dt_chunk * (K / float(chunk))
                key = f"{int(round(rc / 1000.0))}km_{op}"
                breakdown[key] = {
                    "S": int(len(dr.pool_idx)),
                    "M": int(groups.n_groups),
                    "chunk_candidates": int(chunk),
                    "chunk_s": dt_chunk,
                    "full_K_s": dt_full,
                }

    n_traffic = int(cfg["sweep"]["n_traffic_draws"])
    sum_one_traffic = float(sum(v["full_K_s"] for v in breakdown.values()))
    total_all_calls = n_traffic * sum_one_traffic
    mean_per_call = sum_one_traffic / max(len(breakdown), 1)
    return {
        "method": f"Measured scoring.aggregate_templates on MacBook arm64 memmap chunk ({chunk}, {T}, {R}, 3) float32, scaled by K/{chunk}",
        "machine": "MacBook arm64",
        "K": int(K),
        "R": int(R),
        "T": int(T),
        "n_calls_total": int(n_traffic * len(breakdown)),
        "mean_s_per_call": mean_per_call,
        "one_traffic_draw_all_caps_ops_s": sum_one_traffic,
        "total_experiment_s": total_all_calls,
        "breakdown": breakdown,
    }


def _detect_actual_t14_t17(cfg, counts: dict, out_dir: Path) -> dict:
    """Read actual T14 run_log.csv and T17 sweep timing logs if present on disk."""
    stage_dirs = [
        ("library", repo_path(cfg["paths"]["library_dir"])),
        ("truth", repo_path(cfg["paths"]["truth_dir"])),
        ("historic", repo_path(cfg["paths"]["historic_runs_dir"])),
    ]
    t14_wall = 0.0
    t14_solve = 0.0
    t14_runs = 0
    t14_gpus = set()
    t14_files = []
    for _, d in stage_dirs:
        p = d / "run_log.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if len(df) == 0:
            continue
        # Only count real GPU runs (not local CPU smoke logs)
        gpu_rows = df[df["gpu"].astype(str).str.lower() != "cpu"] if "gpu" in df.columns else df
        if len(gpu_rows) == 0:
            continue
        t14_wall += float(gpu_rows["wall_s"].sum())
        t14_solve += float(gpu_rows["solve_s"].sum()) if "solve_s" in gpu_rows.columns else 0.0
        t14_runs += int(gpu_rows["n_runs"].sum())
        if "gpu" in gpu_rows.columns:
            t14_gpus.update(gpu_rows["gpu"].astype(str).unique().tolist())
        t14_files.append(str(p))

    t14_complete = t14_runs >= counts["n_library_build"] and t14_wall > 0.0

    # Check for T17 sweep timing log
    t17_wall = None
    t17_gpu = None
    t17_file = None
    sweep_dir = out_dir / "sweep"
    if sweep_dir.exists():
        for cand_name in ("sweep_log.json", "sweep_summary.json", "run_log.json"):
            sp = sweep_dir / cand_name
            if sp.exists():
                try:
                    sd = json.loads(sp.read_text(encoding="utf-8"))
                    w = sd.get("total_wall_s") or sd.get("wall_s") or sd.get("elapsed_s")
                    if w is not None and float(w) > 0:
                        t17_wall = float(w)
                        t17_gpu = sd.get("gpu") or sd.get("device_name")
                        t17_file = str(sp)
                        break
                except Exception:
                    pass

    return {
        "t14_complete": t14_complete,
        "t14_runs_logged": t14_runs,
        "t14_wall_s": t14_wall if t14_complete else None,
        "t14_solve_s": t14_solve if t14_complete else None,
        "t14_gpus": sorted(t14_gpus),
        "t14_files": t14_files,
        "t17_complete": t17_wall is not None,
        "t17_wall_s": t17_wall,
        "t17_gpu": t17_gpu,
        "t17_file": t17_file,
    }


def build_reshaping_ladder(cfg, out_dir: Path = None, agg_measurement: dict = None) -> dict:
    """Compute all ladder rows, save outputs/reshaping_ladder.json and outputs/reshaping_ladder.png."""
    out_dir = repo_path(cfg["paths"]["outputs"]) if out_dir is None else Path(out_dir)
    cpu_path = out_dir / "benchmark_cpu.json"
    gpu_path = out_dir / "benchmark_gpu.json"
    if not cpu_path.exists():
        raise FileNotFoundError(f"Missing {cpu_path}; run scripts/06_benchmark.py --cpu first.")

    cpu = bench.read_json(cpu_path)
    gpu = bench.read_json(gpu_path) if gpu_path.exists() else None
    counts = load_experiment_counts(cfg)

    def _find(records, name=None, tier=None, E=None):
        sub = records
        if name is not None:
            sub = [r for r in sub if r.get("name") == name]
        if tier is not None:
            sub = [r for r in sub if r.get("tier") == tier]
        if E is not None:
            sub = [r for r in sub if r.get("E") == E]
        return sub[0] if sub else None

    r_sim_cpu = _find(cpu["records"], tier="serial_cpu")
    r_loop = _find(cpu["records"], tier="loop_cpu")
    r_vec = _find(cpu["records"], tier="vec_no_prefix_cpu", E=100) or _find(cpu["records"], tier="vec_no_prefix_cpu")
    r_tcpu = _find(cpu["records"], tier="tensor_cpu", E=100) or _find(cpu["records"], tier="tensor_cpu")

    if gpu is not None:
        gpu_solvers = [r for r in gpu["records"] if r.get("workload") == "solver" and r.get("tier") == "batched_gpu"]
        r_sim_gpu = min(gpu_solvers, key=lambda r: r["seconds_per_scenario"]) if gpu_solvers else None
        r_tgpu = _find(gpu["records"], tier="tensor_gpu", E=100) or _find(gpu["records"], tier="tensor_gpu")
        gpu_name = gpu["device"].get("device_name", "Tesla T4")
    else:
        r_sim_gpu, r_tgpu, gpu_name = None, None, "Tesla T4"

    t_sim_cpu = float(r_sim_cpu["seconds_per_scenario"])
    t_loop = float(r_loop["seconds_per_100_events"]) / 100.0
    t_vec = float(r_vec["seconds_per_100_events"]) / 100.0
    t_tcpu = float(r_tcpu["seconds_per_100_events"]) / 100.0
    t_sim_gpu = float(r_sim_gpu["seconds_per_scenario"]) if r_sim_gpu else None
    b_sim_gpu = int(r_sim_gpu["batch_size"]) if r_sim_gpu else None
    t_tgpu = float(r_tgpu["seconds_per_100_events"]) / 100.0 if r_tgpu else None

    if agg_measurement is None:
        agg_measurement = measure_aggregate_templates(cfg, counts["K"], chunk=8)
    t_agg_total = float(agg_measurement["total_experiment_s"])
    t_agg_per_trial = t_agg_total / counts["n_trials"]

    actual = _detect_actual_t14_t17(cfg, counts, out_dir)

    K = counts["K"]
    n_truth = counts["n_truth"]
    n_traffic = counts["n_traffic_draws"]
    n_lib = counts["n_library_build"]
    n_trials = counts["n_trials"]
    n_agg = counts["n_agg_calls"]

    per_unit_timings = {
        "t_sim_cpu_serial_s_per_scenario": {
            "value": t_sim_cpu,
            "machine": "MacBook arm64",
            "record": r_sim_cpu["name"],
            "source_file": "outputs/benchmark_cpu.json",
        },
        "t_loop_s_per_trial": {
            "value": t_loop,
            "machine": "MacBook arm64",
            "record": r_loop["name"],
            "source_file": "outputs/benchmark_cpu.json",
        },
        "t_vec_no_prefix_s_per_trial": {
            "value": t_vec,
            "machine": "MacBook arm64",
            "record": r_vec["name"],
            "source_file": "outputs/benchmark_cpu.json",
        },
        "t_tensor_cpu_s_per_trial": {
            "value": t_tcpu,
            "machine": "MacBook arm64",
            "record": r_tcpu["name"],
            "source_file": "outputs/benchmark_cpu.json",
        },
        "t_agg_templates_mean_s_per_call": {
            "value": float(agg_measurement["mean_s_per_call"]),
            "total_all_calls_s": t_agg_total,
            "amortised_s_per_trial": t_agg_per_trial,
            "n_calls": n_agg,
            "machine": agg_measurement["machine"],
            "source_file": "scripts/fig_reshaping_ladder.py (live measurement of scoring.aggregate_templates)",
        },
    }
    if t_sim_gpu is not None:
        per_unit_timings["t_sim_gpu_s_per_scenario"] = {
            "value": t_sim_gpu,
            "batch_size": b_sim_gpu,
            "machine": f"Colab {gpu_name}",
            "record": r_sim_gpu["name"],
            "source_file": "outputs/benchmark_gpu.json",
        }
    if t_tgpu is not None:
        per_unit_timings["t_tensor_gpu_s_per_trial"] = {
            "value": t_tgpu,
            "machine": f"Colab {gpu_name}",
            "record": r_tgpu["name"],
            "source_file": "outputs/benchmark_gpu.json",
        }

    # --- Build the 6 ladder rows ---
    lib_cpu_s = n_lib * t_sim_cpu
    score_loop_s = n_trials * t_loop
    score_vec_s = n_trials * t_vec
    score_tcpu_s = n_trials * t_tcpu

    row1_s = n_trials * K * t_sim_cpu
    row2_sim_s = n_truth * n_traffic * K * t_sim_cpu
    row2_s = row2_sim_s + score_loop_s
    row3_s = lib_cpu_s + t_agg_total + score_loop_s
    row4_s = lib_cpu_s + t_agg_total + score_vec_s
    row5_s = lib_cpu_s + t_agg_total + score_tcpu_s

    rows = [
        {
            "row": 1,
            "label": "1. Fully naive\n(re-sim every cand per trial, CPU)",
            "short_label": "Fully naive: re-sim every candidate per trial (MacBook CPU)",
            "machine": "MacBook arm64 CPU",
            "formula": f"trials ({n_trials:,}) * K ({K:,}) * t_sim_cpu ({t_sim_cpu:.3f} s)",
            "sim_s": row1_s,
            "agg_s": 0.0,
            "scoring_s": 0.0,
            "total_s": row1_s,
            "human_total": format_duration(row1_s),
            "status": "extrapolated",
            "rough_estimate_s": 130.0 * SEC_PER_YEAR,
            "rough_estimate_label": "~130 years",
        },
        {
            "row": 2,
            "label": "2. Re-sim per (event, traffic)\n+ Python-loop scoring (CPU)",
            "short_label": "Re-sim per (truth event, traffic draw) + loop scoring (MacBook CPU)",
            "machine": "MacBook arm64 CPU",
            "formula": (
                f"n_truth ({n_truth:,}) * n_traffic ({n_traffic}) * K ({K:,}) * t_sim_cpu ({t_sim_cpu:.3f} s) "
                f"+ trials ({n_trials:,}) * t_loop ({t_loop:.2f} s)"
            ),
            "sim_s": row2_sim_s,
            "agg_s": 0.0,
            "scoring_s": score_loop_s,
            "total_s": row2_s,
            "human_total": f"{format_duration(row2_s)} ({format_duration(row2_sim_s)} sim + {format_duration(score_loop_s)} score)",
            "status": "extrapolated",
            "rough_estimate_s": 200.0 * SEC_PER_DAY + 2.3 * SEC_PER_YEAR,
            "rough_estimate_label": "~200 days sim + ~2.3 yr scoring (~2.85 yr)",
        },
        {
            "row": 3,
            "label": "3. Reusable library\n+ Python-loop scoring (CPU)",
            "short_label": "Template library + template gather + Python-loop scoring (MacBook CPU)",
            "machine": "MacBook arm64 CPU",
            "formula": (
                f"library ({n_lib:,} * {t_sim_cpu:.3f} s) + gather/agg ({n_agg} calls = {t_agg_total:.1f} s) "
                f"+ trials ({n_trials:,}) * t_loop ({t_loop:.2f} s)"
            ),
            "sim_s": lib_cpu_s,
            "agg_s": t_agg_total,
            "scoring_s": score_loop_s,
            "total_s": row3_s,
            "human_total": format_duration(row3_s),
            "status": "extrapolated",
            "rough_estimate_s": 2.3 * SEC_PER_YEAR,
            "rough_estimate_label": "~2.3 years",
        },
        {
            "row": 4,
            "label": "4. Library + vectorised NumPy\nno prefix scan (CPU)",
            "short_label": "Template library + gather + vectorised NumPy no-prefix (MacBook CPU)",
            "machine": "MacBook arm64 CPU",
            "formula": (
                f"library ({n_lib:,} * {t_sim_cpu:.3f} s) + gather/agg ({n_agg} calls = {t_agg_total:.1f} s) "
                f"+ trials ({n_trials:,}) * t_vec ({t_vec:.4f} s)"
            ),
            "sim_s": lib_cpu_s,
            "agg_s": t_agg_total,
            "scoring_s": score_vec_s,
            "total_s": row4_s,
            "human_total": format_duration(row4_s),
            "status": "extrapolated",
            "rough_estimate_s": 44.0 * SEC_PER_HOUR,
            "rough_estimate_label": "~44 h",
        },
        {
            "row": 5,
            "label": "5. Fully reshaped on CPU\n(library + matmul + prefix scan)",
            "short_label": "Fully reshaped: library + gather + matmul + prefix scan (MacBook CPU)",
            "machine": "MacBook arm64 CPU",
            "formula": (
                f"library ({n_lib:,} * {t_sim_cpu:.3f} s) + gather/agg ({n_agg} calls = {t_agg_total:.1f} s) "
                f"+ trials ({n_trials:,}) * t_tensor_cpu ({t_tcpu:.5f} s)"
            ),
            "sim_s": lib_cpu_s,
            "agg_s": t_agg_total,
            "scoring_s": score_tcpu_s,
            "total_s": row5_s,
            "human_total": format_duration(row5_s),
            "status": "extrapolated",
            "rough_estimate_s": 5.0 * SEC_PER_HOUR,
            "rough_estimate_label": "~5 h",
        },
    ]

    # Row 6: Fully reshaped on Colab Tesla T4 GPU
    if actual["t14_complete"] and actual["t17_complete"]:
        row6_sim_s = actual["t14_wall_s"]
        row6_score_s = actual["t17_wall_s"]
        row6_s = row6_sim_s + row6_score_s
        row6_status = "measured"
        row6_formula = (
            f"Measured T14 run_log wall ({row6_sim_s:.1f} s) + measured T17 sweep wall ({row6_score_s:.1f} s; incl. gather/agg)"
        )
        row6_agg_s = 0.0
    elif t_sim_gpu is not None and t_tgpu is not None:
        if actual["t14_complete"]:
            row6_sim_s = actual["t14_wall_s"]
            sim_part = f"measured T14 run_log ({row6_sim_s:.1f} s)"
            row6_status = "extrapolated (T14 measured, T17 extrapolated)"
        else:
            row6_sim_s = n_lib * t_sim_gpu
            sim_part = f"library ({n_lib:,} * {t_sim_gpu:.3f} s at B={b_sim_gpu})"
            row6_status = "extrapolated"
        row6_score_s = n_trials * t_tgpu
        row6_agg_s = t_agg_total
        row6_s = row6_sim_s + row6_agg_s + row6_score_s
        row6_formula = (
            f"{sim_part} + gather/agg ({n_agg} calls = {t_agg_total:.1f} s; MacBook-measured, Colab CPU in practice) "
            f"+ trials ({n_trials:,}) * t_tensor_gpu ({t_tgpu:.5f} s)"
        )
    else:
        row6_s = None

    if row6_s is not None:
        rows.append(
            {
                "row": 6,
                "label": f"6. Fully reshaped on {gpu_name}\n(library + matmul + prefix scan)",
                "short_label": f"Fully reshaped: library + gather + matmul + prefix scan (Colab {gpu_name})",
                "machine": f"Colab {gpu_name}",
                "formula": row6_formula,
                "sim_s": row6_sim_s,
                "agg_s": row6_agg_s,
                "scoring_s": row6_score_s,
                "total_s": row6_s,
                "human_total": format_duration(row6_s),
                "status": row6_status,
                "rough_estimate_s": 45.0 * SEC_PER_MIN,
                "rough_estimate_label": "~45 min",
            }
        )

    # Compute speedups & check against rough estimates (flag any >30% difference in notes)
    notes = []
    for idx, r in enumerate(rows):
        r["speedup_vs_naive"] = float(row1_s / r["total_s"])
        r["step_speedup_vs_prev"] = float(rows[idx - 1]["total_s"] / r["total_s"]) if idx > 0 else 1.0
        est = r["rough_estimate_s"]
        rel_diff = (r["total_s"] - est) / est
        r["rel_diff_vs_rough_estimate"] = float(rel_diff)
        if abs(rel_diff) > 0.30:
            extra = ""
            if r["row"] == 6 and r["agg_s"] > 0:
                extra = (
                    " Including the 20 CPU scoring.aggregate_templates calls (MacBook-measured, Colab CPU in practice; "
                    "reading 31.3 GB x 20 = 626 GB through Library.__getitem__ before slicing active ships) adds "
                    f"{format_duration(r['agg_s'])} on top of {format_duration(r['sim_s'] + r['scoring_s'])} of GPU solve+score."
                )
            notes.append(
                f"Row {r['row']} ({r['human_total']}) differs by {rel_diff:+.1%} from rough estimate {r['rough_estimate_label']}: "
                f"sim={format_duration(r['sim_s'])}, agg={format_duration(r['agg_s'])}, score={format_duration(r['scoring_s'])}.{extra}"
            )

    if not notes:
        notes.append(
            "All 6 rows agree within 30% of the pre-check rough estimates (~130 yr, ~200 d + scoring, ~2.3 yr, ~44 h, ~5 h, ~45 min)."
        )
    notes.append(
        f"Template gather + projection + aggregation (scoring.aggregate_templates) across all {n_agg} "
        f"(traffic draw, range cap, operator) combinations takes {format_duration(t_agg_total)} "
        f"({agg_measurement['mean_s_per_call']:.2f} s/call mean on MacBook arm64, amortised to {t_agg_per_trial*1e6:.1f} us/trial) "
        "and is included in Rows 3-6."
    )
    if not actual["t14_complete"]:
        notes.append(
            "Row 6 currently extrapolates T14 from measured real-grid T4 B=32 (0.418 s/scen in outputs/benchmark_gpu.json) "
            "because local data/*/run_log.csv contains no Colab GPU rows yet; re-running this script after T14+T17 logs land "
            "automatically upgrades Row 6 to direct measured wall-clock."
        )

    # Separate small quiet-day calibration cost (scoring only, no solver runs)
    q_trials = int(counts["n_quiet_trials"])
    q_fleets = int(counts["n_quiet_extra_fleets"])
    q_score_cpu_s = q_trials * t_tcpu
    q_score_gpu_s = q_trials * t_tgpu if t_tgpu is not None else None
    quiet_cal_cost = {
        "n_splits": int(counts["n_quiet_splits"]),
        "n_quiet_traffic_draws_per_split": int(counts["n_quiet_traffic_draws"]),
        "n_quiet_noise_per_traffic": int(counts["n_quiet_noise_per_traffic"]),
        "n_quiet_trials_per_stratum": int(counts["n_quiet_trials_per_stratum"]),
        "n_quiet_trials_total": q_trials,
        "n_quiet_extra_fleets": q_fleets,
        "scoring_only_cpu_s": float(q_score_cpu_s),
        "scoring_only_gpu_s": float(q_score_gpu_s) if q_score_gpu_s is not None else None,
        "note": (
            f"Quiet-day calibration + holdout ({counts['n_quiet_splits']} splits x "
            f"{counts['n_quiet_traffic_draws']} traffic draws x {counts['n_quiet_noise_per_traffic']} noise draws "
            f"x {counts['n_strata']} strata = {q_trials:,} scoring-only trials, no solver runs): "
            f"{format_duration(q_score_cpu_s)} on MacBook CPU"
            + (f" / {format_duration(q_score_gpu_s)} on Colab {gpu_name}" if q_score_gpu_s is not None else "")
            + f", plus {q_fleets} extra quiet traffic fleets (folded into a single library read in sweep.gather_pass)."
        ),
    }
    notes.append(quiet_cal_cost["note"])

    payload = {
        "experiment_counts": counts,
        "per_unit_timings": per_unit_timings,
        "aggregate_templates_measurement": agg_measurement,
        "quiet_day_calibration": quiet_cal_cost,
        "actual_t14_t17_logs": actual,
        "rows": rows,
        "notes": notes,
    }
    bench.write_json(payload, out_dir / "reshaping_ladder.json")

    # ---- Figure: outputs/reshaping_ladder.png ----
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(13.6, 6.4))

    y_pos = np.arange(len(rows))[::-1]
    totals = [r["total_s"] for r in rows]
    palette = [
        viz.COLORS["negative"],   # 1. fully naive
        "#CC79A7",                # 2. re-sim per (event, traffic)
        viz.COLORS["neutral"],    # 3. library + loops
        viz.COLORS["moderate"],   # 4. library + vec no-prefix
        viz.COLORS["positive"],   # 5. fully reshaped CPU
        viz.COLORS["ships"],      # 6. fully reshaped T4 GPU
    ][: len(rows)]

    bars = ax.barh(y_pos, totals, height=0.62, color=palette, edgecolor="#333333", linewidth=0.7)
    ax.set_xscale("log")
    xmin = min(totals) * 0.25
    xmax = max(totals) * 180.0
    ax.set_xlim(xmin, xmax)

    # Human-unit reference ticks on log x-axis
    ref_ticks = [
        (60.0, "1 min"),
        (3600.0, "1 h"),
        (SEC_PER_DAY, "1 day"),
        (30.0 * SEC_PER_DAY, "1 mo"),
        (SEC_PER_YEAR, "1 yr"),
        (100.0 * SEC_PER_YEAR, "100 yr"),
    ]
    visible_ticks = [(v, lbl) for v, lbl in ref_ticks if xmin <= v <= xmax]
    ax.set_xticks([v for v, _ in visible_ticks])
    ax.set_xticklabels([lbl for _, lbl in visible_ticks], fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.8)
    ax.set_xlabel("Total wall-clock time for full experiment (log scale, human units)")
    ax.grid(True, axis="x", which="major", linestyle=":", alpha=0.65)

    for y, r in zip(y_pos, rows):
        v = r["total_s"]
        tag = "measured" if r["status"] == "measured" else "extrapolated"
        sp = r["speedup_vs_naive"]
        if sp < 1.5:
            sp_txt = "1x baseline"
        elif sp >= 1e6:
            sp_txt = f"{sp/1e6:.2f}M x faster"
        elif sp >= 1e3:
            sp_txt = f"{sp/1e3:.1f}k x faster"
        else:
            sp_txt = f"{sp:.0f}x faster"
        txt_lbl = f" {r['human_total']}  ({sp_txt})  [{tag}]"
        ax.text(v * 1.18, y, txt_lbl, va="center", ha="left", fontsize=8.3,
                fontweight="bold" if r["row"] in (1, len(rows)) else "normal")

    size_box = (
        f"Experiment size (from candidates.csv, truth/sources.csv, config.yaml):\n"
        f"• Truth events: {counts['n_truth_map2']:,} Map 2 + {counts['n_truth_twin']:,} twin = {counts['n_truth']:,}\n"
        f"• Strata: {counts['n_strata']} ({counts['n_operators']} op × {counts['n_range_caps']} caps × {counts['n_regimes']} regimes) × {counts['n_draws_per_stratum']} draws = {counts['n_trials']:,} trials\n"
        f"• Candidates: K = {counts['K']:,} (K2 = {counts['K2']:,} w/ Aegean flips); library build = {counts['n_library_build']:,} runs\n"
        f"• Gather/agg ({counts['n_agg_calls']} calls): {format_duration(t_agg_total)} in Rows 3–6 (Row 6: MacBook-measured, Colab CPU in practice)"
    )
    ax.text(
        0.985, 0.04, size_box, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.38", "fc": "#F7FAFD", "ec": viz.COLORS["ships"], "alpha": 0.95},
    )

    viz.save_figure(
        fig,
        out_dir / "reshaping_ladder.png",
        "What Reshaping Saves: Full Eastern Mediterranean OSSE Wall-Clock Ladder",
        f"Computed from measured real-grid per-unit timings (MacBook arm64 CPU & Colab {gpu_name}) for {counts['n_trials']:,} trials, K={counts['K']:,} candidates, and {counts['n_library_build']:,} library+truth+historic runs, including {format_duration(t_agg_total)} for {counts['n_agg_calls']} template gather/aggregation calls (Row 6 gather/agg: MacBook-measured, Colab CPU in practice). All per-unit timings are measured; totals are extrapolated until T14+T17 wall-clock logs land.",
    )
    print(f"Wrote {out_dir / 'reshaping_ladder.json'} and {out_dir / 'reshaping_ladder.png'}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build What Reshaping Saves table and figure")
    parser.add_argument("--out-dir", default=None, help="Output directory (default from config.yaml)")
    parser.add_argument("--chunk", type=int, default=8, help="Candidate chunk size for aggregate_templates timing")
    args = parser.parse_args(argv)
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["outputs"]) if args.out_dir is None else Path(args.out_dir)
    counts = load_experiment_counts(cfg)
    agg = measure_aggregate_templates(cfg, counts["K"], chunk=args.chunk)
    build_reshaping_ladder(cfg, out_dir=out_dir, agg_measurement=agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
