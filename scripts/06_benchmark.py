"""T19: Three-tier benchmark for Solver (Workload A) and Scoring (Workload B).

Workloads are timed separately (PLAN.md Criterion 3 & Phase 6):
  A. Batched linear shallow-water solver:
       1. Naive serial NumPy (solver_reference.run_reference, B = 1)
       2. Vectorised batched NumPy on CPU (Solver.run_library, B = 1, 8, 16)
       3. Batched CuPy on Colab GPU (Solver.run_library, B = 1, 8, 32, 64, 128)
  B. Matched-filter scoring tensor:
       1. Loop-based reference on CPU (scoring.score_loop_reference)
       2. Batched matmul + prefix scan on CPU (scoring.cumulative_stats + detection_scores)
       3. Batched matmul + prefix scan on GPU (CuPy)

Separates algorithmic (reshaping) speedup from hardware speedup, reports wall-clock time,
cell updates/s, achieved memory bandwidth vs GPU peak, and "what a 1-hour budget buys".

Usage:
  PYTHONPATH=src python scripts/06_benchmark.py --cpu     # run CPU tiers -> outputs/benchmark_cpu.json
  PYTHONPATH=src python scripts/06_benchmark.py --gpu     # run GPU tiers -> outputs/benchmark_gpu.json
  PYTHONPATH=src python scripts/06_benchmark.py --plot    # combine & plot -> benchmark.{json,png}, what_speed_buys.png
  PYTHONPATH=src python scripts/06_benchmark.py           # auto-run active backend + plot
"""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami import backend, bench, scoring, ships, sources, viz
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami import solver_reference as ref

# Published context (PLAN.md Criterion 3): Tsunami-HySEA (EDANYA / ChEESE / Macias et al. 2017-2020)
TSUNAMI_HYSEA_CONTEXT = {
    "citation": "Macias et al. (2017) Pure Appl. Geophys. 174:3147; ChEESE / Macias et al. (2019) AGU NH33A-06; Gibbons et al. (2020) Front. Earth Sci. 8",
    "med_10M_cells_8h_1xV100_s": 284.0,
    "med_10M_cells_8h_2xP100_s": 257.0,
    "ftrt_ratio_1xV100": 101.0,
    "notes": "Nonlinear shallow-water FV solver, 1 scenario per GPU (10M cells, 30 arc-sec, 8 h simulated in 284 s on 1x V100).",
}


def _solver_bytes(B, ny, nx, n_steps, R, n_samples):
    """Minimum float32 memory traffic (bytes): dynamic (eta, u, v R/W = 24 B/cell) + 8 static fields (32 B/cell) + receiver writes."""
    stencil_bytes = (24 * B + 32) * ny * nx * n_steps
    sample_bytes = 4 * B * R * 3 * n_samples
    return int(stencil_bytes + sample_bytes)


def _make_eta0_batch(grid, table, cfg, B):
    hel = table[(table["region"] == "hellenic") & np.isclose(table["mw"], 8.0)]
    base = sources.source_eta0(grid, hel.iloc[len(hel) // 2], cfg)
    scales = np.linspace(0.8, 1.2, B, dtype=np.float32)
    return base[None, :, :] * scales[:, None, None]


def _make_scoring_inputs(cfg, pool, table, E, K=None):
    rng = np.random.default_rng([cfg["meta"]["seed"], 19, E])
    draw = ships.traffic_draw(pool, cfg, index=0, purpose=1)
    groups = scoring.build_groups(draw, pool, cfg, "ais")
    S = len(draw.pool_idx)
    M = groups.n_groups
    T = int(cfg["library"]["n_minutes"])
    K_use = len(table) if K is None else int(K)
    flip = table["flip_allowed"].values[:K_use].astype(bool)
    sigma_i = np.full(S, 0.055, dtype=np.float32)
    sigma_shared = 0.08
    G = rng.normal(0.0, 0.05, size=(K_use, M, T)).astype(np.float32)
    D = rng.normal(0.0, 0.10, size=(E, M, T)).astype(np.float32)
    w = scoring.cell_weights(groups, sigma_i, sigma_shared, valid=None, xp=np)
    return draw, groups, flip, sigma_i, sigma_shared, G, D, w


def _scoring_bytes(K, K2, M, E, T):
    """Minimum float32 tensor traffic (bytes) for cumulative_stats + expand_flips + detection_scores + stat."""
    return int(4 * (K * M * T + E * M * T + M * T + E * K * T + K * T + 2 * E * K2 * T + K2 * T + E * T))


def run_cpu_benchmarks(cfg):
    grid = load_grid(cfg)
    pool = ships.load_pool(cfg)
    table = sources.candidate_table(cfg)
    sampler = make_sampler(grid, pool)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    ny, nx = grid.depth.shape
    R = len(pool.lon)

    solver_cpu = Solver(grid, sampler, cfg, xp=np)
    n_steps = int(solver_cpu.n_steps)
    n_samples = int(solver_cpu.n_samples)
    eta1 = _make_eta0_batch(grid, table, cfg, 1)

    records = []

    # ---- Workload A: Solver (CPU) ----
    # Tier 1: naive serial reference (B = 1)
    with bench.Timer("solver_serial_cpu", sync=None) as t:
        _ = ref.run_reference(grid, eta1[0], sampler, cfg)
    records.append(
        bench.record(
            "solver_serial_cpu",
            t.seconds,
            cell_updates=ny * nx * n_steps,
            bytes_moved=_solver_bytes(1, ny, nx, n_steps, R, n_samples),
            workload="solver",
            tier="serial_cpu",
            batch_size=1,
            seconds_per_scenario=t.seconds,
            ny=ny,
            nx=nx,
            n_steps=n_steps,
            n_receivers=R,
        )
    )
    print(f"[CPU] solver_serial_cpu (B=1): {t.seconds:.3f} s/scen")

    # Tier 2: vectorised batched NumPy on CPU
    for B in (1, 8, 16):
        etaB = _make_eta0_batch(grid, table, cfg, B)
        with bench.Timer(f"solver_batched_cpu_B{B}", sync=None) as t:
            _ = solver_cpu.run_library(etaB, gb_idx)
        records.append(
            bench.record(
                f"solver_batched_cpu_B{B}",
                t.seconds,
                cell_updates=B * ny * nx * n_steps,
                bytes_moved=_solver_bytes(B, ny, nx, n_steps, R, n_samples),
                workload="solver",
                tier="batched_cpu",
                batch_size=B,
                seconds_per_scenario=t.seconds / B,
                ny=ny,
                nx=nx,
                n_steps=n_steps,
                n_receivers=R,
            )
        )
        print(f"[CPU] solver_batched_cpu (B={B}): {t.seconds:.3f} s total ({t.seconds/B:.3f} s/scen)")

    # ---- Workload B: Scoring (CPU) ----
    # Two-point slice timing on identical (K_sub, E) to separate O(K2*E) slope from fixed loop overhead
    K_sub, E1, E2 = 8, 2, 4
    draw, groups, flip_sub, sigma_i, sigma_shared, _, _, _ = _make_scoring_inputs(cfg, pool, table, E2, K=K_sub)
    S = len(draw.pool_idx)
    M = groups.n_groups
    T = int(cfg["library"]["n_minutes"])
    rng = np.random.default_rng([cfg["meta"]["seed"], 19, 999])
    d_ship2 = rng.normal(0.0, 0.10, size=(E2, S, T)).astype(np.float32)
    g_ship = rng.normal(0.0, 0.05, size=(K_sub, S, T)).astype(np.float32)
    eps = float(cfg["scoring"]["eps"])

    with bench.Timer("scoring_loop_cpu_slice_E2", sync=None) as t_loop1:
        _ = scoring.score_loop_reference(d_ship2[:E1], g_ship, groups, sigma_i, sigma_shared, flip_sub, eps)
    with bench.Timer("scoring_loop_cpu_slice_E4", sync=None) as t_loop2:
        _ = scoring.score_loop_reference(d_ship2, g_ship, groups, sigma_i, sigma_shared, flip_sub, eps)
    with bench.Timer("scoring_tensor_cpu_slice_E4", sync=None) as t_tens2:
        _ = scoring.score_tensor(d_ship2, g_ship, groups, sigma_i, sigma_shared, flip_sub, eps, xp=np)

    # Marginal cost per event at K2_sub is (t_loop2 - t_loop1) / (E2 - E1); scales linearly with K2
    K_full = len(table)
    K2_full = int(K_full + table["flip_allowed"].sum())
    K2_sub = K_sub + int(flip_sub.sum())
    E_ref = 100
    per_event_at_ksub = max(0.0, (t_loop2.seconds - t_loop1.seconds) / (E2 - E1))
    fixed_ksub = max(0.0, t_loop1.seconds - E1 * per_event_at_ksub)
    loop_100ev_s = (K2_full / K2_sub) * (fixed_ksub + E_ref * per_event_at_ksub)
    records.append(
        bench.record(
            "scoring_loop_cpu",
            loop_100ev_s,
            workload="scoring",
            tier="loop_cpu",
            E=E_ref,
            K=K_full,
            K2=K2_full,
            M=M,
            S=S,
            T=T,
            measured_slice_E2_seconds=t_loop1.seconds,
            measured_slice_E4_seconds=t_loop2.seconds,
            slice_K=K_sub,
            slice_K2=K2_sub,
            slice_tensor_E4_seconds=t_tens2.seconds,
            direct_slice_speedup=t_loop2.seconds / max(t_tens2.seconds, 1e-9),
            extrapolated_from_two_point_slope=True,
            seconds_per_100_events=loop_100ev_s,
        )
    )
    print(
        f"[CPU] scoring_loop_cpu slice (K2={K2_sub}): E=2 {t_loop1.seconds:.3f} s, E=4 {t_loop2.seconds:.3f} s "
        f"vs tensor E=4 {t_tens2.seconds:.4f} s ({t_loop2.seconds/t_tens2.seconds:.0f}x direct; "
        f"extrapolated full K2={K2_full}, E=100: {loop_100ev_s:.0f} s)"
    )

    # Full-catalogue tensor scoring on CPU (E = 20, 100, 500)
    for E in (20, 100, 500):
        _, groups, flip_full, _, _, G, D, w = _make_scoring_inputs(cfg, pool, table, E, K=K_full)
        with bench.Timer(f"scoring_tensor_cpu_E{E}", sync=None) as t:
            gd, gg = scoring.cumulative_stats(G, D, w, xp=np)
            gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip_full)
            S_sc = scoring.detection_scores(gd2, gg2, eps)
            _ = scoring.detection_statistic(S_sc)
        records.append(
            bench.record(
                f"scoring_tensor_cpu_E{E}",
                t.seconds,
                bytes_moved=_scoring_bytes(K_full, K2_full, M, E, T),
                workload="scoring",
                tier="tensor_cpu",
                E=E,
                K=K_full,
                K2=K2_full,
                M=M,
                S=S,
                T=T,
                seconds_per_100_events=t.seconds * (100.0 / E),
            )
        )
        print(f"[CPU] scoring_tensor_cpu (K={K_full}, K2={K2_full}, M={M}, E={E}): {t.seconds:.4f} s ({t.seconds*100/E:.4f} s / 100 ev)")

    payload = {
        "device": bench.device_info(),
        "records": records,
    }
    out = repo_path(cfg["paths"]["outputs"]) / "benchmark_cpu.json"
    bench.write_json(payload, out)
    print(f"Wrote {out}")
    return payload


def run_gpu_benchmarks(cfg):
    if not backend.GPU:
        raise RuntimeError("No CUDA GPU detected (CuPy unavailable). Run with --cpu on this machine.")

    dev = bench.device_info()
    peak_bw = dev.get("peak_bandwidth_gb_s")
    grid = load_grid(cfg)
    pool = ships.load_pool(cfg)
    table = sources.candidate_table(cfg)
    sampler = make_sampler(grid, pool)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    ny, nx = grid.depth.shape
    R = len(pool.lon)

    solver_gpu = Solver(grid, sampler, cfg, xp=backend.xp)
    n_steps = int(solver_gpu.n_steps)
    n_samples = int(solver_gpu.n_samples)

    # Warmup CuPy kernels before timing
    eta1 = _make_eta0_batch(grid, table, cfg, 1)
    _ = solver_gpu.run_library(eta1, gb_idx)
    backend.sync()

    records = []
    # ---- Workload A: Solver (GPU) ----
    for B in (1, 8, 32, 64, 128):
        etaB = _make_eta0_batch(grid, table, cfg, B)
        try:
            with bench.Timer(f"solver_batched_gpu_B{B}", sync=backend.sync) as t:
                _ = solver_gpu.run_library(etaB, gb_idx)
        except Exception as exc:
            print(f"[GPU] Skipping B={B}: {exc}")
            continue
        records.append(
            bench.record(
                f"solver_batched_gpu_B{B}",
                t.seconds,
                cell_updates=B * ny * nx * n_steps,
                bytes_moved=_solver_bytes(B, ny, nx, n_steps, R, n_samples),
                peak_bandwidth_gb_s=peak_bw,
                workload="solver",
                tier="batched_gpu",
                batch_size=B,
                seconds_per_scenario=t.seconds / B,
                ny=ny,
                nx=nx,
                n_steps=n_steps,
                n_receivers=R,
            )
        )
        print(f"[GPU] solver_batched_gpu (B={B}): {t.seconds:.3f} s total ({t.seconds/B:.4f} s/scen)")

    # ---- Workload B: Scoring (GPU) ----
    K_full = len(table)
    K2_full = int(K_full + table["flip_allowed"].sum())
    eps = float(cfg["scoring"]["eps"])
    T = int(cfg["library"]["n_minutes"])

    # Warmup scoring on GPU
    _, groups, flip_full, _, _, G0, D0, w0 = _make_scoring_inputs(cfg, pool, table, 20, K=K_full)
    G_g, D_g, w_g = backend.to_xp(G0), backend.to_xp(D0), backend.to_xp(w0)
    gd, gg = scoring.cumulative_stats(G_g, D_g, w_g, xp=backend.xp)
    gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip_full)
    _ = scoring.detection_statistic(scoring.detection_scores(gd2, gg2, eps))
    backend.sync()

    M = groups.n_groups
    S = len(groups.group)
    for E in (20, 100, 500):
        _, _, flip_full, _, _, G, D, w = _make_scoring_inputs(cfg, pool, table, E, K=K_full)
        G_g, D_g, w_g = backend.to_xp(G), backend.to_xp(D), backend.to_xp(w)
        backend.sync()
        with bench.Timer(f"scoring_tensor_gpu_E{E}", sync=backend.sync) as t:
            gd, gg = scoring.cumulative_stats(G_g, D_g, w_g, xp=backend.xp)
            gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip_full)
            S_sc = scoring.detection_scores(gd2, gg2, eps)
            _ = scoring.detection_statistic(S_sc)
        records.append(
            bench.record(
                f"scoring_tensor_gpu_E{E}",
                t.seconds,
                bytes_moved=_scoring_bytes(K_full, K2_full, M, E, T),
                peak_bandwidth_gb_s=peak_bw,
                workload="scoring",
                tier="tensor_gpu",
                E=E,
                K=K_full,
                K2=K2_full,
                M=M,
                S=S,
                T=T,
                seconds_per_100_events=t.seconds * (100.0 / E),
            )
        )
        print(f"[GPU] scoring_tensor_gpu (E={E}): {t.seconds:.4f} s ({t.seconds*100/E:.4f} s / 100 ev)")

    payload = {
        "device": dev,
        "records": records,
    }
    out = repo_path(cfg["paths"]["outputs"]) / "benchmark_gpu.json"
    bench.write_json(payload, out)
    print(f"Wrote {out}")
    return payload


def combine_and_plot(cfg, out_dir=None):
    out_dir = repo_path(cfg["paths"]["outputs"]) if out_dir is None else out_dir
    cpu_path = out_dir / "benchmark_cpu.json"
    gpu_path = out_dir / "benchmark_gpu.json"

    cpu = bench.read_json(cpu_path) if cpu_path.exists() else None
    gpu = bench.read_json(gpu_path) if gpu_path.exists() else None
    if cpu is None and gpu is None:
        raise FileNotFoundError("Neither benchmark_cpu.json nor benchmark_gpu.json exists. Run --cpu or --gpu first.")

    recs = []
    if cpu:
        recs.extend(cpu["records"])
    if gpu:
        recs.extend(gpu["records"])

    def _best(tier, workload, key):
        sub = [r for r in recs if r.get("workload") == workload and r.get("tier") == tier]
        return min(sub, key=lambda r: r[key]) if sub else None

    s_ser = _best("serial_cpu", "solver", "seconds_per_scenario")
    s_bcpu = _best("batched_cpu", "solver", "seconds_per_scenario")
    s_bgpu = _best("batched_gpu", "solver", "seconds_per_scenario")

    c_loop = _best("loop_cpu", "scoring", "seconds_per_100_events")
    c_tcpu = _best("tensor_cpu", "scoring", "seconds_per_100_events")
    c_tgpu = _best("tensor_gpu", "scoring", "seconds_per_100_events")

    summary = {
        "solver": {
            "serial_cpu_s_per_scen": s_ser["seconds_per_scenario"] if s_ser else None,
            "batched_cpu_s_per_scen": s_bcpu["seconds_per_scenario"] if s_bcpu else None,
            "batched_gpu_s_per_scen": s_bgpu["seconds_per_scenario"] if s_bgpu else None,
            "algo_speedup_cpu": (s_ser["seconds_per_scenario"] / s_bcpu["seconds_per_scenario"]) if (s_ser and s_bcpu) else None,
            "hw_speedup_gpu_over_batched_cpu": (s_bcpu["seconds_per_scenario"] / s_bgpu["seconds_per_scenario"]) if (s_bcpu and s_bgpu) else None,
            "total_speedup_gpu_over_serial": (s_ser["seconds_per_scenario"] / s_bgpu["seconds_per_scenario"]) if (s_ser and s_bgpu) else None,
            "gpu_bandwidth_gb_s": s_bgpu.get("bandwidth_gb_s") if s_bgpu else None,
            "gpu_bandwidth_pct_peak": s_bgpu.get("bandwidth_pct_peak") if s_bgpu else None,
        },
        "scoring": {
            "loop_cpu_s_per_100ev": c_loop["seconds_per_100_events"] if c_loop else None,
            "tensor_cpu_s_per_100ev": c_tcpu["seconds_per_100_events"] if c_tcpu else None,
            "tensor_gpu_s_per_100ev": c_tgpu["seconds_per_100_events"] if c_tgpu else None,
            "algo_speedup_cpu": (c_loop["seconds_per_100_events"] / c_tcpu["seconds_per_100_events"]) if (c_loop and c_tcpu) else None,
            "direct_slice_algo_speedup": c_loop.get("direct_slice_speedup") if c_loop else None,
            "hw_speedup_gpu_over_tensor_cpu": (c_tcpu["seconds_per_100_events"] / c_tgpu["seconds_per_100_events"]) if (c_tcpu and c_tgpu) else None,
            "total_speedup_gpu_over_loop": (c_loop["seconds_per_100_events"] / c_tgpu["seconds_per_100_events"]) if (c_loop and c_tgpu) else None,
        },
    }

    combined = {
        "cpu_device": cpu["device"] if cpu else None,
        "gpu_device": gpu["device"] if gpu else None,
        "summary": summary,
        "tsunami_hysea_context": TSUNAMI_HYSEA_CONTEXT,
        "records": recs,
    }
    bench.write_json(combined, out_dir / "benchmark.json")

    # Dynamic metadata for axis/title labels
    s_meta = s_bgpu or s_bcpu or s_ser or {}
    ny = int(s_meta.get("ny", 390))
    nx = int(s_meta.get("nx", 795))
    n_rec = int(s_meta.get("n_receivers", 20061))
    c_meta = c_tgpu or c_tcpu or c_loop or {}
    K_lbl = int(c_meta.get("K", 1083))
    K2_lbl = int(c_meta.get("K2", 1455))
    T_lbl = int(c_meta.get("T", 120))

    # ---- Figure 1: outputs/benchmark.png (Workloads A and B separately) ----
    viz.apply_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Panel A: Solver seconds/scenario by tier and batch size
    solver_bars = []
    if s_ser:
        solver_bars.append(("Serial CPU\n(ref, B=1)", s_ser["seconds_per_scenario"], viz.COLORS["neutral"], s_ser))
    for r in sorted([r for r in recs if r.get("tier") == "batched_cpu"], key=lambda x: x["batch_size"]):
        solver_bars.append((f"Batched CPU\n(B={r['batch_size']})", r["seconds_per_scenario"], viz.COLORS["moderate"], r))
    for r in sorted([r for r in recs if r.get("tier") == "batched_gpu"], key=lambda x: x["batch_size"]):
        solver_bars.append((f"Batched GPU\n(B={r['batch_size']})", r["seconds_per_scenario"], viz.COLORS["ships"], r))

    xs = np.arange(len(solver_bars))
    vals = [b[1] for b in solver_bars]
    cols = [b[2] for b in solver_bars]
    axA.bar(xs, vals, color=cols, edgecolor="#333333", linewidth=0.6)
    axA.set_yscale("log")
    axA.set_xticks(xs)
    axA.set_xticklabels([b[0] for b in solver_bars], fontsize=8)
    axA.set_ylabel("Wall-clock time per 2 h scenario (s, log scale)")
    axA.set_title(f"Workload A: Shallow-water solver ({ny}\u00d7{nx} grid, {n_rec:,} receivers)")
    axA.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, (_, v, _, rec) in enumerate(solver_bars):
        cu_m = (rec.get("cell_updates_per_s") or 0.0) / 1e6
        bw = rec.get("bandwidth_gb_s")
        pct = rec.get("bandwidth_pct_peak")
        lbl = f"{v:.2f} s\n({cu_m:.0f}M cu/s)"
        if pct is not None:
            lbl += f"\n{bw:.0f} GB/s ({pct:.0f}%)"
        axA.text(x, v * 1.12, lbl, ha="center", va="bottom", fontsize=7.2)
    if not gpu:
        axA.text(0.98, 0.96, "GPU tier pending\nColab slot from B", transform=axA.transAxes,
                 ha="right", va="top", fontsize=8.5, color=viz.COLORS["negative"],
                 bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF4E5", "ec": viz.COLORS["negative"]})

    # Panel B: Scoring seconds per 100 events
    score_bars = []
    if c_loop:
        score_bars.append(("Python Loops\n(CPU, ref)", c_loop["seconds_per_100_events"], viz.COLORS["neutral"], c_loop))
    if c_tcpu:
        score_bars.append((f"Tensor Matmul\n+ Prefix Scan\n(CPU, E={c_tcpu['E']})", c_tcpu["seconds_per_100_events"], viz.COLORS["moderate"], c_tcpu))
    if c_tgpu:
        score_bars.append((f"Tensor Matmul\n+ Prefix Scan\n(GPU, E={c_tgpu['E']})", c_tgpu["seconds_per_100_events"], viz.COLORS["ships"], c_tgpu))

    xb = np.arange(len(score_bars))
    vb = [b[1] for b in score_bars]
    cb = [b[2] for b in score_bars]
    axB.bar(xb, vb, color=cb, edgecolor="#333333", linewidth=0.6, width=0.55)
    axB.set_yscale("log")
    axB.set_xticks(xb)
    axB.set_xticklabels([b[0] for b in score_bars], fontsize=8.5)
    axB.set_ylabel("Wall-clock time per 100 events (s, log scale)")
    axB.set_title(f"Workload B: Matched-filter scoring (K={K_lbl} / K2={K2_lbl}, T={T_lbl} min)")
    axB.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, (_, v, _, rec) in enumerate(score_bars):
        axB.text(x, v * 1.15, f"{v:.3g} s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    if c_loop and c_tcpu:
        algo_x = summary["scoring"]["algo_speedup_cpu"]
        note = f"Algorithmic reshaping (CPU): {algo_x:,.0f}\u00d7"
        if c_tgpu:
            hw_x = summary["scoring"]["hw_speedup_gpu_over_tensor_cpu"]
            tot_x = summary["scoring"]["total_speedup_gpu_over_loop"]
            note += f"\nHardware (GPU / CPU tensor): {hw_x:.1f}\u00d7\nTotal (GPU / loops): {tot_x:,.0f}\u00d7"
        axB.text(0.97, 0.95, note, transform=axB.transAxes, ha="right", va="top", fontsize=8.5,
                 bbox={"boxstyle": "round,pad=0.35", "fc": "#F0F6FC", "ec": viz.COLORS["ships"]})

    gpu_label = gpu["device"].get("device_name", "GPU") if gpu else "GPU pending"
    viz.save_figure(
        fig,
        out_dir / "benchmark.png",
        f"Three-Tier Performance Benchmark: Solver (A) and Scoring (B) \u2014 CPU vs {gpu_label}",
        "Same NumPy/CuPy codebase (float32); context: Tsunami-HySEA solves 10M Med cells (8 h nonlinear) in 284 s on 1\u00d7V100 (~101\u00d7 FTRT).",
    )

    # ---- Figure 2: outputs/what_speed_buys.png (1-hour budget analysis) ----
    budget_s = 3600.0
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    tiers = []
    if s_ser:
        tiers.append(("Serial CPU\n(ref)", s_ser["seconds_per_scenario"],
                      c_loop["seconds_per_100_events"] if c_loop else 1e4, viz.COLORS["neutral"]))
    if s_bcpu:
        tiers.append(("Batched CPU\n(reshaped)", s_bcpu["seconds_per_scenario"],
                      c_tcpu["seconds_per_100_events"] if c_tcpu else 10.0, viz.COLORS["moderate"]))
    if s_bgpu:
        tiers.append((f"Batched GPU\n({gpu_label})", s_bgpu["seconds_per_scenario"],
                      c_tgpu["seconds_per_100_events"] if c_tgpu else 1.0, viz.COLORS["ships"]))

    scens_1h = [budget_s / t[1] for t in tiers]
    n_truth_strata = K_lbl * 12
    draws_per_ev_1h = [max(1.0, (budget_s / (t[2] / 100.0)) / n_truth_strata) for t in tiers]

    x1 = np.arange(len(tiers))
    ax1.bar(x1, scens_1h, color=[t[3] for t in tiers], edgecolor="#333333", linewidth=0.6, width=0.55)
    ax1.set_yscale("log")
    ax1.axhline(K_lbl, color=viz.COLORS["negative"], ls="--", lw=1.2, label=f"Full library ({K_lbl:,} candidates)")
    ax1.axhline(2 * K_lbl + 60, color=viz.COLORS["truth"], ls=":", lw=1.2, label=f"Library + truth + historic (~{2*K_lbl+60:,} runs)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([t[0] for t in tiers], fontsize=9)
    ax1.set_ylabel("2-hour scenarios simulated per hour (log scale)")
    ax1.set_title("A. Scenarios simulated in a 1-hour compute budget")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, v in zip(x1, scens_1h):
        ax1.text(x, v * 1.15, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Monte Carlo uncertainty on the 90% detection probability (Map 2) vs N draws
    n_grid = np.logspace(0.5, 3.5, 200)
    p = 0.90
    half_ci_95 = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / n_grid)
    ax2.plot(n_grid, half_ci_95, color="#333333", lw=1.8, label="95% CI half-width on p = 0.90 threshold")
    ax2.set_xscale("log")
    for (label, _, _, col), n_aff in zip(tiers, draws_per_ev_1h):
        n_clip = min(max(n_aff, 3.2), 3000.0)
        ci = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / n_clip)
        ax2.scatter([n_clip], [ci], s=85, color=col, edgecolors="k", zorder=5,
                    label=f"{label.splitlines()[0]}: ~{n_aff:,.0f} draws/stratum (\u00b1{ci:.1f}%)")
    ax2.axvline(100, color=viz.COLORS["positive"], ls="--", lw=1.1, label="Project target (5 traffic \u00d7 20 noise = 100)")
    ax2.set_xlabel("Traffic \u00d7 noise draws per truth event per stratum (log scale)")
    ax2.set_ylabel("95% binomial CI half-width on 90% detection rate (%)")
    ax2.set_title("B. What scoring speed buys: Map 2 statistical precision in 1 hour")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    viz.save_figure(
        fig2,
        out_dir / "what_speed_buys.png",
        "What the Speed Makes Possible: Experiment Scale and Uncertainty in a 1-Hour Budget",
        f"Computed directly from measured wall-clock rates for the {ny}\u00d7{nx} grid ({K_lbl:,} library candidates, 12 strata); higher draw counts narrow Map 2 detection uncertainty.",
    )
    print(f"Wrote {out_dir / 'benchmark.json'}, benchmark.png, what_speed_buys.png")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description="T19 Solver and Scoring Benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Run CPU benchmark tiers")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmark tiers (requires CuPy/CUDA)")
    parser.add_argument("--plot", action="store_true", help="Combine saved CPU/GPU JSONs and generate plots")
    args = parser.parse_args(argv)

    cfg = load_config()
    if not (args.cpu or args.gpu or args.plot):
        if backend.GPU:
            run_gpu_benchmarks(cfg)
        else:
            run_cpu_benchmarks(cfg)
        combine_and_plot(cfg)
        return 0

    if args.cpu:
        run_cpu_benchmarks(cfg)
    if args.gpu:
        run_gpu_benchmarks(cfg)
    if args.plot:
        combine_and_plot(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
