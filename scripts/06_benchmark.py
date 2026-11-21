"""T19: Three-tier benchmark for Solver (Workload A) and Scoring (Workload B).

Workloads are timed separately (PLAN.md Criterion 3 & Phase 6):
  A. Batched linear shallow-water solver:
       1. Naive serial NumPy on MacBook arm64 (solver_reference.run_reference, B = 1, real 390x795)
       2. Vectorised batched NumPy on MacBook arm64 (Solver.run_library, B = 1, 8, 16, real 390x795)
       3. Same-machine Colab 2-vCPU CPU vs Colab Tesla T4 GPU (from outputs/benchmark_gpu.json
          when available, or Stream A's outputs/validation/gpu_check.json flat 360x750 fallback)
  B. Matched-filter scoring tensor (compared consistently at E = 100 events, K = 1,083 / K2 = 1,455):
       1. Python loop reference on MacBook CPU (scoring.score_loop_reference, two-point slope)
       2. Vectorised NumPy looping over minutes WITHOUT prefix scan (recomputing 0..t sums at each t)
       3. Batched matmul + parallel prefix scan on MacBook CPU (scoring.cumulative_stats)
       4. Batched matmul + parallel prefix scan on Colab GPU (CuPy, when benchmark_gpu.json lands)

Separates vectorisation gain, prefix-scan gain, and same-machine hardware speedup; reports
wall-clock time, cell updates/s, memory bandwidth, exact Tsunami-HySEA citation, and
"what a 1-hour budget buys".

Usage:
  PYTHONPATH=src python scripts/06_benchmark.py --cpu     # run CPU tiers -> outputs/benchmark_cpu.json
  PYTHONPATH=src python scripts/06_benchmark.py --gpu     # run GPU tiers -> outputs/benchmark_gpu.json
  PYTHONPATH=src python scripts/06_benchmark.py --plot    # combine & plot -> benchmark.{json,png}, what_speed_buys.png
  PYTHONPATH=src python scripts/06_benchmark.py           # auto-run active backend + plot
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami import backend, bench, scoring, ships, sources, viz
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami import solver_reference as ref

# Exact verified primary-journal Tsunami-HySEA benchmark citation:
# Estrada et al. (2021), Scientific Reports 11:16253, doi:10.1038/s41598-021-95729-6
TSUNAMI_HYSEA_CONTEXT = {
    "citation": (
        "Estrada, F., Gonzalez-Vida, J. M., Pelaez, J. A., Galindo-Zaldivar, J., Ortega, S., "
        "Macias, J., Vazquez, J. T., & Ercilla, G. (2021), Scientific Reports 11:16253, "
        "doi:10.1038/s41598-021-95729-6 (Methods: Tsunami model)"
    ),
    "verbatim_quote": (
        "By means of this very efficient implementation, the model is able to simulate 8 h of real time "
        "tsunami in the Mediterranean Sea (in a mesh with 10 million volumes and a resolution of 30 arc-sec) "
        "in 257 s using two NVIDIA Tesla P100, or even in 284 s with one NVIDIA Tesla V100."
    ),
    "med_10M_volumes_30arcsec_8h_1xV100_s": 284.0,
    "med_10M_volumes_30arcsec_8h_2xP100_s": 257.0,
    "ftrt_ratio_1xV100": 28800.0 / 284.0,
    "notes": (
        "Nonlinear shallow-water FV solver: 8 h real-time Mediterranean Sea tsunami (10M volumes, "
        "30 arc-sec) in 284 s on 1x NVIDIA Tesla V100 or 257 s on 2x NVIDIA Tesla P100."
    ),
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


def score_vectorised_no_prefix(G, D, w, flip_allowed, eps):
    """Middle scoring tier: vectorised NumPy, looping over minutes t and recomputing full 0..t sums (no prefix scan)."""
    K, M, T = G.shape
    E = D.shape[0]
    Gw = (G * w).astype(np.float32)
    gd = np.empty((E, K, T), dtype=np.float64)
    gg = np.empty((K, T), dtype=np.float64)
    # Recompute the full 0..t sum at every minute t (O(T^2) work; no cumulative prefix scan)
    for t in range(T):
        Gw_t = Gw[:, :, : t + 1].reshape(K, -1)
        D_t = D[:, :, : t + 1].reshape(E, -1)
        G_t = G[:, :, : t + 1].reshape(K, -1)
        gd[:, :, t] = (D_t @ Gw_t.T).astype(np.float64)
        gg[:, t] = np.sum(Gw_t * G_t, axis=1, dtype=np.float64)
    gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip_allowed)
    S_sc = scoring.detection_scores(gd2, gg2, eps)
    return scoring.detection_statistic(S_sc)


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
    machine_lbl = "MacBook arm64"

    # ---- Workload A: Solver (CPU, MacBook arm64, real 390x795 grid) ----
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
            machine=machine_lbl,
            grid_label=f"real {ny}x{nx}, R={R:,}",
            batch_size=1,
            seconds_per_scenario=t.seconds,
            ny=ny,
            nx=nx,
            n_steps=n_steps,
            n_receivers=R,
        )
    )
    print(f"[CPU] solver_serial_cpu (B=1): {t.seconds:.3f} s/scen")

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
                machine=machine_lbl,
                grid_label=f"real {ny}x{nx}, R={R:,}",
                batch_size=B,
                seconds_per_scenario=t.seconds / B,
                ny=ny,
                nx=nx,
                n_steps=n_steps,
                n_receivers=R,
            )
        )
        print(f"[CPU] solver_batched_cpu (B={B}): {t.seconds:.3f} s total ({t.seconds/B:.3f} s/scen)")

    # ---- Workload B: Scoring (CPU, MacBook arm64) ----
    # Tier 1: Python loops (two-point slope E=2 -> E=4 at K2=8, extrapolated to full K2, E=100)
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
            machine=machine_lbl,
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
    print(f"[CPU] scoring_loop_cpu (K2={K2_full}, E=100 extrapolated): {loop_100ev_s:.1f} s")

    # Tier 2 (middle): Vectorised NumPy looping over minutes WITHOUT prefix scan (recomputing 0..t sums) at E=100
    _, groups, flip_full, _, _, G100, D100, w100 = _make_scoring_inputs(cfg, pool, table, E_ref, K=K_full)
    with bench.Timer("scoring_vec_no_prefix_cpu_E100", sync=None) as t_mid:
        _ = score_vectorised_no_prefix(G100, D100, w100, flip_full, eps)
    records.append(
        bench.record(
            "scoring_vec_no_prefix_cpu_E100",
            t_mid.seconds,
            workload="scoring",
            tier="vec_no_prefix_cpu",
            machine=machine_lbl,
            E=E_ref,
            K=K_full,
            K2=K2_full,
            M=M,
            S=S,
            T=T,
            seconds_per_100_events=t_mid.seconds,
        )
    )
    print(f"[CPU] scoring_vec_no_prefix_cpu (K={K_full}, K2={K2_full}, M={M}, E={E_ref}): {t_mid.seconds:.3f} s")

    # Tier 3: Batched matmul + parallel prefix scan on CPU (E = 20, 100, 500)
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
                machine=machine_lbl,
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
    gpu_name = dev.get("device_name", "Colab GPU")
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

    # Also time same-machine Colab CPU (B=1) for apples-to-apples hardware speedup
    solver_cpu = Solver(grid, sampler, cfg, xp=np)
    eta1 = _make_eta0_batch(grid, table, cfg, 1)
    records = []
    with bench.Timer("solver_colab_cpu_B1", sync=None) as t_c:
        _ = solver_cpu.run_library(eta1, gb_idx)
    records.append(
        bench.record(
            "solver_colab_cpu_B1",
            t_c.seconds,
            cell_updates=ny * nx * n_steps,
            bytes_moved=_solver_bytes(1, ny, nx, n_steps, R, n_samples),
            workload="solver",
            tier="colab_cpu",
            machine="Colab CPU",
            grid_label=f"real {ny}x{nx}, R={R:,}",
            batch_size=1,
            seconds_per_scenario=t_c.seconds,
            ny=ny,
            nx=nx,
            n_steps=n_steps,
            n_receivers=R,
        )
    )

    # Warmup CuPy kernels
    _ = solver_gpu.run_library(eta1, gb_idx)
    backend.sync()

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
                machine=gpu_name,
                grid_label=f"real {ny}x{nx}, R={R:,}",
                batch_size=B,
                seconds_per_scenario=t.seconds / B,
                ny=ny,
                nx=nx,
                n_steps=n_steps,
                n_receivers=R,
            )
        )
        print(f"[GPU] solver_batched_gpu (B={B}): {t.seconds:.3f} s total ({t.seconds/B:.4f} s/scen)")

    # Workload B: Scoring (GPU)
    K_full = len(table)
    K2_full = int(K_full + table["flip_allowed"].sum())
    eps = float(cfg["scoring"]["eps"])
    T = int(cfg["library"]["n_minutes"])

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
                machine=gpu_name,
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

    payload = {"device": dev, "records": records}
    out = repo_path(cfg["paths"]["outputs"]) / "benchmark_gpu.json"
    bench.write_json(payload, out)
    print(f"Wrote {out}")
    return payload


def _load_gpu_check_fallback(out_dir: Path):
    """Load Stream A's measured Colab T4 check (outputs/validation/gpu_check.json) if benchmark_gpu.json is absent."""
    p = out_dir / "validation" / "gpu_check.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    tim = raw.get("timing", {})
    if "cupy_B64_s_per_scenario" not in tim:
        return None
    ny, nx, n_steps, R, n_samp = 360, 750, 1440, 20000, 240  # flat 3000 m, 2 h
    peak_bw = 320.0  # Tesla T4 theoretical GDDR6 peak (GB/s)
    glabel = "synthetic flat 360x750, R=20k (gpu_check.json)"
    recs = [
        bench.record(
            "solver_colab_cpu_B1_flat",
            float(tim["numpy_B1_s_per_scenario"]),
            cell_updates=ny * nx * n_steps,
            bytes_moved=_solver_bytes(1, ny, nx, n_steps, R, n_samp),
            workload="solver",
            tier="colab_cpu",
            machine="Colab 2-vCPU",
            grid_label=glabel,
            batch_size=1,
            seconds_per_scenario=float(tim["numpy_B1_s_per_scenario"]),
            ny=ny,
            nx=nx,
            n_steps=n_steps,
            n_receivers=R,
            from_gpu_check_fallback=True,
        ),
    ]
    for B, key in ((16, "cupy_B16_s_per_scenario"), (64, "cupy_B64_s_per_scenario")):
        if key not in tim:
            continue
        s_per = float(tim[key])
        rec = bench.record(
            f"solver_batched_gpu_B{B}_flat",
            s_per * B,
            cell_updates=B * ny * nx * n_steps,
            bytes_moved=_solver_bytes(B, ny, nx, n_steps, R, n_samp),
            peak_bandwidth_gb_s=peak_bw,
            workload="solver",
            tier="batched_gpu",
            machine="Colab T4",
            grid_label=glabel,
            batch_size=B,
            seconds_per_scenario=s_per,
            ny=ny,
            nx=nx,
            n_steps=n_steps,
            n_receivers=R,
            from_gpu_check_fallback=True,
        )
        rec["backend"] = "cupy"
        if B == 64 and "cupy_B64_gcell_updates_per_s" in tim:
            rec["cell_updates_per_s"] = float(tim["cupy_B64_gcell_updates_per_s"]) * 1e9
        recs.append(rec)
    return {
        "device": {
            "backend": "cupy",
            "gpu": True,
            "device_name": "Tesla T4",
            "peak_bandwidth_gb_s": peak_bw,
            "source": "outputs/validation/gpu_check.json (synthetic flat 360x750, R=20,000)",
        },
        "records": recs,
        "is_fallback": True,
    }


def combine_and_plot(cfg, out_dir=None):
    out_dir = repo_path(cfg["paths"]["outputs"]) if out_dir is None else Path(out_dir)
    cpu_path = out_dir / "benchmark_cpu.json"
    gpu_path = out_dir / "benchmark_gpu.json"

    cpu = bench.read_json(cpu_path) if cpu_path.exists() else None
    gpu = bench.read_json(gpu_path) if gpu_path.exists() else None
    gpu_check_fallback = _load_gpu_check_fallback(out_dir)
    gpu_is_fallback = False
    if gpu is None:
        gpu = gpu_check_fallback
        gpu_is_fallback = gpu is not None
    if cpu is None and gpu is None:
        raise FileNotFoundError("No benchmark JSON found. Run --cpu or --gpu first.")

    gpu_name = gpu["device"].get("device_name", "GPU") if gpu else "GPU"

    # Backfill machine & grid_label on records loaded from older GPU/CPU JSONs
    recs = []
    if cpu:
        for r in cpu["records"]:
            rc = dict(r)
            rc.setdefault("machine", "MacBook arm64")
            if rc.get("workload") == "solver" and "grid_label" not in rc and "ny" in rc and "nx" in rc:
                rc["grid_label"] = f"real {rc['ny']}x{rc['nx']}, R={int(rc.get('n_receivers', 0)):,}"
            recs.append(rc)
    if gpu:
        for r in gpu["records"]:
            rc = dict(r)
            if "machine" not in rc:
                rc["machine"] = "Colab 2-vCPU" if rc.get("tier") == "colab_cpu" else f"Colab {gpu_name}"
            if rc.get("workload") == "solver" and "grid_label" not in rc and "ny" in rc and "nx" in rc:
                rc["grid_label"] = f"real {rc['ny']}x{rc['nx']}, R={int(rc.get('n_receivers', 0)):,}"
            recs.append(rc)

    def _best(tier, workload, key):
        sub = [r for r in recs if r.get("workload") == workload and r.get("tier") == tier]
        return min(sub, key=lambda r: r[key]) if sub else None

    def _match_E(tier, E_target=100):
        sub = [r for r in recs if r.get("workload") == "scoring" and r.get("tier") == tier]
        exact = [r for r in sub if r.get("E") == E_target]
        if exact:
            return exact[0]
        return min(sub, key=lambda r: r["seconds_per_100_events"]) if sub else None

    s_ser = _best("serial_cpu", "solver", "seconds_per_scenario")
    s_bcpu = _best("batched_cpu", "solver", "seconds_per_scenario")
    s_ccpu = _best("colab_cpu", "solver", "seconds_per_scenario")
    s_bgpu = _best("batched_gpu", "solver", "seconds_per_scenario")

    c_loop = _match_E("loop_cpu", 100)
    c_mid = _match_E("vec_no_prefix_cpu", 100)
    c_tcpu = _match_E("tensor_cpu", 100)
    c_tgpu = _match_E("tensor_gpu", 100)

    # Same-machine Colab CPU-vs-GPU ratio: use real-grid colab_cpu if measured, else flat-grid gpu_check.json reference
    same_machine_flat_ref = None
    if gpu_check_fallback:
        fb_recs = gpu_check_fallback["records"]
        fb_cpu = next((r for r in fb_recs if r.get("tier") == "colab_cpu"), None)
        fb_gpu = min((r for r in fb_recs if r.get("tier") == "batched_gpu"), key=lambda r: r["seconds_per_scenario"], default=None)
        if fb_cpu and fb_gpu:
            same_machine_flat_ref = {
                "colab_cpu_s_per_scen": fb_cpu["seconds_per_scenario"],
                "colab_gpu_s_per_scen": fb_gpu["seconds_per_scenario"],
                "colab_gpu_batch_size": fb_gpu["batch_size"],
                "speedup": fb_cpu["seconds_per_scenario"] / fb_gpu["seconds_per_scenario"],
                "grid": "synthetic flat 360x750, R=20,000 (outputs/validation/gpu_check.json)",
            }

    if s_ccpu and s_bgpu:
        same_machine_hw_speedup = s_ccpu["seconds_per_scenario"] / s_bgpu["seconds_per_scenario"]
        same_machine_from_flat = bool(gpu_is_fallback)
    elif same_machine_flat_ref is not None:
        same_machine_hw_speedup = same_machine_flat_ref["speedup"]
        same_machine_from_flat = True
    else:
        same_machine_hw_speedup = None
        same_machine_from_flat = False

    summary = {
        "solver": {
            "macbook_serial_cpu_s_per_scen": s_ser["seconds_per_scenario"] if s_ser else None,
            "macbook_batched_cpu_s_per_scen": s_bcpu["seconds_per_scenario"] if s_bcpu else None,
            "colab_cpu_s_per_scen": s_ccpu["seconds_per_scenario"] if s_ccpu else (same_machine_flat_ref["colab_cpu_s_per_scen"] if same_machine_flat_ref else None),
            "colab_gpu_s_per_scen": s_bgpu["seconds_per_scenario"] if s_bgpu else None,
            "colab_gpu_batch_size": s_bgpu["batch_size"] if s_bgpu else None,
            "colab_gpu_cell_updates_per_s": s_bgpu.get("cell_updates_per_s") if s_bgpu else None,
            "gpu_from_flat_fallback": gpu_is_fallback,
            "same_machine_ratio_from_flat_grid": same_machine_from_flat,
            "same_machine_flat_ref": same_machine_flat_ref,
            "algo_speedup_macbook_cpu": (s_ser["seconds_per_scenario"] / s_bcpu["seconds_per_scenario"]) if (s_ser and s_bcpu) else None,
            "hw_speedup_same_machine_colab": same_machine_hw_speedup,
            "gpu_bandwidth_gb_s": s_bgpu.get("bandwidth_gb_s") if s_bgpu else None,
            "gpu_bandwidth_pct_peak": s_bgpu.get("bandwidth_pct_peak") if s_bgpu else None,
        },
        "scoring": {
            "E_comparison": 100,
            "loop_cpu_s_per_100ev": c_loop["seconds_per_100_events"] if c_loop else None,
            "vec_no_prefix_cpu_s_per_100ev": c_mid["seconds_per_100_events"] if c_mid else None,
            "tensor_cpu_s_per_100ev": c_tcpu["seconds_per_100_events"] if c_tcpu else None,
            "tensor_gpu_s_per_100ev": c_tgpu["seconds_per_100_events"] if c_tgpu else None,
            "vectorisation_speedup_cpu": (c_loop["seconds_per_100_events"] / c_mid["seconds_per_100_events"]) if (c_loop and c_mid) else None,
            "prefix_scan_speedup_cpu": (c_mid["seconds_per_100_events"] / c_tcpu["seconds_per_100_events"]) if (c_mid and c_tcpu) else None,
            "algo_speedup_cpu_total": (c_loop["seconds_per_100_events"] / c_tcpu["seconds_per_100_events"]) if (c_loop and c_tcpu) else None,
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

    c_meta = c_tgpu or c_tcpu or c_mid or c_loop or {}
    K_lbl = int(c_meta.get("K", 1083))
    K2_lbl = int(c_meta.get("K2", 1455))
    T_lbl = int(c_meta.get("T", 120))

    # ---- Figure 1: outputs/benchmark.png ----
    viz.apply_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.0, 5.8))

    # Panel A: Solver (LINEAR y-axis, machine labelled on every bar, labels inside axes)
    solver_bars = []
    if s_ser:
        solver_bars.append(("Serial ref\nB=1\nMacBook arm64", s_ser["seconds_per_scenario"], viz.COLORS["neutral"], s_ser))
    for r in sorted([r for r in recs if r.get("tier") == "batched_cpu"], key=lambda x: x["batch_size"]):
        solver_bars.append((f"Batched CPU\nB={r['batch_size']}\nMacBook arm64", r["seconds_per_scenario"], viz.COLORS["moderate"], r))
    if s_ccpu:
        lbl_tag = "Colab 2-vCPU\n(flat 360\u00d7750)" if s_ccpu.get("from_gpu_check_fallback") else "Colab CPU"
        solver_bars.append((f"Batched CPU\nB={s_ccpu['batch_size']}\n{lbl_tag}", s_ccpu["seconds_per_scenario"], "#8C564B", s_ccpu))
    for r in sorted([r for r in recs if r.get("tier") == "batched_gpu"], key=lambda x: x["batch_size"]):
        m_lbl = f"{r.get('machine', f'Colab {gpu_name}')}\n(flat 360\u00d7750)" if r.get("from_gpu_check_fallback") else r.get("machine", f"Colab {gpu_name}")
        solver_bars.append((f"Batched GPU\nB={r['batch_size']}\n{m_lbl}", r["seconds_per_scenario"], viz.COLORS["ships"], r))

    xs = np.arange(len(solver_bars))
    vals = [b[1] for b in solver_bars]
    cols = [b[2] for b in solver_bars]
    axA.bar(xs, vals, color=cols, edgecolor="#333333", linewidth=0.6)
    axA.set_xticks(xs)
    axA.set_xticklabels([b[0] for b in solver_bars], fontsize=7.4)
    axA.set_ylabel("Wall-clock time per 2 h scenario (s, linear scale)")
    axA.set_ylim(0.0, max(vals) * 1.42)
    grid_title = "synthetic flat 360\u00d7750 T4" if gpu_is_fallback else "real 390\u00d7795 grid, R=20,063"
    axA.set_title(f"Workload A: Shallow-water solver ({grid_title})", fontsize=10)
    axA.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, (_, v, _, rec) in enumerate(solver_bars):
        cu_m = (rec.get("cell_updates_per_s") or 0.0) / 1e6
        lbl = f"{v:.2f} s\n({cu_m:.0f}M cu/s)" if cu_m < 950 else f"{v:.2f} s\n({cu_m/1000:.2f}G cu/s)"
        axA.text(x, v + max(vals) * 0.022, lbl, ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold" if rec.get("tier") == "batched_gpu" else "normal")

    box_lines = []
    if s_bgpu and not gpu_is_fallback:
        cu_m = (s_bgpu.get("cell_updates_per_s") or 0.0) / 1e6
        bw_pct = s_bgpu.get("bandwidth_pct_peak") or 0.0
        box_lines.append(
            f"Real 390\u00d7795 absolute: Colab {gpu_name} B={s_bgpu['batch_size']} = "
            f"{s_bgpu['seconds_per_scenario']:.3f} s/scen ({cu_m:.0f}M cu/s, {bw_pct:.1f}% peak BW)"
        )
    if same_machine_hw_speedup:
        if same_machine_from_flat and same_machine_flat_ref:
            box_lines.append(
                f"Same-machine Colab test (flat 360\u00d7750, R=20k, gpu_check.json): "
                f"2-vCPU {same_machine_flat_ref['colab_cpu_s_per_scen']:.2f} s vs "
                f"T4 B={same_machine_flat_ref['colab_gpu_batch_size']} {same_machine_flat_ref['colab_gpu_s_per_scen']:.2f} s = "
                f"{same_machine_hw_speedup:.1f}\u00d7"
            )
        elif s_ccpu and s_bgpu:
            box_lines.append(
                f"Same-machine Colab speedup (real 390\u00d7795): "
                f"Colab CPU {s_ccpu['seconds_per_scenario']:.2f} s vs {gpu_name} B={s_bgpu['batch_size']} "
                f"{s_bgpu['seconds_per_scenario']:.2f} s = {same_machine_hw_speedup:.1f}\u00d7"
            )
    if box_lines:
        axA.text(0.02, 0.96, "\n".join(box_lines), transform=axA.transAxes, ha="left", va="top", fontsize=7.3,
                 bbox={"boxstyle": "round,pad=0.32", "fc": "#F0F6FC", "ec": viz.COLORS["ships"]})

    # Panel B: Scoring (consistent E=100 across ALL bars; splits vectorisation vs prefix scan + GPU)
    score_bars = []
    if c_loop:
        score_bars.append(("Python loops\n(ref, E=100)\nMacBook arm64", c_loop["seconds_per_100_events"], viz.COLORS["neutral"], c_loop))
    if c_mid:
        score_bars.append(("Vectorised NumPy\nno prefix scan (E=100)\nMacBook arm64", c_mid["seconds_per_100_events"], viz.COLORS["moderate"], c_mid))
    if c_tcpu:
        score_bars.append(("Matmul + prefix scan\n(E=100)\nMacBook arm64", c_tcpu["seconds_per_100_events"], viz.COLORS["positive"], c_tcpu))
    if c_tgpu:
        score_bars.append((f"Matmul + prefix scan\n(E=100)\n{c_tgpu.get('machine', f'Colab {gpu_name}')}", c_tgpu["seconds_per_100_events"], viz.COLORS["ships"], c_tgpu))

    xb = np.arange(len(score_bars))
    vb = [b[1] for b in score_bars]
    cb = [b[2] for b in score_bars]
    axB.bar(xb, vb, color=cb, edgecolor="#333333", linewidth=0.6, width=0.56)
    axB.set_yscale("log")
    axB.set_ylim(min(vb) * 0.30, max(vb) * 14.0)
    axB.set_xticks(xb)
    axB.set_xticklabels([b[0] for b in score_bars], fontsize=7.8)
    axB.set_ylabel("Wall-clock time for E = 100 events (s, log scale)")
    axB.set_title(f"Workload B: Matched-filter scoring (K={K_lbl}, K2={K2_lbl}, T={T_lbl} min, E=100)", fontsize=10)
    axB.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, (_, v, _, _) in enumerate(score_bars):
        axB.text(x, v * 1.28, f"{v:,.2f} s" if v < 100 else f"{v:,.0f} s", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    note_lines = []
    if c_loop and c_mid:
        note_lines.append(f"Vectorisation (loops \u2192 vec. per-min, MacBook): {summary['scoring']['vectorisation_speedup_cpu']:,.0f}\u00d7")
    if c_mid and c_tcpu:
        note_lines.append(f"Prefix scan (O(T\u00b2) \u2192 cumsum O(T), MacBook): {summary['scoring']['prefix_scan_speedup_cpu']:.1f}\u00d7")
    if c_loop and c_tcpu:
        note_lines.append(f"Combined CPU reshaping (E=100, MacBook): {summary['scoring']['algo_speedup_cpu_total']:,.0f}\u00d7")
    if c_tgpu:
        note_lines.append(
            f"Colab {gpu_name} vs MacBook CPU (cross-machine, E=100): "
            f"{summary['scoring']['hw_speedup_gpu_over_tensor_cpu']:.1f}\u00d7 "
            f"({summary['scoring']['total_speedup_gpu_over_loop']:,.0f}\u00d7 vs loops)"
        )
    else:
        note_lines.append("GPU scoring tier: pending Colab --gpu run")
    axB.text(0.97, 0.96, "\n".join(note_lines), transform=axB.transAxes, ha="right", va="top", fontsize=7.5,
             bbox={"boxstyle": "round,pad=0.32", "fc": "#F0F6FC", "ec": viz.COLORS["ships"]})

    viz.save_figure(
        fig,
        out_dir / "benchmark.png",
        "Three-Tier Performance Benchmark: Solver (Workload A) and Scoring (Workload B)",
        "Same NumPy/CuPy float32 codebase on real 390x795 grid (R=20,063); same-machine Colab 17.1x ratio is from flat 360x750 gpu_check.json. Published context: Tsunami-HySEA simulates 8 h Med Sea (10M volumes, 30 arc-sec) in 284 s on 1xV100 or 257 s on 2xP100 (Estrada et al. 2021, Sci. Rep. 11:16253, doi:10.1038/s41598-021-95729-6, Methods: Tsunami model).",
    )

    # ---- Figure 2: outputs/what_speed_buys.png ----
    budget_s = 3600.0
    n_strata = 12
    n_event_strata = K_lbl * n_strata
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.5))

    # Panel A: Scenarios simulated in 1 hour (LINEAR axis, real 390x795 GPU bars!)
    s_bars = []
    if s_ccpu:
        tag = "\n(flat 360\u00d7750)" if s_ccpu.get("from_gpu_check_fallback") else ""
        s_bars.append((f"Colab 2-vCPU\nNumPy B={s_ccpu['batch_size']}{tag}", budget_s / s_ccpu["seconds_per_scenario"], "#8C564B"))
    if s_ser:
        s_bars.append(("MacBook arm64\nSerial ref B=1\n(real 390\u00d7795)", budget_s / s_ser["seconds_per_scenario"], viz.COLORS["neutral"]))
    if s_bcpu:
        s_bars.append((f"MacBook arm64\nBatched B={s_bcpu['batch_size']}\n(real 390\u00d7795)", budget_s / s_bcpu["seconds_per_scenario"], viz.COLORS["moderate"]))
    gpu_solver_recs = sorted([r for r in recs if r.get("tier") == "batched_gpu"], key=lambda x: x["batch_size"])
    for idx_g, rg in enumerate(gpu_solver_recs):
        tag = "\n(flat 360\u00d7750)" if rg.get("from_gpu_check_fallback") else "\n(real 390\u00d7795)"
        col_g = "#74ADD1" if idx_g < len(gpu_solver_recs) - 1 else viz.COLORS["ships"]
        s_bars.append((f"Colab {gpu_name}\nCuPy B={rg['batch_size']}{tag}", budget_s / rg["seconds_per_scenario"], col_g))

    x1 = np.arange(len(s_bars))
    y1 = [b[1] for b in s_bars]
    ax1.bar(x1, y1, color=[b[2] for b in s_bars], edgecolor="#333333", linewidth=0.6, width=0.58)
    ax1.axhline(K_lbl, color=viz.COLORS["negative"], ls="--", lw=1.2, label=f"Full library ({K_lbl:,} candidates)")
    ax1.axhline(2 * K_lbl + 60, color=viz.COLORS["truth"], ls=":", lw=1.2, label=f"Library + truth + historic (~{2*K_lbl+60:,} runs)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([b[0] for b in s_bars], fontsize=7.6)
    ax1.set_ylabel("2-hour scenarios simulated per hour (linear scale)")
    ax1.set_ylim(0.0, max(y1) * 1.24)
    ax1.set_title("A. Scenarios simulated in a 1-hour compute budget (real 390\u00d7795)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for x, v in zip(x1, y1):
        ax1.text(x, v + max(y1) * 0.022, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    # Panel B: Scoring draws per truth event per stratum in 1 hour (once library exists)
    n_grid = np.logspace(0.0, 3.6, 250)
    p = 0.90
    half_ci_95 = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / n_grid)
    ax2.plot(n_grid, half_ci_95, color="#333333", lw=1.8, label="95% CI half-width on p = 0.90 threshold")
    ax2.set_xscale("log")
    ax2.set_xlim(0.85, 4500.0)
    ax2.set_ylim(0.0, 65.0)

    def _draws_in_1h(sec_per_100):
        t_per_ev = sec_per_100 / 100.0
        return budget_s / (n_event_strata * t_per_ev)

    if c_loop:
        n_loop = _draws_in_1h(c_loop["seconds_per_100_events"])
        hrs_1draw = (n_event_strata * (c_loop["seconds_per_100_events"] / 100.0)) / 3600.0
        ax2.scatter([1.0], [58.8], marker="<", s=95, color=viz.COLORS["neutral"], edgecolors="k", zorder=5,
                    label=f"Python loops (MacBook): {n_loop:.4f} draws/h (<1; ~{hrs_1draw:.0f} h/draw)")
    if c_mid:
        n_mid = _draws_in_1h(c_mid["seconds_per_100_events"])
        if n_mid >= 1.0:
            ci_mid = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / n_mid)
            ax2.scatter([n_mid], [ci_mid], s=85, color=viz.COLORS["moderate"], edgecolors="k", zorder=5,
                        label=f"Vec. NumPy no-prefix (MacBook, E=100): {n_mid:.1f} draws (\u00b1{ci_mid:.1f}%)")
        else:
            hrs_mid = (n_event_strata * (c_mid["seconds_per_100_events"] / 100.0)) / 3600.0
            ax2.scatter([1.0], [52.0], marker="<", s=85, color=viz.COLORS["moderate"], edgecolors="k", zorder=5,
                        label=f"Vec. NumPy no-prefix (MacBook, E=100): {n_mid:.2f} draws/h ({hrs_mid:.1f} h/draw)")
    if c_tcpu:
        n_tcpu = _draws_in_1h(c_tcpu["seconds_per_100_events"])
        ci_tcpu = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / max(n_tcpu, 1.0))
        ax2.scatter([max(n_tcpu, 1.0)], [ci_tcpu], s=95, color=viz.COLORS["positive"], edgecolors="k", zorder=5,
                    label=f"Matmul+prefix (MacBook, E=100): {n_tcpu:.0f} draws (\u00b1{ci_tcpu:.1f}%)")
    if c_tgpu:
        n_tgpu = _draws_in_1h(c_tgpu["seconds_per_100_events"])
        x_gpu = min(n_tgpu, 4000.0)
        ci_gpu = 100.0 * 1.96 * np.sqrt(p * (1.0 - p) / n_tgpu)
        ax2.scatter([x_gpu], [ci_gpu], s=110, color=viz.COLORS["ships"], edgecolors="k", zorder=6,
                    label=f"Matmul+prefix (Colab {gpu_name}, E=100): {n_tgpu:,.0f} draws (\u00b1{ci_gpu:.1f}%)")

    ax2.axvline(100, color=viz.COLORS["ships"], ls="--", lw=1.1, label="Project target (5 traffic \u00d7 20 noise = 100)")

    calc_lines = [
        "Once library exists, extra draws pay scoring only:",
        f"draws/stratum = 3600 s / ({K_lbl:,} events \u00d7 {n_strata} strata \u00d7 t_ev)",
    ]
    if c_tcpu:
        t_ev_ms = (c_tcpu["seconds_per_100_events"] / 100.0) * 1000.0
        s_per_draw = n_event_strata * (c_tcpu["seconds_per_100_events"] / 100.0)
        calc_lines.append(f"MacBook CPU (E=100): t_ev={t_ev_ms:.1f} ms \u2192 {s_per_draw:.0f} s/draw \u2192 {budget_s/s_per_draw:.0f} draws/h")
    if c_tgpu:
        t_ev_ms_g = (c_tgpu["seconds_per_100_events"] / 100.0) * 1000.0
        s_per_draw_g = n_event_strata * (c_tgpu["seconds_per_100_events"] / 100.0)
        calc_lines.append(f"Colab {gpu_name} (E=100): t_ev={t_ev_ms_g:.2f} ms \u2192 {s_per_draw_g:.0f} s/draw \u2192 {budget_s/s_per_draw_g:,.0f} draws/h")
    else:
        calc_lines.append("GPU scoring point: awaiting B's --gpu run")

    ax2.text(0.98, 0.49, "\n".join(calc_lines), transform=ax2.transAxes, ha="right", va="top", fontsize=7.2,
             bbox={"boxstyle": "round,pad=0.3", "fc": "#FFFDF5", "ec": "#999999"})
    ax2.set_xlabel("Traffic \u00d7 noise draws per truth event per stratum in 1 h (log scale)")
    ax2.set_ylabel("95% binomial CI half-width on 90% detection rate (%)")
    ax2.set_title("B. What scoring speed buys: Map 2 precision in 1 hour")
    ax2.legend(loc="upper right", fontsize=7.2, framealpha=0.92)
    ax2.grid(True, linestyle=":", alpha=0.6)

    cap_left = (
        f"Left: linear scenarios/h on real 390x795 grid (Colab {gpu_name} B={s_bgpu['batch_size']} = {budget_s / s_bgpu['seconds_per_scenario']:,.0f}/h at {s_bgpu['seconds_per_scenario']:.3f} s/scen vs MacBook CPU {budget_s / s_bcpu['seconds_per_scenario']:,.0f}/h)."
        if (s_bgpu and s_bcpu and not gpu_is_fallback)
        else f"Computed from measured wall-clock rates for {K_lbl:,} library candidates and 12 strata."
    )
    cap_right = (
        f" Right: draws/stratum = 3600 / ({K_lbl:,} x 12 x t_ev) at E=100 once library is built — loops <1/h, MacBook CPU { _draws_in_1h(c_tcpu['seconds_per_100_events']):.0f}/h, Colab {gpu_name} {_draws_in_1h(c_tgpu['seconds_per_100_events']):,.0f}/h (exceeds 100-draw target)."
        if (c_tcpu and c_tgpu)
        else ""
    )
    viz.save_figure(
        fig2,
        out_dir / "what_speed_buys.png",
        "What the Speed Makes Possible: Experiment Scale and Map 2 Uncertainty in a 1-Hour Budget",
        cap_left + cap_right,
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
