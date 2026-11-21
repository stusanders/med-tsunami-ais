"""Tests for T08 benchmark harness (src/tsunami/bench.py) and T19 plotting."""
import importlib.util
import time
from pathlib import Path

import numpy as np

from tsunami import bench, scoring
from tsunami.config import load_config, repo_path


def _load_bench_script():
    script_path = repo_path("scripts/06_benchmark.py")
    spec = importlib.util.spec_from_file_location("bench_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_timer_syncs_and_measures_elapsed():
    calls = []

    def fake_sync():
        calls.append(time.perf_counter())

    with bench.Timer("sleep_test", sync=fake_sync) as t:
        time.sleep(0.015)

    assert len(calls) == 2
    assert t.name == "sleep_test"
    assert t.seconds is not None and t.seconds >= 0.01


def test_record_metrics_and_bandwidth():
    rec = bench.record(
        "solver_gpu",
        seconds=2.0,
        cell_updates=1_000_000_000,
        bytes_moved=320_000_000_000,
        peak_bandwidth_gb_s=320.0,
        batch_size=np.int64(64),
        grid_shape=(390, 795),
    )
    assert rec["name"] == "solver_gpu"
    assert np.isclose(rec["cell_updates_per_s"], 5e8)
    assert np.isclose(rec["bandwidth_gb_s"], 160.0)
    assert np.isclose(rec["bandwidth_pct_peak"], 50.0)
    assert rec["batch_size"] == 64
    assert rec["grid_shape"] == [390, 795]


def test_write_and_read_json(tmp_path):
    recs = [
        bench.record("a", 0.5, cell_updates=1000),
        bench.record("b", 0.25, bytes_moved=500_000_000),
    ]
    out = bench.write_json(recs, tmp_path / "sub" / "bench.json")
    assert out.exists()
    loaded = bench.read_json(out)
    assert len(loaded) == 2
    assert loaded[0]["name"] == "a"
    assert np.isclose(loaded[1]["bandwidth_gb_s"], 2.0)


def test_device_info_fields():
    info = bench.device_info()
    assert info["backend"] in ("numpy", "cupy")
    assert isinstance(info["gpu"], bool)
    assert "platform" in info


def test_score_vectorised_no_prefix_matches_cumulative_stats():
    mod = _load_bench_script()
    rng = np.random.default_rng(42)
    K, M, T, E = 12, 18, 25, 6
    G = rng.normal(0.0, 0.05, size=(K, M, T)).astype(np.float32)
    D = rng.normal(0.0, 0.10, size=(E, M, T)).astype(np.float32)
    w = rng.uniform(0.5, 2.0, size=(1, M, T)).astype(np.float32)
    flip = np.array([i % 3 == 0 for i in range(K)], dtype=bool)
    eps = 1e-6

    stat_mid = mod.score_vectorised_no_prefix(G, D, w, flip, eps)
    gd, gg = scoring.cumulative_stats(G, D, w, xp=np)
    gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip)
    stat_ref = scoring.detection_statistic(scoring.detection_scores(gd2, gg2, eps))
    assert np.allclose(stat_mid, stat_ref, rtol=1e-4, atol=1e-5)


def test_combine_and_plot_cpu_plus_gpu(tmp_path):
    mod = _load_bench_script()

    cpu_payload = {
        "device": {"backend": "numpy", "gpu": False, "platform": "test-cpu"},
        "records": [
            bench.record(
                "solver_serial_cpu", 4.0, cell_updates=3.7e8, bytes_moved=2.0e10,
                workload="solver", tier="serial_cpu", machine="MacBook arm64", batch_size=1,
                seconds_per_scenario=4.0, ny=390, nx=795, n_steps=1200, n_receivers=20061,
            ),
            bench.record(
                "solver_batched_cpu_B1", 2.0, cell_updates=3.7e8, bytes_moved=2.0e10,
                workload="solver", tier="batched_cpu", machine="MacBook arm64", batch_size=1,
                seconds_per_scenario=2.0, ny=390, nx=795, n_steps=1200, n_receivers=20061,
            ),
            bench.record(
                "scoring_loop_cpu", 10000.0,
                workload="scoring", tier="loop_cpu", machine="MacBook arm64",
                E=100, K=1083, K2=1455, M=615, T=120,
                seconds_per_100_events=10000.0, direct_slice_speedup=40.0,
            ),
            bench.record(
                "scoring_vec_no_prefix_cpu_E100", 20.0,
                workload="scoring", tier="vec_no_prefix_cpu", machine="MacBook arm64",
                E=100, K=1083, K2=1455, M=615, T=120,
                seconds_per_100_events=20.0,
            ),
            bench.record(
                "scoring_tensor_cpu_E100", 1.0, bytes_moved=5.4e8,
                workload="scoring", tier="tensor_cpu", machine="MacBook arm64",
                E=100, K=1083, K2=1455, M=615, T=120,
                seconds_per_100_events=1.0,
            ),
        ],
    }
    gpu_payload = {
        "device": {"backend": "cupy", "gpu": True, "device_name": "Tesla T4", "peak_bandwidth_gb_s": 320.0},
        "records": [
            bench.record(
                "solver_colab_cpu_B1", 5.0, cell_updates=3.7e8, bytes_moved=2.0e10,
                workload="solver", tier="colab_cpu", machine="Colab 2-vCPU", batch_size=1,
                seconds_per_scenario=5.0, ny=390, nx=795, n_steps=1200, n_receivers=20061,
            ),
            bench.record(
                "solver_batched_gpu_B64", 16.0, cell_updates=64 * 3.7e8, bytes_moved=64 * 2.0e10,
                peak_bandwidth_gb_s=320.0, workload="solver", tier="batched_gpu", machine="Colab Tesla T4",
                batch_size=64, seconds_per_scenario=0.25, ny=390, nx=795, n_steps=1200, n_receivers=20061,
            ),
            bench.record(
                "scoring_tensor_gpu_E100", 0.05, bytes_moved=5.4e8, peak_bandwidth_gb_s=320.0,
                workload="scoring", tier="tensor_gpu", machine="Colab Tesla T4",
                E=100, K=1083, K2=1455, M=615, T=120,
                seconds_per_100_events=0.05,
            ),
        ],
    }
    bench.write_json(cpu_payload, tmp_path / "benchmark_cpu.json")
    bench.write_json(gpu_payload, tmp_path / "benchmark_gpu.json")

    combined = mod.combine_and_plot(load_config(), out_dir=Path(tmp_path))
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "benchmark.png").exists()
    assert (tmp_path / "what_speed_buys.png").exists()
    assert np.isclose(combined["summary"]["solver"]["algo_speedup_macbook_cpu"], 2.0)
    assert np.isclose(combined["summary"]["solver"]["hw_speedup_same_machine_colab"], 20.0)
    assert np.isclose(combined["summary"]["scoring"]["vectorisation_speedup_cpu"], 500.0)
    assert np.isclose(combined["summary"]["scoring"]["prefix_scan_speedup_cpu"], 20.0)
    assert np.isclose(combined["summary"]["scoring"]["algo_speedup_cpu_total"], 10000.0)
    assert np.isclose(combined["summary"]["scoring"]["hw_speedup_gpu_over_tensor_cpu"], 20.0)
    assert np.isclose(combined["summary"]["scoring"]["total_speedup_gpu_over_loop"], 200000.0)
