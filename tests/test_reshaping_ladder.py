"""Tests for What Reshaping Saves ladder (scripts/fig_reshaping_ladder.py)."""
import importlib.util
from pathlib import Path

import numpy as np

from tsunami import bench
from tsunami.config import load_config, repo_path


def _load_ladder_module():
    path = repo_path("scripts/fig_reshaping_ladder.py")
    spec = importlib.util.spec_from_file_location("reshaping_ladder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_experiment_counts_matches_real_files():
    import copy
    mod = _load_ladder_module()
    cfg = load_config()
    counts = mod.load_experiment_counts(cfg)
    sw_tr = int(cfg["sweep"]["n_traffic_draws"])
    sw_nz = int(cfg["sweep"]["n_noise_draws"])
    assert counts["K"] == 1083
    assert counts["K2"] == 1455
    assert counts["n_truth_map2"] == 1083
    assert counts["n_truth_twin"] == 80
    assert counts["n_truth"] == 1163
    assert counts["n_historic"] == 2
    assert counts["n_library_build"] == 2248
    assert counts["n_strata"] == 12
    assert counts["n_traffic_draws"] == sw_tr
    assert counts["n_noise_per_traffic"] == sw_nz
    assert counts["n_draws_per_stratum"] == sw_tr * sw_nz == 100
    assert counts["n_trials"] == counts["n_truth"] * counts["n_strata"] * sw_tr * sw_nz == 1_395_600
    assert counts["n_agg_calls"] == sw_tr * counts["n_range_caps"] * counts["n_operators"] == 20
    # Changing scoring.n_quiet_traffic_draws must not alter sweep trial or gather counts
    cfg2 = copy.deepcopy(cfg)
    cfg2["scoring"]["n_quiet_traffic_draws"] = 50
    counts2 = mod.load_experiment_counts(cfg2)
    assert counts2["n_trials"] == counts["n_trials"]
    assert counts2["n_agg_calls"] == counts["n_agg_calls"]
    assert counts2["n_quiet_traffic_draws"] == 50


def test_format_duration_human_units():
    mod = _load_ladder_module()
    assert mod.format_duration(45.0).endswith("s")
    assert mod.format_duration(2700.0).endswith("min")
    assert mod.format_duration(5.4 * 3600.0).endswith("h")
    assert mod.format_duration(200.0 * 86400.0).endswith("days")
    assert mod.format_duration(137.0 * 365.25 * 86400.0).endswith("years")


def test_build_reshaping_ladder_end_to_end(tmp_path):
    mod = _load_ladder_module()
    cpu_payload = {
        "device": {"backend": "numpy", "gpu": False, "platform": "test-cpu"},
        "records": [
            bench.record("solver_serial_cpu", 2.85, workload="solver", tier="serial_cpu", seconds_per_scenario=2.85),
            bench.record("scoring_loop_cpu", 5623.0, workload="scoring", tier="loop_cpu", E=100, seconds_per_100_events=5623.0),
            bench.record("scoring_vec_no_prefix_cpu_E100", 11.54, workload="scoring", tier="vec_no_prefix_cpu", E=100, seconds_per_100_events=11.54),
            bench.record("scoring_tensor_cpu_E100", 0.934, workload="scoring", tier="tensor_cpu", E=100, seconds_per_100_events=0.934),
        ],
    }
    gpu_payload = {
        "device": {"backend": "cupy", "gpu": True, "device_name": "Tesla T4"},
        "records": [
            bench.record("solver_batched_gpu_B32", 13.38, workload="solver", tier="batched_gpu", batch_size=32, seconds_per_scenario=0.418),
            bench.record("scoring_tensor_gpu_E100", 0.132, workload="scoring", tier="tensor_gpu", E=100, seconds_per_100_events=0.132),
        ],
    }
    bench.write_json(cpu_payload, tmp_path / "benchmark_cpu.json")
    bench.write_json(gpu_payload, tmp_path / "benchmark_gpu.json")

    fake_agg = {
        "method": "mock",
        "machine": "MacBook arm64",
        "K": 1083,
        "R": 20063,
        "T": 120,
        "n_calls_total": 20,
        "mean_s_per_call": 3.0,
        "one_traffic_draw_all_caps_ops_s": 12.0,
        "total_experiment_s": 60.0,
        "breakdown": {},
    }

    payload = mod.build_reshaping_ladder(load_config(), out_dir=Path(tmp_path), agg_measurement=fake_agg)
    assert (tmp_path / "reshaping_ladder.json").exists()
    assert (tmp_path / "reshaping_ladder.png").exists()

    rows = payload["rows"]
    assert len(rows) == 6
    totals = [r["total_s"] for r in rows]
    assert all(totals[i] > totals[i + 1] for i in range(len(totals) - 1))
    for r in rows:
        for k in ("formula", "total_s", "human_total", "status", "speedup_vs_naive"):
            assert k in r
    # Rows 3-6 include the template gather/aggregation cost
    for r in rows[2:]:
        assert np.isclose(r["agg_s"], 60.0)
    # Every per-unit timing records its source file
    for entry in payload["per_unit_timings"].values():
        assert "source_file" in entry and entry["source_file"]
