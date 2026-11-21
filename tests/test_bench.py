"""Tests for T08 benchmark harness (src/tsunami/bench.py)."""
import time

import numpy as np

from tsunami import bench


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
