import numpy as np
import pandas as pd

from tsunami import library_build as LB
from tsunami import solver_reference as ref
from tsunami.config import load_config
from tsunami.solver import Solver
from tsunami.synthetic import BUOY, GAUGE, SHIP, synthetic_grid, synthetic_pool

CFG0 = load_config()
CFG = {**CFG0, "solver": {**CFG0["solver"], "t_end_s": 1800.0}, "library": {**CFG0["library"], "n_minutes": 30}}


def case():
    ny, nx = 40, 56
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 2500.0 - 30.0 * ii + 300.0 * np.sin(jj / 5.0)
    depth[:, 52:] = -10.0
    grid = synthetic_grid(depth, lat0_deg=34.0)
    rng = np.random.default_rng(1)
    n = 12
    pool = synthetic_pool(grid, np.r_[rng.uniform(grid.lon[3], grid.lon[45], n), grid.lon[[50, 8]]],
                          np.r_[rng.uniform(grid.lat[3], grid.lat[35], n), grid.lat[[25, 6]]], [SHIP] * n + [GAUGE, BUOY])
    solver = Solver(grid, ref.make_sampler(grid, pool), CFG, xp=np)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    table = pd.DataFrame({"run_id": 100 + np.arange(7), "cx": [10, 15, 20, 25, 30, 35, 22], "cy": [10, 20, 15, 25, 12, 28, 18],
                          "amp_scale": [1, 1, 1, 1, 1, 2.0, 1]})

    def source_fn(g, row, cfg, kind=None):
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        a = 1.0 if kind != "monopole" else 0.5
        return (a * np.exp(-(((ii - row["cx"]) / 4.0) ** 2 + ((jj - row["cy"]) / 4.0) ** 2))).astype(np.float32)

    return grid, solver, gb_idx, table, source_fn


def run(out, max_chunks=None, batch=3):
    grid, solver, gb_idx, table, fn = case()
    return LB.build_stage("t", table, out, solver, grid, CFG, gb_idx, fn, batch=batch, max_chunks=max_chunks), (grid, solver, gb_idx, table, fn)


def load_all(out, n_chunks):
    return [np.load(out / f"rec_{c:04d}.npy") for c in range(n_chunks)], [np.load(out / f"gb_{c:04d}.npy") for c in range(n_chunks)]


def test_chunks_match_direct_solver_and_index(tmp_path):
    n, (grid, solver, gb_idx, table, fn) = run(tmp_path)
    assert n == 3                                           # 7 runs in chunks of 3: 3, 3, 1
    idx = pd.read_csv(tmp_path / "index.csv")
    assert idx[["chunk", "row"]].values.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0]]
    rec, gb = load_all(tmp_path, 3)
    assert rec[0].shape == (3, 30, 14, 3) and gb[2].shape == (1, 60, 2) and rec[0].dtype == np.float32
    eta0 = LB.source_eta0(table, np.arange(3, 6), grid, CFG, fn)
    assert np.allclose(eta0[2], 2.0 * fn(grid, table.iloc[5].to_dict(), CFG))          # amp_scale applied
    want_rec, want_gb = solver.run_library(eta0, gb_idx)
    np.testing.assert_allclose(rec[1], want_rec, atol=1e-7)
    np.testing.assert_allclose(gb[1], want_gb, atol=1e-7)
    log = LB.read_log(tmp_path)
    assert log["chunk"].tolist() == [0, 1, 2] and (log["gpu"] == "cpu").all() and (log["wall_s"] > 0).all()
    n_ok, bad = LB.verify_stage(tmp_path, 14, 2, CFG, 60)
    assert n_ok == 6 and not bad


def test_resume_after_interruption_is_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(a)
    n1, _ = run(b, max_chunks=1)                            # "killed" after one chunk
    assert n1 == 1
    (b / "rec_0009.npy.tmp").write_bytes(b"junk")           # stale temporary file from the kill
    n2, _ = run(b)
    assert n2 == 2 and not list(b.glob("*.tmp"))
    n3, _ = run(b)
    assert n3 == 0                                          # nothing left to do
    for c in range(3):
        assert np.array_equal(np.load(a / f"rec_{c:04d}.npy"), np.load(b / f"rec_{c:04d}.npy"))
        assert np.array_equal(np.load(a / f"gb_{c:04d}.npy"), np.load(b / f"gb_{c:04d}.npy"))
    assert LB.read_log(b)["chunk"].tolist() == [0, 1, 2]


def test_truncated_chunk_is_rebuilt(tmp_path):
    run(tmp_path)
    good = np.load(tmp_path / "rec_0001.npy")
    p = tmp_path / "rec_0001.npy"
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    n, _ = run(tmp_path)
    assert n == 1
    assert np.array_equal(np.load(p), good)


def test_checksum_detects_corruption(tmp_path):
    run(tmp_path)
    p = tmp_path / "gb_0000.npy"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0xFF
    p.write_bytes(bytes(raw))
    n_ok, bad = LB.verify_stage(tmp_path, 14, 2, CFG, 60)
    assert bad == ["gb_0000.npy"]


def test_kind_column_reaches_source_fn(tmp_path):
    grid, solver, gb_idx, table, fn = case()
    t = table.iloc[:2].copy()
    t["kind"] = ["dipole", "monopole"]
    t["amp_scale"] = 1.0
    eta0 = LB.source_eta0(t, np.arange(2), grid, CFG, fn)
    assert np.isclose(eta0[0].max(), 1.0, atol=0.05) and np.isclose(eta0[1].max(), 0.5, atol=0.05)


def test_gpu_sub_batches_give_identical_chunks(tmp_path):
    grid, solver, gb_idx, table, fn = case()
    a, b = tmp_path / "a", tmp_path / "b"
    LB.build_stage("t", table, a, solver, grid, CFG, gb_idx, fn, batch=4)
    LB.build_stage("t", table, b, solver, grid, CFG, gb_idx, fn, batch=4, gpu_batch=1)
    for c in range(2):
        assert np.array_equal(np.load(a / f"rec_{c:04d}.npy"), np.load(b / f"rec_{c:04d}.npy"))
