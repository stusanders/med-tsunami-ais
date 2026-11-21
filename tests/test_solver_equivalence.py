"""T09: batched solver equals the reference per scenario; linearity; library path."""
import numpy as np
import pytest

from tsunami import backend
from tsunami import solver_reference as ref
from tsunami.config import load_config
from tsunami.solver import Solver, to_library
from tsunami.synthetic import BUOY, GAUGE, SHIP, synthetic_grid, synthetic_pool

CFG = load_config()


def small_case(t_end_s=1800.0):
    ny, nx = 48, 64
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 3000.0 - 40.0 * ii + 500.0 * np.sin(jj / 6.0)
    depth[:, 58:] = -10.0                                   # coast on the east
    depth[20:24, 30:34] = -5.0                              # island
    grid = synthetic_grid(depth, lat0_deg=34.0)
    rng = np.random.default_rng(0)
    n_ship = 30
    lon = rng.uniform(grid.lon[2], grid.lon[50], n_ship)
    lat = rng.uniform(grid.lat[2], grid.lat[45], n_ship)
    pool = synthetic_pool(grid, np.r_[lon, grid.lon[[55, 10]]], np.r_[lat, grid.lat[[30, 5]]],
                          [SHIP] * n_ship + [GAUGE, BUOY])
    sources = []
    for (ci, cj, s, a) in [(15, 20, 4.0, 1.0), (40, 10, 3.0, -0.7), (25, 38, 5.0, 0.4)]:
        sources.append(a * np.exp(-(((ii - ci) / s) ** 2 + ((jj - cj) / s) ** 2)))
    cfg = {**CFG, "solver": {**CFG["solver"], "t_end_s": t_end_s}}
    return grid, pool, np.stack(sources).astype(np.float32), cfg


def test_batched_equals_reference():
    grid, pool, eta0, cfg = small_case()
    sampler = ref.make_sampler(grid, pool)
    batched = Solver(grid, sampler, cfg, xp=np).run(eta0)
    for b in range(eta0.shape[0]):
        single = ref.run_reference(grid, eta0[b], sampler, cfg)
        scale = np.abs(single).max(axis=(0, 1))             # per channel
        np.testing.assert_allclose(batched[b], single, atol=1e-5 * scale.max(), rtol=0)
        assert np.abs(batched[b] - single).max(axis=(0, 1)).max() <= 1e-5 * scale.max()


def test_batched_superposition():
    grid, pool, eta0, cfg = small_case()
    s = Solver(grid, ref.make_sampler(grid, pool), cfg, xp=np)
    out = s.run(np.stack([eta0[0], eta0[1], eta0[0] + eta0[1], 2 * eta0[2]]))
    single2 = s.run(eta0[2:3])[0]
    scale = np.abs(out).max()
    np.testing.assert_allclose(out[2], out[0] + out[1], atol=1e-5 * scale)
    np.testing.assert_allclose(out[3], 2 * single2, atol=1e-5 * scale)


def test_run_library_matches_to_library():
    grid, pool, eta0, cfg = small_case(t_end_s=CFG["solver"]["t_end_s"])
    s = Solver(grid, ref.make_sampler(grid, pool), cfg, xp=np)
    gb_idx = np.where(pool.rtype > 0)[0]
    recf_a, gb_a = to_library(s.run(eta0[:2]), gb_idx, cfg)
    recf_b, gb_b = s.run_library(eta0[:2], gb_idx)
    assert recf_a.shape == (2, CFG["library"]["n_minutes"], len(pool.lon), 3)
    assert gb_a.shape == (2, 240, 2)
    np.testing.assert_allclose(recf_b, recf_a, atol=1e-6 * np.abs(recf_a).max())
    np.testing.assert_array_equal(gb_b, gb_a)


def test_snapshots_and_shapes():
    grid, pool, eta0, cfg = small_case()
    s = Solver(grid, ref.make_sampler(grid, pool), cfg, xp=np)
    snap = s.snapshots(eta0[:1], [0.0, 600.0, 1200.0])
    assert snap.shape == (1, 3) + grid.depth.shape
    np.testing.assert_allclose(snap[0, 0], eta0[0] * grid.wet)
    assert np.all(snap[:, :, ~grid.wet] == 0)
    assert s.n_steps == s.n_samples * s.n_sub


@pytest.mark.skipif(not backend.GPU, reason="CuPy/GPU not available")
def test_cupy_equals_numpy():
    grid, pool, eta0, cfg = small_case()
    sampler = ref.make_sampler(grid, pool)
    cpu = Solver(grid, sampler, cfg, xp=np).run(eta0)
    gpu = backend.to_numpy(Solver(grid, sampler, cfg, xp=backend.xp).run(eta0))
    np.testing.assert_allclose(gpu, cpu, atol=1e-5 * np.abs(cpu).max())
