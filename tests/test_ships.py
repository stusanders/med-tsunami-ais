import numpy as np
import pytest

from tsunami import ships
from tsunami.config import load_config, repo_path
from tsunami.geometry import fold_axis
from tsunami.synthetic import synthetic_grid, synthetic_pool
from tsunami.units import knots_to_ms

cfg = load_config()


def small_pool(n_areas=6, per_area=40, seed=0):
    """Hand-made ship pool: n_areas traffic areas with known expected counts."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(0.5, 12.0, n_areas)
    aid = np.repeat(np.arange(n_areas), per_area)
    R = len(aid)
    w = np.repeat(lam / per_area, per_area).astype(np.float32)
    return ships.ReceiverPool(
        lon=rng.uniform(15, 20, R), lat=rng.uniform(33, 37, R), depth=np.full(R, 500, np.float32),
        lane_dir=rng.uniform(0, np.pi, R).astype(np.float32), rtype=np.zeros(R, np.int8), code=np.full(R, "", "<U24"),
        cell_j=np.zeros(R, np.int32), cell_i=np.zeros(R, np.int32), coast_dist=rng.uniform(5e3, 1e5, R).astype(np.float32),
        weight=w, area_id=aid.astype(np.int32)), lam


def test_draw_counts_match_poisson_expectation():
    pool, lam = small_pool()
    N = 600
    counts = np.zeros(len(lam))
    for k in range(N):
        d = ships.traffic_draw(pool, cfg, k)
        counts += np.bincount(pool.area_id[d.pool_idx], minlength=len(lam))
    mean = counts / N
    assert np.all(np.abs(mean - lam) < 4 * np.sqrt(lam / N))          # 4 standard errors of a Poisson mean


def test_draw_properties():
    pool, _ = small_pool()
    d = ships.traffic_draw(pool, cfg, 3)
    assert len(np.unique(d.pool_idx)) == len(d.pool_idx)                # no repeats
    assert (pool.rtype[d.pool_idx] == ships.SHIP).all()
    assert d.sog.dtype == np.float32 and d.heading.dtype == np.float32
    assert (d.sog >= knots_to_ms(cfg["traffic"]["min_speed_kn"])).all()
    assert (d.heading >= 0).all() and (d.heading < 2 * np.pi).all()
    along = fold_axis(d.heading.astype(np.float64)) - pool.lane_dir[d.pool_idx]
    assert np.allclose(np.mod(along + 1e-6, np.pi), 1e-6, atol=1e-4)    # heading is lane_dir or lane_dir + pi


def test_both_travel_directions_equally_likely():
    pool, _ = small_pool(per_area=400)
    fw = n = 0
    for k in range(50):
        d = ships.traffic_draw(pool, cfg, k)
        fw += (np.abs(d.heading - pool.lane_dir[d.pool_idx]) < 1e-4).sum()
        n += len(d.pool_idx)
    assert abs(fw / n - 0.5) < 4 * np.sqrt(0.25 / n)


def test_seeded_and_reproducible():
    pool, _ = small_pool()
    a, b, c = ships.traffic_draw(pool, cfg, 7), ships.traffic_draw(pool, cfg, 7), ships.traffic_draw(pool, cfg, 8)
    assert np.array_equal(a.pool_idx, b.pool_idx) and np.array_equal(a.heading, b.heading) and np.array_equal(a.sog, b.sog)
    assert not np.array_equal(a.pool_idx, c.pool_idx)
    assert a.seed == 7


def test_range_cap_gives_nested_fleets():
    pool, _ = small_pool()
    d100 = ships.apply_range_cap(ships.traffic_draw(pool, cfg, 1), pool, 1e5)
    d40 = ships.apply_range_cap(d100, pool, 4e4)
    assert set(d40.pool_idx) <= set(d100.pool_idx)
    assert (pool.coast_dist[d40.pool_idx] <= 4e4).all() and len(d40.pool_idx) < len(d100.pool_idx)


def test_count_is_capped_at_points_in_area():
    pool, _ = small_pool(n_areas=1, per_area=3)
    pool.weight[:] = 50.0                                               # expected 150 ships in an area with 3 points
    d = ships.traffic_draw(pool, cfg, 0)
    assert len(d.pool_idx) == 3


def test_gauges_in_pool_after_ships_snapped_to_wet_cells():
    depth = np.full((20, 20), 300.0)
    depth[:, :3] = 0
    g = synthetic_grid(depth)
    p = synthetic_pool(g, [20.5, 20.2, 20.3], [35.2, 35.2, 35.3], [0, 1, 2])
    assert list(p.rtype) == [0, 1, 2] and np.isnan(p.lane_dir[1:]).all() and (p.area_id[1:] == -1).all()


have_pool = repo_path(cfg["paths"]["receiver_pool"]).exists() and repo_path(cfg["paths"]["grid"]).exists()


@pytest.mark.skipif(not have_pool, reason="pool not built")
def test_real_pool_contract():
    from tsunami.grid import load_grid
    g, pool = load_grid(cfg), ships.load_pool(cfg)
    t = cfg["traffic"]
    S = pool.n_ships
    assert abs(S - t["pool_size"]) <= 1
    assert (pool.rtype[:S] == ships.SHIP).all() and (pool.rtype[S:] > 0).all()
    assert (np.diff(pool.rtype[S:]) >= 0).all()                         # gauges then buoys
    s = slice(0, S)
    assert pool.depth[s].min() >= t["min_ship_depth_m"] and pool.coast_dist[s].max() <= t["pool_max_coast_dist_m"]
    assert not g.strait[pool.cell_j[s], pool.cell_i[s]].any()
    assert (pool.lane_dir[s] >= 0).all() and (pool.lane_dir[s] < np.pi).all()
    assert (pool.weight[s] > 0).all() and (pool.weight[S:] == 0).all()
    assert np.array_equal(pool.area_id[s], ships.area_id(pool.lon[s], pool.lat[s], cfg, g))
    gb = S + np.arange(len(pool) - S)
    assert np.allclose(pool.lon[gb], g.lon[pool.cell_i[gb]]) and g.wet[pool.cell_j[gb], pool.cell_i[gb]].all()
    assert len(set(pool.code[gb])) == len(gb)


@pytest.mark.skipif(not have_pool, reason="pool not built")
def test_real_pool_expected_total_matches_density():
    from tsunami.grid import load_grid
    g, pool = load_grid(cfg), ships.load_pool(cfg)
    E = ships.expected_ships(ships.regrid_density(cfg, g), g, cfg)
    total = E[ships._eligible(g, cfg)].sum()
    assert pool.weight[:pool.n_ships].sum() == pytest.approx(total, rel=1e-3)
    n = np.mean([len(ships.traffic_draw(pool, cfg, k).pool_idx) for k in range(100)])
    assert n == pytest.approx(total, rel=0.03)


@pytest.mark.skipif(not have_pool, reason="pool not built")
def test_pool_rebuild_is_identical(tmp_path):
    from tsunami.grid import load_grid
    a = ships.build_pool(cfg, load_grid(cfg), save=False)
    b = ships.load_pool(cfg)
    assert np.array_equal(a.lon, b.lon) and np.array_equal(a.lane_dir, b.lane_dir, equal_nan=True)
