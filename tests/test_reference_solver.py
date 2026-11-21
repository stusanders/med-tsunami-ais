"""T02: analytic checks of the reference solver (flat bottom, closed basin, sloping shelf)."""
import copy

import numpy as np
import pytest

from tsunami import solver_reference as ref
from tsunami.config import load_config
from tsunami.synthetic import GAUGE, SHIP, synthetic_grid, synthetic_pool

CFG = load_config()
G = CFG["physics"]["g_m_s2"]


def cfg_closed():
    """No sponge: every boundary is a reflecting wall (exact conservation)."""
    c = copy.deepcopy(CFG)
    c["solver"]["sponge_width_cells"] = 0
    return c


def peak_time(series, interval):
    """Time of the maximum, refined with a parabola through the 3 samples around it."""
    k = int(np.argmax(series))
    y0, y1, y2 = series[k - 1], series[k], series[k + 1]
    offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return interval * (k + 1 + offset)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_flat_bottom_arrival_time(axis):
    h, sigma, dist = 4000.0, 30e3, 500e3
    n_long, n_short = 300, 5
    shape = (n_short, n_long) if axis == "x" else (n_long, n_short)
    grid = synthetic_grid(np.full(shape, h), lon0_deg=20.0, lat0_deg=33.0)
    c = cfg_closed()
    src = 60
    if axis == "x":
        j = n_short // 2
        step_m = grid.dx[j]
        k_rec = src + int(round(dist / step_m))
        dist = (k_rec - src) * step_m
        x = (np.arange(n_long) - src)[None, :] * grid.dx[:, None]
        eta0 = np.exp(-0.5 * (x / sigma) ** 2)
        pool = synthetic_pool(grid, [grid.lon[k_rec]], [grid.lat[j]], [GAUGE])
    else:
        i = n_short // 2
        k_rec = src + int(round(dist / grid.dy))
        dist = (k_rec - src) * grid.dy
        y = (np.arange(n_long) - src)[:, None] * grid.dy * np.ones((1, n_short))
        eta0 = np.exp(-0.5 * (y / sigma) ** 2)
        pool = synthetic_pool(grid, [grid.lon[i]], [grid.lat[k_rec]], [GAUGE])
    sampler = ref.make_sampler(grid, pool)
    interval = 10.0
    expected = dist / np.sqrt(G * h)
    rec = ref.run_reference(grid, eta0, sampler, c, t_end_s=expected * 1.3, sample_interval_s=interval)
    t_peak = peak_time(rec[:, 0, 0], interval)
    assert abs(t_peak - expected) / expected < 0.02, (t_peak, expected)
    # the split pulse carries half the initial height
    assert rec[:, 0, 0].max() == pytest.approx(0.5, rel=0.05)


def test_closed_basin_volume_conserved():
    ny, nx = 60, 80
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 200.0 + 3800.0 * np.sin(np.pi * ii / nx) * np.sin(np.pi * jj / ny)
    depth[25:32, 40:48] = -50.0                      # an island
    grid = synthetic_grid(depth, lat0_deg=35.0)
    eta0 = np.exp(-(((ii - 25) / 5.0) ** 2 + ((jj - 20) / 5.0) ** 2)).astype(np.float32)
    c = cfg_closed()
    v0 = ref.volume(eta0 * grid.wet, grid)
    eta, u, v = ref.run_fields(grid, eta0, c, n_steps=800)
    v1 = ref.volume(eta, grid)
    assert abs(v1 - v0) / abs(v0) < 0.01, (v0, v1)
    assert np.all(eta[~grid.wet] == 0)
    assert np.abs(eta).max() > 1e-3   # the wave is still there, not damped away


def test_sloping_shelf_greens_law():
    """Pulse running up a gentle 1D slope: amplitude ~ h^(-1/4)."""
    h_deep, h_shallow = 4000.0, 50.0
    ny = 4
    grid0 = synthetic_grid(np.ones((ny, 10)), lat0_deg=0.0)
    dxm = grid0.dx[ny // 2]
    n_deep, n_slope = int(600e3 / dxm), int(600e3 / dxm)
    nx = n_deep + n_slope
    prof = np.full(nx, h_deep)
    prof[n_deep:] = np.linspace(h_deep, h_shallow, n_slope)
    grid = synthetic_grid(np.tile(prof, (ny, 1)), lat0_deg=0.0)
    c = cfg_closed()
    src, sigma = n_deep // 2, 60e3
    x = (np.arange(nx) - src)[None, :] * grid.dx[:, None]
    eta0 = np.exp(-0.5 * (x / sigma) ** 2)

    targets = [h_deep, 2000.0, 1000.0, 500.0, 250.0]
    cells = [n_deep - 20] + [n_deep + int(np.argmin(np.abs(prof[n_deep:] - h))) for h in targets[1:]]
    j = ny // 2
    pool = synthetic_pool(grid, grid.lon[cells], np.full(len(cells), grid.lat[j]), [GAUGE] * len(cells))
    sampler = ref.make_sampler(grid, pool)
    interval = 10.0
    speed = np.sqrt(G * prof)
    t_arr = [np.sum(dxm / speed[src:k]) for k in cells]
    tau = sigma / np.sqrt(G * h_deep)
    rec = ref.run_reference(grid, eta0, sampler, c, t_end_s=max(t_arr) + 3 * tau, sample_interval_s=interval)
    t = interval * (np.arange(rec.shape[0]) + 1)
    amp = []
    for r, ta in enumerate(t_arr):
        win = (t > ta - 2 * tau) & (t < ta + 2 * tau)
        amp.append(rec[win, r, 0].max())
    amp = np.array(amp)
    h_rec = prof[cells]
    ratio = amp / amp[0]
    greens = (h_rec / h_rec[0]) ** -0.25
    np.testing.assert_allclose(ratio, greens, rtol=0.10)


def test_linearity_and_superposition():
    ny, nx = 40, 50
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 1000.0 + 20.0 * ii
    depth[:, :4] = -1.0
    grid = synthetic_grid(depth)
    a = np.exp(-(((ii - 20) / 4.0) ** 2 + ((jj - 15) / 4.0) ** 2))
    b = -0.5 * np.exp(-(((ii - 35) / 3.0) ** 2 + ((jj - 28) / 5.0) ** 2))
    pool = synthetic_pool(grid, grid.lon[[10, 25, 40]] + 0.01, grid.lat[[10, 20, 30]] + 0.01, [SHIP, SHIP, GAUGE])
    s = ref.make_sampler(grid, pool)
    run = lambda e: ref.run_reference(grid, e, s, CFG, t_end_s=1800.0)
    ra, rb, rab, r2a = run(a), run(b), run(a + b), run(2 * a)
    scale = np.abs(rab).max()
    np.testing.assert_allclose(rab, ra + rb, atol=1e-5 * scale)
    np.testing.assert_allclose(r2a, 2 * ra, atol=1e-5 * np.abs(r2a).max())


def test_sampler_bilinear_and_snapped():
    grid = synthetic_grid(np.full((10, 12), 1000.0))
    lon = [grid.lon[3] + 0.3 * grid.dlon_deg, grid.lon[7] + 0.2 * grid.dlon_deg]
    lat = [grid.lat[4] + 0.6 * grid.dlat_deg, grid.lat[5] + 0.4 * grid.dlat_deg]
    pool = synthetic_pool(grid, lon, lat, [SHIP, GAUGE])
    s = ref.make_sampler(grid, pool)
    jj, ii = np.meshgrid(np.arange(10), np.arange(12), indexing="ij")
    eta = (2.0 * ii + 3.0 * jj).astype(np.float32)        # linear field: bilinear is exact
    u = np.zeros((10, 13), np.float32)
    v = np.zeros((11, 12), np.float32)
    out = ref.sample(eta, u, v, s)
    assert out[0, 0] == pytest.approx(2 * 3.3 + 3 * 4.6, rel=1e-6)
    assert out[1, 0] == pytest.approx(eta[5, 7])          # gauge reads its snapped cell (5, 7)


def test_time_step_divides_sample_interval():
    grid = synthetic_grid(np.full((10, 10), 5000.0))
    dt, n_sub = ref.time_step(grid, CFG)
    assert dt * n_sub == pytest.approx(CFG["solver"]["sample_interval_s"])
    cfl_dt = CFG["solver"]["cfl"] * min(grid.dx.min(), grid.dy) / np.sqrt(G * 5000.0)
    assert dt <= cfl_dt


def test_sponge_only_on_configured_edges_and_outside_area_of_interest():
    d = CFG["domain"]
    res = d["resolution_arcmin"] / 60.0
    lon = np.arange(d["lon_min_deg"] + res / 2, d["lon_max_deg"], res)
    lat = np.arange(d["lat_min_deg"] + res / 2, d["lat_max_deg"], res)
    mask = ref.sponge_mask((len(lat), len(lon)), CFG)
    w = CFG["solver"]["sponge_width_cells"]
    assert mask[:, :w].all() and mask[-w:, :].all()           # west and north damped
    assert not mask[w:-w, -w:].any() and not mask[:w, w:-w].any()  # east and south are walls
    a = d["area_of_interest"]
    aoi = ((lon[None, :] >= a["lon_deg"][0]) & (lon[None, :] <= a["lon_deg"][1])
           & (lat[:, None] >= a["lat_deg"][0]) & (lat[:, None] <= a["lat_deg"][1]))
    assert not (mask & aoi).any()
