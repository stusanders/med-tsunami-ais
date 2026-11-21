"""T00: contract helpers (units, geometry conventions, time axis, causal baseline)."""
import numpy as np
import pytest

from tsunami import backend, config, geometry, timeseries, units


def test_imports_and_config():
    cfg = config.load_config()
    assert backend.DTYPE == np.float32
    mags = config.magnitudes(cfg)
    assert mags[0] == 6.5 and mags[-1] == 8.5 and len(mags) == 9
    assert cfg["library"]["n_minutes"] * cfg["library"]["minute_s"] == cfg["solver"]["t_end_s"]


def test_unit_helpers():
    assert units.knots_to_ms(1.0) == pytest.approx(0.514444)
    assert units.ms_to_knots(units.knots_to_ms(12.3)) == pytest.approx(12.3)
    assert units.deg2rad(180.0) == pytest.approx(np.pi)
    assert units.rad2deg(units.deg2rad(37.0)) == pytest.approx(37.0)


def test_heading_normal_is_starboard():
    # Heading north: starboard is east. Heading east: starboard is south.
    e, n = geometry.heading_normal(np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(e, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(n, [0.0, -1.0], atol=1e-12)
    # project_normal equals SOG*sin(COG-HDG) for a small current.
    hdg, sog = np.deg2rad(30.0), 6.0
    cur = np.array([0.05, -0.02])
    he, hn = geometry.heading_unit(hdg)
    vog = sog * np.array([he, hn]) + cur
    cog = np.arctan2(vog[0], vog[1])
    drift = np.hypot(*vog) * np.sin(cog - hdg)
    assert geometry.project_normal(cur[0], cur[1], hdg) == pytest.approx(drift, rel=1e-9)


def test_block_average_minutes():
    x = np.arange(240, dtype=np.float32).reshape(1, 240, 1, 1)
    y = timeseries.block_average(x, 2, axis=1)
    assert y.shape == (1, 120, 1, 1)
    assert y[0, 0, 0, 0] == pytest.approx(0.5)


def test_baseline_removes_constant_after_window():
    W = 45
    x = np.full((3, 120), 2.5, dtype=np.float32)
    y = timeseries.baseline_filter(x, W, axis=1, history=np.full((3, W), 2.5, np.float32))
    np.testing.assert_allclose(y, 0.0, atol=1e-6)
    y0 = timeseries.baseline_filter(x, W, axis=1)  # zero history
    np.testing.assert_allclose(y0[:, W:], 0.0, atol=1e-6)


def test_baseline_is_causal():
    rng = np.random.default_rng(0)
    W, m = 30, 50
    x = rng.normal(size=120).astype(np.float32)
    x2 = x.copy()
    x2[m:] += 100.0  # change the present and the future only
    y, y2 = timeseries.baseline_filter(x, W, axis=0), timeseries.baseline_filter(x2, W, axis=0)
    np.testing.assert_array_equal(y[:m], y2[:m])
    # sample m depends on itself but not on its own trailing mean
    assert y2[m] - y[m] == pytest.approx(100.0, rel=1e-5)


def test_baseline_is_linear():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 4, 120)).astype(np.float32)
    f = lambda z: timeseries.baseline_filter(z, 45, axis=1)
    np.testing.assert_allclose(f(a + b), f(a) + f(b), atol=1e-5)


def test_cell_index_spacing():
    cfg = config.load_config()
    R = cfg["physics"]["earth_radius_m"]
    cj, ci = geometry.cell_index(np.array([20.0, 20.0]), np.array([36.0, 36.0 + 0.5]),
                                 25000.0, 36.0, 12.0, 30.0, R)
    assert cj[1] - cj[0] == 2  # 0.5 deg ~ 55.6 km ~ 2 cells
