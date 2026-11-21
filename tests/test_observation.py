"""T12: observation operators and noise."""
from types import SimpleNamespace

import numpy as np
import pytest

from tsunami import noise, observe
from tsunami.config import load_config
from tsunami.units import knots_to_ms

CFG = load_config()
T = CFG["library"]["n_minutes"]
W = CFG["library"]["baseline_window_min"]


def make_draw(S=400, seed=0, spread_km=400.0):
    rng = np.random.default_rng(seed)
    lon = 20.0 + rng.uniform(0, spread_km / 90.0, S)
    lat = 35.0 + rng.uniform(0, spread_km / 111.0, S)
    pool = SimpleNamespace(lon=lon, lat=lat)
    draw = SimpleNamespace(pool_idx=np.arange(S), heading=rng.uniform(0, 2 * np.pi, S).astype(np.float32),
                           sog=knots_to_ms(rng.uniform(10, 18, S)).astype(np.float32), seed=seed)
    return pool, draw


def test_zero_noise_returns_true_current():
    pool, draw = make_draw(S=3)
    draw.heading = np.array([0.0, np.pi / 2, np.pi], np.float32)       # north, east, south
    truth = np.zeros((T, 3, 3), np.float32)
    truth[:, :, 1] = 0.1                                               # 0.1 m/s eastward everywhere
    sig = observe.ais_signal(observe.gather_truth(truth, draw), draw)
    np.testing.assert_allclose(sig[:, 0], [0.1, 0.0, -0.1], atol=1e-7)  # starboard of north is east
    d = observe.observe(sig, np.zeros((2, 3, T), np.float32))
    np.testing.assert_array_equal(d[1], sig)
    half = observe.ais_signal(truth, draw, response=0.5)
    np.testing.assert_allclose(half, 0.5 * sig, atol=1e-7)


def test_gps_operator_reads_eta():
    _, draw = make_draw(S=2)
    truth = np.zeros((T, 2, 3), np.float32)
    truth[:, :, 0] = np.linspace(0, 0.3, T)[:, None]
    np.testing.assert_allclose(observe.gps_signal(truth, draw)[1], truth[:, 1, 0])


@pytest.mark.parametrize("regime", ["calm", "rough"])
def test_independent_noise_matches_ship_sigma(regime):
    pool, draw = make_draw(S=500)
    rng = np.random.default_rng(1)
    n = noise.ais_noise(draw, pool, regime, CFG, rng, n_draws=40, include_shared=False)
    emp = n.std(axis=(0, 2))
    np.testing.assert_allclose(emp, noise.ship_sigma(draw, regime, "ais", CFG), rtol=0.05)
    assert n.shape == (40, 500, T)


def test_regime_scales_wobble():
    _, draw = make_draw(S=5)
    a = CFG["ais_noise"]
    n_rep = 60.0 / a["report_interval_s"]
    expect = lambda m: np.sqrt((np.deg2rad(a["heading_wobble_sd_deg"]) * m) ** 2
                               + np.deg2rad(a["heading_resolution_deg"]) ** 2 / 12) / np.sqrt(n_rep)
    for regime, mult in CFG["weather_regimes"].items():
        assert noise.heading_sd_per_minute(regime, CFG) == pytest.approx(expect(mult))


def test_shared_noise_matches_shared_sigma_and_correlation():
    pool, draw = make_draw(S=300, spread_km=600.0)
    rng = np.random.default_rng(2)
    sd = CFG["ais_noise"]["shared_sd_m_s"]
    xy = noise.ship_xy_m(pool, draw, CFG)
    f = noise.shared_field(xy, sd, CFG, rng, n_draws=60, n_times=W + T)
    assert f[:, 0].std() == pytest.approx(sd, rel=0.1)                     # raw field level
    n = noise.ais_noise(draw, pool, "calm", CFG, rng, n_draws=60, include_bias=False)
    ind = noise.ship_sigma(draw, "calm", "ais", CFG)
    shared_emp = np.sqrt(max(n.var() - np.mean(ind**2), 0.0))
    assert shared_emp == pytest.approx(noise.shared_sigma("calm", "ais", CFG), rel=0.15)
    # spatial correlation of the raw u field at ~L
    L = CFG["ais_noise"]["shared_corr_length_m"]
    r = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    u = f[:, 0, :, 0]                                                        # (E, S) at t = 0
    c = np.corrcoef(u.T)
    near = (r > 0.8 * L) & (r < 1.2 * L)
    assert c[near].mean() == pytest.approx(np.exp(-0.5), abs=0.1)


def test_baseline_removes_constant_drift():
    pool, draw = make_draw(S=50)
    with_bias = noise.ais_noise(draw, pool, "calm", CFG, np.random.default_rng(3), 4, include_shared=False)
    rng = np.random.default_rng(3)
    S = len(draw.sog)
    raw = np.asarray(draw.sog)[None, :, None] * noise.heading_sd_per_minute("calm", CFG) \
        * rng.standard_normal((4, S, W + T))
    without = noise.filter_with_history(raw.astype(np.float32), CFG)
    np.testing.assert_allclose(with_bias, without, atol=1e-5)


def test_noise_filter_never_uses_future():
    rng = np.random.default_rng(4)
    raw = rng.standard_normal((2, 3, W + T)).astype(np.float32)
    m = 70
    raw2 = raw.copy()
    raw2[..., W + m:] += 5.0
    a, b = noise.filter_with_history(raw, CFG), noise.filter_with_history(raw2, CFG)
    np.testing.assert_array_equal(a[..., :m], b[..., :m])
    assert not np.allclose(a[..., m], b[..., m])


def test_gps_noise_level():
    _, draw = make_draw(S=300)
    n = noise.gps_noise(draw, "rough", CFG, np.random.default_rng(5), 30)
    assert n.std() == pytest.approx(noise.ship_sigma(draw, "rough", "gps", CFG).mean(), rel=0.05)


def test_gaps():
    _, draw = make_draw(S=1000)
    rng = np.random.default_rng(6)
    off = np.zeros(1000, bool)
    off[:10] = True
    v = observe.gaps(draw, CFG, rng, drop_frac=0.1, off_mask=off)
    assert v.shape == (1000, T)
    assert not v[:10].any()
    assert 1 - v[10:].mean() == pytest.approx(0.1, abs=0.01)
    assert observe.gaps(draw, CFG, rng).all()
