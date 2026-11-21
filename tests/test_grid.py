import numpy as np
import pytest

from tsunami import grid as G
from tsunami.config import load_config, repo_path
from tsunami.geometry import haversine_m

cfg = load_config()
R = cfg["physics"]["earth_radius_m"]


def test_block_mean_and_bands():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert G.block_mean(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    depth = np.array([[0, 10, 49.9, 50, 200, 200.1]])
    wet = depth > 0
    assert G.depth_bands(depth, wet, [50.0, 200.0]).tolist() == [[-1, 0, 0, 1, 1, 2]]


def test_ocean_connected_drops_inland_lake():
    wet = np.zeros((6, 10), bool)
    wet[:, :4] = True
    wet[2:4, 7:9] = True          # lake
    lon, lat = np.arange(10.0), np.arange(6.0)
    keep = G.ocean_connected(wet, lon, lat, 1.0, 1.0)
    assert keep[:, :4].all() and not keep[2:4, 7:9].any()


def test_distance_to_land_matches_haversine():
    lon, lat = np.linspace(20, 22, 9), np.linspace(34, 36, 9)
    land = np.zeros((9, 9), bool)
    land[:, 0] = True
    d = G.distance_to_land_m(land, lon, lat, R)
    assert (d[:, 0] == 0).all()
    j, i = 4, 6
    assert d[j, i] == pytest.approx(haversine_m(lon[i], lat[j], lon[0], lat[j], R), rel=1e-6)


def test_metrics_formulae():
    lat = np.array([30.0, 36.0])
    dx, dy, cos_c, cos_v = G.metrics(lat, 1 / 30, 1 / 30, R)
    assert dy == pytest.approx(R * np.deg2rad(1 / 30))
    assert dx[1] == pytest.approx(R * np.cos(np.deg2rad(36)) * np.deg2rad(1 / 30))
    assert cos_v.shape == (3,) and cos_v[1] == pytest.approx(np.cos(np.deg2rad(36 - 1 / 60)))
    assert cos_v[2] == pytest.approx(np.cos(np.deg2rad(36 + 1 / 60)))


have_data = repo_path(cfg["paths"]["gebco"]).exists()


@pytest.mark.skipif(not have_data, reason="GEBCO file not present")
def test_build_grid_contract(tmp_path):
    g = G.build_grid(cfg, save=False)
    ny, nx = 360, 750
    assert g.depth.shape == g.wet.shape == g.strait.shape == g.coast_dist.shape == g.depth_band.shape == (ny, nx)
    assert g.lon.shape == (nx,) and g.lat.shape == (ny,) and g.dx.shape == (ny,) and g.cos_c.shape == (ny,)
    assert g.cos_v.shape == (ny + 1,)
    assert g.depth.dtype == np.float32 and g.wet.dtype == bool and g.depth_band.dtype == np.int8
    assert g.lon[0] == pytest.approx(12 + 1 / 60) and g.lat[0] == pytest.approx(30 + 1 / 60)
    assert np.all(np.diff(g.lat) > 0) and np.all(np.diff(g.lon) > 0)
    assert g.depth[g.wet].min() >= cfg["domain"]["min_wet_depth_m"]
    assert (g.depth[~g.wet] == 0).all()
    assert (g.coast_dist[~g.wet] == 0).all() and (g.coast_dist[g.wet] > 0).all()
    assert ((g.depth_band == -1) == ~g.wet).all()
    assert g.strait.any() and 0.3 < g.wet.mean() < 0.7


@pytest.mark.skipif(not have_data, reason="GEBCO file not present")
def test_known_places():
    g = G.build_grid(cfg, save=False)
    j = lambda la: int(np.argmin(abs(g.lat - la)))
    i = lambda lo: int(np.argmin(abs(g.lon - lo)))
    assert g.depth[j(35.0), i(20.0)] > 2500                  # Ionian abyssal plain (about 3 to 4 km)
    assert not g.wet[j(41.5), i(32.0)]                      # Black Sea is excluded
    assert not g.wet[j(30.5), i(27.5)]                      # Qattara depression is not ocean
    assert g.strait[j(38.15), i(15.62)]                     # Messina


@pytest.mark.skipif(not have_data, reason="GEBCO file not present")
def test_save_load_roundtrip():
    g = G.build_grid(cfg, save=True)
    h = G.load_grid(cfg)
    for f in g.__dataclass_fields__:
        a, b = getattr(g, f), getattr(h, f)
        assert np.array_equal(a, b), f
