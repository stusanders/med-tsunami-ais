"""T03: earthquake sources (EFSM20 arcs, Aegean grid, Gaussian dipole)."""
import numpy as np
import pytest

from tsunami import sources
from tsunami.config import load_config, magnitudes, repo_path

CFG = load_config()
R = CFG["physics"]["earth_radius_m"]
pytestmark = pytest.mark.skipif(not repo_path(CFG["paths"]["grid"]).exists(), reason="needs data/processed/grid.npz (T01)")


@pytest.fixture(scope="module")
def grid():
    from tsunami.grid import load_grid
    return load_grid(CFG)


@pytest.fixture(scope="module")
def table(grid):
    return sources.candidate_table(CFG, grid, save=False)


def dist_m(lon1, lat1, lon2, lat2):
    e, n = sources._enu(lon2, lat2, lon1, lat1, R)
    return np.hypot(e, n)


def test_table_structure(table):
    assert list(table.cand_id) == list(range(len(table)))
    assert set(table.region) == set(sources.REGIONS)
    assert (table.flip_allowed == (table.region == "aegean")).all()
    assert (table.polarity == 1).all()
    mags = magnitudes(CFG)
    assert set(np.round(table[table.region != "aegean"].mw, 4)) == set(mags)
    assert table[table.region == "aegean"].mw.max() <= CFG["sources"]["aegean_mw_max"] + 1e-9
    a = CFG["domain"]["area_of_interest"]
    assert table.trench_lon_deg.between(*a["lon_deg"]).all() and table.trench_lat_deg.between(*a["lat_deg"]).all()


@pytest.mark.parametrize("region", ["hellenic", "calabrian", "cyprus"])
def test_arc_spacing_and_dip_direction(table, region):
    pos = table[table.region == region].drop_duplicates("pos_id")
    d = dist_m(pos.trench_lon_deg.values[:-1], pos.trench_lat_deg.values[:-1],
               pos.trench_lon_deg.values[1:], pos.trench_lat_deg.values[1:])
    spacing = CFG["sources"]["arc_spacing_m"]
    assert np.median(d) == pytest.approx(spacing, rel=0.1)
    # strike agrees with EFSM20's own (Aki-Richards) triangle strikes near each trench point
    _, tris = sources.arc_mesh(CFG["sources"]["arcs"][region], CFG)
    for p in pos.itertuples():
        e, n = sources._enu(tris.lon_c.values, tris.lat_c.values, p.trench_lon_deg, p.trench_lat_deg, R)
        near = (np.hypot(e, n) < CFG["sources"]["arc_local_radius_m"]) & (tris.depth_c_m.values <= CFG["sources"]["arc_max_depth_for_dip_m"])
        ang = np.deg2rad(tris.strike.values[near])
        mean = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        diff = (p.strike_deg - mean + 180.0) % 360.0 - 180.0
        assert abs(diff) < 45.0, (p.pos_id, p.strike_deg, mean)
    assert pos.dip_deg.between(0.5, 60).all()


def test_aegean_grid_avoids_historic_epicentres(table):
    aeg = table[table.region == "aegean"].drop_duplicates("pos_id")
    for ep in CFG["sources"]["exclusion_epicentres"].values():
        d = dist_m(aeg.lon_deg.values, aeg.lat_deg.values, ep["lon_deg"], ep["lat_deg"])
        assert d.min() >= CFG["sources"]["aegean_exclusion_radius_m"]


def test_scaling_monotonic_and_slip_from_moment():
    m = np.array([6.5, 7.5, 8.5])
    L, W, D = np.array([sources.rupture_size(x, "hellenic", CFG) for x in m]).T
    assert np.all(np.diff(L) > 0) and np.all(np.diff(W) > 0) and np.all(np.diff(D) > 0)
    m0 = CFG["sources"]["rigidity_pa"] * L * W * D
    np.testing.assert_allclose(np.log10(m0), 1.5 * m + 9.1, rtol=1e-9)
    assert sources.rupture_size(7.75, "aegean", CFG)[1] <= CFG["sources"]["crustal_max_width_m"]


def test_mw8_dipole_plausible(grid, table):
    row = table[(table.region == "hellenic") & np.isclose(table.mw, 8.0)].iloc[20]
    eta = sources.source_eta0(grid, row, CFG)
    assert eta.dtype == np.float32 and eta.shape == grid.depth.shape
    assert 0.3 < eta.max() < 3.0                               # m, order of published Mw 8 scenarios
    ratio = -eta.min() / eta.max()
    assert ratio == pytest.approx(CFG["sources"]["subsidence_ratio"], rel=0.15)
    up, down, net = sources.dipole_volume(eta, grid)
    assert up > 0 > down and net > 0
    mono = sources.source_eta0(grid, row, CFG, kind="monopole")
    assert mono.min() >= 0
    np.testing.assert_allclose(sources.source_eta0(grid, row, CFG, polarity=-1), -eta)
    # subsidence lobe lies landward (down-dip) of the uplift peak
    j_up, i_up = np.unravel_index(eta.argmax(), eta.shape)
    j_dn, i_dn = np.unravel_index(eta.argmin(), eta.shape)
    e, n = sources._enu(grid.lon[i_dn], grid.lat[j_dn], grid.lon[i_up], grid.lat[j_up], R)
    dx, dy = sources._bearing_vec(row.strike_deg + 90.0)
    assert dx * e + dy * n > 0


def test_perturb_moves_along_strike_and_differs_from_library(table):
    row = table[(table.region == "cyprus") & np.isclose(table.mw, 7.5)].iloc[5]
    half = CFG["sources"]["arc_spacing_m"] / 2
    p = sources.perturb(row, half, 0.1, CFG)
    assert dist_m(row.trench_lon_deg, row.trench_lat_deg, p["trench_lon_deg"], p["trench_lat_deg"]) == pytest.approx(half, rel=1e-3)
    assert p["mw"] == pytest.approx(7.6) and p["length_m"] > row.length_m
    same = (np.isclose(table.lon_deg, p["lon_deg"]) & np.isclose(table.lat_deg, p["lat_deg"]) & np.isclose(table.mw, p["mw"]))
    assert not same.any()


def test_custom_normal_fault_is_subsidence_first(grid):
    s = sources.custom_source(26.8, 37.9, 270.0, 45.0, 7.0, "normal", CFG, name="samos-like")
    eta = sources.source_eta0(grid, s, CFG)
    assert s["polarity"] == -1 and -eta.min() > eta.max() > 0
    t = sources.custom_source(26.8, 37.9, 270.0, 45.0, 7.0, "thrust", CFG)
    assert t["polarity"] == 1
    with pytest.raises(ValueError):
        sources.custom_source(26.8, 37.9, 270.0, 45.0, 7.0, "strike-slip", CFG)
