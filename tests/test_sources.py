"""Tests for T03 earthquake sources (src/tsunami/sources.py)."""
import numpy as np
import pytest

from tsunami import sources
from tsunami.config import load_config
from tsunami.geometry import haversine_m
from tsunami.synthetic import synthetic_grid


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def table(cfg):
    return sources.candidate_table(cfg)


def test_candidate_table_contract_and_regions(table, cfg):
    expected_cols = [
        "cand_id",
        "pos_id",
        "region",
        "lon_deg",
        "lat_deg",
        "strike_deg",
        "dip_deg",
        "mw",
        "length_m",
        "width_m",
        "slip_m",
        "polarity",
        "flip_allowed",
    ]
    assert list(table.columns) == expected_cols
    assert np.array_equal(table["cand_id"].values, np.arange(len(table)))
    assert set(table["region"].unique()) == {"hellenic", "calabrian", "cyprus", "aegean"}
    assert (table["polarity"] == 1).all()

    # Arcs: 9 bins (6.5..8.5), flip_allowed False; Aegean: 6 bins (6.5..7.75), flip_allowed True
    arcs = table[table["region"] != "aegean"]
    aeg = table[table["region"] == "aegean"]
    assert not arcs["flip_allowed"].any()
    assert aeg["flip_allowed"].all()
    assert np.isclose(arcs["mw"].min(), 6.5) and np.isclose(arcs["mw"].max(), 8.5)
    assert np.isclose(aeg["mw"].min(), 6.5) and np.isclose(aeg["mw"].max(), cfg["sources"]["aegean_mw_max"])
    assert len(arcs["mw"].unique()) == 9
    assert len(aeg["mw"].unique()) == 6

    # All candidates inside domain.area_of_interest
    aoi = cfg["domain"]["area_of_interest"]
    assert (table["lon_deg"] >= aoi["lon_deg"][0]).all() and (table["lon_deg"] <= aoi["lon_deg"][1]).all()
    assert (table["lat_deg"] >= aoi["lat_deg"][0]).all() and (table["lat_deg"] <= aoi["lat_deg"][1]).all()


def test_aegean_excluded_from_historic_epicentres(table, cfg):
    aeg_pos = table[table["region"] == "aegean"].drop_duplicates("pos_id")
    r_excl = cfg["sources"]["aegean_exclusion_radius_m"]
    radius_m = cfg["physics"]["earth_radius_m"]
    # Samos 2020 (26.784, 37.897) and Amorgos 1956 (25.76, 36.72)
    for ex_lo, ex_la in ((26.784, 37.897), (25.76, 36.72)):
        dists = haversine_m(aeg_pos["lon_deg"].values, aeg_pos["lat_deg"].values, ex_lo, ex_la, radius_m)
        assert (dists >= r_excl).all()


def test_source_eta0_dipole_monopole_polarity_and_volume(table, cfg):
    # Open flat ocean synthetic grid covering eastern Med
    grid = synthetic_grid(np.full((360, 750), 2000.0), lon0_deg=12.0, lat0_deg=30.0)
    hel8 = table[(table["region"] == "hellenic") & np.isclose(table["mw"], 8.0)].iloc[15]

    eta_dip = sources.source_eta0(grid, hel8, cfg, kind="dipole")
    eta_mono = sources.source_eta0(grid, hel8, cfg, kind="monopole")
    eta_flip = sources.source_eta0(grid, hel8, cfg, kind="dipole", polarity=-1)

    assert eta_dip.dtype == np.float32
    assert eta_dip.shape == grid.depth.shape
    # Plausible Mw 8 uplift and subsidence sizes
    assert 0.3 < float(eta_dip.max()) < 5.0
    assert -2.0 < float(eta_dip.min()) < -0.05
    # Monopole has no negative lobe
    assert float(eta_mono.min()) >= 0.0
    # Polarity flip is exact negation (matches scoring.expand_flips)
    assert np.allclose(eta_flip, -eta_dip, atol=1e-6)

    # Dipole volume check
    up_v, sub_v, net_v = sources.dipole_volume(eta_dip, grid)
    assert up_v > 0.0 and sub_v > 0.0
    assert np.isclose(net_v, up_v - sub_v, rtol=1e-9)
    # Ratio of subsidence to uplift volumes should be roughly subsidence_ratio (0.3)
    assert 0.15 < sub_v / up_v < 0.45


def test_perturb_moves_along_strike_and_rescales(table, cfg):
    row = table[(table["region"] == "hellenic") & np.isclose(table["mw"], 7.5)].iloc[10]
    p = sources.perturb(row, along_strike_m=12500.0, dmw=0.1, cfg=cfg)
    dist = float(haversine_m(row["lon_deg"], row["lat_deg"], p["lon_deg"], p["lat_deg"], cfg["physics"]["earth_radius_m"]))
    assert np.isclose(dist, 12500.0, rtol=0.02)
    assert np.isclose(p["mw"], 7.6)
    assert p["length_m"] > row["length_m"]
    assert p["slip_m"] > row["slip_m"]


def test_custom_source_normal_and_thrust(cfg):
    samos = sources.custom_source(26.784, 37.897, 276.0, 48.0, 7.0, "normal", cfg, length_m=32000.0, width_m=15000.0)
    assert samos["polarity"] == -1
    assert samos["length_m"] == 32000.0
    assert samos["width_m"] == 15000.0
    # M0 conservation check: mu * L * W * slip == M0(Mw=7.0)
    m0_expected = sources.seismic_moment_nm(7.0)
    m0_got = float(cfg["sources"]["rigidity_pa"]) * samos["length_m"] * samos["width_m"] * samos["slip_m"]
    assert np.isclose(m0_got, m0_expected, rtol=1e-3)

    thrust = sources.custom_source(23.5, 34.5, 290.0, 15.0, 8.0, "thrust", cfg)
    assert thrust["polarity"] == 1

    with pytest.raises(ValueError):
        sources.custom_source(23.5, 34.5, 290.0, 15.0, 8.0, "strike-slip", cfg)
