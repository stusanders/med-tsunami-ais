import numpy as np
import pandas as pd
import pytest

from tsunami import truth_events as TE
from tsunami.config import load_config
from tsunami.geometry import haversine_m

cfg = load_config()
R = cfg["physics"]["earth_radius_m"]


class FakeSources:
    """Stand-in with the docs/interfaces.md section 7 signatures (Stream C's sources.py replaces it)."""

    @staticmethod
    def perturb(row, along_strike_m, dmw, cfg):
        d = dict(row)
        az = np.deg2rad(row["strike_deg"])
        d["lon_deg"] = row["lon_deg"] + np.rad2deg(along_strike_m * np.sin(az) / (R * np.cos(np.deg2rad(row["lat_deg"]))))
        d["lat_deg"] = row["lat_deg"] + np.rad2deg(along_strike_m * np.cos(az) / R)
        d["mw"] = row["mw"] + dmw
        return d

    @staticmethod
    def custom_source(lon_deg, lat_deg, strike_deg, dip_deg, mw, fault, cfg, length_m=None, width_m=None, slip_m=None):
        return dict(lon_deg=lon_deg, lat_deg=lat_deg, strike_deg=strike_deg, dip_deg=dip_deg, mw=mw,
                    polarity=-1 if fault == "normal" else 1, region="custom", flip_allowed=False)


def candidates():
    rows, cid = [], 0
    for region, lon0, lat0, step, mws in [("hellenic", 22.0, 35.0, 25e3, np.arange(6.5, 8.51, 0.25)),
                                           ("aegean", 25.0, 38.0, 50e3, np.arange(6.5, 7.76, 0.25))]:
        for p in range(5):
            for mw in mws:
                rows.append(dict(cand_id=cid, pos_id=p + (0 if region == "hellenic" else 100), region=region,
                                 lon_deg=lon0 + 0.2 * p, lat_deg=lat0 + 0.1 * p, strike_deg=90.0 + 5 * p, dip_deg=20.0,
                                 mw=float(mw), polarity=1, flip_allowed=region == "aegean"))
                cid += 1
    return pd.DataFrame(rows)


def test_map2_offset_and_magnitude():
    cand = candidates()
    ev = pd.DataFrame(TE.map2_events(cand, FakeSources, cfg))
    assert len(ev) == len(cand)
    d = haversine_m(ev.lon_deg.values, ev.lat_deg.values, cand.lon_deg.values, cand.lat_deg.values, R)
    want = np.where(cand.region == "aegean", cfg["sources"]["aegean_spacing_m"], cfg["sources"]["arc_spacing_m"]) / 2
    np.testing.assert_allclose(d, want, rtol=2e-3)
    np.testing.assert_allclose(ev.mw - cand.mw, cfg["truth"]["map2_dmw"])
    assert (ev.parent_cand_id.values == cand.cand_id.values).all() and (ev.run_kind == "truth_map2").all()
    assert (ev.loc[ev.region == "aegean", "polarity"] == cfg["truth"]["aegean_truth_polarity"]).all()
    assert (ev.loc[ev.region != "aegean", "polarity"] == 1).all()


def test_twin_events_off_grid_and_reproducible():
    cand = candidates()
    a = pd.DataFrame(TE.twin_events(cand, FakeSources, cfg))
    b = pd.DataFrame(TE.twin_events(cand, FakeSources, cfg))
    assert len(a) == cfg["truth"]["n_twin"] and a.equals(b)
    par = cand.set_index("cand_id").loc[a.parent_cand_id]
    off = haversine_m(a.lon_deg.values, a.lat_deg.values, par.lon_deg.values, par.lat_deg.values, R)
    spacing = np.where(par.region.values == "aegean", cfg["sources"]["aegean_spacing_m"], cfg["sources"]["arc_spacing_m"])
    assert (off >= 0.1 * spacing * 0.99).all() and (off <= 0.45 * spacing * 1.01).all()
    for region, g in a.groupby("region"):
        lim = cand[cand.region == region].mw
        assert g.mw.between(lim.min() - 1e-9, lim.max() + 1e-9).all()
    assert set(a.loc[a.region == "aegean", "polarity"]) <= {-1, 1}


def test_no_truth_source_coincides_with_a_library_source():
    cand = candidates()
    truth = TE.truth_table(cand, FakeSources, cfg)
    assert TE.coincidences(truth, cand, cfg).empty
    assert truth.run_id.is_unique and set(truth.run_kind) == {"truth_map2", "truth_twin"}
    # the check does catch a planted duplicate
    dup = truth.iloc[:1].copy()
    dup[["lon_deg", "lat_deg", "strike_deg", "mw"]] = cand.loc[7, ["lon_deg", "lat_deg", "strike_deg", "mw"]].values
    assert len(TE.coincidences(dup, cand, cfg)) == 1


def test_quiet_days_disjoint_and_large_enough():
    q = TE.quiet_days(cfg)
    cal, hold = q[q.split == "cal"], q[q.split == "hold"]
    assert len(cal) >= cfg["scoring"]["n_quiet_calibration"] and len(hold) >= cfg["scoring"]["n_quiet_holdout"]
    assert not set(cal.traffic_seed) & set(hold.traffic_seed)
    assert not set(cal.noise_seed) & set(hold.noise_seed)
    assert q.noise_seed.is_unique and q.quiet_id.is_unique
    assert cal.traffic_seed.nunique() == cfg["scoring"]["n_quiet_traffic_draws"]


def test_historic_table_uses_polarity_from_fault_type(monkeypatch):
    c = {**cfg, "historic_sources": [dict(name="samos", lon_deg=26.8, lat_deg=37.9, strike_deg=270.0, dip_deg=37.0, mw=7.0, fault="normal"),
                                     dict(name="crete", lon_deg=23.0, lat_deg=35.0, strike_deg=315.0, dip_deg=30.0, mw=8.3, fault="thrust")]}
    h = TE.historic_table(FakeSources, c)
    assert h.name.tolist() == ["samos", "crete"] and h.polarity.tolist() == [-1, 1] and (h.run_kind == "historic").all()
