"""Tests for T18 Maps 2 and 3 (scripts/06_maps.py) against synthetic sweep parquet."""
import importlib.util

import numpy as np
import pandas as pd

from tsunami.config import load_config, repo_path


def _load_maps_module():
    path = repo_path("scripts/06_maps.py")
    spec = importlib.util.spec_from_file_location("maps_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_map2_and_map3_synthetic_end_to_end(tmp_path):
    mod = _load_maps_module()
    cfg = load_config()

    # Synthetic truth index matching docs/interfaces.md sections 7 & 13
    truth_rows = []
    run_id = 0
    for pos_id, (region, lon, lat) in enumerate(
        [("hellenic", 23.0, 35.0), ("calabrian", 16.8, 37.5), ("aegean", 25.5, 37.2)]
    ):
        for mw in (6.6, 7.1, 7.6, 8.1):
            truth_rows.append(
                dict(
                    run_id=run_id,
                    run_kind="truth_map2",
                    region=region,
                    pos_id=pos_id,
                    lon_deg=lon,
                    lat_deg=lat,
                    mw=mw,
                )
            )
            run_id += 1
    truth_df = pd.DataFrame(truth_rows)

    # Synthetic gauge table matching gauges.load_gauges
    gauge_table = pd.DataFrame(
        [
            dict(code="G1", name="Crete", lon_deg=25.1, lat_deg=35.3, rtype=1),
            dict(code="G2", name="Catania", lon_deg=15.1, lat_deg=37.5, rtype=1),
            dict(code="INGV01", name="Ionian South", lon_deg=16.5, lat_deg=36.83, rtype=2),
        ]
    )

    # Synthetic sweep dataframe matching docs/interfaces.md section 14
    rng = np.random.default_rng(42)
    sweep_rows = []
    for _, tr in truth_df.iterrows():
        for op in ("ais", "gps"):
            for rcap in (40000.0, 100000.0):
                for reg, shift in (("calm", 0.0), ("moderate", 0.3), ("rough", 0.6)):
                    # Higher Mw and GPS detect more reliably
                    mw_eff = tr["mw"] - shift + (0.4 if op == "gps" else 0.0)
                    for det in ("network", "single_calibrated"):
                        for draw_i in range(10):
                            detected = bool(mw_eff >= (7.3 if det == "network" else 7.8))
                            confirm_s = float(600.0 + rng.uniform(0, 300)) if detected else np.nan
                            sweep_rows.append(
                                dict(
                                    run_id=int(tr["run_id"]),
                                    traffic_seed=draw_i % 2,
                                    noise_seed=draw_i,
                                    regime=reg,
                                    range_cap_m=rcap,
                                    operator=op,
                                    detector=det,
                                    detected=detected,
                                    confirm_s=confirm_s,
                                    best_cand_id=0,
                                    best_polarity=1,
                                    best_lon_deg=tr["lon_deg"],
                                    best_lat_deg=tr["lat_deg"],
                                    best_mw=tr["mw"],
                                    loc_err_m=10000.0,
                                    mw_err=0.1,
                                    top3_cand_ids="0,1,2",
                                    n_ships=900,
                                    first_gauge_s=900.0,
                                    nearest_gauge_code="G1" if tr["region"] != "calabrian" else "G2",
                                    nearest_gauge_s=1200.0,
                                    first_instrument_s=800.0,
                                )
                            )
    sweep_df = pd.DataFrame(sweep_rows)

    map2_df = mod.compute_map2_summary(sweep_df, truth_df, detector="network")
    assert not map2_df.empty
    # In calm AIS, mw >= 7.3 -> smallest detected should be 7.6
    calm_ais = map2_df[
        (map2_df["operator"] == "ais")
        & (map2_df["regime"] == "calm")
        & np.isclose(map2_df["range_cap_m"], 100000.0)
    ]
    assert np.allclose(calm_ais["mw_90"].to_numpy(), 7.6)

    mod.plot_map2(map2_df, cfg, tmp_path / "map2.png", range_cap_m=100000.0)
    mod.plot_map2_uncertainty(sweep_df, truth_df, cfg, tmp_path / "map2_uncertainty.png", range_cap_m=100000.0)
    assert (tmp_path / "map2.png").exists()
    assert (tmp_path / "map2_uncertainty.png").exists()

    map3_df = mod.compute_map3_summary(sweep_df, gauge_table, regime="moderate", operator="ais", detector="network")
    assert not map3_df.empty
    assert (map3_df["gain_med_min_nearest_gauge_s"] > 0).all()
    mod.plot_map3(map3_df, gauge_table, cfg, tmp_path / "map3.png")
    assert (tmp_path / "map3.png").exists()
