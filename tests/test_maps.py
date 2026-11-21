"""Tests for T18 Maps 2 and 3 + Detector Comparison (scripts/06_maps.py) against synthetic sweep parquet."""
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


def test_load_sweep_excludes_gauge_triggers(tmp_path):
    mod = _load_maps_module()
    pd.DataFrame({"run_id": [0, 1], "detector": ["network", "network"]}).to_parquet(
        tmp_path / "part_0_40000_ais_calm.parquet"
    )
    pd.DataFrame({"run_id": [0], "code": ["peir"], "trigger_s": [300.0]}).to_parquet(
        tmp_path / "gauge_triggers.parquet"
    )
    df = mod.load_sweep(tmp_path)
    assert len(df) == 2
    assert "code" not in df.columns


def test_map2_and_map3_synthetic_end_to_end(tmp_path):
    mod = _load_maps_module()
    cfg = load_config()

    # Synthetic truth table matching data/truth/sources.csv
    truth_rows = []
    run_id = 0
    for pos_id, (region, lon, lat) in enumerate(
        [("hellenic", 23.0, 35.0), ("calabrian", 16.8, 37.5), ("aegean", 25.5, 37.2)]
    ):
        for mw in (6.6, 7.1, 7.6, 8.1):
            # Simulate downdip centroid shift with Mw on arcs (must still collapse to 1 position per pos_id!)
            shift_deg = 0.02 * (mw - 6.6) if region != "aegean" else 0.0
            truth_rows.append(
                dict(
                    run_id=run_id,
                    run_kind="truth_map2",
                    region=region,
                    pos_id=pos_id,
                    lon_deg=lon + shift_deg,
                    lat_deg=lat + shift_deg,
                    mw=mw,
                )
            )
            run_id += 1
    # Add 2 twin events
    truth_rows.append(dict(run_id=run_id, run_kind="truth_twin", region="aegean", pos_id=99, lon_deg=25.2, lat_deg=37.0, mw=7.7))
    truth_df = pd.DataFrame(truth_rows)

    # Synthetic gauge table matching gauges.load_gauges
    gauge_table = pd.DataFrame(
        [
            dict(code="G1", name="Crete", lon_deg=25.1, lat_deg=35.3, rtype=1),
            dict(code="G2", name="Catania", lon_deg=15.1, lat_deg=37.5, rtype=1),
            dict(code="INGV01", name="Ionian South", lon_deg=16.5, lat_deg=36.83, rtype=2),
        ]
    )

    # Synthetic thresholds matching outputs/thresholds.csv
    thr_rows = []
    for op in ("ais", "gps"):
        for rcap in (40000.0, 100000.0):
            for reg in ("calm", "moderate", "rough"):
                thr_rows.append(dict(operator=op, range_cap_m=rcap, regime=reg, detector="network", threshold=300.0, n_cal=500, far_cal=0.01, n_hold=500, far_hold=0.016 if op == "ais" else 0.006))
                thr_rows.append(dict(operator=op, range_cap_m=rcap, regime=reg, detector="single_calibrated", threshold=5.2, n_cal=500, far_cal=0.01, n_hold=500, far_hold=0.012))
                thr_rows.append(dict(operator=op, range_cap_m=rcap, regime=reg, detector="single_literal", threshold=3.0, n_cal=500, far_cal=1.0, n_hold=500, far_hold=1.0))
    thr_df = pd.DataFrame(thr_rows)

    # Synthetic sweep dataframe matching docs/interfaces.md section 14
    rng = np.random.default_rng(42)
    sweep_rows = []
    for _, tr in truth_df.iterrows():
        for op in ("ais", "gps"):
            for rcap in (40000.0, 100000.0):
                for reg, shift in (("calm", 0.0), ("moderate", 0.3), ("rough", 0.6)):
                    mw_eff = tr["mw"] - shift + (0.4 if op == "gps" else 0.0)
                    for det in ("network", "single_calibrated", "single_literal"):
                        for draw_i in range(10):
                            if det == "single_literal":
                                detected = True
                            else:
                                detected = bool(mw_eff >= (7.3 if det == "network" else 7.8))
                            confirm_s = float(600.0 + rng.uniform(0, 300)) if detected else np.nan
                            is_net = det == "network"
                            loc_err = float(25000.0 if draw_i < 8 else 650000.0) if (is_net and detected) else np.nan
                            sweep_rows.append(
                                dict(
                                    run_id=int(tr["run_id"]),
                                    run_kind=tr["run_kind"],
                                    traffic_seed=draw_i % 2,
                                    noise_seed=draw_i,
                                    regime=reg,
                                    range_cap_m=rcap,
                                    operator=op,
                                    detector=det,
                                    detected=detected,
                                    confirm_s=confirm_s,
                                    best_cand_id=0 if is_net else -1,
                                    best_polarity=1 if is_net else 0,
                                    best_lon_deg=tr["lon_deg"] if is_net else np.nan,
                                    best_lat_deg=tr["lat_deg"] if is_net else np.nan,
                                    best_mw=tr["mw"] if is_net else np.nan,
                                    loc_err_m=loc_err,
                                    mw_err=0.1 if (is_net and detected) else np.nan,
                                    top3_cand_ids="0,1,2" if is_net else "",
                                    n_ships=682,
                                    first_gauge_s=900.0,
                                    nearest_gauge_code="G1" if tr["region"] != "calabrian" else "G2",
                                    nearest_gauge_s=1200.0,
                                    first_instrument_s=800.0,
                                )
                            )
    sweep_df = pd.DataFrame(sweep_rows)

    map2_df = mod.compute_map2_summary(sweep_df, truth_df, detector="network", thresholds_df=thr_df)
    assert not map2_df.empty
    calm_ais = map2_df[
        (map2_df["operator"] == "ais")
        & (map2_df["regime"] == "calm")
        & np.isclose(map2_df["range_cap_m"], 100000.0)
    ]
    assert len(calm_ais) == 3  # 3 unique positions even when arc centroid shifts with Mw
    assert np.allclose(calm_ais["mw_90"].to_numpy(), 7.6)
    assert np.allclose(calm_ais["far_hold"].to_numpy(), 0.016)

    mod.plot_map2(map2_df, cfg, tmp_path / "map2.png", range_cap_m=100000.0)
    mod.plot_map2_uncertainty(sweep_df, truth_df, cfg, tmp_path / "map2_uncertainty.png", range_cap_m=100000.0, thresholds_df=thr_df)
    assert (tmp_path / "map2.png").exists()
    assert (tmp_path / "map2_uncertainty.png").exists()

    # Detector comparison: split by operator & region, single-ship has NaN loc_err, network has pct_within_100km
    det_cmp = mod.compute_detector_comparison(sweep_df, truth_df, thresholds_df=thr_df)
    assert set(det_cmp["operator"]) == {"ais", "gps"}
    assert set(det_cmp["region"]) == {"hellenic", "calabrian", "aegean"}
    single_rows = det_cmp[det_cmp["detector"] != "network"]
    assert single_rows["med_loc_err_km"].isna().all()
    assert single_rows["pct_within_100km"].isna().all()
    net_rows = det_cmp[(det_cmp["detector"] == "network") & (det_cmp["n_detected"] > 0)]
    assert np.allclose(net_rows["pct_within_100km"].to_numpy(), 80.0)
    lit_rows = det_cmp[det_cmp["detector"] == "single_literal"]
    assert np.allclose(lit_rows["far_hold"].to_numpy(), 1.0)

    # Twin location summary
    twin_loc = mod.compute_twin_location_summary(sweep_df, thresholds_df=thr_df)
    assert not twin_loc.empty
    assert "pct_within_100km" in twin_loc.columns
    assert "far_hold" in twin_loc.columns

    map3_df = mod.compute_map3_summary(sweep_df, gauge_table, regime="moderate", operator="ais", detector="network")
    assert not map3_df.empty
    assert (map3_df["gain_med_min_nearest_gauge_s"] > 0).all()
    mod.plot_map3(map3_df, gauge_table, cfg, tmp_path / "map3.png", operator="ais", regime="moderate")
    assert (tmp_path / "map3.png").exists()
