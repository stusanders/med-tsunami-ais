"""Tests for T23 Piraeus AIS yaw correction (scripts/yaw_correction_piraeus.py)."""
import importlib.util

import numpy as np
import pandas as pd

from tsunami.config import load_config, repo_path
from tsunami.units import deg2rad, knots_to_ms


def _load_yaw_module():
    spec = importlib.util.spec_from_file_location("yaw_script", repo_path("scripts/yaw_correction_piraeus.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrap_deg_across_zero():
    mod = _load_yaw_module()
    diffs = mod.wrap_deg(np.array([359.0 - 1.0, 1.0 - 359.0, 180.0, -180.0]))
    np.testing.assert_allclose(diffs, [-2.0, 2.0, -180.0, -180.0])


def test_yaw_correction_recovers_exact_offset_and_cancels_turn_drift():
    mod = _load_yaw_module()
    cfg = load_config()

    # Synthesise a ship turning smoothly through 360 deg (crossing north) at 14 kn
    dt_s = 10.0
    n = 40
    t_ms = np.arange(n) * int(dt_s * 1000)
    rot_deg_s = 0.5  # +30 deg/min clockwise turn
    hdg_deg = (350.0 + rot_deg_s * np.arange(n) * dt_s) % 360.0
    sog_kn = np.full(n, 14.0)
    sog_ms = knots_to_ms(sog_kn)
    L_true = 85.0
    rot_rad_s = deg2rad(rot_deg_s)
    # Inazu et al. (2020): clockwise turn swings stern/GNSS to port -> v_n = -L * ROT
    v_n_true = -L_true * rot_rad_s
    dc_deg = np.rad2deg(np.arcsin(v_n_true / sog_ms))
    cog_deg = (hdg_deg + dc_deg) % 360.0

    df = pd.DataFrame(
        {
            "t": t_ms,
            "vessel_id": "ship_1",
            "heading": hdg_deg,
            "course": cog_deg,
            "speed": sog_kn,
        }
    )
    # Add duplicate rows to verify deduplication
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    sub, meta = mod.compute_turn_table(df, cfg)
    assert meta["n_dedup"] == n
    assert len(sub) == n - 2
    np.testing.assert_allclose(sub["rot_rad_s"].to_numpy(), rot_rad_s, rtol=1e-6)

    # Constant ROT has zero variance, so test fit_antenna_offset with alternating +/- turns
    rot_alt = np.concatenate([np.full(20, rot_rad_s), np.full(20, -rot_rad_s)])
    vn_alt = -L_true * rot_alt
    L_est = mod.fit_antenna_offset(vn_alt, rot_alt)
    assert np.isclose(L_est, L_true, rtol=1e-6)

    v_corr = mod.apply_yaw_correction(sub["v_n"], sub["rot_rad_s"], L_true)
    np.testing.assert_allclose(v_corr, 0.0, atol=1e-5)

    # Steady ship (ROT == 0) is unchanged by yaw correction
    v_steady = np.array([0.12, -0.08, 0.03])
    np.testing.assert_allclose(mod.apply_yaw_correction(v_steady, np.zeros(3), L_true), v_steady)
