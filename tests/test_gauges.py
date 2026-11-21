import numpy as np
import pytest

from tsunami import gauges as G
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver_reference import sponge_mask

cfg = load_config()
have = repo_path(cfg["paths"]["grid"]).exists() and repo_path(cfg["paths"]["ioc_gauges"]).exists()


def table(n=3, thr=(0.05, 0.05, 0.03), delay=(180.0, 180.0, 60.0)):
    import pandas as pd
    return pd.DataFrame({"threshold_m": thr[:n], "delay_s": delay[:n]})


def test_trigger_first_crossing_plus_delay_and_inf():
    gb = np.zeros((2, 240, 3), np.float32)
    gb[0, 9, 0] = 0.06                      # sample k=9 -> t = 30 * 10 = 300 s, + 180 s delay
    gb[0, 4, 0] = 0.049                     # below threshold: ignored
    gb[0, 20, 1] = -0.2                     # negative eta counts (absolute value)
    gb[0, 3, 2] = 0.031                     # buoy threshold 0.03, delay 60 -> 120 + 60
    gb[1, :, :] = 0.01                      # nothing triggers
    t = G.trigger_times(gb, table(), sample_interval_s=30.0)
    assert t.shape == (2, 3)
    assert t[0].tolist() == [480.0, 30 * 21 + 180.0, 180.0]
    assert np.isinf(t[1]).all()


def test_trigger_threshold_is_strict():
    gb = np.full((1, 240, 1), 0.05, np.float32)
    assert np.isinf(G.trigger_times(gb, table(1), 30.0)).all()


@pytest.mark.skipif(not have, reason="grid or IOC list missing")
def test_load_gauges_contract():
    g = load_grid(cfg)
    t = G.load_gauges(g, cfg)
    assert {"code", "lon_deg", "lat_deg", "cell_j", "cell_i", "rtype", "threshold_m", "delay_s"} <= set(t.columns)
    assert g.wet[t["cell_j"], t["cell_i"]].all()
    assert np.allclose(t["lon_deg"], g.lon[t["cell_i"]]) and np.allclose(t["lat_deg"], g.lat[t["cell_j"]])
    assert not sponge_mask(g.depth.shape, cfg)[t["cell_j"], t["cell_i"]].any()
    a = cfg["domain"]["area_of_interest"]
    assert t["lon_deg"].between(*a["lon_deg"]).all() and t["lat_deg"].between(*a["lat_deg"]).all()
    assert t["code"].is_unique and (np.diff(t["rtype"]) >= 0).all()
    assert (t["threshold_m"] == cfg["gauges"]["threshold_m"]).all() or (t["rtype"] == 2).any()
    codes = set(t["code"])
    assert {"kos3", "amor"} <= codes                         # Samos and Amorgos replays need these
    dropped = dict(t.attrs["dropped"])
    assert "ista" in dropped and "ista" not in codes         # Black Sea station dropped, not snapped across the land
