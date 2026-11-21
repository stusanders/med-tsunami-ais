"""T15: scoring. Tensor == loop; planted source recovered; opposite headings add; sign-flipped
template scores zero; subsidence-first Aegean event caught by its flipped candidate; no NaN early;
quiet-day threshold gives the target false-alarm rate on held-out draws."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsunami import scoring
from tsunami import solver_reference as ref
from tsunami.config import load_config
from tsunami.solver import Solver
from tsunami.synthetic import SHIP, synthetic_grid, synthetic_pool

CFG = load_config()
EPS = CFG["scoring"]["eps"]


def random_case(E=3, S=40, K=5, T=12, seed=0, spread_deg=1.0):
    rng = np.random.default_rng(seed)
    lon = 20.0 + rng.uniform(0, spread_deg, S)
    lat = 35.0 + rng.uniform(0, spread_deg, S)
    pool = SimpleNamespace(lon=lon, lat=lat)
    draw = SimpleNamespace(pool_idx=np.arange(S), heading=rng.uniform(0, 2 * np.pi, S),
                           sog=rng.uniform(5, 9, S))
    d = rng.normal(size=(E, S, T)).astype(np.float32)
    g = rng.normal(size=(K, S, T)).astype(np.float32)
    sigma = rng.uniform(0.03, 0.1, S).astype(np.float32)
    return pool, draw, d, g, sigma


# ------------------------------------------------------------------ equivalence

@pytest.mark.parametrize("with_gaps", [False, True])
def test_tensor_equals_loop(with_gaps):
    pool, draw, d, g, sigma = random_case()
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    assert 1 < groups.n_groups < len(draw.heading)           # some cells hold several ships
    flip = np.array([False, True, False, True, False])
    valid = None
    if with_gaps:
        valid = np.random.default_rng(1).random((d.shape[1], d.shape[2])) > 0.3
    a = scoring.score_tensor(d, g, groups, sigma, 0.05, flip, EPS, valid=valid, xp=np)
    b = scoring.score_loop_reference(d, g, groups, sigma, 0.05, flip, EPS, valid=valid)
    for x, y in zip(a[:4], b[:4]):                            # gd2, gg2, S, stat
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=1e-6 * np.abs(y).max())
    np.testing.assert_array_equal(a[4], b[4])
    np.testing.assert_array_equal(a[5], b[5])


def test_gps_groups_are_cells_only():
    pool, draw, *_ = random_case()
    gr = scoring.build_groups(draw, pool, CFG, "gps")
    assert np.all(gr.sign == 1) and np.all(gr.key[:, 2] == 0)


# ------------------------------------------------------------------ sign alignment

def test_opposite_heading_ships_in_one_cell_add():
    theta = 0.3
    pool = SimpleNamespace(lon=np.array([20.01, 20.02]), lat=np.array([35.01, 35.02]))
    draw = SimpleNamespace(pool_idx=np.arange(2), heading=np.array([theta, theta + np.pi]), sog=np.array([7.0, 7.0]))
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    assert groups.n_groups == 1 and groups.sign[0] == -groups.sign[1]
    u, v = 0.08, -0.05
    from tsunami.geometry import project_normal
    obs = np.array([project_normal(u, v, theta), project_normal(u, v, theta + np.pi)])
    assert obs[0] == pytest.approx(-obs[1])                   # raw readings cancel
    d = np.tile(obs[None, :, None], (1, 1, 4)).astype(np.float32)
    D = scoring.aggregate(d, groups, xp=np)
    assert abs(D[0, 0, 0]) == pytest.approx(abs(obs[0]), rel=1e-6)   # aligned: they add, mean = full signal
    gd, gg = scoring.cumulative_stats(scoring.aggregate(d[0][None], groups, xp=np), D,
                                      scoring.cell_weights(groups, np.array([0.05, 0.05]), 0.02, xp=np), xp=np)
    assert gd[0, 0, -1] > 0


def test_projection_uses_each_ships_own_normal():
    """The bin centre only decides the sign: aligned data equal |u . n_i|, not u . n_bin."""
    theta = np.deg2rad([10.0, 30.0])                          # same 45 deg bin, different normals
    pool = SimpleNamespace(lon=np.array([20.01, 20.02]), lat=np.array([35.01, 35.02]))
    draw = SimpleNamespace(pool_idx=np.arange(2), heading=theta, sog=np.array([7.0, 7.0]))
    block = np.zeros((1, 3, 2, 3), np.float32)
    block[..., 1], block[..., 2] = 0.1, 0.0                   # eastward current
    g = scoring.project_templates(block, draw, "ais")
    np.testing.assert_allclose(g[0, :, 0], 0.1 * np.cos(theta), rtol=1e-6)


# ------------------------------------------------------------------ clamp and flips

def test_sign_flipped_template_scores_zero():
    pool, draw, _, g, sigma = random_case(K=1, E=1)
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    d = -g                                                    # data is exactly the template upside down
    _, _, S, stat, _, _ = scoring.score_tensor(d, g, groups, sigma, 0.05, np.array([False]), EPS, xp=np)
    assert np.all(S == 0) and np.all(stat == 0)


def test_subsidence_first_event_found_by_flipped_candidate():
    pool, draw, _, g, sigma = random_case(K=4, E=1, seed=3)
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    flip = np.array([False, False, True, True])               # last two: Aegean
    d = -g[2][None]                                           # subsidence-first version of candidate 2
    gd2, gg2, S, stat, parent, polarity = scoring.score_tensor(d, g, groups, sigma, 0.05, flip, EPS, xp=np)
    best = int(S[0, :, -1].argmax())
    assert parent[best] == 2 and polarity[best] == -1
    assert S[0, 2, -1] == 0                                   # unflipped version clamped to zero


def test_no_nan_at_early_minutes():
    pool, draw, d, g, sigma = random_case(T=10)
    g[:, :, :4] = 0.0                                         # wave has not reached any ship yet
    d[:, :, :4] = 0.0
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    gd2, gg2, S, stat, *_ = scoring.score_tensor(d, g, groups, sigma, 0.05, np.zeros(5, bool), EPS, xp=np)
    assert np.isfinite(S).all() and np.all(S[:, :, :4] == 0)
    L = scoring.log_likelihood(gd2, gg2)
    assert np.isfinite(L).all()


# ------------------------------------------------------------------ planted source from a real mini-library

def mini_library():
    """Solver-built library on a small synthetic sea: 4 positions x 3 magnitudes, sizes grow with Mw."""
    ny, nx = 40, 56
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 2500.0 + 30.0 * jj - 10.0 * ii
    grid = synthetic_grid(depth, lon0_deg=20.0, lat0_deg=35.0)
    rng = np.random.default_rng(7)
    S = 60
    lon = rng.uniform(grid.lon[3], grid.lon[-4], S)
    lat = rng.uniform(grid.lat[3], grid.lat[-4], S)
    pool = synthetic_pool(grid, lon, lat, [SHIP] * S, lane_dir=rng.uniform(0, np.pi, S))
    rows, eta0 = [], []
    for p, (pi_, pj) in enumerate([(12, 12), (40, 10), (15, 30), (42, 28)]):
        for mw in (7.0, 7.5, 8.0):
            amp = 10 ** (0.5 * (mw - 7.0))                    # height and width both grow with magnitude
            sig = 2.0 * 10 ** (0.25 * (mw - 7.0))
            eta0.append(amp * np.exp(-(((ii - pi_) / sig) ** 2 + ((jj - pj) / (0.6 * sig)) ** 2)))
            rows.append(dict(cand_id=len(rows), pos_id=p, lon_deg=grid.lon[pi_], lat_deg=grid.lat[pj], mw=mw,
                             flip_allowed=p >= 2))
    cfg = copy.deepcopy(CFG)
    cfg["solver"]["t_end_s"] = 3600.0
    cfg["library"]["n_minutes"] = 60
    cfg["library"]["baseline_window_min"] = 20
    s = Solver(grid, ref.make_sampler(grid, pool), cfg, xp=np)
    recf, _ = s.run_library(np.stack(eta0).astype(np.float32), np.array([], int))
    draw = SimpleNamespace(pool_idx=np.arange(S), heading=pool.lane_dir + np.pi * (rng.random(S) < 0.5),
                           sog=np.full(S, 7.0))
    return pool, draw, recf, pd.DataFrame(rows)


@pytest.mark.parametrize("operator", ["ais", "gps"])
def test_noiseless_planted_source_recovered_exactly(operator):
    pool, draw, lib, table = mini_library()
    groups = scoring.build_groups(draw, pool, CFG, operator)
    valid = np.ones((len(draw.pool_idx), lib.shape[1]), bool)
    G = scoring.aggregate_templates(lib, table.cand_id.to_numpy(), draw, groups, valid, operator, chunk=5)
    sigma = np.full(len(draw.pool_idx), 0.05, np.float32)
    w = scoring.cell_weights(groups, sigma, 0.02, valid, xp=np)
    for truth in (4, 7, 11):                                  # each truth = one library entry
        d = scoring.project_templates(lib[truth][None], draw, operator)
        D = scoring.aggregate(d, groups, valid, xp=np)
        gd, gg = scoring.cumulative_stats(G, D, w, xp=np)
        gd2, gg2, parent, polarity = scoring.expand_flips(gd, gg, table.flip_allowed.to_numpy())
        S = scoring.detection_scores(gd2, gg2, EPS)
        best_pos = table.pos_id[parent[int(S[0, :, -1].argmax())]]
        assert best_pos == table.pos_id[truth]                # location from the normalised score
        L = scoring.log_likelihood(gd2, gg2)
        est = scoring.estimate_source(L, np.array([-1]), table, parent, polarity, top_n=3)
        top = est[est["rank"] == 0].iloc[0]
        assert top.cand_id == truth and top.polarity == 1     # position AND magnitude, exactly
        assert len(est) == 3


# ------------------------------------------------------------------ thresholds and false alarms

def test_threshold_controls_false_alarm_rate_on_held_out_draws():
    pool, draw, _, g, sigma = random_case(S=60, K=6, T=20, seed=5)
    groups = scoring.build_groups(draw, pool, CFG, "ais")
    flip = np.array([False, False, False, True, True, True])
    rng = np.random.default_rng(11)

    def quiet(n):
        noise = rng.normal(size=(n, len(sigma), 20)).astype(np.float32) * sigma[None, :, None]
        return scoring.score_tensor(noise, g, groups, sigma, 0.0, flip, EPS, xp=np)[3]

    far = CFG["scoring"]["false_alarm_rate"]
    thr = scoring.calibrate_threshold(quiet(4000), far)
    held = quiet(4000)
    rate = np.mean(scoring.confirmation_minute(held, thr) >= 0)
    assert rate == pytest.approx(far, abs=0.006)


def test_confirmation_minute():
    stat = np.array([[0.0, 1.0, 5.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(scoring.confirmation_minute(stat, 3.0), [2, -1])


def test_library_reader_roundtrip(tmp_path):
    arr = np.random.default_rng(0).normal(size=(5, 4, 3, 3)).astype(np.float32)
    np.save(tmp_path / "rec_0000.npy", arr[:3])
    np.save(tmp_path / "rec_0001.npy", arr[3:])
    pd.DataFrame(dict(run_id=range(5), chunk=[0, 0, 0, 1, 1], row=[0, 1, 2, 0, 1])).to_csv(tmp_path / "index.csv", index=False)
    lib = scoring.Library(tmp_path)
    np.testing.assert_array_equal(lib[[4, 1]], arr[[4, 1]])
