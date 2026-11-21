import itertools

import numpy as np
import pandas as pd
import pytest

from tsunami import library_build as LB
from tsunami import scoring, noise, observe
from tsunami import solver_reference as ref
from tsunami import sweep as SW
from tsunami.config import load_config
from tsunami.solver import Solver
from tsunami.synthetic import BUOY, GAUGE, SHIP, synthetic_grid, synthetic_pool

CFG0 = load_config()
CFG = {**CFG0, "sweep": {**CFG0["sweep"], "n_noise_draws": 3, "event_batch": 2, "n_traffic_draws": 2, "gather_chunk": 2},
       "traffic": {**CFG0["traffic"], "range_caps_m": [40000.0, 100000.0]}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    ny, nx = 36, 60
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 2000.0 - 15.0 * ii + 200.0 * np.sin(jj / 5.0)
    depth[:, 56:] = -10.0
    grid = synthetic_grid(depth, lat0_deg=34.0)
    rng = np.random.default_rng(3)
    n = 60
    lon, lat = rng.uniform(grid.lon[4], grid.lon[48], n), rng.uniform(grid.lat[4], grid.lat[30], n)
    pool = synthetic_pool(grid, np.r_[lon, grid.lon[[54, 6]]], np.r_[lat, grid.lat[[20, 8]]], [SHIP] * n + [GAUGE, BUOY],
                          lane_dir=np.r_[rng.uniform(0, np.pi, n), 0, 0])
    pool.weight[:n] = 0.5                                   # expected 30 ships per draw
    solver = Solver(grid, ref.make_sampler(grid, pool), CFG, xp=np)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    centres = [(12, 10), (20, 14), (30, 22), (40, 18), (25, 26), (45, 8)]
    cand = pd.DataFrame({"cand_id": np.arange(6), "cx": [c[0] for c in centres], "cy": [c[1] for c in centres],
                         "lon_deg": [grid.lon[c[0]] for c in centres], "lat_deg": [grid.lat[c[1]] for c in centres],
                         "mw": 7.0, "flip_allowed": [False, False, True, True, False, False], "run_id": np.arange(6)})

    def source_fn(g, row, cfg, kind=None):
        a = float(row.get("polarity", 1))
        return (300.0 * a * np.exp(-(((ii - row["cx"]) / 5.0) ** 2 + ((jj - row["cy"]) / 5.0) ** 2))).astype(np.float32)

    LB.build_stage("lib", cand, root / "lib", solver, grid, CFG, gb_idx, source_fn, batch=3)
    truth = cand.iloc[[1, 2, 3]].copy()
    truth["run_id"] = [0, 1, 2]
    truth["polarity"] = [1, -1, 1]                          # run 1 is subsidence-first: needs the flipped candidate 2
    truth["run_kind"] = ["truth_map2", "truth_twin", "truth_twin"]
    truth["parent"] = [1, 2, 3]
    LB.build_stage("truth", truth, root / "truth", solver, grid, CFG, gb_idx, source_fn, batch=2)
    return dict(root=root, grid=grid, pool=pool, cand=cand, truth=truth, gb_idx=gb_idx)


def quiet_table():
    rows, q = [], 0
    for split, tseed in (("cal", 0), ("hold", 1)):
        for k in range(20):
            rows.append((q, split, tseed, 100 * tseed + k))
            q += 1
    return pd.DataFrame(rows, columns=["quiet_id", "split", "traffic_seed", "noise_seed"])


def test_gather_matches_per_ship_scoring(world):
    w = world
    f = SW.make_fleet(w["pool"], CFG, SW.PURPOSE_TRAFFIC, 0)
    lib = scoring.Library(w["root"] / "lib")
    SW.gather_pass(lib, w["cand"].cand_id.values, [f], CFG, w["root"] / "c1", "lib")
    for cap, op in itertools.product(sorted(CFG["traffic"]["range_caps_m"]), CFG["operators"]):
        G = np.asarray(SW.load_G(w["root"] / "c1", "lib", f, cap, op))
        d = f.draws[cap]
        groups = f.groups[(cap, op)]
        G_ref = scoring.aggregate_templates(lib, w["cand"].cand_id.values, d, groups, np.ones((len(d.pool_idx), 120), bool), op)
        np.testing.assert_allclose(G, G_ref, atol=1e-6)
    # network stat on aggregated observations equals the per-ship tensor path
    cap, op, regime = 100000.0, "ais", "calm"
    d, groups = f.draws[cap], f.groups[(cap, op)]
    si, ss, st = SW.sigmas(d, regime, op, CFG)
    rng = np.random.default_rng(0)
    nz = SW.make_noise(w["pool"], f, cap, op, regime, CFG, rng, 2)
    G = np.asarray(SW.load_G(w["root"] / "c1", "lib", f, cap, op))
    wts = scoring.cell_weights(groups, si, ss)
    flip = w["cand"].flip_allowed.values
    _, _, stat, _, _ = SW.network_trials(G, scoring.aggregate(nz, groups, None, xp=np), wts, flip, CFG["scoring"]["eps"])
    g_ship = scoring.project_templates(lib[w["cand"].cand_id.values], d, op)
    out = scoring.score_tensor(nz, g_ship, groups, si, ss, flip, CFG["scoring"]["eps"], xp=np)
    np.testing.assert_allclose(np.asarray(stat), np.asarray(out[3]), rtol=1e-4, atol=1e-6)


def test_full_sweep_recovers_planted_events(world):
    w = world
    pool, cand, truth = w["pool"], w["cand"], w["truth"]
    cache = w["root"] / "c2"
    lib, tlib = scoring.Library(w["root"] / "lib"), scoring.Library(w["root"] / "truth")
    quiet = quiet_table()
    qf = {i: SW.make_fleet(pool, CFG, SW.PURPOSE_QTRAFFIC, i) for i in (0, 1)}
    SW.gather_pass(lib, cand.cand_id.values, list(qf.values()), CFG, cache, "quiet")
    qs = SW.quiet_stats(cache, "quiet", qf, quiet, pool, CFG, cand)
    tdf, thr = SW.thresholds(qs, CFG)
    assert len(tdf) == 12 * 3 and set(tdf.detector) == set(SW.DETECTORS)
    assert (tdf.threshold > 0).all() and (tdf.n_cal == 20).all() and (tdf.n_hold == 20).all()
    lit = tdf[tdf.detector == "single_literal"]
    assert (lit.threshold == 3.0).all()
    ef = [SW.make_fleet(pool, CFG, SW.PURPOSE_TRAFFIC, i) for i in range(2)]
    SW.gather_pass(lib, cand.cand_id.values, ef, CFG, cache, "lib")
    SW.gather_pass(tlib, truth.run_id.values, ef, CFG, cache, "truth", per_ship=True)
    from tsunami import gauges
    gt = pd.DataFrame({"code": pool.code[w["gb_idx"]], "lon_deg": pool.lon[w["gb_idx"]], "lat_deg": pool.lat[w["gb_idx"]],
                       "rtype": pool.rtype[w["gb_idx"]], "threshold_m": 0.05, "delay_s": 180.0})
    trig, first = SW.gauge_table(truth, CFG, w["root"] / "truth", gt)
    gcols = SW.gauge_columns(truth, trig, CFG, gt)
    assert len(gcols) == 3 and (gcols.first_instrument_s <= gcols.first_gauge_s).all()
    cap, op, regime = 100000.0, "ais", "calm"
    f = ef[0]
    G = np.asarray(SW.load_G(cache, "lib", f, cap, op))
    Ds = SW.load_G(cache, "truth", f, cap, op)
    sig = np.load(cache / f"truth_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r")
    df = SW.event_shard(pool, f, cap, op, regime, truth, Ds, sig, G, cand, thr[(op, cap, regime)], gcols, CFG, 7)
    N = CFG["sweep"]["n_noise_draws"]
    assert len(df) == 3 * N * 3
    need = ["run_id", "traffic_seed", "noise_seed", "regime", "range_cap_m", "operator", "detector", "detected", "confirm_s", "best_cand_id",
            "best_polarity", "best_lon_deg", "best_lat_deg", "best_mw", "loc_err_m", "mw_err", "top3_cand_ids", "n_ships", "first_gauge_s",
            "nearest_gauge_code", "nearest_gauge_s", "first_instrument_s"]
    assert set(need) <= set(df.columns)
    net = df[df.detector == "network"]
    assert net.detected.all()                                                       # planted events are huge (300 m sources) so the plumbing is tested, not the noise
    assert (net.confirm_s % 60 == 0).all() and (net.confirm_s <= 7200).all()
    # truth run 0 is library candidate 1 at the same place with the same polarity: recovered exactly
    r0 = net[net.run_id == 0]
    assert (r0.best_cand_id == 1).all() and (r0.best_polarity == 1).all() and (r0.loc_err_m < 1.0).all()
    # truth run 1 is subsidence-first at candidate 2's place: only the flipped candidate fits (candidate 2 allows flips)
    r1 = net[net.run_id == 1]
    assert (r1.best_cand_id == 2).all() and (r1.best_polarity == -1).all()
    assert set(df.detector) == set(SW.DETECTORS)
    assert (df[df.detector == "single_literal"].best_cand_id == -1).all()
    assert df.noise_seed.nunique() == N
    assert np.isfinite(df.n_ships).all()


def test_gap_and_scaling_paths_run_and_detect(world):
    w = world
    pool, cand, truth = w["pool"], w["cand"], w["truth"]
    cache = w["root"] / "c3"
    lib, tlib = scoring.Library(w["root"] / "lib"), scoring.Library(w["root"] / "truth")
    cap, op, regime = 100000.0, "ais", "calm"
    valid_for = lambda f, c, o: observe.gaps(f.draws[c], CFG, np.random.default_rng([1, 7, f.idx]), drop_frac=0.1)
    qf = {i: SW.make_fleet(pool, CFG, SW.PURPOSE_QTRAFFIC, i) for i in (0, 1)}
    SW.gather_pass(lib, cand.cand_id.values, list(qf.values()), CFG, cache, "qg", valid_fn=valid_for)
    qs = SW.quiet_stats(cache, "qg", qf, quiet_table(), pool, CFG, cand, strata=[(op, cap, regime)], valid_fn=valid_for)
    assert list(qs) == [(op, cap, regime)]
    tdf, thr = SW.thresholds(qs, CFG)
    assert len(tdf) == 3
    f = SW.make_fleet(pool, CFG, SW.PURPOSE_TRAFFIC, 0)
    SW.gather_pass(lib, cand.cand_id.values, [f], CFG, cache, "lg", valid_fn=valid_for)
    SW.gather_pass(tlib, truth.run_id.values, [f], CFG, cache, "tr", per_ship=True)
    v = valid_for(f, cap, op)
    assert 0.8 < v.mean() < 0.97
    sig = np.load(cache / f"tr_ship_{f.purpose}_{f.idx}_{op}.npy", mmap_mode="r")
    groups = f.groups[(cap, op)]
    Dsv = scoring.aggregate(np.asarray(sig)[:, f.masks[cap], :], groups, v, xp=np)
    G = np.asarray(SW.load_G(cache, "lg", f, cap, op))
    from tsunami import gauges
    gt = pd.DataFrame({"code": pool.code[w["gb_idx"]], "lon_deg": pool.lon[w["gb_idx"]], "lat_deg": pool.lat[w["gb_idx"]],
                       "rtype": pool.rtype[w["gb_idx"]], "threshold_m": 0.05, "delay_s": 180.0})
    trig, _ = SW.gauge_table(truth, CFG, w["root"] / "truth", gt)
    gcols = SW.gauge_columns(truth, trig, CFG, gt)
    df = SW.event_shard(pool, f, cap, op, regime, truth, Dsv, sig, G, cand, thr[(op, cap, regime)], gcols, CFG, 3, valid=v)
    net = df[df.detector == "network"]
    assert net.detected.mean() > 0.9
    # a linear rescale of the truth signal by 0.0 must not be detected; Scaled is lazy and exact
    Dz = SW.Scaled(SW.load_G(cache, "tr", f, cap, op), 0.0)
    assert np.all(np.asarray(Dz[slice(0, 2)]) == 0)
    z = SW.event_shard(pool, f, cap, op, regime, truth, Dz, SW.Scaled(sig, 0.0), np.asarray(SW.load_G(cache, "lg", f, cap, op)), cand,
                       thr[(op, cap, regime)], gcols, CFG, 3, valid=v)
    assert z[z.detector == "network"].detected.mean() < 0.5
    lv = SW.map2_levels(df.assign(run_kind="truth_map2"), truth.assign(pos_id=truth.run_id, region="r"))
    assert len(lv) == 3


def test_uplift_scale_is_applied_to_library_and_gauges(world):
    w = world
    lib = scoring.Library(w["root"] / "lib")
    scale = pd.Series([0.5, 2.0, 1.0, 3.0, 0.25, 1.5], index=np.arange(6))
    slib = SW.ScaledLibrary(lib, scale)
    ids = np.array([4, 1, 3])
    np.testing.assert_allclose(slib[ids], lib[ids] * np.array([0.25, 2.0, 3.0], np.float32)[:, None, None, None], rtol=1e-6)
    assert len(slib) == len(lib) and (slib.index.run_id.values == lib.index.run_id.values).all()
    f = SW.make_fleet(w["pool"], CFG, SW.PURPOSE_TRAFFIC, 0)
    SW.gather_pass(lib, w["cand"].cand_id.values, [f], CFG, w["root"] / "u0", "a")
    SW.gather_pass(slib, w["cand"].cand_id.values, [f], CFG, w["root"] / "u1", "a")
    for op in CFG["operators"]:
        G0 = np.asarray(SW.load_G(w["root"] / "u0", "a", f, 100000.0, op))
        G1 = np.asarray(SW.load_G(w["root"] / "u1", "a", f, 100000.0, op))
        np.testing.assert_allclose(G1, G0 * scale.values.astype(np.float32)[:, None, None], rtol=1e-4, atol=1e-8)
    pool = w["pool"]
    gt = pd.DataFrame({"code": pool.code[w["gb_idx"]], "lon_deg": pool.lon[w["gb_idx"]], "lat_deg": pool.lat[w["gb_idx"]],
                       "rtype": pool.rtype[w["gb_idx"]], "threshold_m": 0.05, "delay_s": 180.0})
    truth = w["truth"]
    t0, _ = SW.gauge_table(truth, CFG, w["root"] / "truth", gt)
    t1, _ = SW.gauge_table(truth, CFG, w["root"] / "truth", gt, scale=pd.Series([1000.0, 1000.0, 1000.0], index=[0, 1, 2]))
    assert (t1 <= t0).all() and (t1 < t0).any()                  # a huge amplification can only trigger earlier
    zero = SW.gauge_table(truth, CFG, w["root"] / "truth", gt, scale=pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2]))[0]
    assert np.isinf(zero).all()


def test_scaled_legacy_run_equals_new_rule_run(world):
    """A run built with the legacy uplift fraction, multiplied by uplift_scale, equals a run built with the new rule (linearity)."""
    from tsunami import sources
    w = world
    grid, pool = w["grid"], w["pool"]
    cfg = {**CFG, "solver": {**CFG["solver"], "t_end_s": 1800.0}, "library": {**CFG["library"], "n_minutes": 30}}
    legacy = cfg["sources"]["legacy_uplift_fraction_of_slip"]
    row = dict(lon_deg=float(grid.lon[25]), lat_deg=float(grid.lat[18]), strike_deg=40.0, dip_deg=20.0, width_m=40000.0, length_m=80000.0,
               slip_m=2.0, polarity=1)
    new_row = dict(row, uplift_dip_deg=15.0)                                          # new rule: slip x sin(15 deg)
    old_row = dict(row, uplift_dip_deg=float(np.rad2deg(np.arcsin(legacy))))          # legacy: slip x 0.3
    scale = sources.uplift_scale(new_row, cfg)
    assert scale == pytest.approx(np.sin(np.deg2rad(15.0)) / legacy)
    eta_new, eta_old = sources.source_eta0(grid, new_row, cfg), sources.source_eta0(grid, old_row, cfg)
    np.testing.assert_allclose(eta_new, eta_old * scale, rtol=1e-5, atol=1e-7)       # same shape, scaled amplitude
    solver = Solver(grid, ref.make_sampler(grid, pool), cfg, xp=np)
    gb_idx = np.nonzero(pool.rtype > 0)[0]
    rec_new, gb_new = solver.run_library(eta_new[None], gb_idx)
    rec_old, gb_old = solver.run_library(eta_old[None], gb_idx)
    tol = 1e-5 * float(np.abs(rec_new).max())
    np.testing.assert_allclose(np.asarray(rec_old) * scale, np.asarray(rec_new), atol=tol)       # rec of the legacy run, scaled
    np.testing.assert_allclose(np.asarray(gb_old) * scale, np.asarray(gb_new), atol=1e-5 * float(np.abs(gb_new).max()))   # and the raw gauge series
    # and through the loader used in T17: a one-run library written with the legacy source, scaled on load
    d = w["root"] / "legacy_one"
    t = pd.DataFrame({"run_id": [0], **{k: [v] for k, v in old_row.items()}})
    LB.build_stage("t", t, d, solver, grid, cfg, gb_idx, lambda g, r, c, kind=None: sources.source_eta0(g, r, c), batch=1)
    loaded = SW.ScaledLibrary(scoring.Library(d), pd.Series([scale], index=[0]))[[0]]
    np.testing.assert_allclose(loaded, np.asarray(rec_new), atol=tol)
