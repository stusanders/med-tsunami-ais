"""T27 demo data: Hellenic Mw 8 showcase (scenes 8, 9, 11). Seeded and re-runnable; NumPy backend, Mac CPU.

    cd ~/projects/mts-demo && TSUNAMI_BACKEND=numpy PYTHONPATH=src \
        ../med-tsunami-ais/.venv/bin/python scripts/demo/make_showcase_data.py [--workers 3] [--quick]

What it does (docs/DEMO.md Part 3, "Data needed that does not exist yet"):
  1. Full-field eta snapshots of the showcase event (cand_id from outputs/first_detection.json, i.e. 186,
     the T13/T24 event) every minute for 0-120 min, float16, cropped to the area of interest
     -> outputs/demo_data/showcase_snapshots.npz
  2. A LOCAL candidate subset (the 1,083-candidate library lives only on Colab and is not used): the real solver
     (solver.Solver.run_library, current sources.py with uplift = slip x sin(mean dip), so no uplift_scale
     rescaling is applied; that factor is only for the legacy 0.3 x slip T14 library) is run for every Hellenic
     position at Mw 7.25-8.5 plus Calabrian and Cyprus positions at Mw 7.5 and 8.5 and Aegean positions at Mw 7.5.
     Only per-ship projected templates for the showcase traffic draw are kept, plus per-candidate quiet-day
     detection maxima for the threshold.
  3. Showcase traffic draw (ships.traffic_draw index 0, purpose 1, 100 km range cap) + one AIS noise draw
     (purpose 2, index 0), calm regime, AIS operator: per-ship cleaned drift, true signal, SNR; network statistic
     (scoring.score_tensor), its threshold and confirmation minute, top-three sources (scoring.estimate_source);
     Inazu's literal single-ship 3-sigma rule (baseline_detector); gauge and buoy trigger times (gauges).
     -> outputs/demo_data/showcase_traffic.npz, showcase_templates.npy
  4. Threshold: outputs/thresholds.csv (T17) if present for (ais, 100 km, calm, network); otherwise a DEV threshold
     calibrated with scoring.calibrate_threshold at 1 % FAR on the quiet-day calibration split of
     data/truth/quiet_days.csv (traffic purpose 5, noise purpose 6), scored against the same local subset.
     Measured false-alarm rates of the network and the literal single-ship rule on the held-out split.
  5. Quiet-day example with the showcase traffic (noise purpose 6, index QUIET_EXAMPLE_NOISE outside the
     cal/hold ranges): single-ship 3-sigma exceedances (times, positions) and the network statistic.
     -> outputs/demo_data/quiet_day_example.npz
  6. Two-ship inset (pool 3179 shelf, pool 2557 deep; the T24 ships) for scene 8.
  7. outputs/demo_data/showcase_meta.json: every scalar the scenes print, plus provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TSUNAMI_BACKEND", "numpy")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tsunami import baseline_detector as bd  # noqa: E402
from tsunami import gauges, noise, observe, scoring, ships, sources  # noqa: E402
from tsunami.config import load_config, magnitudes, repo_path  # noqa: E402
from tsunami.grid import load_grid  # noqa: E402
from tsunami.solver import Solver, make_sampler  # noqa: E402
from tsunami.units import knots_to_ms  # noqa: E402

OUT = REPO / "outputs" / "demo_data"
REGIME, OPERATOR = "calm", "ais"
PURPOSE_TRAFFIC, PURPOSE_NOISE, PURPOSE_QTRAFFIC, PURPOSE_QNOISE = 1, 2, 5, 6     # docs/interfaces.md section 0
SHOW_TRAFFIC_INDEX, SHOW_NOISE_INDEX = 0, 0
INSET_NOISE_INDEX = 1                     # AIS-noise stream for the two inset ships (index 0 = showcase draw)
QUIET_EXAMPLE_NOISE = 1000                # quiet-day noise stream index for scene 11 (cal/hold use 0..999)
INSET_POOL = {"shelf": 3179, "deep": 2557}   # the T24 cleaning-figure ships (Cretan shelf 61 m, deep 2,766 m)

# Local candidate subset: region -> magnitudes
SUBSET = {"hellenic": [7.25, 7.5, 7.75, 8.0, 8.25, 8.5],
          "calabrian": [7.5, 8.5], "cyprus": [7.5, 8.5], "aegean": [7.5]}
SUBSET_QUICK = {"hellenic": [7.75, 8.0, 8.25], "aegean": [7.5]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------- setup

def showcase_cand_id(cfg):
    fd = json.loads((repo_path(cfg["paths"]["outputs"]) / "first_detection.json").read_text())
    return int(fd["source"]["cand_id"])


def candidate_subset(cand, cfg, cid, quick=False):
    spec = SUBSET_QUICK if quick else SUBSET
    keep = np.zeros(len(cand), bool)
    for region, mws in spec.items():
        keep |= (cand.region == region).values & np.isin(np.round(cand.mw.values, 4), np.round(mws, 4))
    if quick:   # every 4th position only (pipeline test)
        keep &= (cand.pos_id.values % 4 == 0)
    keep |= cand.cand_id.values == cid
    return cand[keep]


def range_cap(cfg):
    return float(cfg["traffic"]["range_caps_m"][-1])


def showcase_draw(pool, cfg):
    return ships.apply_range_cap(ships.traffic_draw(pool, cfg, SHOW_TRAFFIC_INDEX, purpose=PURPOSE_TRAFFIC), pool,
                                 range_cap(cfg))


def quiet_draw(pool, cfg, traffic_seed):
    return ships.apply_range_cap(ships.traffic_draw(pool, cfg, int(traffic_seed), purpose=PURPOSE_QTRAFFIC), pool,
                                 range_cap(cfg))


def inset_draw(pool, cfg):
    """The two T24 ships as a 2-ship draw: heading = lane direction, speed = mid of traffic.speed_kn."""
    idx = np.array([INSET_POOL["shelf"], INSET_POOL["deep"]], np.int64)
    sog = knots_to_ms(float(np.mean(cfg["traffic"]["speed_kn"])))
    return ships.TrafficDraw(idx, pool.lane_dir[idx].astype(np.float32), np.full(2, sog, np.float32), -1)


def sigma_total(draw, cfg):
    si = noise.ship_sigma(draw, REGIME, OPERATOR, cfg).astype(np.float64)
    ss = noise.shared_sigma(REGIME, OPERATOR, cfg)
    return si, ss, np.sqrt(si**2 + ss**2)


# ------------------------------------------------------------------------------------------- quiet days

def build_quiet(cfg, pool, quick):
    """Noise-only observations for the quiet-day list, aggregated per traffic draw (network) and scored by the
    single-ship rule. Returns dict with per-draw arrays (written to a /tmp npz for the workers)."""
    q = pd.read_csv(repo_path(cfg["paths"]["truth_dir"]) / "quiet_days.csv")
    if quick:
        q = q.groupby(["split", "traffic_seed"]).head(8).reset_index(drop=True)
    seed = cfg["meta"]["seed"]
    per_draw, ss_stat, n_ships = {}, np.zeros((len(q), int(cfg["library"]["n_minutes"]))), {}
    order = []
    for ts, grp in q.groupby("traffic_seed", sort=True):
        draw = quiet_draw(pool, cfg, ts)
        groups = scoring.build_groups(draw, pool, cfg, OPERATOR)
        si, ss, st = sigma_total(draw, cfg)
        nz = np.concatenate([noise.ais_noise(draw, pool, REGIME, cfg, np.random.default_rng([seed, PURPOSE_QNOISE, int(ns)]), 1)
                             for ns in grp.noise_seed.values])                                     # (E, S, T)
        rows = grp.index.values
        ss_stat[rows] = bd.single_ship_stat(nz, st.astype(np.float32))
        D = scoring.aggregate(nz, groups, None, xp=np)                                             # (E, M, T)
        w = scoring.cell_weights(groups, si, ss, None, xp=np)
        per_draw[int(ts)] = dict(D=D, w=w)
        n_ships[int(ts)] = len(draw.pool_idx)
        order.append((int(ts), rows))
        log(f"quiet traffic draw {ts}: {len(draw.pool_idx)} ships, {groups.n_groups} groups, {len(rows)} noise draws")
    return q, per_draw, ss_stat, n_ships, order


# ------------------------------------------------------------------------------------------- workers

_W = {}


def _init_worker(quiet_path, quick):
    cfg = load_config()
    grid, pool = load_grid(cfg), ships.load_pool(cfg)
    _W.update(cfg=cfg, grid=grid, pool=pool, solver=Solver(grid, make_sampler(grid, pool), cfg, xp=np),
              gb_idx=np.nonzero(pool.rtype > 0)[0], show=showcase_draw(pool, cfg))
    z = np.load(quiet_path, allow_pickle=True)
    meta = z["meta"].item()
    _W["quiet"] = []
    for ts in meta["traffic_seeds"]:
        draw = quiet_draw(pool, cfg, ts)
        _W["quiet"].append(dict(draw=draw, groups=scoring.build_groups(draw, pool, cfg, OPERATOR),
                                D=z[f"D_{ts}"], w=z[f"w_{ts}"]))


def _run_candidate(row):
    cfg, grid, pool, solver = _W["cfg"], _W["grid"], _W["pool"], _W["solver"]
    t0 = time.time()
    eta0 = sources.source_eta0(grid, row, cfg)
    recf, gb30 = solver.run_library(eta0[None], _W["gb_idx"])                    # (1, 120, R, 3), (1, 240, R_gb)
    t_solve = time.time() - t0
    g_show = scoring.project_templates(recf, _W["show"], OPERATOR)[0]           # (S, 120) == observe.ais_signal
    eps = float(cfg["scoring"]["eps"])
    flip = np.array([bool(row["flip_allowed"])])
    q_stat = []
    for qd in _W["quiet"]:
        G = scoring.aggregate(scoring.project_templates(recf, qd["draw"], OPERATOR), qd["groups"], None, xp=np)
        gd, gg = scoring.cumulative_stats(G, qd["D"], qd["w"], xp=np)
        gd2, gg2, _, _ = scoring.expand_flips(gd, gg, flip)
        stat, _ = scoring.detection_statistic(scoring.detection_scores(gd2, gg2, eps))
        q_stat.append(stat.astype(np.float32))                                   # (E_q, T)
    extra = {}
    if int(row["cand_id"]) == int(row["_showcase"]):
        extra = dict(recf_inset=recf[0][:, [INSET_POOL["shelf"], INSET_POOL["deep"]], :].copy(), gb30=gb30[0].copy())
    return int(row["cand_id"]), g_show.astype(np.float32), q_stat, t_solve, time.time() - t0, extra


# ------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="small subset and few quiet days (pipeline test)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    cfg = load_config()
    seed = cfg["meta"]["seed"]
    grid, pool = load_grid(cfg), ships.load_pool(cfg)
    cand = pd.read_csv(repo_path(cfg["paths"]["candidates"]))
    cid = showcase_cand_id(cfg)
    row_show = cand[cand.cand_id == cid].iloc[0].to_dict()
    sub = candidate_subset(cand, cfg, cid, a.quick).reset_index(drop=True)
    assert cid in set(sub.cand_id), "showcase candidate must be in the subset"
    K = len(sub)
    log(f"showcase cand {cid} ({row_show['region']} Mw {row_show['mw']}); local subset K = {K} candidates "
        f"({sub.groupby('region').size().to_dict()})")

    # ---- 1. snapshots (one extra solver call)
    t = time.time()
    solver = Solver(grid, make_sampler(grid, pool), cfg, xp=np)
    eta0 = sources.source_eta0(grid, row_show, cfg)
    n_min = int(cfg["library"]["n_minutes"])
    times_s = np.arange(0, n_min + 1) * cfg["library"]["minute_s"]
    snaps = solver.snapshots(eta0[None], times_s)[0]                                    # (121, ny, nx)
    aoi = cfg["domain"]["area_of_interest"]
    jj = np.nonzero((grid.lat >= aoi["lat_deg"][0]) & (grid.lat <= aoi["lat_deg"][1]))[0]
    ii = np.nonzero((grid.lon >= aoi["lon_deg"][0]) & (grid.lon <= aoi["lon_deg"][1]))[0]
    crop = (slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1))
    np.savez_compressed(out / "showcase_snapshots.npz", eta=snaps[:, crop[0], crop[1]].astype(np.float16),
                        times_min=(times_s / 60).astype(np.int16), lon=grid.lon[crop[1]], lat=grid.lat[crop[0]],
                        wet=grid.wet[crop], depth=grid.depth[crop], cand_id=cid,
                        eta0_max_m=float(eta0.max()), eta0_min_m=float(eta0.min()))
    log(f"snapshots: {snaps.shape} -> {(out / 'showcase_snapshots.npz').stat().st_size / 1e6:.1f} MB in {time.time() - t:.1f}s")
    del snaps

    # ---- quiet days (noise only), aggregated per quiet traffic draw
    q, per_draw, ss_stat_quiet, q_nships, q_order = build_quiet(cfg, pool, a.quick)
    quiet_path = Path("/tmp/t27_showcase_quiet.npz")
    np.savez(quiet_path, meta=np.array(dict(traffic_seeds=sorted(per_draw)), dtype=object),
             **{f"D_{ts}": v["D"] for ts, v in per_draw.items()}, **{f"w_{ts}": v["w"] for ts, v in per_draw.items()})
    del per_draw

    # ---- 2. candidate subset through the solver (parallel workers)
    show = showcase_draw(pool, cfg)
    S = len(show.pool_idx)
    g_show = np.lib.format.open_memmap(out / "showcase_templates.npy", mode="w+", dtype=np.float32, shape=(K, S, n_min))
    stat_quiet = np.zeros((len(q), n_min), np.float32)
    rows = [dict(r, _showcase=cid) for r in sub.to_dict("records")]
    pos = {c: k for k, c in enumerate(sub.cand_id)}
    extra_show, t_solve, done = None, [], 0
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(str(quiet_path), a.quick)) as P:
        for c, g, q_stat, ts, tt, extra in P.imap_unordered(_run_candidate, rows):
            k = pos[c]
            g_show[k] = g
            for (tsd, qrows), st in zip(q_order, q_stat):
                stat_quiet[qrows] = np.maximum(stat_quiet[qrows], st)
            if extra:
                extra_show = extra
            t_solve.append(ts)
            done += 1
            if done % 10 == 0 or done == K:
                el = time.time() - T0
                log(f"candidates {done}/{K} (solver {np.mean(t_solve):.2f}s/run, total {tt:.2f}s); elapsed {el / 60:.1f} min")
    g_show.flush()
    quiet_path.unlink(missing_ok=True)
    solver_cpu_s = float(np.sum(t_solve))

    # ---- 4. threshold
    far = float(cfg["scoring"]["false_alarm_rate"])
    cal, hold = (q.split == "cal").values, (q.split == "hold").values
    thr_file = repo_path(cfg["paths"]["outputs"]) / "thresholds.csv"
    thr_kind, thr = None, None
    if thr_file.exists():
        th = pd.read_csv(thr_file)
        m = th[(th.operator == OPERATOR) & np.isclose(th.range_cap_m, range_cap(cfg)) & (th.regime == REGIME)
               & (th.detector == "network")]
        if len(m):
            thr, thr_kind = float(m.threshold.iloc[0]), "outputs/thresholds.csv (T17, full library)"
    thr_dev = scoring.calibrate_threshold(stat_quiet[cal], far)
    if thr is None:
        thr, thr_kind = thr_dev, f"dev threshold: local {K}-candidate subset"
    k_lit = float(cfg["single_ship"]["literal_k_sigma"])
    k_cal = bd.calibrated_threshold(ss_stat_quiet[cal], far)
    rates = dict(
        network_far_cal=float(np.mean(stat_quiet[cal].max(axis=-1) > thr)),
        network_far_hold=float(np.mean(stat_quiet[hold].max(axis=-1) > thr)),
        single_literal_far_cal=bd.false_alarm_rate(ss_stat_quiet[cal], k_lit),
        single_literal_far_hold=bd.false_alarm_rate(ss_stat_quiet[hold], k_lit),
        single_calibrated_k=k_cal,
        single_calibrated_far_hold=bd.false_alarm_rate(ss_stat_quiet[hold], k_cal),
        n_cal=int(cal.sum()), n_hold=int(hold.sum()),
        quiet_fleet_sizes={int(k): int(v) for k, v in q_nships.items()},
    )
    log(f"threshold {thr:.3f} ({thr_kind}); rates {rates}")

    # ---- 3. showcase observation and scoring
    eps = float(cfg["scoring"]["eps"])
    groups = scoring.build_groups(show, pool, cfg, OPERATOR)
    si, ss, st = sigma_total(show, cfg)
    k_show = pos[cid]
    sig = np.array(g_show[k_show])                                                       # (S, T) true signal
    nz = noise.ais_noise(show, pool, REGIME, cfg, np.random.default_rng([seed, PURPOSE_NOISE, SHOW_NOISE_INDEX]), 1)
    d = observe.observe(sig, nz)                                                         # (1, S, T)
    flip = sub.flip_allowed.values.astype(bool)
    G_all = np.asarray(g_show)
    gd2, gg2, Ssc, stat, parent, polarity = scoring.score_tensor(d, G_all, groups, si, ss, flip, eps, xp=np)
    stat = np.asarray(stat)[0]
    confirm = int(scoring.confirmation_minute(stat[None], thr)[0])
    L = scoring.log_likelihood(gd2, gg2)
    t_est = np.array([confirm])
    top = scoring.estimate_source(L, t_est, sub, parent, polarity, int(cfg["scoring"]["top_n"]))
    top_final = scoring.estimate_source(L, np.array([-1]), sub, parent, polarity, int(cfg["scoring"]["top_n"]))
    argk = np.asarray(Ssc)[0].argmax(axis=0)                                             # (T,) leader over time
    ss_show = bd.single_ship_stat(d, st.astype(np.float32))[0]
    lit_min = int(bd.alarm_minute(ss_show[None], k_lit)[0])
    cal_min = int(bd.alarm_minute(ss_show[None], k_cal)[0])
    z = np.abs(d[0]) / st[:, None]
    lit_ship = int(np.argmax(z[:, lit_min])) if lit_min >= 0 else -1
    snr_t = np.abs(sig) / st[:, None]

    gt = gauges.load_gauges(grid, cfg)
    trig = gauges.trigger_times(extra_show["gb30"][None], gt)[0]
    is_g, is_b = (gt.rtype == gauges.GAUGE).values, (gt.rtype == gauges.BUOY).values

    def first(sel):
        tt = trig[sel]
        if not np.isfinite(tt).any():
            return None, None
        k = int(np.argmin(tt))
        return float(tt[k]), str(gt.code.values[sel][k])

    fg_s, fg_code = first(is_g)
    fb_s, fb_code = first(is_b)

    # ---- inset ships
    ins = inset_draw(pool, cfg)
    sig_in = observe.ais_signal(extra_show["recf_inset"], ins)                          # (2, T) (already gathered)
    nz_in = noise.ais_noise(ins, pool, REGIME, cfg, np.random.default_rng([seed, PURPOSE_NOISE, INSET_NOISE_INDEX]), 1)[0]
    _, _, st_in = sigma_total(ins, cfg)
    snr_in = np.abs(sig_in).max(axis=1) / st_in

    np.savez_compressed(
        out / "showcase_traffic.npz",
        pool_idx=show.pool_idx, lon=pool.lon[show.pool_idx], lat=pool.lat[show.pool_idx],
        depth=pool.depth[show.pool_idx], heading=show.heading, sog=show.sog, sigma_total=st.astype(np.float32),
        cleaned_drift=d[0], true_signal=sig, snr_t=snr_t.astype(np.float32), peak_snr=snr_t.max(axis=1).astype(np.float32),
        net_stat=stat.astype(np.float32), threshold=thr, confirm_minute=confirm, leader_k=argk.astype(np.int32),
        sub_cand_id=sub.cand_id.values, sub_lon=sub.lon_deg.values, sub_lat=sub.lat_deg.values, sub_mw=sub.mw.values,
        sub_region=sub.region.values.astype("U12"),
        single_stat=ss_show.astype(np.float32),
        gauge_code=gt.code.values.astype("U24"), gauge_lon=gt.lon_deg.values, gauge_lat=gt.lat_deg.values,
        gauge_rtype=gt.rtype.values.astype(np.int8), gauge_trigger_s=trig,
        inset_pool_idx=ins.pool_idx, inset_lon=pool.lon[ins.pool_idx], inset_lat=pool.lat[ins.pool_idx],
        inset_depth=pool.depth[ins.pool_idx], inset_true=sig_in, inset_cleaned=(sig_in + nz_in).astype(np.float32),
        inset_sigma=st_in.astype(np.float32), inset_snr=snr_in.astype(np.float32),
        quiet_stat_cal=stat_quiet[cal], quiet_stat_hold=stat_quiet[hold],
        quiet_single_cal=ss_stat_quiet[cal].astype(np.float32), quiet_single_hold=ss_stat_quiet[hold].astype(np.float32),
    )

    # ---- 5. quiet-day example with the showcase traffic
    nzq = noise.ais_noise(show, pool, REGIME, cfg, np.random.default_rng([seed, PURPOSE_QNOISE, QUIET_EXAMPLE_NOISE]), 1)
    _, _, _, stat_q, _, _ = scoring.score_tensor(nzq, G_all, groups, si, ss, flip, eps, xp=np)
    stat_q = np.asarray(stat_q)[0]
    zq = np.abs(nzq[0]) / st[:, None]
    ex_s, ex_m = np.nonzero(zq > k_lit)
    ss_q = bd.single_ship_stat(nzq, st.astype(np.float32))[0]
    q_lit_min = int(bd.alarm_minute(ss_q[None], k_lit)[0])
    np.savez_compressed(out / "quiet_day_example.npz", ship=ex_s.astype(np.int32), minute=ex_m.astype(np.int16),
                        z=zq[ex_s, ex_m].astype(np.float32), lon=pool.lon[show.pool_idx][ex_s],
                        lat=pool.lat[show.pool_idx][ex_s], net_stat=stat_q.astype(np.float32), threshold=thr,
                        ship_lon=pool.lon[show.pool_idx], ship_lat=pool.lat[show.pool_idx])

    # ---- 6. meta
    def topdf(df):
        return [dict(rank=int(r.rank) + 1, cand_id=int(r.cand_id), lon_deg=float(r.lon_deg), lat_deg=float(r.lat_deg),
                     mw=float(r.mw), polarity=int(r.polarity), minute=int(r.minute),
                     region=str(sub.region.values[pos[int(r.cand_id)]])) for r in df.itertuples()]

    meta = dict(
        generated_by="scripts/demo/make_showcase_data.py", generated_at=time.strftime("%Y-%m-%d %H:%M"),
        quick=bool(a.quick), backend=os.environ.get("TSUNAMI_BACKEND"),
        showcase=dict(cand_id=cid, region=row_show["region"], mw=float(row_show["mw"]), lon_deg=float(row_show["lon_deg"]),
                      lat_deg=float(row_show["lat_deg"]), peak_uplift_m=float(eta0.max()),
                      uplift_rule=cfg["sources"]["uplift_rule"], uplift_dip_deg=float(row_show["uplift_dip_deg"]),
                      note="truth = library candidate itself (it is in the local subset): best case for the matched filter"),
        regime=REGIME, operator=OPERATOR, range_cap_m=range_cap(cfg),
        seeds=dict(base=seed, traffic=[PURPOSE_TRAFFIC, SHOW_TRAFFIC_INDEX], noise=[PURPOSE_NOISE, SHOW_NOISE_INDEX],
                   inset_noise=[PURPOSE_NOISE, INSET_NOISE_INDEX], quiet_example_noise=[PURPOSE_QNOISE, QUIET_EXAMPLE_NOISE],
                   quiet_days="data/truth/quiet_days.csv (traffic purpose 5, noise purpose 6)"),
        subset=dict(n_candidates=K, spec=(SUBSET_QUICK if a.quick else SUBSET),
                    by_region={k: int(v) for k, v in sub.groupby("region").size().items()},
                    n_flip=int(flip.sum()), solver_cpu_s=solver_cpu_s),
        n_ships=int(S), n_groups=int(groups.n_groups), sigma_total_median_m_s=float(np.median(st)),
        threshold=dict(value=float(thr), kind=thr_kind, dev_value=float(thr_dev), far_target=far,
                       label=thr_kind if thr_kind.startswith("dev") else "threshold: T17 thresholds.csv"),
        network=dict(confirm_minute_index=confirm, confirm_s=(60.0 * (confirm + 1) if confirm >= 0 else None),
                     peak_stat=float(stat.max()), peak_stat_minute=int(stat.argmax()) + 1,
                     final_stat=float(stat[-1]), top3=topdf(top), top3_final=topdf(top_final),
                     top3_minute=int(top.minute.iloc[0]) + 1),
        single_ship=dict(k_literal=k_lit, literal_alarm_minute_index=lit_min,
                         literal_alarm_s=(60.0 * (lit_min + 1) if lit_min >= 0 else None),
                         literal_alarm_ship=lit_ship, literal_alarm_ship_true_snr=(float(snr_t[lit_ship, lit_min]) if lit_min >= 0 else None),
                         literal_alarm_ship_lon_lat=([float(pool.lon[show.pool_idx][lit_ship]), float(pool.lat[show.pool_idx][lit_ship])] if lit_min >= 0 else None),
                         calibrated_k=k_cal, calibrated_alarm_s=(60.0 * (cal_min + 1) if cal_min >= 0 else None)),
        ships=dict(max_peak_snr=float(snr_t.max()), n_peak_snr_gt1=int((snr_t.max(axis=1) > 1).sum()),
                   n_peak_snr_gt3=int((snr_t.max(axis=1) > k_lit).sum())),
        instruments=dict(first_gauge_s=fg_s, first_gauge_code=fg_code, first_buoy_s=fb_s, first_buoy_code=fb_code,
                         n_gauges=int(is_g.sum()), n_gauges_triggered=int(np.isfinite(trig[is_g]).sum()),
                         n_buoys=int(is_b.sum()), n_buoys_triggered=int(np.isfinite(trig[is_b]).sum()),
                         window_s=float(cfg["solver"]["t_end_s"])),
        inset=dict(pool_idx=[int(x) for x in ins.pool_idx], depth_m=[float(x) for x in pool.depth[ins.pool_idx]],
                   snr=[float(x) for x in snr_in], sigma_total_m_s=[float(x) for x in st_in],
                   peak_true_m_s=[float(x) for x in np.abs(sig_in).max(axis=1)],
                   speed_kn=float(np.mean(cfg["traffic"]["speed_kn"])), heading="pool lane direction"),
        quiet=dict(rates, example=dict(n_exceedances=int(len(ex_s)), n_ships_exceeding=int(len(np.unique(ex_s))),
                                       literal_first_alarm_s=(60.0 * (q_lit_min + 1) if q_lit_min >= 0 else None),
                                       net_peak_stat=float(stat_q.max()), net_crosses=bool(stat_q.max() > thr),
                                       n_ships=int(S))),
        t17_thresholds_used=thr_kind.startswith("outputs/"),
        runtime_s=float(time.time() - T0),
    )
    (out / "showcase_meta.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"done in {(time.time() - T0) / 60:.1f} min -> {out}")
    log(json.dumps({k: meta[k] for k in ("threshold", "network", "single_ship", "ships", "instruments", "inset")}, indent=1, default=str))


if __name__ == "__main__":
    main()
