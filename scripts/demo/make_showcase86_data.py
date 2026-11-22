"""T27 demo data for scenes 8, 9, 9b (trial S, Hellenic Mw 8.6) and 9c (trial Q, Hellenic Mw 8.1): two REAL T17 sweep trials
reproduced locally, exactly as src/tsunami/sweep.py::event_shard and scripts/05_sweep.py compute them.

    cd ~/projects/mts-demo && TSUNAMI_BACKEND=numpy PYTHONPATH=src \
        ../med-tsunami-ais/.venv/bin/python scripts/demo/make_showcase86_data.py [--library subset|full] [--workers 4]

Code: the T17 sweep code (sweep.py, gauges.table_from_pool, config.yaml `sweep:`) is on main, not in this worktree's older src,
so this script imports tsunami from ../med-tsunami-ais/src (read only) and reads ../med-tsunami-ais/config.yaml. That is the code
the sweep ran with (git diff 671b433..main touches only a replay placeholder in config.yaml).

Trials (chosen from outputs/sweep/*.parquet, see choose() / the meta JSON "choice" entry):
  S: run_id 170 (hellenic truth_map2 pos 18, Mw 8.6), traffic draw 1, noise draw j = 15, calm, 40 km cap.
  Q: run_id 168 (hellenic truth_map2 pos 18, Mw 8.1), traffic draw 0, noise draw j = 15, calm, 40 km cap.
Reproduction (05_sweep.py, event stage):
  fleet  = sweep.make_fleet(pool, cfg, PURPOSE_TRAFFIC=1, traffic_seed)
  combo  = position of (traffic_seed, cap, operator, regime) in itertools.product(range(5), caps, operators, regimes)
  noise  = sweep.make_noise(..., default_rng([seed, PURPOSE_AIS|GPS, 1_000_000 + combo]), N = 20); draw j (noise_seed = combo*1000 + j)
  truth  = fresh local solver run of the truth row (uplift = slip x sin(uplift_dip_deg), uplift_dip_deg from data/truth/uplift_scale.csv;
           sources.csv has no uplift_dip_deg column and would otherwise fall back to the shape dip). No uplift_scale multiplier: that
           factor only converts the stored legacy runs (sweep.ScaledLibrary; tests/test_sweep.py::test_scaled_legacy_run_equals_new_rule_run).
  templates = same gather as sweep.gather_pass for every library candidate (local solver runs, cached per candidate under
           outputs/demo_data/cache86/); `--library subset` = candidates within SUBSET_KM of either trial's source, `full` = all 1,083.
  scoring = sweep.network_trials / scoring.confirmation_minute / sweep.top3_at with outputs/thresholds.csv (op, 40 km, calm, network).
Gauges: outputs/sweep/gauge_triggers.parquet for the run (plus a local recomputation from the truth run as a check).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TSUNAMI_BACKEND", "numpy")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[2]                    # this worktree (outputs/ and data/ are symlinks to the main folder)
MAIN = WT.parent / "med-tsunami-ais"                        # main checkout: T17 sweep code and config
sys.path.insert(0, str(MAIN / "src"))

from tsunami import gauges, scoring, ships, sources, sweep as SW  # noqa: E402
from tsunami.config import load_config  # noqa: E402
from tsunami.geometry import haversine_m  # noqa: E402
from tsunami.grid import load_grid  # noqa: E402
from tsunami.solver import Solver, make_sampler  # noqa: E402

OUT = WT / "outputs" / "demo_data"
CACHE = OUT / "cache86"
PROGRESS = OUT / "showcase86_progress.txt"
SWEEP = WT / "outputs" / "sweep"
TRUTH = WT / "data" / "truth"
CAND = WT / "data" / "processed" / "candidates.csv"
THR = WT / "outputs" / "thresholds.csv"

REGIME, CAP = "calm", 40000.0
SUBSET_KM = 400.0
# trials: prefix -> (run_id, traffic_seed, noise draw j); chosen by choose() below from the parquet (printed on every run)
TRIALS = {"showcase86": dict(run_id=170, traffic_seed=1, j=15, scenes="8, 9, 9b"),
          "showcase80": dict(run_id=168, traffic_seed=0, j=15, scenes="9c")}
# Cretan shelf: depth < SHELF_DEPTH_M inside the box around Crete and Gavdos (excludes Kythira, Kasos, Karpathos)
SHELF_BOX = dict(lon_deg=[23.3, 26.5], lat_deg=[34.7, 35.8])
SHELF_DEPTH_M = 200.0
SNR_LEVEL = 1.0


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)


def progress(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + stage + "\n")
    log("PROGRESS:", stage)


def jdump(o):
    def conv(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            v = x.item()
            return None if isinstance(v, float) and not np.isfinite(v) else v
        if isinstance(x, np.ndarray):
            return [conv(v) for v in x.tolist()]
        return str(x)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    return json.dumps(clean(json.loads(json.dumps(o, default=conv))), indent=1)


# ------------------------------------------------------------------------------------------- parquet pass (choice)

def load_net(cap=CAP, regime=REGIME):
    parts = sorted(SWEEP.glob(f"part_*_{int(cap)}_*_{regime}.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df[df.detector == "network"].reset_index(drop=True), [str(p.relative_to(WT)) for p in parts]


def choose(net, truth):
    """Print why TRIALS were chosen (the numbers come from the parquet). Returns a dict for the meta JSON."""
    t = truth.set_index("run_id")
    d = net.join(t[["region", "pos_id", "lon_deg", "lat_deg"]], on="run_id")
    d["j"] = d.noise_seed % 1000
    out = {}
    s = TRIALS["showcase86"]
    a = d[(d.operator == "ais") & (d.run_kind == "truth_map2") & (d.region == "hellenic") & np.isclose(d.truth_mw, 8.6)]
    pos = int(t.loc[s["run_id"], "pos_id"])
    ap = a[a.pos_id == pos]
    row = ap[(ap.traffic_seed == s["traffic_seed"]) & (ap.j == s["j"])].iloc[0]
    by_draw = ap.groupby("traffic_seed").apply(lambda x: x[x.detected].confirm_s.median() / 60, include_groups=False)
    out["S"] = dict(
        rule="hellenic truth_map2 Mw 8.6, pos 15-19 (~100% AIS detection); traffic draw whose median confirm at this position is "
             "closest to the stratum median; noise draw with confirm near the position median and the typical loc error; best "
             "candidate = the Mw 8.5 library source at the same position (GPS picks the same one)",
        pos_id=pos, pos_det_frac=float(ap.detected.mean()), pos_med_confirm_min=float(ap[ap.detected].confirm_s.median() / 60),
        pos_med_loc_km=float(ap[ap.detected].loc_err_m.median() / 1e3),
        pos_med_confirm_min_by_draw={int(k): float(v) for k, v in by_draw.items()},
        trial_confirm_min=float(row.confirm_s / 60), trial_loc_km=float(row.loc_err_m / 1e3), trial_best_cand=int(row.best_cand_id))
    q = TRIALS["showcase80"]
    aq = d[(d.operator == "ais") & (d.run_id == q["run_id"]) & (d.traffic_seed == q["traffic_seed"])]
    rq = aq[aq.j == q["j"]].iloc[0]
    out["Q"] = dict(
        rule="hellenic truth_map2 near Mw 8 (bins 7.85, 8.1: no 8.0 bin) south-west of Crete; traffic draw with no ships on the "
             "Cretan shelf; AIS network not confirmed within 2 h; estimate at 120 min within the typical ~13 km of the true source",
        mw=float(rq.truth_mw), n_undetected_of_20=int((~aq.detected).sum()),
        n_undetected_loc_lt_20km=int(((~aq.detected) & (aq.loc_err_m < 20e3)).sum()),
        trial_loc_km=float(rq.loc_err_m / 1e3), trial_best_cand=int(rq.best_cand_id))
    log("CHOICE S:", json.dumps(out["S"], default=str))
    log("CHOICE Q:", json.dumps(out["Q"], default=str))
    return out


def typical(net, truth, gtrig=None):
    t = truth.set_index("run_id")
    d = net.join(t[["region"]], on="run_id")
    s = d[(d.run_kind == "truth_map2") & (d.region == "hellenic") & np.isclose(d.truth_mw, 8.6)]
    a, g = s[s.operator == "ais"], s[s.operator == "gps"]
    if gtrig is None:
        gtrig = pd.read_parquet(SWEEP / "gauge_triggers.parquet")
    g3 = (gtrig[gtrig.rtype == 1].sort_values(["run_id", "trigger_s"])
          .groupby("run_id").nth(2).set_index("run_id")["trigger_s"])
    third_gauge_med_min = float(a[["run_id"]].join(g3.rename("third_s"), on="run_id")["third_s"].median() / 60)
    return dict(
        definition="outputs/sweep/part_*_40000_*_calm.parquet, detector network, hellenic truth_map2 Mw 8.6; medians over detected trials",
        n_trials_ais=int(len(a)), n_trials_gps=int(len(g)),
        ais_det_frac=float(a.detected.mean()), ais_med_confirm_min=float(a[a.detected].confirm_s.median() / 60),
        ais_med_loc_km=float(a[a.detected].loc_err_m.median() / 1e3), first_gauge_med_min=float(a.first_gauge_s.median() / 60),
        third_gauge_med_min=third_gauge_med_min,
        gps_det_frac=float(g.detected.mean()), gps_med_confirm_min=float(g[g.detected].confirm_s.median() / 60),
        gps_med_loc_km=float(g[g.detected].loc_err_m.median() / 1e3))


# ------------------------------------------------------------------------------------------- helpers

def combo_index(cfg, traffic_seed, cap, op, regime):
    """Index n of the shard in 05_sweep.py's enumeration (its noise rng and noise_seed = n*1000 + j)."""
    caps = sorted(cfg["traffic"]["range_caps_m"])
    combos = list(itertools.product(range(cfg["sweep"]["n_traffic_draws"]), caps, cfg["operators"], list(cfg["weather_regimes"])))
    return combos.index((traffic_seed, cap, op, regime))


def fleet_sig(f):
    h = hashlib.sha1()
    d = f.draws[CAP]
    for a in (d.pool_idx, d.heading):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def truth_row(truth, tscale, run_id):
    r = truth[truth.run_id == run_id].iloc[0].to_dict()
    r["uplift_dip_deg"] = float(tscale.set_index("run_id").loc[run_id, "uplift_dip_deg"])
    return r


# ------------------------------------------------------------------------------------------- workers

_W = {}


def _init_worker(traffic_seeds):
    cfg = load_config(MAIN / "config.yaml")
    grid, pool = load_grid(cfg), ships.load_pool(cfg)
    fleets = {i: SW.make_fleet(pool, cfg, SW.PURPOSE_TRAFFIC, i) for i in traffic_seeds}
    _W.update(cfg=cfg, grid=grid, pool=pool, fleets=fleets, solver=Solver(grid, make_sampler(grid, pool), cfg, xp=np),
              gb_idx=np.nonzero(np.asarray(pool.rtype) > 0)[0], gtable=gauges.table_from_pool(pool, cfg))


def _project(recf):
    """Per-ship projected signal (1, S40, 120) for every fleet and operator: the gather of sweep.gather_pass for the 40 km cap."""
    out = {}
    for i, f in _W["fleets"].items():
        sub = recf[:, :, f.draw100.pool_idx, :]
        sc = sub[:, :, f.masks[CAP], :]
        for op in _W["cfg"]["operators"]:
            out[f"{i}_{op}"] = scoring.project_templates(sc, f.draws[CAP], op)[0]
        out[f"sig_{i}"] = np.array(fleet_sig(f))
    return out


def _task(job):
    kind, key, row, path = job
    cfg, grid, solver = _W["cfg"], _W["grid"], _W["solver"]
    t0 = time.time()
    eta0 = sources.source_eta0(grid, row, cfg)
    if kind == "snap":
        n_min = int(cfg["library"]["n_minutes"])
        times_s = np.arange(0, n_min + 1) * float(cfg["library"]["minute_s"])
        snaps = np.asarray(solver.snapshots(eta0[None], times_s))[0]
        aoi = cfg["domain"]["area_of_interest"]
        jj = np.nonzero((grid.lat >= aoi["lat_deg"][0]) & (grid.lat <= aoi["lat_deg"][1]))[0]
        ii = np.nonzero((grid.lon >= aoi["lon_deg"][0]) & (grid.lon <= aoi["lon_deg"][1]))[0]
        cj, ci = slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1)
        tmp = Path(str(path) + ".tmp.npz")
        np.savez_compressed(tmp, eta=snaps[:, cj, ci].astype(np.float16), lon=grid.lon[ci], lat=grid.lat[cj],
                            t_min=(times_s / 60).astype(np.int16), wet=grid.wet[cj, ci], depth=grid.depth[cj, ci],
                            run_id=int(row["run_id"]), eta0_max_m=float(eta0.max()), eta0_min_m=float(eta0.min()),
                            eta_note=np.array("raw solver eta(t) (sea-surface height incl. the static initial offset), m"))
        tmp.rename(path)
        return kind, key, time.time() - t0
    recf, gb30 = solver.run_library(eta0[None], _W["gb_idx"])
    recf = np.asarray(recf)
    out = _project(recf)
    if kind == "truth":
        out["gb30"] = np.asarray(gb30)[0]
        out["eta0_gb"] = gauges.eta0_at_receivers(eta0, _W["gtable"])
        out["eta0_max_m"] = np.array(float(eta0.max()))
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **out)
    tmp.rename(path)
    return kind, key, time.time() - t0


def cache_ok(path, sigs):
    if not path.exists():
        return False
    try:
        z = np.load(path)
        return all(str(z[f"sig_{i}"]) == s for i, s in sigs.items())
    except Exception:
        return False


def run_jobs(jobs, workers, traffic_seeds, label):
    if not jobs:
        return
    from multiprocessing import get_context
    t0, ts, n = time.time(), [], 0
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(traffic_seeds,)) as P:
        for kind, key, dt in P.imap_unordered(_task, jobs):
            n += 1
            ts.append(dt)
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                log(f"{label}: {n}/{len(jobs)} ({np.mean(ts):.1f} s/run/process, elapsed {el/60:.1f} min, "
                    f"eta {el/n*(len(jobs)-n)/60:.1f} min)")
            if n % 100 == 0:
                progress(f"{label}: {n}/{len(jobs)} solver runs done")


# ------------------------------------------------------------------------------------------- per-trial scoring

def score_trial(prefix, spec, cfg, pool, grid, fleets, cand_k, G_ship, truth, net, thr_df, gtrig, gtable, lib_label, choice, typ,
                files_read):
    T = int(cfg["library"]["n_minutes"])
    N = int(cfg["sweep"]["n_noise_draws"])
    R = cfg["physics"]["earth_radius_m"]
    eps = float(cfg["scoring"]["eps"])
    run_id, ts, j = spec["run_id"], spec["traffic_seed"], spec["j"]
    f = fleets[ts]
    d = f.draws[CAP]
    S = len(d.pool_idx)
    tr = truth[truth.run_id == run_id].iloc[0]
    zt = np.load(CACHE / f"truth_{run_id}.npz")
    flip = cand_k["flip_allowed"].values.astype(bool)
    res, arrays = {}, {}
    plon, plat, pdep = pool.lon[d.pool_idx], pool.lat[d.pool_idx], pool.depth[d.pool_idx]
    for op in cfg["operators"]:
        n_combo = combo_index(cfg, ts, CAP, op, REGIME)
        groups = f.groups[(CAP, op)]
        si, ss, st = SW.sigmas(d, REGIME, op, cfg)
        w = scoring.cell_weights(groups, si, ss, None)
        purpose = SW.PURPOSE_AIS if op == "ais" else SW.PURPOSE_GPS
        rng = np.random.default_rng([cfg["meta"]["seed"], purpose, 1_000_000 + n_combo])
        nz = SW.make_noise(pool, f, CAP, op, REGIME, cfg, rng, N)                        # (N, S, 120): all 20 draws of the shard
        Dn = scoring.aggregate(nz, groups, None, xp=np)
        sig = zt[f"{ts}_{op}"][None]                                                      # (1, S, 120) true signal
        Ds = scoring.aggregate(sig, groups, None, xp=np)                                  # as gather_pass does for the truth runs
        D = (Ds[:, None] + Dn[None]).reshape(N, groups.n_groups, -1)
        G = np.asarray(scoring.aggregate(G_ship[f"{ts}_{op}"], groups, None, xp=np), np.float32)
        tr_thr = thr_df[(thr_df.operator == op) & np.isclose(thr_df.range_cap_m, CAP) & (thr_df.regime == REGIME)
                        & (thr_df.detector == "network")].threshold
        thr = float(tr_thr.iloc[0])
        gd2, gg2, stat, parent, polarity = SW.network_trials(G, D, w, flip, eps)
        stat = np.asarray(stat)
        c_net = scoring.confirmation_minute(stat, thr)
        t_idx = np.where(c_net >= 0, c_net, stat.shape[1] - 1)
        top = SW.top3_at(gd2, gg2, t_idx)
        L = 2.0 * np.asarray(gd2)[j] - np.asarray(gg2)                                    # (K2, T) leader per minute for trial j
        lead = parent[L.argmax(axis=0)]
        bp_all = parent[top[:, 0]]
        # compare all N draws of this (event, traffic draw, op) with the parquet
        pq = net[(net.run_id == run_id) & (net.traffic_seed == ts) & (net.operator == op)].copy()
        pq["j"] = pq.noise_seed - n_combo * 1000
        pq = pq.set_index("j").sort_index()
        assert (pq.noise_seed // 1000 == n_combo).all(), "combo enumeration does not match the parquet noise_seed"
        loc_conf = np.where(c_net >= 0, 60.0 * (c_net + 1), np.inf)
        best_all = cand_k.cand_id.values[bp_all]
        m_conf = np.array([loc_conf[k] == pq.loc[k, "confirm_s"] for k in range(N)])
        m_best = np.array([best_all[k] == pq.loc[k, "best_cand_id"] for k in range(N)])
        row = pq.loc[j]
        bp = bp_all[j]
        cr = cand_k.iloc[bp]
        loc_err = float(haversine_m(cr.lon_deg, cr.lat_deg, tr.lon_deg, tr.lat_deg, R))
        top3 = [dict(cand_id=int(cand_k.cand_id.values[parent[k2]]), lon=float(cand_k.lon_deg.values[parent[k2]]),
                     lat=float(cand_k.lat_deg.values[parent[k2]]), mw=float(cand_k.mw.values[parent[k2]]),
                     polarity=int(polarity[k2])) for k2 in top[j]]
        conf_s = float(loc_conf[j])
        res[op] = dict(
            detected=bool(c_net[j] >= 0), confirm_s=conf_s, confirm_min=(conf_s / 60 if np.isfinite(conf_s) else None),
            threshold=thr, best_cand_id=int(cr.cand_id), best_lon=float(cr.lon_deg), best_lat=float(cr.lat_deg), best_mw=float(cr.mw),
            loc_err_km=loc_err / 1e3, estimate_minute=int(t_idx[j]) + 1, top3=top3,
            peak_stat=float(stat[j].max()), peak_stat_minute=int(stat[j].argmax()) + 1, stat_at_120=float(stat[j, -1]),
            parquet=dict(confirm_s=float(row.confirm_s), best_cand_id=int(row.best_cand_id), loc_err_km=float(row.loc_err_m / 1e3),
                         top3_cand_ids=str(row.top3_cand_ids), noise_seed=int(row.noise_seed)),
            parquet_match=dict(confirm=bool(m_conf[j]), best_cand=bool(m_best[j]),
                               top3=";".join(str(t["cand_id"]) for t in top3) == str(row.top3_cand_ids),
                               all_noise_draws_confirm=f"{int(m_conf.sum())}/{N}", all_noise_draws_best_cand=f"{int(m_best.sum())}/{N}",
                               mismatched_draws=[int(k) for k in range(N) if not (m_conf[k] and m_best[k])]),
            library=lib_label, combo_index=n_combo, noise_seed=int(n_combo * 1000 + j), n_groups=int(groups.n_groups))
        snr = np.abs(sig[0]) / st[:, None]
        arrays[op] = dict(true=sig[0].astype(np.float32), obs=(sig[0] + nz[j]).astype(np.float32), st=st.astype(np.float32),
                          snr=snr.astype(np.float32), stat=stat[j].astype(np.float32), thr=thr,
                          lead_lon=cand_k.lon_deg.values[lead], lead_lat=cand_k.lat_deg.values[lead], lead_mw=cand_k.mw.values[lead],
                          lead_id=cand_k.cand_id.values[lead])
        log(f"{prefix} {op}: confirm {res[op]['confirm_min']} min (parquet {row.confirm_s/60}), best {int(cr.cand_id)} "
            f"(parquet {int(row.best_cand_id)}), loc {loc_err/1e3:.2f} km; all {N} draws: confirm {m_conf.sum()}/{N}, best {m_best.sum()}/{N}")

    # ---- shelf ships, ships above noise
    box = ((plon >= SHELF_BOX["lon_deg"][0]) & (plon <= SHELF_BOX["lon_deg"][1]) & (plat >= SHELF_BOX["lat_deg"][0])
           & (plat <= SHELF_BOX["lat_deg"][1]))
    shelf = box & (pdep < SHELF_DEPTH_M)
    # "south of Crete": walking north along the ship's grid column, land is met before the box's northern edge
    south = np.zeros(S, bool)
    for k in np.nonzero(box)[0]:
        ci = int(np.argmin(np.abs(grid.lon - plon[k])))
        col = (grid.lat > plat[k]) & (grid.lat <= SHELF_BOX["lat_deg"][1])
        south[k] = bool((~grid.wet[col, ci].astype(bool)).any())
    snr_a = arrays["ais"]["snr"]
    peak = snr_a.max(axis=1)
    above = peak > SNR_LEVEL
    near_box = box
    # ---- gauges
    gt = gtrig[gtrig.run_id == run_id].set_index("code").loc[gtable.code.values]
    trig = gt.trigger_s.values.astype(float)
    is_g, is_b = gtable.rtype.values == gauges.GAUGE, gtable.rtype.values == gauges.BUOY
    trig_local = gauges.trigger_times(zt["gb30"][None], gtable, zt["eta0_gb"][None], cfg["solver"]["sample_interval_s"])[0]
    pq_fg = float(net[net.run_id == run_id].first_gauge_s.iloc[0])

    def first(sel):
        tt = np.where(sel, trig, np.inf)
        k = int(np.argmin(tt))
        return (float(tt[k]), str(gtable.code.values[k])) if np.isfinite(tt[k]) else (None, None)

    fg_s, fg_code = first(is_g)
    fb_s, fb_code = first(is_b)
    trial = dict(
        run_id=int(run_id), region=str(tr.region), pos_id=int(tr.pos_id), run_kind=str(tr.run_kind), mw=float(tr.mw),
        lon=float(tr.lon_deg), lat=float(tr.lat_deg), strike_deg=float(tr.strike_deg), dip_deg=float(tr.dip_deg),
        uplift_dip_deg=float(truth_row(truth, pd.read_csv(TRUTH / "uplift_scale.csv"), run_id)["uplift_dip_deg"]),
        length_km=float(tr.length_m / 1e3), width_km=float(tr.width_m / 1e3), slip_m=float(tr.slip_m),
        peak_uplift_m=float(zt["eta0_max_m"]), depth_note="sources.csv has no hypocentre depth (Gaussian surface source)",
        traffic_seed=int(ts), noise_draw_j=int(j), noise_seed_ais=res["ais"]["noise_seed"], noise_seed_gps=res["gps"]["noise_seed"],
        noise_seed_note="noise_seed = combo_index*1000 + j; the GPS shard has its own combo_index (3 later in 05_sweep's enumeration), "
                        "so the GPS trial is the same event and traffic draw with draw j of the GPS shard's noise",
        regime=REGIME, range_cap_m=CAP, n_ships=int(S),
        shelf=dict(definition=f"pool depth < {SHELF_DEPTH_M:.0f} m inside lon {SHELF_BOX['lon_deg']} x lat {SHELF_BOX['lat_deg']} "
                              "(Crete and Gavdos; excludes Kythira, Kasos, Karpathos)",
                   south_definition="shelf ship with Cretan land due north of it inside the box (grid column walk)",
                   n_ships_in_box=int(box.sum()), n_shelf_ships=int(shelf.sum()), n_shelf_ships_south=int((shelf & south).sum()),
                   n_box_ships_south_of_crete=int((box & south).sum()),
                   box_ships=[dict(lon=float(plon[k]), lat=float(plat[k]), depth_m=float(pdep[k]), south=bool(south[k]),
                                   peak_snr_ais=float(peak[k])) for k in np.nonzero(near_box)[0]]),
        n_shelf_ships=int(shelf.sum()))
    gps_peak = arrays["gps"]["snr"].max(axis=1)
    meta = dict(
        trial=trial, ais=res["ais"], gps=res["gps"],
        gauges=dict(n_gauges=int(is_g.sum()), n_buoys=int(is_b.sum()), first_gauge_s=fg_s, first_gauge_min=(fg_s / 60 if fg_s else None),
                    first_gauge_name=fg_code, first_buoy_s=fb_s, first_buoy_min=(fb_s / 60 if fb_s else None), first_buoy_name=fb_code,
                    n_gauges_triggered=int(np.isfinite(trig[is_g]).sum()), n_buoys_triggered=int(np.isfinite(trig[is_b]).sum()),
                    window_s=float(cfg["solver"]["t_end_s"]),
                    trigger_s={str(c): (float(v) if np.isfinite(v) else None) for c, v in zip(gtable.code.values, trig)},
                    parquet_first_gauge_s=pq_fg, first_gauge_matches_parquet=bool(fg_s == pq_fg),
                    local_recomputed_match=bool(np.array_equal(trig_local, trig)),
                    local_recomputed_n_differ=int((trig_local != trig).sum()),
                    source="outputs/sweep/gauge_triggers.parquet (eta(t) - eta(0)); local truth run recomputed as a check"),
        ships_above_noise=dict(definition=f"AIS: peak over 120 min of |true cleaned signal| / sigma_total > {SNR_LEVEL:g}",
                               n_above=int(above.sum()), n_total=int(S), max_snr=float(peak.max()),
                               max_snr_ship=dict(lon=float(plon[peak.argmax()]), lat=float(plat[peak.argmax()]),
                                                 depth_m=float(pdep[peak.argmax()])),
                               n_above_3=int((peak > 3).sum()),
                               above_depth_m_median=(float(np.median(pdep[above])) if above.any() else None),
                               above_n_depth_lt_200=int((above & (pdep < 200)).sum()),
                               above_ships=[dict(lon=float(plon[k]), lat=float(plat[k]), depth_m=float(pdep[k]), peak_snr=float(peak[k]))
                                            for k in np.argsort(-peak)[:int(above.sum())]],
                               gps_n_above=int((gps_peak > SNR_LEVEL).sum()), gps_max_snr=float(gps_peak.max())),
        typical=typ, choice=choice,
        provenance=dict(script="scripts/demo/make_showcase86_data.py", date=time.strftime("%Y-%m-%d %H:%M"),
                        git_commit_worktree=_git(WT), git_commit_code=_git(MAIN), code_path=str(MAIN / "src"),
                        files_read=files_read, library=lib_label, n_candidates=int(len(cand_k)),
                        backend=os.environ.get("TSUNAMI_BACKEND"),
                        note="truth signal and templates from fresh local NumPy solver runs (current uplift rule, no uplift_scale); "
                             "the sweep used stored legacy runs x uplift_scale on Colab, equal by linearity up to float32 rounding"),
    )
    (OUT / f"{prefix}_meta.json").write_text(jdump(meta))
    gl = gtable
    np.savez_compressed(
        OUT / f"{prefix}_traffic.npz",
        lon=plon.astype(np.float32), lat=plat.astype(np.float32), depth_m=pdep.astype(np.float32), pool_idx=d.pool_idx,
        heading=np.asarray(d.heading), sog=np.asarray(d.sog), shelf=shelf, south_of_crete=south,
        sigma_total=arrays["ais"]["st"], true_drift=arrays["ais"]["true"], obs_drift=arrays["ais"]["obs"], snr_t=arrays["ais"]["snr"],
        gps_sigma_total=arrays["gps"]["st"], gps_true=arrays["gps"]["true"], gps_obs=arrays["gps"]["obs"], gps_snr_t=arrays["gps"]["snr"],
        stat_ais=arrays["ais"]["stat"], stat_gps=arrays["gps"]["stat"], thr_ais=arrays["ais"]["thr"], thr_gps=arrays["gps"]["thr"],
        lead_lon_ais=arrays["ais"]["lead_lon"], lead_lat_ais=arrays["ais"]["lead_lat"], lead_mw_ais=arrays["ais"]["lead_mw"],
        lead_cand_ais=arrays["ais"]["lead_id"], lead_lon_gps=arrays["gps"]["lead_lon"], lead_lat_gps=arrays["gps"]["lead_lat"],
        lead_mw_gps=arrays["gps"]["lead_mw"], lead_cand_gps=arrays["gps"]["lead_id"],
        g_lon=gl.lon_deg.values, g_lat=gl.lat_deg.values, g_trigger_s=trig, g_is_buoy=is_b, g_code=gl.code.values.astype("U24"),
        g_trigger_local_s=trig_local, t_min=np.arange(1, T + 1, dtype=np.int16),
        src_lon=float(tr.lon_deg), src_lat=float(tr.lat_deg), src_mw=float(tr.mw),
        t_min_note=np.array("minute index k holds minute k+1 after origin (confirmation time = 60 (k+1) s)"))
    return meta


def _git(p):
    try:
        return subprocess.check_output(["git", "-C", str(p), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# ------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=["subset", "full"], default="subset")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="pipeline test: nearest N candidates, outputs to /tmp/t27_smoke86")
    a = ap.parse_args()
    global OUT, PROGRESS
    if a.smoke:
        OUT = Path("/tmp/t27_smoke86")
        PROGRESS = OUT / "progress.txt"
    assert a.workers <= 4, "at most 4 worker processes on this machine"
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    progress(f"start --library {a.library} (workers {a.workers})")
    cfg = load_config(MAIN / "config.yaml")
    grid, pool = load_grid(cfg), ships.load_pool(cfg)
    truth = pd.read_csv(TRUTH / "sources.csv").sort_values("run_id").reset_index(drop=True)
    tscale = pd.read_csv(TRUTH / "uplift_scale.csv")
    cand = pd.read_csv(CAND).sort_values("cand_id").reset_index(drop=True)
    thr_df = pd.read_csv(THR)
    net, parts = load_net()
    gtrig = pd.read_parquet(SWEEP / "gauge_triggers.parquet")
    gtable = gauges.table_from_pool(pool, cfg)
    files_read = parts + ["outputs/sweep/gauge_triggers.parquet", "outputs/thresholds.csv", "data/truth/sources.csv",
                          "data/truth/uplift_scale.csv", "data/processed/candidates.csv", "data/processed/receiver_pool.npz",
                          "data/processed/grid.npz", "../med-tsunami-ais/config.yaml"]
    choice = choose(net, truth)
    typ = typical(net, truth, gtrig)
    log("typical:", typ)

    seeds = sorted({s["traffic_seed"] for s in TRIALS.values()})
    fleets = {i: SW.make_fleet(pool, cfg, SW.PURPOSE_TRAFFIC, i) for i in seeds}
    sigs = {i: fleet_sig(f) for i, f in fleets.items()}
    R = cfg["physics"]["earth_radius_m"]
    dist = np.full(len(cand), np.inf)
    for s in TRIALS.values():
        tr = truth[truth.run_id == s["run_id"]].iloc[0]
        dist = np.minimum(dist, haversine_m(cand.lon_deg.values, cand.lat_deg.values, tr.lon_deg, tr.lat_deg, R) / 1e3)
    keep = np.ones(len(cand), bool) if a.library == "full" else dist <= SUBSET_KM
    if a.smoke:
        keep = dist <= np.sort(dist)[a.smoke - 1]
    cand_k = cand[keep].reset_index(drop=True)
    lib_label = "full 1083" if (a.library == "full" and len(cand_k) == len(cand)) else f"subset {len(cand_k)} (within {SUBSET_KM:.0f} km)"
    log(f"library: {lib_label}")

    # ---- solver jobs: truth runs + snapshots first, then candidates nearest first
    jobs = []
    for prefix, s in TRIALS.items():
        r = truth_row(truth, tscale, s["run_id"])
        p = CACHE / f"truth_{s['run_id']}.npz"
        if not cache_ok(p, sigs):
            jobs.append(("truth", s["run_id"], r, p))
        ps = OUT / f"{prefix}_snapshots.npz"
        if not ps.exists():
            jobs.append(("snap", s["run_id"], r, ps))
    order = np.argsort(dist[keep])
    n_cached = 0
    for k in order:
        r = cand_k.iloc[k].to_dict()
        p = CACHE / f"cand_{int(r['cand_id']):04d}.npz"
        if cache_ok(p, sigs):
            n_cached += 1
        else:
            jobs.append(("cand", int(r["cand_id"]), r, p))
    log(f"solver jobs: {len(jobs)} ({n_cached} candidates cached)")
    progress(f"{len(jobs)} solver runs to do ({n_cached} candidates already cached), library {lib_label}")
    run_jobs(jobs, a.workers, seeds, a.library)
    progress("solver runs done; scoring")

    # ---- load per-ship templates
    G_ship = {}
    for i in seeds:
        for op in cfg["operators"]:
            G_ship[f"{i}_{op}"] = np.empty((len(cand_k), len(fleets[i].draws[CAP].pool_idx), int(cfg["library"]["n_minutes"])), np.float32)
    for k, c in enumerate(cand_k.cand_id.values):
        z = np.load(CACHE / f"cand_{int(c):04d}.npz")
        for key in G_ship:
            G_ship[key][k] = z[key]

    metas = {}
    for prefix, s in TRIALS.items():
        metas[prefix] = score_trial(prefix, s, cfg, pool, grid, fleets, cand_k, G_ship, truth, net, thr_df, gtrig, gtable, lib_label,
                                    choice["S" if prefix == "showcase86" else "Q"], typ if prefix == "showcase86" else
                                    dict(typ, note="typical values are for the Mw 8.6 stratum of scenes 8-9"), files_read)
        m = metas[prefix]
        progress(f"{prefix} files written ({lib_label}): AIS confirm {m['ais']['confirm_min']} min "
                 f"(parquet {m['ais']['parquet']['confirm_s']/60:g}), best {m['ais']['best_cand_id']} (parquet {m['ais']['parquet']['best_cand_id']}), "
                 f"loc {m['ais']['loc_err_km']:.2f} km; GPS confirm {m['gps']['confirm_min']} min, best {m['gps']['best_cand_id']} "
                 f"(parquet {m['gps']['parquet']['best_cand_id']}); match AIS {m['ais']['parquet_match']} GPS {m['gps']['parquet_match']}")
    progress(f"done --library {a.library} in {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
