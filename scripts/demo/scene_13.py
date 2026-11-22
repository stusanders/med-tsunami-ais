"""Scene 13 (Results): what ships could confirm, and how fast.

On-screen text is docs/DEMO_SCRIPT.md scene 13, word for word; every number in it is generated here from the files.

Data (read at render time, ~0.2 s; no cache needed):
  * outputs/sweep/part_*_<cap>_*.parquet (T17, docs/interfaces.md section 14): network rows only, joined on run_id
    with data/truth/sources.csv rows of run_kind "truth_map2" (region, pos_id, mw). Positions are pos_id (one
    source position, nine tested magnitudes; the truth lon/lat moves slightly with Mw, so the dot sits at the
    median over magnitudes). Cap = the 40 km range cap (smallest of config traffic.range_caps_m), as in the
    script's source table ("map2 events, calm, 40 km cap, network").
  * outputs/map3_table.csv (T18, Stream C): per coast (nearest tide gauge of each event), column
    gain_med_min_first_instrument_s = median over detected events of (first tide gauge or buoy - ship confirmation)
    in minutes, same cap; NaN = never confirmed.

Definitions:
  * Left map, per position: smallest tested Mw at which the network confirms the event on at least half of the
    traffic x noise draws (the T18 mw_50 floor, recomputed from the sweep per pos_id). Tabs: AIS calm, moderate,
    rough, then precise GPS (calm; GPS noise does not depend on the weather in config.yaml). The right map follows
    the same tab.
  * Key line 1 (AIS calm): per region, the magnitude at which the region's detection rate (all map2 events in
    the region pooled) crosses 50 %, linearly interpolated between tested magnitudes and rounded to the nearest
    0.5. "only near Mw X" when the crossing falls in the region's last tested interval (just below its largest
    tested Mw), otherwise "from Mw X". Regions sharing a value and wording are joined ("Hellenic and Calabrian
    Arcs"); ordered by value, ties in config.yaml sources.arcs order.
    (T17: Aegean 7.56 -> 7.5, Cyprus 8.11 -> 8, Calabrian 8.42 -> 8.5, Hellenic 8.49 -> 8.5.)
    Suffix from T26 (outputs/historic_replays_summary.csv, historic_thresholds_arcs_plus_aegean.csv): AIS network
    rows, catalogue arcs_plus_aegean, both range caps, calm; an event is "not detected" when every such row is at
    or below the false-alarm level, i.e. its detections (detected_frac x trials) do not exceed the 95th percentile
    of Binomial(trials, far_hold). (T26: Amorgos 1%/1%, Samos 1%/1% of 100 trials vs far_hold 0.2%/1.6%.)
    Names in file order; the region word from data/historic_runs/sources.csv.
  * Key line 2 (AIS calm): share of detected map2 events with confirm_s < first_gauge_s, per region; the maximum
    and its region. (T17: Aegean 11.8 %.)
  * Key line 3 (GPS calm): the largest regional 50 % crossing, rounded to 0.5 ("from Mw X everywhere"), and the
    min to max over regions of the same before-the-first-gauge share. (T17: 6.94 -> 7; 72.5 to 91.8 %.)
These match outputs/detector_comparison.csv (Stream C) det_rate_* and pct_before_first_gauge.
"""
from __future__ import annotations

import os
from pathlib import Path

import common as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch

SCENE = 13
RES = Path(os.environ.get("T27_RESULTS_DIR") or cm.OUTPUTS)
LEVEL = 0.5                                     # detection-rate level that defines "can confirm"
STATES = [("ais", "calm", "AIS drift: Calm"), ("ais", "moderate", "AIS drift: Moderate"), ("ais", "rough", "AIS drift: Rough"),
          ("gps", "calm", "Ships with precise GPS")]
MAP3_COL = "gain_med_min_first_instrument_s"    # against tide gauges and buoys
GAIN_LIM = 30.0                                 # colour bar +-30 min (extends beyond)
REGION_NAME = {"aegean": "Aegean", "cyprus": "Cyprus Arc", "hellenic": "Hellenic Arc", "calabrian": "Calabrian Arc"}
MW_BOUNDS = [6.5, 7.25, 7.75, 8.25, 8.75]
MW_COLS = ["#00E676", "#FFE066", "#FF8C24", "#D92B4B"]
MW_CMAP = mcolors.ListedColormap(MW_COLS)
MW_NORM = mcolors.BoundaryNorm(MW_BOUNDS, len(MW_COLS))
GAIN_CMAP = mcolors.LinearSegmentedColormap.from_list("gain", [cm.C["bad"], "#C9CFDA", cm.C["good"]])
NONE_EDGE = mcolors.to_rgba(cm.MUTED, 0.9)


# ------------------------------------------------------------------ data (also used by scene 15)
def cap_m(cfg):
    return float(min(cfg["traffic"]["range_caps_m"]))


def load_map2_events(cfg, cap=None):
    """Network rows of the T17 sweep for map2 truth events at one range cap, with region, pos_id, mw. None if absent."""
    cap = cap_m(cfg) if cap is None else cap
    files = sorted((RES / "sweep").glob(f"part_*_{cap:.0f}_*.parquet")) or sorted((RES / "sweep").glob("part_*.parquet"))
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        return None
    cols = ["run_id", "operator", "regime", "range_cap_m", "detector", "detected", "confirm_s", "first_gauge_s",
            "first_instrument_s"]
    d = pd.concat([pd.read_parquet(f, columns=cols, filters=[("detector", "==", "network")]) for f in files],
                  ignore_index=True)
    d = d[np.isclose(d.range_cap_m, cap)]
    tr = pd.read_csv(cm.REPO / "data/truth/sources.csv")
    tr = tr[tr.run_kind == "truth_map2"][["run_id", "region", "pos_id", "mw", "lon_deg", "lat_deg"]]
    ev = d.merge(tr, on="run_id", how="inner")
    return ev if len(ev) else None


def positions(ev):
    return ev.groupby("pos_id", as_index=False).agg(region=("region", "first"), lon_deg=("lon_deg", "median"),
                                                    lat_deg=("lat_deg", "median"))


def position_floor(ev, op, reg, level=LEVEL):
    """Per pos_id: smallest tested Mw with detection rate >= level (NaN if none)."""
    s = ev[(ev.operator == op) & (ev.regime == reg)]
    fr = s.groupby(["pos_id", "mw"]).detected.mean().reset_index()
    return fr[fr.detected >= level].groupby("pos_id").mw.min().rename("mw_floor")


def region_crossing(ev, op, reg, level=LEVEL):
    """Per region: dict(x=interpolated Mw where the pooled detection rate crosses `level`, last=crossing in the last
    tested interval, below=already >= level at the smallest tested Mw); x NaN if never reached."""
    s = ev[(ev.operator == op) & (ev.regime == reg)]
    out = {}
    for region, g in s.groupby("region"):
        r = g.groupby("mw").detected.mean().sort_index()
        mw, p = r.index.to_numpy(float), r.to_numpy(float)
        hit = np.flatnonzero(p >= level)
        if not len(hit):
            out[region] = dict(x=np.nan, last=False, below=False, mw_max=mw[-1])
            continue
        i = hit[0]
        x = mw[0] if i == 0 else mw[i - 1] + (mw[i] - mw[i - 1]) * (level - p[i - 1]) / (p[i] - p[i - 1])
        out[region] = dict(x=float(x), last=bool(len(mw) > 1 and x > mw[-2]), below=i == 0, mw_max=mw[-1])
    return out


def pct_before_first_gauge(ev, op, reg):
    """Per region: % of detected map2 events confirmed before the first tide gauge triggers."""
    s = ev[(ev.operator == op) & (ev.regime == reg) & ev.detected.astype(bool)]
    return 100.0 * (s.confirm_s < s.first_gauge_s).groupby(s.region).mean()


def _half(x):
    return f"{np.round(2 * x) / 2:g}"


def _arc_order(cfg):
    arcs = list((cfg.get("sources", {}) or {}).get("arcs", {}) or [])
    return {r: i for i, r in enumerate(arcs + [r for r in REGION_NAME if r not in arcs])}


def _join_names(regions):
    names = [REGION_NAME.get(r, r.title()) for r in regions]
    if len(names) == 1:
        return names[0]
    if all(n.endswith(" Arc") for n in names):
        stems = [n[:-4] for n in names]
        return ", ".join(stems[:-1]) + " and " + stems[-1] + " Arcs"
    return ", ".join(names[:-1]) + " and " + names[-1]


def _drift_groups(ev, cfg):
    """[[Mw, 'from' | 'only near' | 'not up to', [regions]], ...] for AIS calm, smallest Mw first."""
    order = _arc_order(cfg)
    cr = region_crossing(ev, "ais", "calm")
    items = []
    for r, c in cr.items():
        if np.isfinite(c["x"]):
            kind, v = ("only near" if c["last"] else "from"), float(np.round(2 * c["x"]) / 2)
        else:
            kind, v = "not up to", float(c["mw_max"])
        items.append((v, kind, order.get(r, 99), r))
    items.sort()
    groups = []
    for v, kind, _, r in items:
        if groups and groups[-1][0] == v and groups[-1][1] == kind:
            groups[-1][2].append(r)
        else:
            groups.append([v, kind, [r]])
    return groups


def key_bullets(ev, cfg):
    """Short on-screen versions of key_lines (same numbers): [(text, 'ais' | 'gps'), ...]."""
    out = []
    conf = [g for g in _drift_groups(ev, cfg) if g[1] != "not up to"]
    if conf:
        lo, hi = conf[0], conf[-1]
        txt = f"Ship drift: from Mw {lo[0]:g} ({_join_names(lo[2])})"
        if hi is not lo:
            txt += f" up to Mw {hi[0]:g} ({_join_names(hi[2])})"
        out.append((txt, "ais"))
    names, regions = historic_not_detected(cfg)
    if names:
        where = (REGION_NAME.get(regions[0], regions[0].title()) + " ") if len(regions) == 1 else ""
        out.append((f"Real {where}faults replayed ({', '.join(event_label(n) for n in names)}): not detected", "ais"))
    pb = pct_before_first_gauge(ev, "ais", "calm")
    if len(pb):
        out.append((f"Ship drift beats the first tide gauge in at most {pb.max():.0f}% of detections", "ais"))
    cg = region_crossing(ev, "gps", "calm")
    xs = [c["x"] for c in cg.values()]
    pg = pct_before_first_gauge(ev, "gps", "calm")
    if xs and all(np.isfinite(xs)) and len(pg):
        out.append((f"Precise GPS: from Mw {_half(max(xs))} everywhere, first in {pg.min():.0f} to {pg.max():.0f}% "
                    f"of detections", "gps"))
    return out


def key_lines(ev, cfg):
    """The three key-number lines of the script, generated from the sweep (see the module docstring)."""
    groups = _drift_groups(ev, cfg)
    phrases = []
    for v, kind, rs in groups:
        what = f"not confirmed up to Mw {v:g}" if kind == "not up to" else f"{kind} Mw {v:g}"
        phrases.append(f"{_join_names(rs)} {what}")
    l1 = "Ship drift: " + ", ".join(phrases)
    names, regions = historic_not_detected(cfg)
    if names:
        where = (REGION_NAME.get(regions[0], regions[0].title()) + " ") if len(regions) == 1 else ""
        l1 += (f", for earthquakes shaped like the catalogue; real {where}faults replayed "
               f"({', '.join(event_label(n) for n in names)}) were not detected")

    pb = pct_before_first_gauge(ev, "ais", "calm")
    l2 = (f"Ship drift beats the first tide gauge in at most {pb.max():.0f}% of detections "
          f"({REGION_NAME.get(pb.idxmax(), pb.idxmax())})") if len(pb) else ""

    cg = region_crossing(ev, "gps", "calm")
    xs = [c["x"] for c in cg.values()]
    pg = pct_before_first_gauge(ev, "gps", "calm")
    l3 = ""
    if xs and all(np.isfinite(xs)) and len(pg):
        l3 = (f"Precise GPS: from Mw {_half(max(xs))} everywhere, and before the first tide gauge in "
              f"{pg.min():.0f} to {pg.max():.0f}% of detections")
    return l1, l2, l3


HIST_CATALOGUE = "arcs_plus_aegean"            # T26 catalogue with the Aegean grid (the sweep's library)


def historic_rows(cfg, operator, regimes=("calm",), catalogue=HIST_CATALOGUE):
    """T26 network rows (outputs/historic_replays_summary.csv) for one operator, both range caps, given regimes,
    with n_det = detected_frac x trials, far_hold from outputs/historic_thresholds_<catalogue>.csv and
    at_far = n_det <= 95th percentile of Binomial(trials, far_hold), i.e. "at or below the false-alarm level"."""
    p = RES / "historic_replays_summary.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    caps = [float(c) for c in cfg["traffic"]["range_caps_m"]]
    t = t[(t["catalogue"] == catalogue) & (t["operator"] == operator) & (t["detector"] == "network")
          & t["regime"].isin(regimes) & t["range_cap_m"].astype(float).isin(caps)].copy()
    if t.empty:
        return None
    th = RES / f"historic_thresholds_{catalogue}.csv"
    if th.exists():
        far = pd.read_csv(th)[["operator", "range_cap_m", "regime", "detector", "far_hold"]]
        t = t.merge(far, on=["operator", "range_cap_m", "regime", "detector"], how="left")
    else:
        t["far_hold"] = np.nan
    from scipy.stats import binom
    t["n_det"] = np.round(t["detected_frac"] * t["trials"]).astype(int)
    q95 = binom.ppf(0.95, t["trials"].astype(int), t["far_hold"].fillna(0.01))
    t["at_far"] = t["n_det"] <= q95
    return t


def event_label(name):
    return str(name).split("_")[0].title()          # samos_2020 -> Samos


def historic_not_detected(cfg):
    """Names of T26 events whose AIS network rows (arcs_plus_aegean, both caps, calm) are all at the false-alarm level,
    in file order, and their source region(s) from data/historic_runs/sources.csv."""
    t = historic_rows(cfg, "ais")
    if t is None:
        return [], []
    names = [n for n in dict.fromkeys(t["name"]) if t[t["name"] == n]["at_far"].all()]
    regions = []
    sp = cm.REPO / "data/historic_runs/sources.csv"
    if sp.exists() and names:
        hs = pd.read_csv(sp)
        regions = list(dict.fromkeys(hs[hs["name"].isin(names)]["region"]))
    return names, regions


def load_map3(cfg, cap=None):
    p = RES / "map3_table.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    cap = cap_m(cfg) if cap is None else cap
    t = t[np.isclose(t.range_cap_m.astype(float), cap)]
    return t if (len(t) and MAP3_COL in t) else None


# ------------------------------------------------------------------ scene
def _weights(t, times, xf=0.6):
    """Weight of each tab state at time t (cross-fade of xf seconds at each switch). Pure function of t."""
    w = np.zeros(len(times))
    cur = max([i for i, ts in enumerate(times) if t >= ts], default=0)
    if cur == 0:
        w[0] = 1.0
    else:
        x = cm.ramp(t, times[cur], times[cur] + xf)
        w[cur], w[cur - 1] = x, 1 - x
    return w, cur


def build(cut, ctx):
    cfg = ctx.cfg
    dur = cm.duration(SCENE, cut)
    k = dur / 23.0
    T = lambda x: x * k                                     # schedule written for 23 s, compressed for the 3-min cut
    t_tabs = [T(1.6), T(8.0), T(10.6), T(13.2)]
    t_l1, t_l2, t_l3 = T(3.0), T(5.2), T(14.2)
    fig, chrome = cm.new_figure(SCENE, caption="With today's reports, ships confirm only the largest earthquakes and "
                                               "rarely before the tide gauges,\nwhile precise GPS on the same ships "
                                               "would confirm much smaller ones, usually first.")
    chrome["caption"].set_y(0.045); chrome["caption"].set_linespacing(1.35)
    anim = []

    def show(a, t_in, amax=1.0, d=0.6):
        anim.append((a, t_in, amax, d))
        a.set_alpha(0)
        return a

    ev = load_map2_events(cfg)
    ext = cm.aoi_extent(cfg)
    pool = ctx.pool()
    d_hi = float(cfg["domain"]["depth_band_edges_m"][1])

    ax2 = cm.map_axes(fig, (0.02, 0.19, 0.56, 0.55), extent=ext)
    for a in list(ax2.images) + list(ax2.collections) + list(ax2.patches):
        show(a, 0.2)
    sh = ((pool.rtype == 0) if hasattr(pool, "rtype") else np.ones(len(pool.lon), bool)) & (pool.depth < d_hi)
    show(ax2.scatter(pool.lon[sh], pool.lat[sh], s=1.3, color=cm.C["ship"], lw=0, zorder=1), 0.6, 0.28)
    x_l = ax2.get_position().x0
    x_r = 0.605

    # ------------------------------------------------------------ headings and tabs
    show(fig.text(x_l, 0.825, "The smallest earthquake ships can confirm, by source (≥50% of days)",
                  fontsize=cm.FS_SMALL + 2, weight="bold", color=cm.TEXT, va="center"), 0.3)
    rend = fig.canvas.get_renderer()
    tabs, tx = [], x_l
    for i, (_, _, lab) in enumerate(STATES):
        probe = fig.text(0, 0, lab, fontsize=cm.FS_LABEL)
        w = probe.get_window_extent(rend).width / cm.W_PX + 0.018
        probe.remove()
        bx = fig.add_axes((tx, 0.758, w, 0.038)); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
        patch = bx.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.25",
                                            facecolor=cm.PANEL_EDGE, edgecolor="none", mutation_aspect=0.25))
        txt = bx.text(0.5, 0.5, lab, ha="center", va="center", fontsize=cm.FS_LABEL, color=cm.MUTED)
        tabs.append((patch, txt))
        tx += w + (0.022 if i == 2 else 0.008)

    # ------------------------------------------------------------ Map 2 (left) + live count badge
    m2_sc = m3_sc = None
    st2, st3 = [], []
    counts = []
    if ev is not None:
        pos = positions(ev)
        for op, reg, lab in STATES:
            mw = pos.pos_id.map(position_floor(ev, op, reg)).to_numpy(float)
            ok = np.isfinite(mw)
            counts.append((lab, int(ok.sum()), len(mw), int((mw <= 7.1).sum())))
            fc = np.zeros((len(mw), 4)); fc[ok] = MW_CMAP(MW_NORM(mw[ok]))
            ec = np.tile(mcolors.to_rgba(cm.BG), (len(mw), 1)); ec[~ok] = mcolors.to_rgba("#9AA6B8", 0.95)
            st2.append((fc, ec))
        m2_sc = ax2.scatter(pos.lon_deg, pos.lat_deg, s=95, facecolors=st2[0][0], edgecolors=st2[0][1], lw=1.4, zorder=6)
        bullets = key_bullets(ev, cfg)
        # Discrete 4-tier legend pills under the map
        lbl = fig.text(x_l, 0.148, "Smallest Mw confirmed:", fontsize=cm.FS_LABEL, color=cm.TEXT, weight="bold", va="center")
        tier_specs = [
            ("#00E676", "≤ Mw 7.0 (small & large)"),
            ("#FFE066", "Mw 7.5"),
            ("#FF8C24", "Mw 8.0"),
            ("#D92B4B", "Mw 8.5 only"),
        ]
        cb_arts = [lbl]
        lx = x_l + 0.165
        for tcol, ttxt in tier_specs:
             sw = fig.add_axes((lx, 0.134, 0.015, 0.028)); sw.axis("off")
             sw.scatter([0.5], [0.5], s=110, facecolor=tcol, edgecolor=cm.BG, lw=1.2); sw.set_xlim(0, 1); sw.set_ylim(0, 1)
             tl = fig.text(lx + 0.017, 0.148, ttxt, fontsize=cm.FS_LABEL - 1, color=tcol, weight="bold", va="center")
             cb_arts.extend([*sw.collections, tl])
             lx += 0.024 + len(ttxt) * 0.0054
        ring = fig.add_axes((x_l + 0.165, 0.095, 0.015, 0.028)); ring.axis("off")
        ring.scatter([0.5], [0.5], s=100, facecolor="none", edgecolor="#9AA6B8", lw=1.6); ring.set_xlim(0, 1); ring.set_ylim(0, 1)
        nlab = fig.text(x_l + 0.184, 0.109, "Hollow ring = not confirmed (even at Mw 8.6)", fontsize=cm.FS_LABEL - 1,
                        color=cm.MUTED, va="center")
        cb_arts.extend([*ring.collections, nlab])
    else:
        bullets = []
        cb_arts = []
        pax = fig.add_axes(ax2.get_position().bounds, zorder=5); pax.patch.set_visible(False)
        for a in cm.placeholder(pax, "The smallest earthquake ships can confirm, by source"):
            show(a, 1.0)
    for a in cb_arts:
        show(a, t_tabs[0])

    # ------------------------------------------------------------ Right column: comparison & key takeaways
    rcard = cm.card(fig, (x_r, 0.56, 0.37, 0.25))
    show(rcard.patches[0], t_tabs[0])
    n_pos = counts[0][2] if counts else 0
    show(rcard.text(0.05, 0.87, f"Source positions confirmed (out of {n_pos})", fontsize=cm.FS_SMALL,
                    weight="bold", color=cm.TEXT, va="center"), t_tabs[0])
    for idx, (lab, n_ok, n_tot, n_small) in enumerate(counts):
        yr = 0.68 - idx * 0.19
        col = cm.C["gps"] if "GPS" in lab else (cm.C["good"] if "Calm" in lab else (cm.C["moderate"] if "Moderate" in lab else cm.C["bad"]))
        show(rcard.text(0.05, yr, lab, fontsize=cm.FS_SMALL, weight="bold", color=col, va="center"), t_tabs[idx])
        show(rcard.text(0.95, yr, f"{n_ok}", fontsize=cm.FS_SMALL + 2, weight="bold", color=cm.TEXT, ha="right",
                        va="center"), t_tabs[idx])

    import textwrap
    y = 0.51
    n_ais = 0
    for line, op in bullets:
        if op == "gps":
            t_in, col = t_l3, cm.C["gps"]
        else:
            t_in, col = (t_l1, t_l1 + T(1.0), t_l2)[min(n_ais, 2)], cm.TEXT
            n_ais += 1
        show(fig.text(x_r, y, "•", fontsize=cm.FS_SMALL, color=col, va="top"), t_in)
        tt = show(fig.text(x_r + 0.016, y, "\n".join(textwrap.wrap(line, 46)), fontsize=cm.FS_SMALL,
                           color=col, va="top", linespacing=1.2), t_in)
        y -= tt.get_window_extent(rend).height / cm.H_PX + 0.022

    for kk in ("title", "section", "num"):
        chrome[kk].set_alpha(0)
    chrome["caption"].set_alpha(0)
    tab_on = np.array(mcolors.to_rgba(cm.TEXT)); tab_off = np.array(mcolors.to_rgba(cm.MUTED))
    for patch, txt in tabs:
        patch.set_alpha(0); txt.set_alpha(0)

    def update(t):
        for kk in ("title", "section", "num"):
            chrome[kk].set_alpha(cm.fade(t, 0.0, dur=0.7))
        chrome["caption"].set_alpha(cm.fade(t, T(16.0), dur=0.8))
        for a, t_in, amax, d in anim:
            a.set_alpha(amax * cm.fade(t, t_in, dur=d))
        w, _ = _weights(t, t_tabs)
        on = cm.fade(t, t_tabs[0] - 0.6)
        for i, (patch, txt) in enumerate(tabs):
            patch.set_alpha(w[i] * cm.fade(t, t_tabs[0]))
            txt.set_alpha(on)
            txt.set_color((tab_off + (tab_on - tab_off) * w[i] * cm.fade(t, t_tabs[0]))[:3])
            txt.set_weight("bold" if w[i] > 0.5 and t >= t_tabs[0] else "normal")
        a0 = cm.fade(t, t_tabs[0] - 0.2)
        for sc, st in ((m2_sc, st2), (m3_sc, st3)):
            if sc is None:
                continue
            fc = sum(w[i] * st[i][0] for i in range(len(STATES)))
            ec = sum(w[i] * st[i][1] for i in range(len(STATES)))
            fc[:, 3] *= a0; ec[:, 3] *= a0
            sc.set_facecolors(fc); sc.set_edgecolors(ec)

    return cm.Scene(fig, update, still_t=dur - 0.8)


if __name__ == "__main__":
    cm.main(__file__)
