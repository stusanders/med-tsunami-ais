"""Scene 9 (Simulation): the network confirms the tsunami, but after the tide gauges.

Focus is split cleanly between the two key result charts:
  1. Top chart: Evidence from the whole fleet (how well ship drift matches the tsunami wave), relative to the
     1% false-alarm threshold, plus the location accuracy badge (places the source within ~13 km).
  2. Bottom chart: Time taken to confirm the tsunami on a typical day (Ship network vs 3 tide gauges agree vs First buoy).
"""
import common as cm
import matplotlib
import numpy as np
import showcase_trials as st

SCENE = 9


def evidence_axes(fig, rect, stat, thr, color, xlabel=True, ymax=None):
    """Statistic / alarm level against minutes; alarm line at 1. Returns (ax, rel, line, head, alarm_line, ymax)."""
    rel = np.asarray(stat, float) / float(thr)
    ax = fig.add_axes(rect)
    ymax = ymax or max(1.0, float(np.nanmax(rel))) * 1.25
    ax.set_xlim(0, len(rel)); ax.set_ylim(0, ymax)
    ax.set_xticks(list(range(0, len(rel) + 1, 30)))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if xlabel:
        ax.set_xlabel("Minutes after the earthquake", fontsize=cm.FS_LABEL)
    alarm = ax.axhline(1.0, color=cm.C["alarm"], lw=2.4, ls=(0, (6, 4)))
    line, = ax.plot([], [], color=color, lw=3.6, solid_capstyle="round", zorder=4)
    head = ax.scatter([], [], s=90, color=color, zorder=5)
    return ax, rel, line, head, alarm, ymax


def draw_curve(line, head, rel, m):
    """Curve drawn up to clock minute m (rel[k] is the value at the end of minute k+1)."""
    n = int(np.floor(m))
    if n >= 1:
        x = np.r_[0, np.arange(1, n + 1)]
        line.set_data(x, np.r_[0, rel[:n]])
        head.set_offsets([[n, rel[n - 1]]])
    else:
        line.set_data([], []); head.set_offsets(np.empty((0, 2)))


def km_to_deg_lat(km, R_m):
    return km * 1000.0 / R_m * 180.0 / np.pi


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    D = st.load("showcase86")
    meta, tr = D["meta"], D["traffic"]
    fig, chrome = cm.new_figure(SCENE)
    if meta is None or tr is None:
        cm.placeholder(fig, "Evidence vs alarm level, time to confirm",
                       needs="showcase86_meta.json + showcase86_traffic.npz", rect=(0.1, 0.15, 0.8, 0.65))
        return cm.Scene(fig, lambda t: None, still_t=dur - 1)

    trial, ais, typ, gau = meta["trial"], meta["ais"], meta["typical"], meta["gauges"]
    typ_km = float(typ["ais_med_loc_km"])
    typ_conf = float(typ["ais_med_confirm_min"])
    chrome["caption"].set_text(
        f"The fleet builds enough evidence to confirm this earthquake on {st.pct(typ['ais_det_frac'])} of simulated "
        f"days and places it within about {typ_km:.0f} km,\nbut only after about {typ_conf:.0f} minutes, "
        f"long after the nearest tide gauges.")
    chrome["caption"].set_linespacing(1.35)
    chrome["caption"].set_position((0.5, 0.042))
    _fb = gau.get("first_buoy_s")
    _later = "tide gauges and buoys" if (_fb is not None and _fb / 60.0 < typ_conf) else "tide gauges"
    cm.set_title(fig, chrome, f"Result: the ship network detects the tsunami, but later than {_later}")

    # timing (scene seconds)
    T_CURVE0, T_CURVE1 = 0.8, min(9.0, 0.45 * dur)
    T_TL0 = T_CURVE1 + 0.4
    T_TL = [T_TL0, T_TL0 + 1.0, T_TL0 + 2.0]

    def t_of_min(mm):
        return T_CURVE0 + (T_CURVE1 - T_CURVE0) * min(mm, 120) / 120.0

    # ------------------------------------------------------------------ top chart: evidence vs alarm level
    fig.text(0.04, 0.83, "1. Evidence from the whole fleet, relative to the alarm level",
             fontsize=cm.FS_H2 - 2, color=cm.TEXT, weight="bold", ha="left", va="center")
    axs, rel, line, head, alarm, ymax = evidence_axes(fig, (0.04, 0.50, 0.92, 0.29), tr["stat_ais"],
                                                      tr["thr_ais"], cm.C["network"])
    alarm_lbl = axs.text(2, 1.0 + 0.03 * ymax, "1% false-alarm threshold", color=cm.C["alarm"],
                         fontsize=cm.FS_SMALL + 1, weight="bold", ha="left", va="bottom", zorder=7)
    if ais.get("detected") and ais.get("confirm_s") is not None:
        c_min = float(ais["confirm_s"]) / 60.0
        k = int(np.clip(round(c_min), 1, len(rel)))
        conf_txt = f"Confirmed after {c_min:.0f} minutes (showcase run; typical {typ_conf:.0f} minutes)"
        cross_xy = (k, rel[k - 1])
    else:
        c_min, cross_xy = None, None
        conf_txt = "Alarm level not reached within 2 hours"
    cross_pt = axs.scatter([], [], s=360, facecolors="none", edgecolors=cm.C["alarm"], lw=3.2, zorder=6)
    conf_lbl = fig.text(0.04, 0.418, conf_txt, fontsize=cm.FS_SMALL + 1, color=cm.TEXT, weight="bold", ha="left",
                        va="center", alpha=0)
    est_txt = f"Places source within {typ_km:.0f} km (typical)"
    est_lbl = fig.text(0.96, 0.418, est_txt, fontsize=cm.FS_SMALL, color=cm.C["network"], weight="bold",
                       ha="right", va="center", alpha=0)
    for a in (alarm, alarm_lbl):
        a.set_alpha(0)

    # ------------------------------------------------------------------ bottom chart: timeline (typical day)
    fig.text(0.04, 0.34, "2. Time taken to confirm the tsunami (typical day)",
             fontsize=cm.FS_H2 - 2, color=cm.TEXT, weight="bold", ha="left", va="center")
    fb = gau.get("first_buoy_s")
    third_g = float(typ["third_gauge_med_min"])
    rows = [("Ship network", typ_conf, cm.C["network"], f"{st.fmt_min(typ_conf)} min"),
            ("3 tide gauges agree", third_g, cm.C["gauge"],
             f"{st.fmt_min(third_g)} min")]
    if fb is not None:
        rows.append(("First buoy", fb / 60.0, cm.C["buoy"], f"{st.fmt_min(fb / 60.0)} min"))
    else:
        rows.append(("First buoy", None, cm.C["buoy"], "none within 2 hours"))
    axt = fig.add_axes((0.22, 0.11, 0.74, 0.195))
    axt.set_xlim(0, 120); axt.set_ylim(-0.6, 2.6)
    axt.set_yticks([]); axt.set_xticks([])
    for sp in axt.spines.values():
        sp.set_visible(False)
    tl_axis = axt.axvline(0, color=cm.FAINT, lw=1.4, alpha=0)
    fig.text(0.96, 0.342, "Ship network and tide gauges: median day (gauges: 3rd gauge to trigger).\nBuoy: the single run shown above.",
             fontsize=cm.FS_LABEL - 1, color=cm.MUTED, ha="right", va="center", linespacing=1.25)
    bars = []
    for r, (name, mm, col, txt) in enumerate(rows):
        y = 2 - r
        lab = fig.text(0.205, 0.11 + 0.195 * (y + 0.6) / 3.2, name, fontsize=cm.FS_BODY, color=col, weight="bold",
                       ha="right", va="center", alpha=0)
        bar = axt.barh([y], [0], height=0.62, color=col, alpha=0)
        inside = False                     # every value label sits outside its bar, same style
        val = axt.text(0, y, txt, fontsize=cm.FS_SMALL + 2, color=cm.BG if inside else cm.TEXT,
                       ha="right" if inside else "left", va="center", alpha=0, fontweight="bold")
        bars.append((lab, bar[0], val, mm, inside))

    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50)
    veil_ax.axis("off")
    veil = veil_ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG,
                                                          alpha=1.0, lw=0))

    def update(t):
        veil.set_alpha(1.0 - cm.ramp(t, 0.0, 0.5))
        alarm.set_alpha(cm.fade(t, 0.4)); alarm_lbl.set_alpha(cm.fade(t, 0.4))
        m = 120.0 * float(np.clip((t - T_CURVE0) / (T_CURVE1 - T_CURVE0), 0, 1))
        draw_curve(line, head, rel, m)
        t_conf = t_of_min(c_min) if c_min is not None else T_CURVE1
        a_c = cm.fade(t, t_conf, dur=0.5)
        if cross_xy is not None and a_c > 0:
            cross_pt.set_offsets([cross_xy]); cross_pt.set_alpha(a_c)
        else:
            cross_pt.set_offsets(np.empty((0, 2)))
        conf_lbl.set_alpha(a_c)
        est_lbl.set_alpha(cm.fade(t, t_conf, dur=0.7))
        tl_axis.set_alpha(cm.fade(t, T_TL[0], dur=0.5))
        for i, (lab, bar, val, mm, inside) in enumerate(bars):
            a = cm.fade(t, T_TL[i], dur=0.5)
            g = cm.ramp(t, T_TL[i], T_TL[i] + 1.0)
            lab.set_alpha(a)
            w = (mm or 0.0) * g
            bar.set_width(w); bar.set_alpha(0.9 * a)
            val.set_x((w - 1.2 if inside else w + 1.2) if mm is not None else 0.5)
            val.set_alpha(a * (g if mm is not None else 1))

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
