"""Scene 11 (Results): why one ship on its own is not enough.

On-screen text is DEMO_SCRIPT.md scene 11 only (title, three labels, caption).

Visual: one quiet day (noise only, no tsunami) from outputs/demo_data/quiet_day_example.npz (make_showcase_data.py:
traffic draw 0, AIS, 100 km range cap, calm; operator / cap / regime read from showcase_meta.json). Left, the map:
every ship-minute where the literal single-ship rule (|drift| > 3 sigma of the ship's own noise) fires flashes.
Right, the network score on the same noise against the alarm threshold.

Numbers / thresholds (all from outputs/thresholds.csv, read at render time):
  * "Single-ship rule: alarm on X% of quiet days": far_hold of detector == "single_literal", operator "ais", all
    strata (shown as one value if min == max, else min to max).
  * "Ship network: alarm on A to B% of quiet days": far_hold min and max of detector == "network", operator "ais",
    both range caps (40, 100 km), all regimes (same definition as scene 10).
  * Threshold line: threshold of the (operator, range_cap_m, regime) row of the quiet example (ais, 100 km, calm,
    network). The dev threshold stored in quiet_day_example.npz / showcase_meta.json ("threshold") is NOT used.
    Caveat: the example's net_stat was computed by make_showcase_data.py on its local candidate subset, while the
    thresholds.csv value is calibrated on the full library (more candidates, so a larger maximum on quiet days).
"""
import common as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

SCENE = 11
FLASH_MIN = 5.0           # clock minutes a flash ring lasts

CAPTION = ("Judging each ship on its own raises an alarm almost every day, whereas treating the fleet as one "
           "network keeps false alarms rare.")
LBL_QUIET = "A quiet day: noise only, no tsunami"


def _pct(x):
    """Whole percent from 10% up, one decimal below; never round a measured rate to 0% or up to 100%."""
    if x == 1.0 or 0.1 <= x <= 0.99:
        return f"{100 * x:.0f}"
    return f"{100 * x:.1f}" if x > 0 else "0"


def _span(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    return f"{_pct(lo)}%" if _pct(lo) == _pct(hi) else f"{_pct(lo)} to {_pct(hi)}%"


def thresholds(meta):
    """(single-ship rate text, network rate text, network threshold for the example's stratum) or None pieces."""
    p = cm.OUTPUTS / "thresholds.csv"
    if not p.exists():
        return None, None, None
    th = pd.read_csv(p)
    ais = th[th.operator == "ais"]
    lit, net = ais[ais.detector == "single_literal"], ais[ais.detector == "network"]
    row = th[(th.detector == "network") & (th.operator == meta["operator"])
             & np.isclose(th.range_cap_m, meta["range_cap_m"]) & (th.regime == meta["regime"])]
    return (_span(lit.far_hold) if len(lit) else None, _span(net.far_hold) if len(net) else None,
            float(row.threshold.iloc[0]) if len(row) else None)


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    T_CLOCK0, T_CLOCK1 = 0.5, max(4.0, dur - 3.5)
    T_SINGLE, T_NET = 1.6, 4.2
    s8 = cm.load_scene_module(8)
    D = s8.load_showcase()
    meta, qd = D["meta"], D["quiet"]
    fig, chrome = cm.new_figure(SCENE, caption=CAPTION)
    if meta is None or qd is None:
        cm.placeholder(fig, "Quiet day: single-ship false alarms vs the network",
                       needs="outputs/demo_data/quiet_day_example.npz + showcase_meta.json (make_showcase_data.py)",
                       rect=(0.1, 0.15, 0.8, 0.65))
        return cm.Scene(fig, lambda t: None, still_t=dur - 1.0)
    single_txt, net_txt, thr = thresholds(meta)
    if single_txt and net_txt:
        cm.set_title(fig, chrome, f"Comparison with previous work: single-ship rule false-alarms on {single_txt} "
                                  f"of quiet days; our network on {net_txt}")

    ax = cm.map_axes(fig, (0.012, 0.085, 0.615, 0.78))
    e0 = cm.aoi_extent()
    ax.add_patch(Rectangle((e0[0], e0[2]), e0[1] - e0[0], e0[3] - e0[2], color=cm.BG, alpha=0.5, lw=0, zorder=0.3))
    ax.scatter(qd["ship_lon"], qd["ship_lat"], s=10, color=cm.C["ship"], lw=0, alpha=0.45, zorder=3)
    ex_lon, ex_lat, ex_min = qd["lon"], qd["lat"], qd["minute"].astype(float) + 1.0     # alarm at end of minute
    n_ex = len(ex_lon)
    col = np.array(mcolors.to_rgba(cm.C["single"]))
    dots = ax.scatter(ex_lon, ex_lat, s=np.full(n_ex, 26.0), color=cm.C["single"], lw=0, zorder=4)
    rings = ax.scatter(ex_lon, ex_lat, s=np.zeros(n_ex), facecolors="none", edgecolors=cm.C["alarm"], lw=1.8, zorder=5)
    ax.text(0.015, 0.03, LBL_QUIET, transform=ax.transAxes, fontsize=cm.FS_BODY, color=cm.TEXT, ha="left",
            va="bottom", zorder=8, weight="bold", path_effects=[pe.withStroke(linewidth=5, foreground=cm.BG)])

    # ------------------------------------------------------------------ right column
    X0 = 0.648
    fades = []                                         # (artist, t_in)

    def label(y, head, rule_desc, rate, colour, t_in):
        if rate is None:
            arts = cm.placeholder(fig, head, needs="outputs/thresholds.csv", rect=(X0, y - 0.07, 0.32, 0.11))
        else:
            arts = [fig.text(X0, y, head, fontsize=cm.FS_SMALL + 1, weight="bold", color=colour, va="center"),
                    fig.text(X0, y - 0.028, rule_desc, fontsize=cm.FS_LABEL - 1, color=cm.MUTED, va="top",
                             linespacing=1.18),
                    fig.text(X0, y - 0.105, f"False alarm on {rate} of quiet days", fontsize=cm.FS_SMALL,
                             weight="bold", color=cm.TEXT, va="center")]
        for a in arts:
            a.set_alpha(0)
            fades.append((a, t_in))

    label(0.835, "Single-ship 3σ rule (Inazu et al., 2020):",
          "Triggers whenever any single ship's drift\nexceeds 3× its normal noise (random turns)",
          single_txt, cm.C["single"], T_SINGLE)
    label(0.645, "Our ship network (matched filter):",
          "Triggers only when neighbouring ships\nmatch a real spreading tsunami wave",
          net_txt, cm.C["network"], T_NET)

    chart_head = fig.text(X0, 0.455, "Fleet network score on this quiet day",
                          fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.TEXT, va="center", alpha=0)
    fades.append((chart_head, T_NET))

    axs = fig.add_axes((X0 + 0.015, 0.135, 0.305, 0.285))
    stat = qd["net_stat"].astype(float)
    line, = axs.plot([], [], color=cm.C["network"], lw=2.8)
    axs.set_xlim(0, 120)
    axs.set_xticks([0, 60, 120]); axs.set_xticklabels(["0", "60", "120 min"])
    axs.set_yticks([])
    axs.set_ylabel("Fleet match score", fontsize=cm.FS_LABEL - 1, color=cm.MUTED)
    thr_lbls = []
    if thr is not None:
        ymax = max(stat.max(), thr) * 1.18
        axs.axhline(thr, color=cm.C["alarm"], lw=2.2, ls=(0, (6, 4)))
        tl1 = axs.text(3, thr * 1.03, "1% alarm threshold", fontsize=cm.FS_LABEL - 1, weight="bold",
                       color=cm.C["alarm"], va="bottom")
        tl2 = axs.text(62, stat.max() * 1.08, "Network score stays below alarm", fontsize=cm.FS_LABEL - 2,
                       weight="bold", color=cm.C["network"], ha="center", va="bottom")
        thr_lbls = [tl1, tl2]
        net_cross = int(np.argmax(stat > thr)) + 1 if (stat > thr).any() else None
        if net_cross is not None:
            print(f"scene_11 NOTE: the example's network score crosses the threshold at minute {net_cross}")
    else:
        ymax = stat.max() * 1.15
        print("scene_11 WARNING: no (ais, 100 km, calm, network) row in outputs/thresholds.csv; no threshold line")
    axs.set_ylim(0, ymax)
    axs_arts = [axs.patch, axs.yaxis.label, *axs.spines.values(), *axs.get_xticklabels(), *axs.get_xticklines(),
                *axs.lines, *thr_lbls]

    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50)                   # gentle fade-in from the background colour
    veil_ax.axis("off")
    veil = veil_ax.add_patch(Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG, alpha=1.0, lw=0))

    def update(t):
        veil.set_alpha(1.0 - cm.ramp(t, 0.0, 0.5))
        m = 120.0 * float(np.clip((t - T_CLOCK0) / (T_CLOCK1 - T_CLOCK0), 0, 1))
        on = ex_min <= m
        age = np.where(on, m - ex_min, np.inf)
        x = np.clip(age / FLASH_MIN, 0, 1)
        rings.set_sizes(np.where(age < FLASH_MIN, 30 + 700 * x, 0))
        rc = np.tile(np.array(mcolors.to_rgba(cm.C["alarm"])), (n_ex, 1))
        rc[:, 3] = np.where(age < FLASH_MIN, 1 - x, 0)
        rings.set_edgecolors(rc)
        dc = np.tile(col, (n_ex, 1))
        dc[:, 3] = np.where(on, 0.35 + 0.65 * np.clip(1 - age / FLASH_MIN, 0, 1), 0)
        dots.set_facecolors(dc)
        n_now = int(np.floor(m))
        line.set_data(np.arange(1, n_now + 1), stat[:n_now])
        for a, t_in in fades:
            a.set_alpha(cm.fade(t, t_in, dur=0.5))
        a_ax = cm.fade(t, T_NET, dur=0.5)
        for a in axs_arts:
            a.set_alpha(a_ax)
        line.set_alpha(a_ax)
        chrome["caption"].set_alpha(cm.fade(t, 0.8, dur=0.6))

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
