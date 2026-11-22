"""Scene 12 (Results): what reshaping the problem makes possible.

On-screen text is DEMO_SCRIPT.md scene 12 only (title, ladder heading, three rungs, small label, caption).

Ladder (outputs/reshaping_ladder.json "rows", total_s, all rows extrapolated from measured per-unit timings):
  * "The straightforward way, on a laptop": row 2 (re-simulate per truth event and traffic draw + Python-loop
    scoring, MacBook arm64 CPU), 3.1 years -> "about 3 years".
  * "Reshaped, on a laptop": row 5 (library + matmul + prefix scan, MacBook arm64 CPU), 5.6 h -> "about 5.6 hours".
  * "Reshaped, on a GPU": row 6 (same, Colab Tesla T4), 60.4 min -> "about 1 hour".
  Rounding (fmt_duration): years to a whole number; hours to 0.1 h; under 1.5 h to a whole hour.
  Bars: log time axis (tick labels 1 hour / 1 day / 1 year only).
Small label:
  * "Same code and same answers on the CPU and the GPU" only if outputs/validation.md check 5 is PASS and
    outputs/validation/gpu_check.json "pytest" says passed.
  * "N times faster on the same machine": outputs/benchmark.json summary.solver.hw_speedup_same_machine_colab
    (17.06 = Colab CPU NumPy 5.29 s / Colab T4 CuPy 0.31 s per scenario, both from gpu_check.json; measured on a
    synthetic flat 360 x 750 grid at batch 64, summary.solver.same_machine_flat_ref); falls back to
    gpu_check.json timing numpy_B1_s_per_scenario / cupy_B64_s_per_scenario.
"""
import re

import common as cm
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation

SCENE = 12
TICK_FONT = "DejaVu Sans"
YEAR_S = 365.25 * 86400.0

CAPTION = ("Reshaping the problem turns years into hours, while the GPU turns hours into minutes,\n"
           "which allowed the whole experiment to be rerun each time an assumption was corrected.")
HEADING = "Time to run the full experiment"
RUNGS = [(2, "The straightforward way, on a laptop:", "cpu"), (5, "Reshaped, on a laptop:", "cpu"),
         (6, "Reshaped, on a GPU:", "gpu")]                       # (reshaping_ladder.json row, label, colour key)


def fmt_duration(s):
    years, hours = s / YEAR_S, s / 3600.0
    if years >= 1.5:
        return f"about {years:.0f} years"
    if hours >= 1.5:
        return f"about {hours:.1f} hours"
    h = round(hours)
    return f"about {h:.0f} hour" + ("" if h == 1 else "s")


def same_answers():
    vp = cm.OUTPUTS / "validation.md"
    ok5 = vp.exists() and re.search(r"^\|\s*5\s+[^|]+\|\s*PASS\s*\|", vp.read_text(), re.M) is not None
    gc = cm.load_json("outputs/validation/gpu_check.json")
    return ok5 and gc is not None and "passed" in str(gc.get("pytest", ""))


def same_machine_speedup():
    b = cm.load_json("outputs/benchmark.json")
    try:
        return float(b["summary"]["solver"]["hw_speedup_same_machine_colab"])
    except (TypeError, KeyError, ValueError):
        pass
    gc = cm.load_json("outputs/validation/gpu_check.json")
    try:
        return float(gc["timing"]["numpy_B1_s_per_scenario"]) / float(gc["timing"]["cupy_B64_s_per_scenario"])
    except (TypeError, KeyError, ValueError, ZeroDivisionError):
        return None


def build(cut, ctx):
    fig, chrome = cm.new_figure(SCENE, caption=CAPTION)
    anim = []

    def show(a, t_in, amax=1.0, dur=0.6):
        anim.append((a, t_in, amax, dur))
        a.set_alpha(0)
        return a

    show(fig.text(0.04, 0.80, HEADING, fontsize=cm.FS_H2, weight="bold", color=cm.TEXT, va="center"), 0.5)

    lad = cm.load_json("outputs/reshaping_ladder.json")
    rows = {int(r["row"]): r for r in (lad or {}).get("rows", [])}
    X_LO, X_HI = 1800.0, 3.5 * YEAR_S                                # log axis: 10 minutes .. 10 years
    BX, BW = 0.04, 0.92
    bars = []
    FS = cm.FS_BODY
    for k, (rid, lab, ck) in enumerate(RUNGS):
        y = 0.69 - k * 0.165
        t0 = 1.4 + 3.0 * k
        r = rows.get(rid)
        if r is None:
            for a in cm.placeholder(fig, lab, needs="outputs/reshaping_ladder.json", rect=(BX, y - 0.05, 0.5, 0.1)):
                show(a, t0)
            continue
        s = float(r["total_s"])
        col = cm.C[ck]
        t1 = show(fig.text(BX, y + 0.035, lab, fontsize=FS, color=cm.TEXT, va="center"), t0)
        show(fig.add_artist(Annotation(fmt_duration(s), xy=(1, 0.5), xycoords=t1, xytext=(8, 0),
                                       textcoords="offset points", fontsize=FS, color=col, weight="bold",
                                       va="center")), t0 + 1.0)
        frac = np.log(s / X_LO) / np.log(X_HI / X_LO)
        show(fig.add_artist(Rectangle((BX, y - 0.04), BW, 0.05, transform=fig.transFigure, color=cm.PANEL, lw=0, zorder=1)),
             t0)
        bar = fig.add_artist(Rectangle((BX, y - 0.04), 0.0, 0.05, transform=fig.transFigure, color=col, lw=0, zorder=3))
        bars.append((bar, BW * float(np.clip(frac, 0, 1)), t0 + 0.2))

    # minimal log time axis under the bars
    y_ax = 0.69 - 2 * 0.165 - 0.075
    for s_tick, txt in ((3600.0, "1 hour"), (86400.0, "1 day"), (30.44 * 86400.0, "1 month"), (YEAR_S, "1 year"), (3.0 * YEAR_S, "3 years")):
        x = BX + BW * np.log(s_tick / X_LO) / np.log(X_HI / X_LO)
        for k in range(len(RUNGS)):                       # grid segments over the bar bands only (not the text)
            yb = 0.69 - k * 0.165
            show(fig.add_artist(Line2D([x, x], [yb - 0.04, yb + 0.01], transform=fig.transFigure,
                                       color=cm.FAINT, lw=1.2, ls=(0, (3, 3)), zorder=2)), 1.4 + 3.0 * k)
        show(fig.text(x, y_ax - 0.01, txt, fontsize=cm.FS_LABEL, color=cm.MUTED, ha="center", va="top"), 1.2)

    # small label
    parts = []
    if same_answers():
        parts.append("Same code and same answers on the CPU and the GPU")
    sp = same_machine_speedup()
    if sp is not None:
        parts.append(f"{sp:.0f} times faster on the same machine")
    if parts:
        t_s = 11.0
        y_s = 0.155
        show(fig.add_artist(Line2D([BX + 0.011], [y_s], transform=fig.transFigure, marker="o", ms=24, ls="",
                                          color=cm.C["good"], mew=0)), t_s)
        show(fig.text(BX + 0.011, y_s, "✓", ha="center", va="center", fontsize=cm.FS_SMALL - 2, color=cm.BG,
                      weight="bold", family=TICK_FONT), t_s)
        show(fig.text(BX + 0.03, y_s, "; ".join(parts), fontsize=cm.FS_SMALL, color=cm.TEXT, va="center"), t_s)

    for k in ("title", "section", "num"):
        chrome[k].set_alpha(0)

    def update(t):
        for k in ("title", "section", "num"):
            chrome[k].set_alpha(cm.fade(t, 0.0, dur=0.7))
        chrome["caption"].set_alpha(cm.fade(t, 1.0, dur=0.8))
        for a, t_in, amax, d in anim:
            a.set_alpha(amax * cm.fade(t, t_in, dur=d))
        for bar, w, t0 in bars:
            g = cm.ramp(t, t0, t0 + 1.2)
            bar.set_width(w * g)
            bar.set_alpha(1.0 if g > 0 else 0.0)

    return cm.Scene(fig, update, still_t=17.0)


if __name__ == "__main__":
    cm.main(__file__)
