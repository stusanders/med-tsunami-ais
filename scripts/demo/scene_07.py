"""Scene 7 (Method): the reshaping. Three obvious-way-against-ours pairs, each with its check.

On screen (docs/DEMO_SCRIPT.md scene 7): the three pairs, the tick label under each, and the caption. A tick
is drawn only where a file in outputs/ confirms the check passed (pair 1: validation.md check 5; pair 2:
validation.md check 4, linearity; pair 3: outputs/demo_data/checks.md, test_tensor_equals_loop).
Speed numbers are still loaded (benchmark.json) but no longer shown.

Built from:
  outputs/validation.md (T11): check 5 "Batched == reference; CuPy == NumPy" (max difference and limit),
      check 4 "Linearity and superposition" (superposition error and limit)
  outputs/validation/gpu_check.json: pytest line naming tests/test_solver_equivalence.py::test_cupy_equals_numpy
  outputs/benchmark.json (T19): solver_batched_gpu_B1 / _B<n> s per scenario (Colab T4, real grid);
      scoring_vec_no_prefix_cpu_E100 / scoring_tensor_cpu_E100 (MacBook CPU)
  data/processed/candidates.csv: library size
  Loop-vs-tensor scoring equivalence (tests/test_scoring.py::test_tensor_equals_loop): shown as
      "check pending" unless a file in outputs/ records its result.
"""
import re

import common as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCENE = 7
_s06 = cm.load_scene_module(6)
Group, mix = _s06.Group, _s06.mix
N_MIN = [120]      # minutes per window; set from config.yaml in build()


# --------------------------------------------------------------------------------------
# Evidence (read at render time)
# --------------------------------------------------------------------------------------
def _num(pattern, text):
    m = re.search(pattern, text, flags=re.S)
    return m.group(1) if m else None


def load_evidence(ctx):
    ev = {}
    vpath = cm.OUTPUTS / "validation.md"
    vt = vpath.read_text() if vpath.exists() else ""
    rows = dict((int(n), (name.strip(), res)) for n, name, res in
                re.findall(r"\|\s*(\d)\s+([^|]+?)\s*\|\s*(PASS|FAIL)\s*\|", vt))
    # pair 1: batched == reference, GPU == CPU
    ev["batch_pass"] = rows.get(5, ("", ""))[1] == "PASS"
    ev["batch_diff"] = _num(r"batched vs reference[^*]*\*\*([0-9.eE+-]+)\*\*", vt)
    ev["batch_limit"] = _num(r"batched vs reference[^*]*\*\*[0-9.eE+-]+\*\*\s*\(limit ([0-9.eE+-]+)\)", vt)
    gc = cm.load_json("outputs/validation/gpu_check.json") or {}
    pyt = str(gc.get("pytest", ""))
    ev["gpu_pass"] = ("test_cupy_equals_numpy" in pyt and "passed" in pyt and "failed" not in pyt)
    ev["gpu_name"] = str(gc.get("gpu", "")).split(" (")[0] if gc else None
    # pair 2: linearity (what makes one reusable library valid)
    ev["lin_pass"] = rows.get(4, ("", ""))[1] == "PASS"
    ev["lin_err"] = _num(r"superposition error[^*]*\*\*([0-9.eE+-]+)\*\*", vt)
    ev["lin_limit"] = _num(r"superposition error.*?\(limit ([0-9.eE+-]+)\)", vt)
    # pair 3: loop vs cumulative-sum scoring; only a recorded result counts
    ev["scan_pass"], ev["scan_src"] = find_scoring_equivalence()
    # speeds
    bm = cm.load_json("outputs/benchmark.json")
    recs = {r["name"]: r for r in (bm or {}).get("records", [])}
    gpu = sorted([r for r in recs.values() if r.get("tier") == "batched_gpu" and r.get("workload") == "solver"],
                 key=lambda r: r.get("batch_size", 0))
    ev["speed1"] = None
    if len(gpu) >= 2:
        a, b = gpu[0], gpu[-1]
        ev["speed1"] = (a["seconds_per_scenario"], b["seconds_per_scenario"], a.get("batch_size"),
                        b.get("batch_size"), b.get("machine", "GPU").replace("Colab ", ""))
    a, b = recs.get("scoring_vec_no_prefix_cpu_E100"), recs.get("scoring_tensor_cpu_E100")
    ev["speed3"] = (a["seconds"], b["seconds"], a.get("E"), a.get("machine", "CPU")) if a and b else None
    cpath = cm.REPO / ctx.cfg["paths"]["candidates"]
    ev["K"] = len(pd.read_csv(cpath)) if cpath.exists() else None
    return ev


def find_scoring_equivalence():
    """(passed, source file) if any file in outputs/ records the loop-vs-tensor scoring check."""
    bm = cm.load_json("outputs/benchmark.json") or {}
    sc = (bm.get("summary") or {}).get("scoring") or {}
    for k, v in sc.items():
        if any(w in k for w in ("equal", "match", "equiv")) and isinstance(v, bool):
            return v, "outputs/benchmark.json"
    for p in sorted(cm.OUTPUTS.glob("*.md")) + sorted(cm.OUTPUTS.glob("*.json")) + \
            sorted((cm.OUTPUTS / "validation").glob("*")) + sorted(cm.DEMO_DATA.glob("checks.md")):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for m in re.finditer(r"test_tensor_equals_loop[^\n]{0,80}", txt):
            seg = m.group(0).lower()
            if "fail" in seg:
                return False, str(p.relative_to(cm.REPO))
            if "pass" in seg:
                return True, str(p.relative_to(cm.REPO))
    return None, None


# --------------------------------------------------------------------------------------
# Icons (animated ones return an update function)
# --------------------------------------------------------------------------------------
def tiles(ax, nx, ny, x0, x1, y0, y1, gap=0.18):
    w, h = (x1 - x0) / nx, (y1 - y0) / ny
    out = []
    for j in range(ny):
        for i in range(nx):
            out.append(ax.add_patch(mpatches.FancyBboxPatch(
                (x0 + i * w + gap * w / 2, y0 + j * h + gap * h / 2), w * (1 - gap), h * (1 - gap),
                boxstyle=f"round,pad=0,rounding_size={0.15 * min(w, h)}", fc=cm.FAINT, ec="none")))
    return out


def wave_glyph(ax, cx, cy, s, color, lw=2):
    x = np.linspace(-1, 1, 30)
    return ax.plot(cx + s * x, cy + s * 0.45 * np.exp(-(x / 0.4) ** 2) - s * 0.15, color=color, lw=lw)[0]


def icon_serial_tiles(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ts = tiles(ax, 8, 1, 0.2, 9.8, 0.6, 2.4)
    gl = [wave_glyph(ax, p.get_x() + p.get_width() / 2, 1.5, 0.4, cm.MUTED, 1.5) for p in ts]

    def upd(t, a):
        k = int((t * 4) % len(ts))
        for i, (p, g) in enumerate(zip(ts, gl)):
            on = i == k
            p.set_facecolor(cm.C["cpu"] if on else cm.FAINT)
            p.set_alpha(a)
            g.set_color(cm.BG if on else cm.MUTED)
            g.set_alpha(a)
    return upd


def icon_batch_tiles(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ts = tiles(ax, 16, 4, 0.2, 9.8, 0.2, 2.8, gap=0.22)

    def upd(t, a):
        b = 0.75 + 0.25 * np.sin(2 * np.pi * t * 0.8)
        for p in ts:
            p.set_facecolor(mix(cm.FAINT, cm.C["gpu"], b))
            p.set_alpha(a)
    return upd


def icon_resimulate(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ts = tiles(ax, 5, 1, 0.2, 9.8, 0.5, 2.5)
    gl = [wave_glyph(ax, p.get_x() + p.get_width() / 2, 1.5, 0.5, cm.MUTED, 1.5) for p in ts]

    def upd(t, a):
        k = int((t * 2.5) % len(ts))
        for i, (p, g) in enumerate(zip(ts, gl)):
            on = i == k
            p.set_facecolor(cm.C["cpu"] if on else cm.FAINT)
            p.set_alpha(a); g.set_alpha(a)
            g.set_color(cm.BG if on else cm.MUTED)
    return upd


def icon_library(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    arts = []
    # database cylinder
    cx, w, y0, y1 = 1.4, 2.0, 0.45, 2.45
    arts.append(ax.add_patch(mpatches.Rectangle((cx - w / 2, y0), w, y1 - y0, fc="#16223B", ec="none")))
    for yy in (y0, y1):
        arts.append(ax.add_patch(mpatches.Ellipse((cx, yy), w, 0.5, fc="#16223B", ec=cm.C["source"], lw=2)))
    arts += ax.plot([cx - w / 2] * 2, [y0, y1], color=cm.C["source"], lw=2)
    arts += ax.plot([cx + w / 2] * 2, [y0, y1], color=cm.C["source"], lw=2)
    arts.append(wave_glyph(ax, cx, 1.45, 0.6, cm.C["source"], 2))
    days = tiles(ax, 1, 5, 7.4, 9.6, 0.05, 2.95, gap=0.25)
    links = []
    for p in days:
        yy = p.get_y() + p.get_height() / 2
        links += ax.plot([cx + w / 2 + 0.1, 7.45], [1.45, yy], color=cm.C["source"], lw=1.4)

    def upd(t, a):
        for art in arts:
            art.set_alpha(a)
        ph = (t * 1.2) % 1.0
        for i, (p, ln) in enumerate(zip(days, links)):
            p.set_facecolor(mix(cm.FAINT, cm.C["gpu"], 0.8))
            p.set_alpha(a)
            ln.set_alpha(a * (0.35 + 0.65 * (1 - abs(ph - i / 5))))
    return upd


def _evidence_steps(n=12):
    rng = np.random.default_rng(7)
    inc = rng.uniform(0.25, 1.0, n)
    return inc / inc.sum()


def icon_minute_loop(ax):
    """The obvious way: at every minute, all past minutes are summed again from minute 1."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    n = 12
    inc = _evidence_steps(n)
    xs = np.linspace(0.6, 9.4, n)
    w = (xs[1] - xs[0]) * 0.62
    base, = ax.plot([0.3, 9.7], [1.1, 1.1], color=cm.MUTED, lw=1.5)
    bars = [ax.add_patch(mpatches.Rectangle((x - w / 2, 1.15), w, 0.2 + 0.75 * v / inc.max(), fc=cm.FAINT,
                                            ec="none")) for x, v in zip(xs, inc)]
    brk, = ax.plot([], [], color=cm.C["cpu"], lw=2.5)
    note = ax.text(0.3, 0.45, "", fontsize=cm.FS_LABEL - 1, color=cm.TEXT, ha="left", va="center")

    def upd(t, a):
        k = int((t * 3) % n)
        for i, b in enumerate(bars):
            b.set_facecolor(cm.C["cpu"] if i <= k else cm.FAINT)
            b.set_alpha(a * (1.0 if i <= k else 0.6))
        x0, x1 = xs[0] - w / 2, xs[k] + w / 2
        brk.set_data([x0, x0, x1, x1], [2.25, 2.4, 2.4, 2.25])
        note.set_text(f"Minute {k + 1}: re-reads & sums all {k + 1} minutes (1..{k + 1}) — 7,260 steps")
        for art in (base, brk, note):
            art.set_alpha(a)
    return upd


def icon_cumsum(ax):
    """Ours: each minute adds only the new minute (observed drift vs predicted wave) to the previous total."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    n = 12
    inc = _evidence_steps(n)
    xs = np.linspace(0.6, 9.4, n)
    w = (xs[1] - xs[0]) * 0.62
    tot = np.cumsum(inc)
    base, = ax.plot([0.3, 9.7], [1.1, 1.1], color=cm.MUTED, lw=1.5)
    bars = [ax.add_patch(mpatches.Rectangle((x - w / 2, 1.15), w, 0.2 + 0.75 * v / inc.max(), fc=cm.FAINT,
                                            ec="none")) for x, v in zip(xs, inc)]
    ys = 1.95 + 0.65 * tot / tot[-1]
    st, = ax.step(xs, ys, where="mid", color=cm.C["gpu"], lw=2.6)
    pts = ax.scatter(xs, ys, s=30, color=cm.C["gpu"], lw=0)
    note = ax.text(0.3, 0.45, "", fontsize=cm.FS_LABEL - 1, color=cm.C["gpu"], weight="bold", ha="left", va="center")

    def upd(t, a):
        k = int((t * 3) % n)
        for i, b in enumerate(bars):
            if i == k:
                b.set_facecolor(cm.C["gpu"])
                b.set_alpha(a)
            elif i < k:
                b.set_facecolor(cm.MUTED)
                b.set_alpha(0.35 * a)
            else:
                b.set_facecolor(cm.FAINT)
                b.set_alpha(0.25 * a)
        st.set_data(xs[:k + 1], ys[:k + 1])
        pts.set_offsets(np.column_stack([xs[:k + 1], ys[:k + 1]]))
        note.set_text(f"Minute {k + 1}: Score({k + 1}) = Score({k}) + min {k + 1} (1 add — 120 steps)")
        for art in (base, st, pts, note):
            art.set_alpha(a)
    return upd


def tick_icon(ax, ok):
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
    if ok:
        c = ax.add_patch(mpatches.Circle((0, 0), 0.9, fc=cm.C["good"], ec="none"))
        ln, = ax.plot([-0.45, -0.1, 0.5], [0.0, -0.38, 0.4], color=cm.BG, lw=5, solid_capstyle="round",
                      solid_joinstyle="round")
        return [c, ln]
    if ok is False:
        c = ax.add_patch(mpatches.Circle((0, 0), 0.9, fc=cm.C["bad"], ec="none"))
        l1, = ax.plot([-0.4, 0.4], [-0.4, 0.4], color=cm.BG, lw=5, solid_capstyle="round")
        l2, = ax.plot([-0.4, 0.4], [0.4, -0.4], color=cm.BG, lw=5, solid_capstyle="round")
        return [c, l1, l2]
    c = ax.add_patch(mpatches.Circle((0, 0), 0.85, fc="none", ec=cm.MUTED, lw=2.5, ls=(0, (3, 2))))
    h1, = ax.plot([0, 0], [0, 0.5], color=cm.MUTED, lw=3, solid_capstyle="round")
    h2, = ax.plot([0, 0.35], [0, 0], color=cm.MUTED, lw=3, solid_capstyle="round")
    return [c, h1, h2]


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------
def _batch_size(cfg):
    """Tsunamis advanced together in the sweep (config.yaml batch_size, wherever it sits)."""
    def walk(d):
        if isinstance(d, dict):
            if isinstance(d.get("batch_size"), int):
                return d["batch_size"]
            for v in d.values():
                r = walk(v)
                if r:
                    return r
        return None
    return walk(cfg) or 64


def build(cut, ctx):
    ev = load_evidence(ctx)
    N_MIN[0] = int(ctx.cfg["library"]["n_minutes"])
    BATCH = _batch_size(ctx.cfg)
    fig, chrome = cm.new_figure(SCENE, caption="Each change gives the same answer as the simple version, so the GPU can "
                                               f"score all {N_MIN[0]} minutes of every test in one pass.")
    chrome["caption"].set_position((0.5, 0.045))

    XL, WL = 0.025, 0.41         # the obvious way
    XR, WR = 0.525, 0.45         # ours
    H, Y = 0.165, [0.665, 0.425, 0.185]
    TICK_LABEL = "Gives the same answer as the simple version"

    # (the obvious way, ours, icons, check confirmed by a file: True / False / None)
    pairs = [
        ("One tsunami at a time", f"{BATCH} tsunamis advanced together on the GPU", (icon_serial_tiles, icon_batch_tiles),
         bool(ev["batch_pass"] and ev["batch_diff"] is not None) or None),
        ("Re-simulate the wave for every day tested", "Simulate each quake once, reuse it every day", (icon_resimulate, icon_library),
         bool(ev["lin_pass"] and ev["lin_err"] is not None) or None),
        ("Re-sum all past minutes at every step", "Update score incrementally (prev + new min)", (icon_minute_loop, icon_cumsum), ev["scan_pass"]),
    ]
    for (_, ours, _, ok) in pairs:
        if ok is not True:
            print(f"scene 07: check for '{ours}' not confirmed by a file in outputs/; no tick drawn")

    rows = []
    for k, ((serial, ours, icons, ok), y) in enumerate(zip(pairs, Y)):
        R = {}
        g_ser, g_res, g_arr, g_chk = Group(), Group(), Group(), Group()
        cs = cm.card(fig, (XL, y, WL, H))
        g_ser.add(cs, cs.text(0.04, 0.80, serial, fontsize=cm.FS_BODY, weight="bold", color=cm.TEXT, va="center"))
        ia = fig.add_axes((XL + 0.015, y + 0.012, WL - 0.03, H * 0.58))
        R["upd_s"] = icons[0](ia)
        g_ser.add(ia)
        cr = cm.card(fig, (XR, y, WR, H), edgecolor=cm.PANEL_EDGE)
        g_res.add(cr, cr.text(0.04, 0.80, ours, fontsize=cm.FS_BODY, weight="bold", color=cm.TEXT, va="center"))
        ib = fig.add_axes((XR + 0.015, y + 0.012, WR - 0.03, H * 0.58))
        R["upd_r"] = icons[1](ib)
        g_res.add(ib)
        arr = fig.add_artist(mpatches.FancyArrowPatch((XL + WL + 0.012, y + H / 2), (XR - 0.012, y + H / 2),
                                                      transform=fig.transFigure,
                                                      arrowstyle="-|>,head_width=8,head_length=12",
                                                      color=cm.MUTED, lw=3))
        g_arr.add(arr)
        if ok is True:      # tick only where a file confirms the check
            it = fig.add_axes((XR, y - 0.05, 0.025, 0.04))
            g_chk.add(it, tick_icon(it, True))
            g_chk.add(fig.text(XR + 0.032, y - 0.03, TICK_LABEL, fontsize=cm.FS_SMALL, color=cm.C["good"],
                               weight="bold", va="center"))
        R.update(ser=g_ser, res=g_res, arr=g_arr, chk=g_chk, card_r=cr.patches[0])
        rows.append(R)

    T0 = [0.6, 3.4, 6.2]

    def update(t):
        chrome["caption"].set_alpha(cm.fade(t, 0.4, dur=0.8))
        for k, R in enumerate(rows):
            r0 = T0[k]
            a_s = cm.fade(t, r0, dur=0.5)
            a_r = cm.fade(t, r0 + 1.0, dur=0.5)
            R["ser"].set(a_s)
            R["upd_s"](t, a_s)
            R["arr"].set(cm.fade(t, r0 + 0.7, dur=0.4))
            R["res"].set(a_r)
            R["upd_r"](t, a_r)
            R["chk"].set(cm.fade(t, r0 + 1.9, dur=0.5))
            act = cm.ramp(t, r0 + 1.0, r0 + 1.4) * (1 - cm.ramp(t, r0 + 2.8, r0 + 3.2))
            R["card_r"].set_edgecolor(mix(cm.PANEL_EDGE, cm.C["gpu"], act))
            R["card_r"].set_linewidth(1.5 + 1.5 * act)

    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
