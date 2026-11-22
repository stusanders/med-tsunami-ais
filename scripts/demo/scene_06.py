"""Scene 6 (Method): how the simulation works, five steps lighting up in order.

Built from:
  step 1  fault-slip cartoon; outputs/demo_data/showcase86_snapshots.npz (the Hellenic Arc Mw 8.6 run of
          scenes 8-9: sea surface every minute, first 45 minutes shown)
  step 2  outputs/demo_data/showcase86_traffic.npz (that trial's 521 ships); one ship's drift reports built from
          config.yaml ais_noise (wobble, whole-degree headings, fixed offset, shared field), an illustration
  step 3  outputs/demo_data/cleaning_traces.npz (scripts/demo/make_cleaning_traces.py: real solver run of
          cand 186, shelf ship pool 3179, deep ship pool 2557); badge: outputs/yaw_correction.md (51/58 vessels,
          median robust SD 0.492 -> 0.242 m/s) and outputs/piraeus_calibration.json (rows_dedup)
  step 4  showcase86_traffic.npz (obs_drift at minute 80, lead candidate, stat_ais vs thr_ais),
          data/processed/candidates.csv (library size and positions); the scan ring is schematic
  step 5  the four compared instruments and the two measures (schematic; 1 in 100 = scoring.false_alarm_rate)
"""
import json
import textwrap

import common as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

SCENE = 6
ACCENT = "#DCE6F5"        # neutral highlight for the active step (no data meaning)
BASELINE_C = "#C3B1FF"    # the ship's "normal drift" (45-min trailing mean); no palette entry for this meaning
WATER = "#173B66"
CRUST = "#262D3D"
PLATE = "#323A4E"


# --------------------------------------------------------------------------------------
# Fading groups (create artists once, then only set_alpha)
# --------------------------------------------------------------------------------------
# Each artist's own alpha is stored on the artist when a group first collects it (shared leaves
# keep one value). Not a global id() map: ids are reused once a figure is freed, which broke
# frames when one process built a scene more than once (parallel chunks, build_all workers).
_BASE_ATTR = "_demo_base_alpha"


class Group:
    """A set of artists faded together; remembers each leaf's own alpha.

    Leaves are collected lazily on the first set(), skipping excluded artists (those animated
    separately), so groups can be filled in any order during build().
    """

    def __init__(self):
        self.roots, self.axes, self.skip = [], [], set()
        self.items = None        # [(artist, base_alpha)]
        self.a = None

    def add(self, *arts):
        for art in arts:
            if art is None:
                continue
            if isinstance(art, (list, tuple)):
                self.add(*art)
                continue
            if isinstance(art, Axes):
                self.axes.append(art)
            self.roots.append(art)
        return arts[0] if len(arts) == 1 else arts

    def exclude(self, *arts):
        for art in arts:
            if isinstance(art, (list, tuple)):
                self.exclude(*art)
            elif art is not None:
                self.skip.add(id(art))

    def _collect(self, art, out, seen):
        if id(art) in self.skip or id(art) in seen:
            return
        seen.add(id(art))
        if not isinstance(art, Axes):
            if not hasattr(art, _BASE_ATTR):
                base = art.get_alpha()
                setattr(art, _BASE_ATTR, 1.0 if base is None else base)
            out.append((art, getattr(art, _BASE_ATTR)))
        for k in (art.get_children() if hasattr(art, "get_children") else []):
            self._collect(k, out, seen)
        # an Annotation's arrow is not among its children and ignores the text's alpha
        ap = getattr(art, "arrow_patch", None)
        if ap is not None:
            self._collect(ap, out, seen)

    def set(self, a):
        if self.items is None:
            self.items, seen = [], set()
            for r in self.roots:
                self._collect(r, self.items, seen)
        # No "unchanged, skip" shortcut: leaves can sit in two groups (a card and a label on it),
        # and the frame must depend on t alone so chunks rendered in parallel match.
        a = float(np.clip(a, 0, 1))
        self.a = a
        vis = a > 0.004
        for ax in self.axes:
            ax.set_visible(vis)
        for art, base in self.items:
            art.set_alpha(base * a)


def mix(c1, c2, x):
    import matplotlib.colors as mc
    a, b = np.array(mc.to_rgb(c1)), np.array(mc.to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(x, 0, 1)))


def icon_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
    return ax


def draw_icon(ax, kind, color):
    """Consistent line-art icons for the five steps (and reused in step panels)."""
    arts = []
    lw = 3
    if kind == "wave":
        x = np.linspace(-0.85, 0.85, 80)
        arts += ax.plot(x, 0.18 + 0.32 * np.exp(-(x / 0.35) ** 2), color=color, lw=lw, solid_capstyle="round")
        arts += ax.plot([-0.85, 0.85], [-0.45, -0.45], color=color, lw=lw, alpha=0.6)
        arts += ax.plot(x, -0.45 + 0.22 * np.exp(-(x / 0.35) ** 2), color=color, lw=lw * 0.6, alpha=0.6)
    elif kind == "ship":
        hull = np.array([[-0.8, 0.0], [0.8, 0.0], [0.55, -0.35], [-0.65, -0.35]])
        arts.append(ax.add_patch(mpatches.Polygon(hull, closed=True, fc="none", ec=color, lw=lw)))
        arts.append(ax.add_patch(mpatches.Rectangle((-0.45, 0.0), 0.5, 0.3, fc="none", ec=color, lw=lw * 0.8)))
        xs = np.linspace(-0.8, 0.8, 40)
        arts += ax.plot(xs, -0.6 + 0.08 * np.sin(xs * 9), color=color, lw=lw * 0.6, alpha=0.6)
    elif kind == "clean":
        x = np.linspace(-0.85, -0.05, 18)
        y = 0.25 + 0.35 * np.array([0, .9, -.6, .7, -.8, .4, -.9, .8, -.3, .6, -.7, .9, -.5, .3, -.8, .6, -.4, 0])
        arts += ax.plot(x, y, color=color, lw=lw * 0.6, alpha=0.7)
        arts += ax.plot([0.05, 0.85], [0.25, 0.25], color=color, lw=lw)
        arts += ax.plot([-0.85, 0.85], [-0.55, -0.55], color=color, lw=lw * 0.6, alpha=0.6)
    elif kind == "score":
        arts.append(ax.add_patch(mpatches.FancyBboxPatch((-0.3, -0.85), 0.6, 1.7,
                    boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=color, lw=lw)))
        arts.append(ax.add_patch(mpatches.Rectangle((-0.3, -0.85), 0.6, 1.25, fc=color, ec="none", alpha=0.45)))
        arts += ax.plot([-0.6, 0.6], [0.1, 0.1], color=color, lw=lw * 0.7, ls=(0, (2, 1.5)))
    elif kind == "compare":
        for x, h in [(-0.55, 0.9), (0.0, 1.4), (0.55, 0.6)]:
            arts.append(ax.add_patch(mpatches.Rectangle((x - 0.18, -0.75), 0.36, h, fc="none", ec=color, lw=lw)))
        arts += ax.plot([-0.9, 0.9], [-0.75, -0.75], color=color, lw=lw)
    return arts


def nice_axes(ax, ylabel=None, xlabel=None):
    ax.set_facecolor(cm.PANEL)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(cm.FAINT)
    ax.tick_params(colors=cm.MUTED, labelsize=cm.FS_LABEL, length=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=cm.FS_LABEL, color=cm.MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=cm.FS_LABEL, color=cm.MUTED)


# --------------------------------------------------------------------------------------
# Timelines (seconds). Step windows, then per-step internal events.
# --------------------------------------------------------------------------------------

def timeline(cut):
    """Step windows (scene seconds). Step 1 events are scene times; steps 2-5 events are relative to the step start."""
    if cut == "3min":
        return dict(steps=[(0.6, 3.4), (3.4, 6.2), (6.2, 10.2), (10.2, 12.4), (12.4, 13.8)], all_lit=13.8,
                    xs=0.6, slip=0.8, bump=(0.9, 1.5), prop=(1.6, 3.1), clip=(1.0, 3.2), clip_in=0.8,
                    ships=(0.1, 0.8), ring=0.9, noise=(1.1, 1.4, 1.7, 2.0), noise_dur=0.3,
                    raw=(0.0, 0.8), avg=(0.8, 1.5), base=(1.5, 2.2), clean=(2.2, 3.1), badge=3.2,
                    fleet=0.1, cands=0.4, scan=(0.5, 1.0), meter=(1.0, 1.9),
                    who=0.1, how=0.4, days=(0.5, 1.2))
    return dict(steps=[(1.0, 6.0), (6.0, 11.0), (11.0, 17.0), (17.0, 21.0), (21.0, 23.6)], all_lit=23.6,
                xs=1.0, slip=1.4, bump=(1.6, 2.6), prop=(2.8, 5.5), clip=(2.0, 5.8), clip_in=1.6,
                ships=(0.3, 1.5), ring=1.6, noise=(2.0, 2.6, 3.2, 3.8), noise_dur=0.45,
                raw=(0.0, 1.2), avg=(1.2, 2.2), base=(2.2, 3.2), clean=(3.2, 4.5), badge=4.7,
                fleet=0.2, cands=0.8, scan=(1.0, 2.0), meter=(2.0, 3.6),
                who=0.2, how=0.8, days=(1.0, 2.2))


STEPS = [
    ("Simulate every candidate earthquake once", "wave"),
    ("Place a day's ships and add realistic noise", "ship"),
    ("Clean up each ship's position reports", "clean"),
    ("Match the fleet against every candidate earthquake", "score"),
    ("Compare with single ships, tide gauges and buoys", "compare"),
]
STEP3_LABELS = ("Raw reports", "1-minute average", "Usual drift removed", "Compared with the true current")
NOISE_LABELS = ("Heading jitter (±{wob:g}°)", "Whole-degree headings", "Fixed compass offset",
                "Wind, currents (±{sh:g} m/s)")

RX0, RX1, RY0, RY1 = 0.33, 0.975, 0.10, 0.845     # right-hand detail area (figure coords)
RW = RX1 - RX0
HEAD = dict(fontsize=cm.FS_SMALL + 2, weight="bold", color=cm.TEXT, va="center")
LBL = dict(fontsize=cm.FS_LABEL + 1, color=cm.TEXT, va="center")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_inputs(ctx):
    cfg = ctx.cfg
    d = {"cfg": cfg}
    cand_path = cm.REPO / cfg["paths"]["candidates"]
    d["cand"] = pd.read_csv(cand_path) if cand_path.exists() else None
    p = cm.DEMO_DATA / "cleaning_traces.npz"
    d["tr"] = dict(np.load(p)) if p.exists() else None
    pj = cm.DEMO_DATA / "cleaning_traces.json"
    d["tr_meta"] = json.loads(pj.read_text()) if pj.exists() else None
    # the Hellenic Arc Mw 8.6 showcase trial of scenes 8 and 9 (wave every minute, ships, fleet evidence)
    p = cm.DEMO_DATA / "showcase86_snapshots.npz"
    d["snap"] = dict(np.load(p)) if p.exists() else None
    p = cm.DEMO_DATA / "showcase86_traffic.npz"
    d["trf"] = dict(np.load(p)) if p.exists() else None
    pm = cm.DEMO_DATA / "showcase86_meta.json"
    d["meta86"] = json.loads(pm.read_text()) if pm.exists() else None
    pp = cm.REPO / "outputs" / "piraeus_calibration.json"
    d["piraeus"] = json.loads(pp.read_text()) if pp.exists() else None
    return d


def wave_rgba(eta, lim):
    """WAVE_CMAP colour with alpha rising with |eta| (transparent at rest), as scene 8."""
    x = np.clip(np.nan_to_num(eta.astype(np.float32)) / lim, -1, 1)
    rgba = cm.WAVE_CMAP((x + 1) / 2)
    rgba[..., 3] = np.clip((np.abs(x) - 0.03) / 0.45, 0, 1) ** 0.8 * 0.95
    return rgba


# --------------------------------------------------------------------------------------
# Step 1 panel: fault-slip cartoon, then the real simulated wave spreading
# --------------------------------------------------------------------------------------
def build_step1(fig, d, T):
    sub = {k: Group() for k in ("xs", "uplift_lbl", "slip", "clip")}

    # (a) cross-section cartoon (full width)
    card = cm.card(fig, (RX0, 0.495, RW, 0.35))
    sub["xs"].add(card)
    sub["xs"].add(card.text(0.02, 0.9, "The fault slips and the seafloor rises", **HEAD))
    ax = fig.add_axes((RX0 + 0.012, 0.505, RW - 0.024, 0.28))
    ax.set_xlim(0, 16.5); ax.set_ylim(-4.2, 2.6); ax.axis("off")
    xmax = 16.5
    x = np.linspace(0, xmax, 400)
    trench = 3.4
    floor0 = np.where(x < trench, -2.2, -2.2 + (x - trench) * (1.6 / (xmax - trench)))
    hang = x >= trench
    bump_shape = np.where(hang, np.exp(-((x - 5.0) / 1.1) ** 2), 0.0)
    A_floor, A_sea, sea0 = 0.75, 0.75, 1.2
    crust = ax.fill_between(x, -4.2, floor0, color=CRUST, lw=0)
    crust_bot = np.full_like(x, -4.2)
    ax.add_patch(mpatches.Polygon([[0, -2.2], [trench, -2.2], [9.0, -4.2], [0, -4.2]], closed=True, fc=PLATE,
                                  ec="none"))
    water = ax.fill_between(x, floor0, np.full_like(x, sea0), color=WATER, lw=0)
    floor_ln, = ax.plot(x, floor0, color=cm.MUTED, lw=2.5)
    sea_ln, = ax.plot(x, np.full_like(x, sea0), color=cm.C["wave_pos"], lw=3)
    ax.plot([trench, 9.0], [-2.2, -4.2], color=cm.C["source"], lw=3, ls=(0, (5, 3)))
    slip_up = ax.annotate("", xy=(5.4, -2.75), xytext=(7.0, -3.35),
                          arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", color=cm.C["source"], lw=3))
    slip_dn = ax.annotate("", xy=(7.6, -4.05), xytext=(6.0, -3.45),
                          arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6", color=cm.MUTED, lw=2))
    f5 = floor0[np.argmin(abs(x - 5.0))]
    upl = ax.annotate("", xy=(5.0, f5 + A_floor), xytext=(5.0, f5),
                      arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.18", color=cm.TEXT, lw=1.8,
                                      shrinkA=0, shrinkB=0))
    old_floor, = ax.plot(x[hang], floor0[hang], color=cm.MUTED, lw=1.5, ls=(0, (2, 2)))
    sub["xs"].add(ax)
    sub["uplift_lbl"].add(upl, old_floor)
    sub["slip"].add(slip_up, slip_dn)
    sub["xs"].exclude(upl, old_floor, slip_up, slip_dn)
    geo = dict(crust=crust, crust_bot=crust_bot, x=x, floor0=floor0, bump=bump_shape, A_floor=A_floor, A_sea=A_sea,
               sea0=sea0, water=water, floor_ln=floor_ln, sea_ln=sea_ln, xmax=xmax)

    # (b) the real simulated wave of the showcase earthquake, spreading over the real seafloor
    y_b, h_b = RY0, 0.38
    cb = cm.card(fig, (RX0, y_b, RW, h_b))
    sub["clip"].add(cb)
    clip = {}
    snap, meta = d["snap"], d["meta86"]
    if snap is not None:
        tri = (meta or {}).get("trial", {})
        mw = tri.get("mw", 8.6)
        sub["clip"].add(cb.text(0.02, 0.9, f"Simulated over the real seafloor: Hellenic Arc, Mw {mw:.1f}", **HEAD))
        clock = cb.text(0.98, 0.9, "", fontsize=cm.FS_SMALL + 2, color=cm.MUTED, va="center", ha="right")
        sub["clip"].exclude(clock)
        mrect = (RX0 + 0.008, y_b + 0.012, RW - 0.016, h_b - 0.085)
        lon0 = float(tri.get("lon", 22.74))
        lat0 = float(tri.get("lat", 34.62)) + 0.6
        aspect = (mrect[2] * cm.W_PX) / (mrect[3] * cm.H_PX)
        half_lon = 7.0
        half_lat = half_lon * np.cos(np.deg2rad(lat0)) / aspect
        ext = (lon0 - half_lon, lon0 + half_lon, lat0 - half_lat, lat0 + half_lat)
        mx = cm.map_axes(fig, mrect, extent=ext)
        lon, lat = snap["lon"], snap["lat"]
        im = mx.imshow(wave_rgba(snap["eta"][0], 0.5), origin="lower", zorder=2, interpolation="bilinear",
                       extent=(lon[0], lon[-1], lat[0], lat[-1]))
        mx.set_xlim(ext[0], ext[1]); mx.set_ylim(ext[2], ext[3])
        star, = mx.plot(float(tri.get("lon", 22.74)), float(tri.get("lat", 34.62)), marker="*", ms=18,
                        color=cm.C["source"], mec=cm.BG, zorder=4)
        sub["clip"].add(mx)
        sub["clip"].exclude(im)
        clip.update(im=im, eta=snap["eta"], clock=clock, n=len(snap["eta"]))
    else:
        sub["clip"].add(cm.placeholder(cb, "Simulated wave", needs="outputs/demo_data/showcase86_snapshots.npz"))

    def upd(t, a):
        for k, grp in sub.items():
            t_in = {"xs": T["xs"], "slip": T["slip"], "uplift_lbl": T["bump"][1] - 0.2, "clip": T["clip_in"]}[k]
            grp.set(a * cm.fade(t, t_in, dur=0.5))
        s_up = cm.ramp(t, *T["bump"])
        tau = np.clip((t - T["prop"][0]) / (T["prop"][1] - T["prop"][0]), 0, 1)
        x, b = geo["x"], geo["bump"]
        floor = geo["floor0"] + geo["A_floor"] * s_up * b
        if tau <= 0:
            sea = geo["sea0"] + geo["A_sea"] * s_up * b
        else:
            dist = tau * (geo["xmax"] - 5.0) * 0.72
            depth = np.clip(geo["sea0"] - geo["floor0"], 0.3, None)
            shoal = (depth.max() / depth) ** 0.25
            gg = lambda c: np.exp(-((x - c) / 1.1) ** 2)
            sea = geo["sea0"] + geo["A_sea"] * 0.5 * (gg(5.0 + dist) * shoal + gg(5.0 - dist))
        geo["floor_ln"].set_ydata(floor)
        geo["sea_ln"].set_ydata(sea)
        geo["water"].set_verts([np.concatenate([np.column_stack([x, floor]), np.column_stack([x[::-1], sea[::-1]])])])
        geo["crust"].set_verts([np.concatenate([np.column_stack([x, geo["crust_bot"]]),
                                                np.column_stack([x[::-1], floor[::-1]])])])
        if clip:
            p = cm.ramp(t, *T["clip"])
            minute = int(round(45 * p ** 1.2))
            clip["im"].set_data(wave_rgba(clip["eta"][min(minute, clip["n"] - 1)], 0.5))
            ca = a * cm.fade(t, T["clip_in"], dur=0.5)
            clip["im"].set_alpha(ca)
            clip["clock"].set_text(f"{minute} min after the earthquake")
            clip["clock"].set_alpha(ca)

    return upd


# --------------------------------------------------------------------------------------
# Step 2 panel: a day's ships appear, then one ship's drift picks up each kind of noise
# --------------------------------------------------------------------------------------
def noise_stages(cfg, n_min=30, seed=6):
    """One ship's drift reports with no tsunami, as each noise term of config.yaml ais_noise is added in turn.

    drift = SOG * sin(error in heading); terms: per-report heading wobble, whole-degree AIS headings, a fixed
    per-ship heading offset, then the slowly varying shared field (wind, currents). Illustration of the model's
    noise recipe with its own settings, not a saved trace.
    """
    an = cfg["ais_noise"]
    dt = float(an["report_interval_s"])
    sog = 14.0 * 0.514444
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_min * 60, dt)
    wob = rng.normal(0, an["heading_wobble_sd_deg"], len(t))
    frac = 0.37                                  # true heading's fractional degree
    bias = 0.8 * an["heading_bias_sd_deg"]
    phi = np.exp(-60.0 / an["shared_corr_time_s"])
    sh = np.zeros(n_min + 1)
    sh[0] = rng.normal(0, an["shared_sd_m_s"])
    for k in range(1, n_min + 1):
        sh[k] = phi * sh[k - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, an["shared_sd_m_s"])
    shared = np.interp(t / 60.0, np.arange(n_min + 1), sh)
    v = lambda e: sog * np.sin(np.deg2rad(e))
    e1 = wob
    e2 = np.round(frac + wob) - frac
    e3 = np.round(frac + wob + bias) - frac
    stages = [np.zeros_like(t), v(e1), v(e2), v(e3), v(e3) + shared]
    return t / 60.0, stages, dict(wob=an["heading_wobble_sd_deg"], sh=an["shared_sd_m_s"])


def build_step2(fig, d):
    g = Group()
    cfg = d["cfg"]
    anim = {}
    # (a) map: the day's traffic
    mx = cm.map_axes(fig, (RX0 + 0.005, 0.425, RW - 0.01, RY1 - 0.43))
    mx.apply_aspect()
    pos = mx.get_position()
    mc = cm.card(fig, (pos.x0 - 0.006, pos.y0 - 0.01, pos.width + 0.012, pos.height + 0.02))
    mc.set_zorder(mx.get_zorder() - 1)
    g.add(mc, mx)
    trf = d["trf"]
    if trf is not None:
        lon, lat = trf["lon"], trf["lat"]
        n = len(lon)
        order = np.random.default_rng(1).permutation(n)
        sc = mx.scatter(lon[order], lat[order], s=14, color=cm.C["ship"], lw=0, alpha=0.9, zorder=3)
        g.exclude(sc)
        # the ship whose reports are shown below: a shelf ship south of Crete (as scene 8)
        cand = np.where(trf["south_of_crete"] & trf["shelf"])[0]
        i = int(cand[0]) if len(cand) else int(np.argmin(trf["depth_m"]))
        ring, = mx.plot(lon[i], lat[i], "o", ms=20, mfc="none", mec=cm.C["signal"], mew=2.5, zorder=5)
        rl = mx.annotate("this ship", (lon[i], lat[i]), xytext=(18, -18), textcoords="offset points",
                         fontsize=cm.FS_LABEL + 1, color=cm.C["signal"], weight="bold", va="top", zorder=6)
        lab = mc.text(0.015, 0.94, f"One simulated day: {n} merchant ships at sea", **HEAD)
        g.exclude(ring, rl)
        anim.update(sc=sc, xy=np.column_stack([lon[order], lat[order]]), ring=[ring, rl], lab=lab)
    # (b) that ship's drift reports with no tsunami, one noise term at a time
    y_c, h_c = RY0, 0.305
    cc = cm.card(fig, (RX0, y_c, RW, h_c))
    g.add(cc)
    g.add(cc.text(0.015, 0.89, "Quiet sea (no tsunami): true drift = 0 m/s — all spread is ship & weather noise", fontsize=cm.FS_SMALL, weight="bold", color=cm.TEXT, va="center"))
    tmin, stages, par = noise_stages(cfg)
    ax = fig.add_axes((RX0 + 0.045, y_c + 0.02, RW * 0.58, h_c - 0.085))
    nice_axes(ax, ylabel="m/s")
    ax.set_xlim(0, tmin[-1] + 0.2); ax.set_ylim(-0.75, 0.75); ax.set_yticks([-0.5, 0, 0.5]); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    true_ln, = ax.plot([0, tmin[-1] + 0.2], [0, 0], color=cm.TEXT, lw=2.5, zorder=4)
    ax.text(0.4, -0.10, "true tsunami drift = 0 m/s (quiet sea: no wave pushing the ship)", fontsize=cm.FS_LABEL, color=cm.TEXT, zorder=5, va="top", weight="bold")
    pts = ax.scatter(tmin, stages[0], s=7, color=cm.C["ship"], lw=0, alpha=0.9, zorder=3)
    g.add(ax)
    g.exclude(pts)
    chips = []
    cx0 = RX0 + 0.045 + RW * 0.58 + 0.02
    cw = RX1 - 0.012 - cx0
    ch = (h_c - 0.10) / 4 - 0.008
    for k, s in enumerate(NOISE_LABELS):
        cy = y_c + h_c - 0.075 - (k + 1) * (ch + 0.008) + 0.008
        ca = cm.card(fig, (cx0, cy, cw, ch), radius=0.2)
        tx = ca.text(0.05, 0.5, "+ " + s.format(**par), fontsize=cm.FS_LABEL + 1, color=cm.TEXT, va="center",
                     weight="bold")
        g.add(ca)
        g.exclude(tx)
        chips.append((ca, tx))
    anim.update(pts=pts, tmin=tmin, stages=stages, chips=chips)
    return g, anim


def update_step2(t, a, T, t0, anim):
    lt = t - t0
    if "sc" in anim:
        n = len(anim["xy"])
        m = int(round(cm.ramp(lt, *T["ships"]) * n))
        anim["sc"].set_offsets(anim["xy"][:max(m, 1)])
        anim["sc"].set_alpha(a * 0.9 * (m > 0))
        ra = a * cm.fade(lt, T["ring"], dur=0.4)
        for art in anim["ring"]:
            art.set_alpha(ra)
    st = anim["stages"]
    y = st[0]
    for k, t_k in enumerate(T["noise"]):
        p = cm.ramp(lt, t_k, t_k + T["noise_dur"])
        y = y + (st[k + 1] - st[k]) * p
        ca, tx = anim["chips"][k]
        on = cm.ramp(lt, t_k, t_k + 0.25)
        patch = ca.patches[0]
        patch.set_edgecolor(mix(cm.PANEL_EDGE, ACCENT, on))
        patch.set_facecolor(mix(cm.PANEL, "#1C2944", on))
        tx.set_alpha(a * (0.3 + 0.7 * on))
    anim["pts"].set_offsets(np.column_stack([anim["tmin"], y]))
    anim["pts"].set_alpha(a * 0.9)


# --------------------------------------------------------------------------------------
# Step 3 panel: the cleaning, one shelf ship and one deep-water ship
# --------------------------------------------------------------------------------------
def build_step3(fig, d, T):
    g = Group()
    tr = d["tr"]
    anim = {}
    g.add(fig.text(RX0, 0.825, "Every few seconds each ship radios its position, heading and speed (AIS).",
                   fontsize=cm.FS_LABEL + 2, color=cm.MUTED, va="center"))
    glyphs = [(cm.C["ship"], "dot"), (cm.C["ship"], "line"), (BASELINE_C, "dash"), (cm.TEXT, "line")]
    chips = []
    cw = (RW - 0.03) / 4
    for k, s in enumerate(STEP3_LABELS):
        cx = RX0 + k * (cw + 0.01)
        ca = cm.card(fig, (cx, 0.735, cw, 0.058), radius=0.25)
        col, kind = glyphs[k]
        if kind == "dot":
            gl = ca.scatter([0.07, 0.10, 0.13], [0.42, 0.6, 0.48], s=22, color=col, lw=0)
        else:
            gl, = ca.plot([0.05, 0.14], [0.5, 0.5], color=col, lw=3, ls=(0, (4, 2)) if kind == "dash" else "-")
        tx = ca.text(0.57, 0.5, s if len(s) < 22 else s.replace(" the ", "\nthe ", 1), fontsize=cm.FS_LABEL + 1,
                     color=cm.TEXT, ha="center", va="center", weight="bold", linespacing=1.05)
        g.add(ca)
        g.exclude(gl)
        chips.append((ca, tx, gl))
    anim["chips"] = chips
    if tr is None:
        g.add(cm.placeholder(fig, "AIS cleaning traces", needs="outputs/demo_data/cleaning_traces.npz",
                             rect=(RX0, RY0, RW, 0.6)))
        return g, anim
    tm = tr["t_min_end"]
    tmid = tm - 0.5
    trep = tr["t_report_s"] / 60.0
    ships = (d["tr_meta"] or {}).get("ships", {})
    dep0 = ships.get("shelf", {}).get("depth_m")
    title0 = (f"Shelf ship ({dep0:,.0f} m deep): cleaning AIS reports to recover tsunami drift"
              if dep0 else "Shelf ship: cleaning AIS reports to recover tsunami drift")
    xcol, wcol, y1, hh = RX0 + 0.048, RW - 0.06, 0.235, 0.435
    g.add(fig.text(xcol, y1 + hh + 0.025, title0, fontsize=cm.FS_SMALL, weight="bold", color=cm.TEXT, va="center"))
    a1 = fig.add_axes((xcol, y1, wcol, hh))
    nice_axes(a1, ylabel="Sideways drift (m/s)", xlabel="Minutes after earthquake")
    a1.set_xlim(0, 120); a1.set_ylim(-0.85, 0.85); a1.set_xticks([0, 30, 60, 90, 120])
    a1.set_xticklabels(["0", "30", "60", "90", "120"])
    a1.set_yticks([-0.5, 0, 0.5])
    sig = float(tr["sigma_total"][0])
    a1.axhspan(-sig, sig, color=cm.FAINT, alpha=0.30, lw=0)
    a1.axhline(0, color=cm.FAINT, lw=1.2)
    raw = a1.scatter(trep, tr["raw_report"][0], s=10, color=cm.C["ship"], lw=0, alpha=0.55, zorder=2)
    avg, = a1.plot(tmid, tr["raw_min"][0], color=cm.C["ship"], lw=2.2, alpha=0.45, zorder=3)
    base, = a1.plot(tmid, tr["baseline"][0], color=BASELINE_C, lw=3.0, ls=(0, (4, 2)), zorder=4)
    cln, = a1.plot(tmid, tr["clean"][0], color=cm.C["ship"], lw=3.0, zorder=5)
    true, = a1.plot(tmid, tr["true_filtered"][0], color=cm.TEXT, lw=3.2, zorder=6)
    g.add(a1)
    anim[0] = dict(raw=raw, avg=avg, base=base, cln=cln, true=true, trep=trep,
                   raw_xy=np.column_stack([trep, tr["raw_report"][0]]), tmid=tmid,
                   avg_y=tr["raw_min"][0], base_y=tr["baseline"][0], cln_y=tr["clean"][0],
                   true_y=tr["true_filtered"][0])
    g.exclude([anim[0][k] for k in ("raw", "avg", "base", "cln", "true")])
    g.exclude([c[1] for c in chips])
    # real-data check of the turn correction (outputs/yaw_correction.md, T23; Piraeus row count from
    # outputs/piraeus_calibration.json)
    bg = Group()
    n_rep = (d["piraeus"] or {}).get("rows_dedup")
    rep = f"{n_rep / 1e6:.1f} million" if n_rep else "2.5 million"
    bc = cm.card(fig, (RX0, RY0 - 0.005, RW, 0.06), edgecolor=cm.C["good"], lw=2, radius=0.2)
    bc.text(0.012, 0.5, "Real data:", fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.C["good"], va="center")
    bc.text(0.118, 0.5, f"on {rep} Piraeus reports, turn correction halved drift noise "
                        f"for 51 of 58 turning ships", fontsize=cm.FS_LABEL, color=cm.TEXT, va="center")
    bg.add(bc)
    anim["badge"] = bg
    return g, anim


def update_step3(t, a, T, t0, anim):
    lt = t - t0
    stage_w = [T["raw"], T["avg"], T["base"], T["clean"]]
    for k, (ca, tx, gl) in enumerate(anim["chips"]):
        on = cm.ramp(lt, stage_w[k][0], stage_w[k][0] + 0.3)
        patch = ca.patches[0]
        patch.set_edgecolor(mix(cm.PANEL_EDGE, ACCENT, on))
        patch.set_facecolor(mix(cm.PANEL, "#1C2944", on))
        tx.set_alpha(a * (0.35 + 0.65 * on))
        gl.set_alpha(a * (0.35 + 0.65 * on))
    if "badge" in anim:
        anim["badge"].set(a * cm.fade(lt, T["badge"], dur=0.5))
    for s in range(2):
        if s not in anim:
            continue
        A = anim[s]
        n = int(round(cm.ramp(lt, *T["raw"]) * len(A["trep"])))
        A["raw"].set_offsets(A["raw_xy"][:max(n, 1)])
        A["raw"].set_alpha(a * (0.55 - 0.35 * cm.ramp(lt, T["avg"][0], T["avg"][0] + 0.6)) * (n > 0))
        for key, ykey, w in (("avg", "avg_y", T["avg"]), ("base", "base_y", T["base"]), ("cln", "cln_y", T["clean"]),
                             ("true", "true_y", T["clean"])):
            m = int(round(cm.ramp(lt, *w) * len(A["tmid"])))
            A[key].set_data(A["tmid"][:max(m, 1)], A[ykey][:max(m, 1)])
            dim = (1.0 - 0.65 * cm.ramp(lt, T["clean"][0], T["clean"][0] + 0.5)) if key in ("avg", "base") else 1.0
            A[key].set_alpha(a * dim * (m > 0))


# --------------------------------------------------------------------------------------
# Step 4 panel: the fleet's drift pattern against the library, then evidence vs the alarm level
# --------------------------------------------------------------------------------------
def build_step4(fig, d):
    g = Group()
    anim = {}
    trf, cand = d["trf"], d["cand"]
    y_top, h_top = 0.47, 0.375
    wa = 0.31
    xb = RX0 + wa + 0.035
    wb = RX1 - xb
    # (a) what the fleet sees: each ship's cleaned drift at one minute, coloured by direction
    ca = cm.card(fig, (RX0, y_top, wa, h_top))
    g.add(ca)
    g.add(ca.text(0.04, 0.92, "1. Observed fleet drift", fontsize=cm.FS_SMALL, weight="bold", color=cm.TEXT, va="center"))
    g.add(ca.text(0.04, 0.10, "each dot: one ship's cleaned drift", fontsize=cm.FS_LABEL, color=cm.MUTED,
                  va="center"))
    ext = cm.aoi_extent(d["cfg"])
    if trf is not None:
        m = 80
        mxa = cm.map_axes(fig, (RX0 + 0.008, y_top + 0.06, wa - 0.016, h_top - 0.13), extent=ext)
        v = trf["obs_drift"][:, m]
        col = cm.WAVE_CMAP(np.clip(v / 0.3, -1, 1) * 0.5 + 0.5)
        mxa.scatter(trf["lon"], trf["lat"], s=6 + 60 * np.clip(np.abs(v) / 0.3, 0, 1), c=col, lw=0, zorder=3)
        g.add(mxa)
    # (b) library of candidate earthquakes, scanned for the best match
    cb = cm.card(fig, (xb, y_top, wb, h_top))
    g.add(cb)
    n_c = len(cand) if cand is not None else 0
    g.add(cb.text(0.03, 0.92, f"2. Score {n_c:,} simulated quakes", fontsize=cm.FS_SMALL, weight="bold", color=cm.TEXT, va="center"))
    g.add(cb.text(0.03, 0.10, "find which quake pattern fits the fleet", fontsize=cm.FS_LABEL,
                  color=cm.MUTED, va="center"))
    if cand is not None and trf is not None:
        mxb = cm.map_axes(fig, (xb + 0.008, y_top + 0.06, wb - 0.016, h_top - 0.13), extent=ext)
        pos = cand.drop_duplicates(["region", "pos_id"])[["lon_deg", "lat_deg"]].values
        mxb.scatter(pos[:, 0], pos[:, 1], s=16, color=cm.C["source"], alpha=0.55, lw=0, zorder=3)
        best = np.array([float(trf["lead_lon_ais"][-1]), float(trf["lead_lat_ais"][-1])])
        order = np.random.default_rng(4).permutation(len(pos))[:14]
        scan_xy = np.vstack([pos[order], best[None]])
        ring, = mxb.plot(*scan_xy[0], "o", ms=18, mfc="none", mec=cm.TEXT, mew=2.5, zorder=5)
        bstar, = mxb.plot(*best, marker="*", ms=22, color=cm.C["network"], mec=cm.BG, zorder=6)
        blab = mxb.annotate("Closest fit to fleet", best, xytext=(12, -26), textcoords="offset points",
                            fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.C["network"], zorder=7)
        g.add(mxb)
        g.exclude(ring, bstar, blab)
        anim.update(ring=ring, scan_xy=scan_xy, best=[bstar, blab])
    # (c) evidence for the best match, minute by minute, against the alarm level (real showcase run)
    cc = cm.card(fig, (RX0, RY0, RW, 0.335))
    g.add(cc)
    g.add(cc.text(0.015, 0.88, "3. Test detection: fleet match score vs 1% false-alarm threshold", fontsize=cm.FS_SMALL, weight="bold", color=cm.TEXT, va="center"))
    if trf is not None:
        stat, thr = trf["stat_ais"].astype(float), float(trf["thr_ais"])
        tmn = trf["t_min"].astype(float)
        ax = fig.add_axes((RX0 + 0.05, RY0 + 0.055, RW - 0.08, 0.2))
        nice_axes(ax)
        ax.set_xlim(0, 120); ax.set_ylim(0, max(stat.max(), thr) * 1.08)
        ax.set_xticks([0, 30, 60, 90, 120]); ax.set_xticklabels(["0", "30", "60", "90", "120 min"])
        ax.set_yticks([])
        ax.axhline(thr, color=cm.C["alarm"], lw=2.5, ls=(0, (5, 3)))
        ax.text(2, thr * 1.04, "Alarm level (1 false alarm per 100 quiet days)", fontsize=cm.FS_LABEL + 1,
                color=cm.C["alarm"], va="bottom")
        ln, = ax.plot(tmn[:1], stat[:1], color=cm.C["network"], lw=3)
        k_x = int(np.argmax(stat > thr)) if (stat > thr).any() else None
        mk = None
        if k_x is not None:
            mk, = ax.plot(tmn[k_x], thr, "o", ms=16, mfc="none", mec=cm.C["alarm"], mew=2.5)
            mlab = ax.annotate("Alarm", (tmn[k_x], thr), xytext=(-60, 14), textcoords="offset points",
                               fontsize=cm.FS_LABEL + 2, weight="bold", color=cm.C["alarm"])
            g.exclude(mk, mlab)
            mk = [mk, mlab]
        g.add(ax)
        g.exclude(ln)
        anim.update(ln=ln, tmn=tmn, stat=stat, k_x=k_x, mk=mk)
    for xa, ya, xb_, yb_ in ((RX0 + wa + 0.004, y_top + h_top / 2, xb - 0.004, y_top + h_top / 2),
                             (xb + wb / 2, y_top - 0.004, xb + wb / 2, RY0 + 0.335 + 0.004)):
        g.add(fig.add_artist(mpatches.FancyArrowPatch((xa, ya), (xb_, yb_), transform=fig.transFigure,
                                                      arrowstyle="-|>,head_width=6,head_length=9", color=cm.MUTED,
                                                      lw=2.5)))
    return g, anim


def update_step4(t, a, T, t0, anim):
    lt = t - t0
    if "ring" in anim:
        p = cm.ramp(lt, *T["scan"])
        xy = anim["scan_xy"]
        k = min(int(p * (len(xy) - 1) + 1e-9), len(xy) - 1)
        anim["ring"].set_data([xy[k, 0]], [xy[k, 1]])
        anim["ring"].set_alpha(a * cm.fade(lt, T["scan"][0], dur=0.2))
        ba = a * cm.fade(lt, T["scan"][1], dur=0.3)
        for art in anim["best"]:
            art.set_alpha(ba)
    if "ln" in anim:
        p = cm.ramp(lt, *T["meter"])
        n = max(int(round(p * len(anim["tmn"]))), 1)
        anim["ln"].set_data(anim["tmn"][:n], anim["stat"][:n])
        anim["ln"].set_alpha(a * (p > 0))
        if anim["mk"]:
            on = 1.0 if (anim["k_x"] is not None and n > anim["k_x"]) else 0.0
            for art in anim["mk"]:
                art.set_alpha(a * on)


# --------------------------------------------------------------------------------------
# Step 5 panel: who is compared, and on what
# --------------------------------------------------------------------------------------
def build_step5(fig, d):
    g = Group()
    anim = {}
    items = [(cm.C["network"], "network", "Our ship network"), (cm.C["single"], "single", "One ship on its own"),
             (cm.C["gauge"], "gauge", "Coastal tide gauges"), (cm.C["buoy"], "buoy", "Deep-sea buoys")]
    who = Group()
    gap = 0.015
    w = (RW - gap * 3) / 4
    y0, h = 0.57, 0.275
    for i, (col, kind, name) in enumerate(items):
        x0 = RX0 + i * (w + gap)
        c = cm.card(fig, (x0, y0, w, h), edgecolor=col, lw=2)
        who.add(c)
        who.add(c.text(0.5, 0.14, name, fontsize=cm.FS_LABEL + 2, weight="bold", color=col, ha="center",
                       va="center"))
        ia = icon_axes(fig, (x0 + w / 2 - 0.05, y0 + 0.08, 0.10, 0.17))
        _instrument_icon(ia, kind, col)
        who.add(ia)
    # two comparative cards showing the direct comparison across the systems
    how = Group()
    hw = (RW - 0.02) / 2
    yh, hh = RY0, 0.43
    # (a) How fast: direct timeline comparison (minutes to confirm)
    c1 = cm.card(fig, (RX0, yh, hw, hh))
    how.add(c1)
    how.add(c1.text(0.05, 0.90, "1. How fast does each confirm?", **HEAD))
    how.add(c1.text(0.05, 0.79, "Compare confirmation time after each earthquake:",
                    fontsize=cm.FS_LABEL, color=cm.MUTED, va="center"))
    speed_rows = [
        ("Coastal tide gauges", 0.32, cm.C["gauge"], "when wave hits coast"),
        ("Deep-sea buoys", 0.48, cm.C["buoy"], "2 buoys in Ionian only"),
        ("Ship drift network", 0.78, cm.C["network"], "when wave reaches shelf"),
        ("Ships w/ precise GPS", 0.14, cm.C["gps"], "in minutes, over source"),
    ]
    s_bars = []
    for idx, (lbl, w_frac, col, note) in enumerate(speed_rows):
        yr = 0.64 - idx * 0.145
        how.add(c1.text(0.05, yr + 0.038, lbl, fontsize=cm.FS_LABEL, weight="bold", color=col, va="center"))
        how.add(c1.text(0.95, yr + 0.038, note, fontsize=cm.FS_LABEL - 2, color=cm.MUTED, ha="right", va="center"))
        how.add(c1.add_patch(mpatches.Rectangle((0.05, yr - 0.032), 0.90, 0.038, fc=cm.FAINT, ec="none", alpha=0.45)))
        b = c1.add_patch(mpatches.Rectangle((0.05, yr - 0.032), 0.0, 0.038, fc=col, ec="none"))
        how.exclude(b)
        s_bars.append((b, 0.90 * w_frac))

    # (b) How often wrong: false-alarm comparison on quiet days (single ship vs ship network)
    c2 = cm.card(fig, (RX0 + hw + 0.02, yh, hw, hh))
    how.add(c2)
    how.add(c2.text(0.05, 0.90, "2. How often wrong on quiet days?", **HEAD))
    how.add(c2.text(0.05, 0.79, "False-alarm rate on days with no tsunami:",
                    fontsize=cm.FS_LABEL, color=cm.MUTED, va="center"))
    far_rows = [
        ("One ship on its own (3σ rule)", 1.0, cm.C["single"], "100% of quiet days (random ship turns)"),
        ("Our fleet network (matched filter)", 0.04, cm.C["network"], "1% target (requires wave pattern across ships)"),
    ]
    f_bars = []
    for idx, (lbl, w_frac, col, note) in enumerate(far_rows):
        yr = 0.58 - idx * 0.27
        how.add(c2.text(0.05, yr + 0.065, lbl, fontsize=cm.FS_LABEL + 1, weight="bold", color=col, va="center"))
        how.add(c2.add_patch(mpatches.Rectangle((0.05, yr - 0.025), 0.90, 0.052, fc=cm.FAINT, ec="none", alpha=0.45)))
        b = c2.add_patch(mpatches.Rectangle((0.05, yr - 0.025), 0.0, 0.052, fc=col, ec="none"))
        how.exclude(b)
        how.add(c2.text(0.05, yr - 0.075, note, fontsize=cm.FS_LABEL - 1, color=cm.TEXT, va="center"))
        f_bars.append((b, 0.90 * w_frac))

    anim.update(who=who, how=how, s_bars=s_bars, f_bars=f_bars)
    return g, anim


def update_step5(t, a, T, t0, anim):
    lt = t - t0
    anim["who"].set(a * cm.fade(lt, T["who"], dur=0.4))
    ha = a * cm.fade(lt, T["how"], dur=0.4)
    anim["how"].set(ha)
    p = cm.ramp(lt, *T["days"])
    for b, wmax in anim["s_bars"] + anim["f_bars"]:
        b.set_width(wmax * p)
        b.set_alpha(ha)


def _instrument_icon(ax, kind, col):
    if kind == "network":
        rng = np.random.default_rng(3)
        gx, gy = np.meshgrid([-0.6, 0.0, 0.6], [-0.55, 0.0, 0.55])
        pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, 0.12, (9, 2))
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.hypot(*(pts[i] - pts[j])) < 0.75:
                    ax.plot(*pts[[i, j]].T, color=col, lw=1.2, alpha=0.5)
        ax.scatter(*pts.T, s=60, color=col, lw=0)
    elif kind == "single":
        ax.scatter([0], [0], s=90, color=col, lw=0)
        for r in (0.4, 0.7):
            ax.add_patch(mpatches.Circle((0, 0), r, fc="none", ec=col, lw=2, alpha=0.7 if r < 0.5 else 0.4))
    elif kind == "gauge":
        ax.add_patch(mpatches.Rectangle((-0.08, -0.9), 0.16, 1.7, fc=col, ec="none"))
        for yv in np.linspace(-0.6, 0.6, 5):
            ax.plot([0.08, 0.3], [yv, yv], color=col, lw=2)
        xx = np.linspace(-1, 1, 60)
        ax.plot(xx, -0.1 + 0.08 * np.sin(xx * 8), color=cm.C["wave_neg"], lw=2.5)
    elif kind == "buoy":
        ax.add_patch(mpatches.Polygon([[-0.45, -0.15], [0.45, -0.15], [0.3, -0.45], [-0.3, -0.45]], fc=col, ec="none"))
        ax.plot([0, 0], [-0.15, 0.6], color=col, lw=3)
        ax.scatter([0], [0.65], s=50, color=col, lw=0)
        xx = np.linspace(-1, 1, 60)
        ax.plot(xx, -0.3 + 0.08 * np.sin(xx * 8), color=cm.C["wave_neg"], lw=2.5)
        ax.plot([0, 0], [-0.45, -0.95], color=col, lw=1.5, ls=(0, (2, 2)))



# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------
def build(cut, ctx):
    T = timeline(cut)
    d = load_inputs(ctx)
    fig, chrome = cm.new_figure(SCENE, caption="Each earthquake is simulated once over the real seafloor; every simulated "
                                               "day then adds ships and noise to test whether the fleet spots it.")
    chrome["caption"].set_position((0.5, 0.035))

    # ---- left column: five step cards
    steps = []
    x0, wc, hc, gap = 0.025, 0.285, 0.118, 0.031
    for k, (title, icon) in enumerate(STEPS):
        y = RY1 - hc - k * (hc + gap)
        c = cm.card(fig, (x0, y, wc, hc), radius=0.08)
        bx = fig.add_axes((x0 + 0.008, y + 0.02, 0.045, hc - 0.04)); bx.axis("off")
        bx.set_xlim(-1, 1); bx.set_ylim(-1, 1); bx.set_aspect("equal")
        circ = bx.add_patch(mpatches.Circle((0, 0), 0.82, fc=cm.FAINT, ec="none"))
        num = bx.text(0, -0.02, str(k + 1), fontsize=cm.FS_SMALL + 2, weight="bold", color=cm.TEXT, ha="center",
                      va="center")
        t1 = c.text(0.215, 0.5, "\n".join(textwrap.wrap(title, 24)), fontsize=cm.FS_SMALL, weight="bold",
                    color=cm.TEXT, va="center", linespacing=1.2)
        ia = icon_axes(fig, (x0 + wc - 0.041, y + 0.03, 0.033, hc - 0.06))
        ic = draw_icon(ia, icon, cm.TEXT)
        steps.append(dict(card=c, patch=c.patches[0], circ=circ, num=num, t1=t1, icon=ic))
        if k < 4:
            fig.add_artist(mpatches.FancyArrowPatch((x0 + 0.04, y - 0.003), (x0 + 0.04, y - gap + 0.003),
                                                    transform=fig.transFigure, arrowstyle="-|>,head_width=4,head_length=6",
                                                    color=cm.FAINT, lw=2))

    # ---- right-hand panels
    upd1 = build_step1(fig, d, T)
    p2, anim2 = build_step2(fig, d)
    p3, anim3 = build_step3(fig, d, T)
    p4, anim4 = build_step4(fig, d)
    p5, anim5 = build_step5(fig, d)
    starts = [s for s, _ in T["steps"]]

    def emph(k, t):
        s, e = T["steps"][k]
        on = cm.ramp(t, s - 0.15, s + 0.35) * (1 - cm.ramp(t, e - 0.15, e + 0.35))
        return max(on, cm.ramp(t, T["all_lit"], T["all_lit"] + 0.6))

    def panel_alpha(k, t):
        s, e = T["steps"][k]
        a = cm.ramp(t, s, s + 0.45)
        if k < 4:
            a *= 1 - cm.ramp(t, e - 0.35, e)
        return a

    def update(t):
        intro = cm.fade(t, 0.0, dur=0.7)
        for k, st in enumerate(steps):
            e = emph(k, t)
            base = 0.35 + 0.65 * e
            st["patch"].set_edgecolor(mix(cm.PANEL_EDGE, ACCENT, e))
            st["patch"].set_facecolor(mix(cm.PANEL, "#1C2944", e))
            st["patch"].set_linewidth(1.5 + 1.5 * e)
            st["patch"].set_alpha(intro)
            st["circ"].set_facecolor(mix(cm.FAINT, "#3E6FB0", e))
            st["circ"].set_alpha(intro)
            for art in (st["num"], st["t1"]):
                art.set_alpha(intro * base)
            for art in st["icon"]:
                art.set_alpha(intro * base * (0.6 if isinstance(art, Line2D) and art.get_linewidth() < 2.5 else 1.0))
        upd1(t, panel_alpha(0, t))
        a2 = panel_alpha(1, t)
        p2.set(a2)
        update_step2(t, a2, T, starts[1], anim2)
        a3 = panel_alpha(2, t)
        p3.set(a3)
        update_step3(t, a3, T, starts[2], anim3)
        a4 = panel_alpha(3, t)
        p4.set(a4)
        update_step4(t, a4, T, starts[3], anim4)
        a5 = panel_alpha(4, t)
        p5.set(a5)
        update_step5(t, a5, T, starts[4], anim5)
        chrome["caption"].set_alpha(cm.fade(t, 0.5, dur=0.8))

    return cm.Scene(fig, update, still_t=starts[2] + T["badge"] + 0.8 if cut == "full" else None)


if __name__ == "__main__":
    cm.main(__file__)
