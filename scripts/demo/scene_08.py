"""Scene 8 (Simulation): a simulated Mw 8.6 earthquake on the Hellenic Arc reaches the fleet.

Built from the T17 showcase trial (scripts/demo/make_showcase86_data.py, loaded by showcase_trials.py):
  outputs/demo_data/showcase86_snapshots.npz  eta every minute 0-120 (wave field)
  outputs/demo_data/showcase86_traffic.npz    lon, lat, snr_t per ship; g_lon, g_lat, g_trigger_s, g_is_buoy, stat_ais, thr_ais
  outputs/demo_data/showcase86_meta.json      trial.mw/lon/lat, gauges.n_gauges, ships_above_noise

Includes:
  * Bright cyan shallow-shelf contour (< 200 m) across the Aegean, Libyan, and Nile shelves so it is visually
    unmistakable that the glowing ships sit on shallow shelves (< 200 m).
  * Live network evidence-vs-threshold chart on the bottom right showing the fleet evidence crossing the
    1% false-alarm threshold as the wave reaches the shallow Aegean shelves.
"""
import json
import os
from pathlib import Path

import common as cm
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import showcase_trials as st

SCENE = 8
DATA = Path(os.environ.get("T27_SHOWCASE_DIR", cm.DEMO_DATA))
T_WAVE0 = 0.6                       # clock starts; it reaches 120 min HOLD_S before the end of the scene
HOLD_S = 2.5
SNR_ABOVE = 1.0                     # "above its noise": peak signal / noise > 1 (as ships_above_noise.definition)
SHELF_DEPTH_M = 200.0


def load_showcase(data_dir=DATA):
    """Old Mw 8.0 showcase (scene 11 uses it): dict with 'meta', 'snap', 'traffic', 'quiet'."""
    out = {}
    p = data_dir / "showcase_meta.json"
    out["meta"] = json.loads(p.read_text()) if p.exists() else None
    for key, name in (("snap", "showcase_snapshots.npz"), ("traffic", "showcase_traffic.npz"),
                      ("quiet", "quiet_day_example.npz")):
        p = data_dir / name
        out[key] = dict(np.load(p, allow_pickle=False)) if p.exists() else None
    return out


def minute_of(t, dur):
    """Clock (minutes after origin, float) at scene time t."""
    x = float(np.clip((t - T_WAVE0) / (dur - HOLD_S - T_WAVE0), 0, 1))
    return 120.0 * x ** 1.4          # slower early, while the wave is near the source


def wave_lut(lim_frac=0.03):
    """RGBA lookup (256) for eta in [-lim, lim]: WAVE_CMAP colour, alpha rising with |eta|."""
    x = np.linspace(-1, 1, 256)
    rgba = cm.WAVE_CMAP((x + 1) / 2)
    rgba[:, 3] = np.clip((np.abs(x) - lim_frac) / 0.45, 0, 1) ** 0.8 * 0.92
    return rgba.astype(np.float32)


class WaveField:
    """Bathymetry + wave composited into one RGB image per clock minute (pure function of the minute)."""

    def __init__(self, snap, grid, eta_lim, dim=0.42):
        self.eta = snap["eta"]
        wet, depth = st.snapshot_base(snap, grid)
        self.wet = wet
        lon, lat = snap["lon"], snap["lat"]
        dl, dp = lon[1] - lon[0], lat[1] - lat[0]
        self.extent = (lon[0] - dl / 2, lon[-1] + dl / 2, lat[0] - dp / 2, lat[-1] + dp / 2)
        self.lut = wave_lut()
        self.scale = 127.5 / eta_lim
        d = np.where(wet, np.nan_to_num(depth, nan=10.0), 10.0)
        bathy = cm.DEPTH_CMAP(np.clip((np.log10(np.clip(d, 10, 5000)) - 1.0) / 2.7, 0, 1))[..., :3]
        bg = np.array(mcolors.to_rgb(cm.BG))
        self.base = (bathy * (1 - dim) + bg * dim).astype(np.float32)
        self.base[~wet] = mcolors.to_rgb(cm.LAND)

    def eta_at(self, m):
        k0 = int(np.floor(m)); k1 = min(k0 + 1, self.eta.shape[0] - 1); f = m - k0
        return (1 - f) * self.eta[k0].astype(np.float32) + f * self.eta[k1].astype(np.float32)

    def rgb(self, m):
        idx = np.clip(self.eta_at(m) * self.scale + 127.5, 0, 255).astype(np.uint8)
        w = self.lut[idx]
        a = np.where(self.wet, w[..., 3], 0.0)[..., None]
        return (255 * (self.base * (1 - a) + w[..., :3] * a)).astype(np.uint8)


def eta_limit(snap, grid):
    """Colour limit (m): 98th percentile of |eta| over wet cells at 30 min, rounded up to 1/2/5 x 10^k."""
    wet, _ = st.snapshot_base(snap, grid)
    e = np.abs(snap["eta"][min(30, len(snap["eta"]) - 1)].astype(float))[wet]
    return st.nice_ceil(np.percentile(e, 98)) if e.size else 0.1


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    D = st.load("showcase86")
    meta, snap, tr = D["meta"], D["snap"], D["traffic"]
    sa = (meta or {}).get("ships_above_noise", {})
    if sa.get("n_above") and sa.get("above_n_depth_lt_200") is not None:
        caption = (f"The wave barely moves deep water but speeds up over shallow shelves (<200 m, cyan): "
                   f"{sa['above_n_depth_lt_200']} of the {sa['n_above']} ships that feel it sit in water under 200 m "
                   f"(median {sa['above_depth_m_median']:.0f} m).")
    else:
        caption = "The wave barely moves deep water but speeds up over shallow shelves, so mainly ships there feel it."
    fig, chrome = cm.new_figure(SCENE, caption=caption)
    for k in chrome.values():
        k.set_zorder(20)

    map_rect = (0.012, 0.085, 0.705, 0.78)
    ax = cm.map_axes(fig, map_rect, depth=snap is None)
    ax.set_zorder(1)
    ext0 = cm.aoi_extent()

    # ---------------------------------------------------------------- wave field
    if snap is not None:
        eta_lim = eta_limit(snap, ctx.grid)
        wf = WaveField(snap, ctx.grid, eta_lim)
        im = ax.imshow(wf.rgb(0.0), origin="lower", extent=wf.extent, interpolation="bilinear", zorder=0.4,
                       aspect="auto")
        ax.set_xlim(*ext0[:2]); ax.set_ylim(*ext0[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (ext0[2] + ext0[3]))))
    else:
        im, eta_lim = None, 0.1
        cm.placeholder(fig, "Wave field every minute, 0-120 min", needs="showcase86_snapshots.npz",
                       rect=(0.1, 0.45, 0.55, 0.3))[0].axes.set_zorder(10)

    # ---------------------------------------------------------------- visible shallow shelves (< 200 m)
    g = ctx.grid
    dep = np.where(g.wet, g.depth, np.nan)
    SHELF_COL = cm.C["depth_shallow"]   # bright cyan for < 200 m shelves
    shelf_cs = ax.contourf(g.lon, g.lat, dep, levels=[0.0, SHELF_DEPTH_M], colors=[SHELF_COL],
                           alpha=0.28, zorder=1.8, antialiased=True)
    shelf_ln = ax.contour(g.lon, g.lat, np.nan_to_num(dep, nan=1e4), levels=[SHELF_DEPTH_M],
                          colors=[SHELF_COL], linewidths=1.3, alpha=0.85, zorder=1.9)

    # ---------------------------------------------------------------- source
    if meta is not None:
        sx, sy = meta["trial"]["lon"], meta["trial"]["lat"]
        src_ring = ax.scatter([sx], [sy], s=60, facecolors="none", edgecolors=cm.C["source"], lw=2.5, zorder=6)
        src_dot = ax.scatter([sx], [sy], s=220, marker="*", color=cm.C["source"], edgecolors=cm.BG, lw=0.8,
                             zorder=7)
    else:
        src_ring = src_dot = None

    # ---------------------------------------------------------------- ships
    if tr is not None:
        slon, slat = tr["lon"], tr["lat"]
        snr_t = tr["snr_t"].astype(np.float32)                                         # minute m ends at m+1
        snr_t0 = np.concatenate([np.zeros((len(slon), 1), np.float32), snr_t], axis=1)  # clock minute 0..120
        snr_hi = max(10.0, st.nice_ceil(float(np.nanmax(snr_t))))
        snr_lo = snr_hi / 300.0 if snr_hi > 10 else 0.03
        snr_norm = mcolors.LogNorm(snr_lo, snr_hi)
        halo = ax.scatter(slon, slat, s=np.zeros(len(slon)), color=cm.C["signal"], lw=0, zorder=3)
        core = ax.scatter(slon, slat, s=14, c=np.full(len(slon), snr_lo), cmap=cm.SNR_CMAP, norm=snr_norm,
                          lw=0, zorder=4)
        first_above = np.argmax(snr_t0 > SNR_ABOVE, axis=1).astype(float)               # clock minute of first > 1
        first_above[~(snr_t0 > SNR_ABOVE).any(axis=1)] = np.inf
        n_total = len(slon)
        if meta is not None:
            sa = meta.get("ships_above_noise", {})
            n_final = int(np.isfinite(first_above).sum())
            if sa.get("n_above") is not None and (sa["n_above"] != n_final or sa.get("n_total", n_total) != n_total):
                print(f"scene 08 WARNING: ships above noise from snr_t = {n_final} of {n_total}, meta says "
                      f"{sa['n_above']} of {sa.get('n_total')} ({sa.get('definition')})")
    else:
        halo = core = None
        snr_norm = mcolors.LogNorm(0.03, 10)
        cm.placeholder(fig, "Ships coloured by signal relative to noise", needs="showcase86_traffic.npz",
                       rect=(0.1, 0.12, 0.55, 0.25))[0].axes.set_zorder(10)

    # ---------------------------------------------------------------- gauges and buoys
    if tr is not None:
        g_lon, g_lat, g_t = tr["g_lon"], tr["g_lat"], tr["g_trigger_s"].astype(float) / 60.0
        is_b = tr["g_is_buoy"].astype(bool)
        g_t = np.where(np.isfinite(g_t), g_t, np.inf)
        markers = []
        for sel, mk, s_off, s_on, col in ((~is_b, "^", 46, 70, cm.C["gauge"]), (is_b, "*", 150, 230, cm.C["buoy"])):
            off = ax.scatter(g_lon[sel], g_lat[sel], marker=mk, s=s_off, facecolors="none", edgecolors=cm.MUTED,
                             lw=1.1, zorder=5, alpha=0.8)
            on = ax.scatter(g_lon[sel], g_lat[sel], marker=mk, s=s_on, color=col, edgecolors=cm.BG, lw=0.7,
                            zorder=5.5)
            flash = ax.scatter(g_lon[sel], g_lat[sel], marker="o", s=np.zeros(sel.sum()), facecolors="none",
                               edgecolors=col, lw=2.0, zorder=5.4)
            markers.append((on, flash, g_t[sel], np.array(matplotlib.colors.to_rgba(col))))
        tg = g_t[~is_b]
        n_gauges = meta["gauges"]["n_gauges"] if meta is not None else int((~is_b).sum())
        if n_gauges != int((~is_b).sum()):
            print(f"scene 08 WARNING: meta n_gauges {n_gauges} != gauges in traffic file {(~is_b).sum()}")

    # ---------------------------------------------------------------- right column: clock, colour bars, counters, threshold chart
    X0 = 0.735
    clock_big = fig.text(X0, 0.82, "", fontsize=cm.FS_HERO + 2, weight="bold", color=cm.TEXT, ha="left",
                         va="center")
    fig.text(X0 + 0.085, 0.82, "min after\nearthquake", fontsize=cm.FS_LABEL, color=cm.MUTED, ha="left",
             va="center", linespacing=1.15)

    max_eta_coast = float(np.nanmax(snap["eta"])) if snap is not None else 9.4
    fig.text(X0, 0.73, "Open-sea wave height (clipped)", fontsize=cm.FS_LABEL, color=cm.TEXT, ha="left", va="center", weight="bold")
    fig.text(X0, 0.703, f"Deep sea ~0.2–0.5 m · grows to {max_eta_coast:.1f} m at coast", fontsize=cm.FS_LABEL - 2,
             color=cm.C["signal"], ha="left", va="center")
    cax_w = fig.add_axes((X0, 0.662, 0.235, 0.020))
    sm = matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(-eta_lim, eta_lim), cmap=cm.WAVE_CMAP)
    cb = fig.colorbar(sm, cax=cax_w, orientation="horizontal", extend="both")
    cb.set_ticks([-eta_lim, 0, eta_lim]); cb.set_ticklabels([f"≤ −{eta_lim:g} m", "0", f"≥ +{eta_lim:g} m"])
    cb.ax.tick_params(labelsize=cm.FS_LABEL - 1, colors=cm.MUTED, length=3)
    cb.outline.set_edgecolor(cm.FAINT)

    fig.text(X0, 0.60, "Ship signal relative to its noise", fontsize=cm.FS_LABEL, color=cm.TEXT, ha="left",
             va="center", weight="bold")
    cax_s = fig.add_axes((X0, 0.558, 0.235, 0.020))
    sm2 = matplotlib.cm.ScalarMappable(norm=snr_norm, cmap=cm.SNR_CMAP)
    cb2 = fig.colorbar(sm2, cax=cax_s, orientation="horizontal", extend="both")
    ticks = [v for v in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if snr_norm.vmin <= v <= snr_norm.vmax]
    cb2.set_ticks(ticks); cb2.set_ticklabels([f"{v:g}×" for v in ticks])
    cb2.ax.tick_params(labelsize=cm.FS_LABEL - 1, colors=cm.MUTED, length=3)
    cb2.outline.set_edgecolor(cm.FAINT)

    # Compact side-by-side counters
    fig.text(X0, 0.485, "Ships > noise (<200 m)", fontsize=cm.FS_LABEL - 1, color=cm.TEXT, ha="left", va="center")
    cnt_ships = fig.text(X0, 0.442, "", fontsize=cm.FS_H2 - 1, color=cm.C["signal"], weight="bold", ha="left", va="center")
    fig.text(X0 + 0.13, 0.485, "Gauges triggered", fontsize=cm.FS_LABEL - 1, color=cm.TEXT, ha="left", va="center")
    cnt_gauges = fig.text(X0 + 0.13, 0.442, "", fontsize=cm.FS_H2 - 1, color=cm.C["gauge"], weight="bold", ha="left", va="center")

    # Live fleet evidence vs 1% false-alarm threshold chart (bottom right)
    s9 = cm.load_scene_module(9)
    if tr is not None and "stat_ais" in tr and "thr_ais" in tr:
        eax, rel, e_line, e_dot, e_alarm, ymax = s9.evidence_axes(
            fig, (X0 + 0.03, 0.135, 0.21, 0.235), tr["stat_ais"], tr["thr_ais"], cm.C["network"]
        )
        eax.set_title("Fleet evidence vs alarm threshold", fontsize=cm.FS_LABEL, color=cm.TEXT,
                      weight="bold", loc="left", pad=6)
        eax.set_xlabel("Minutes after quake", fontsize=cm.FS_LABEL - 2, labelpad=2)
        eax.set_ylabel("× threshold", fontsize=cm.FS_LABEL - 2, labelpad=2)
        eax.tick_params(labelsize=cm.FS_LABEL - 2)
        eax.text(3, 1.0 + 0.04 * ymax, "1% false-alarm threshold", color=cm.C["alarm"],
                 fontsize=cm.FS_LABEL - 2, weight="bold", ha="left", va="bottom", zorder=7)
    else:
        rel = e_line = e_dot = None

    # ---------------------------------------------------------------- what you're seeing (over Anatolia, land)
    key = cm.card(fig, (0.435, 0.730, 0.275, 0.128))
    key.set_zorder(8)
    key.text(0.05, 0.82, "What you're seeing", fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.TEXT, va="center")
    key.text(0.05, 0.58, "Cyan shading: shallow shelves (<200 m)", fontsize=cm.FS_LABEL - 1, color=SHELF_COL, weight="bold", va="center")
    key.text(0.05, 0.35, "Orange/blue: open-sea wave height", fontsize=cm.FS_LABEL - 1, color=cm.MUTED, va="center")
    key.text(0.05, 0.12, "Glowing dots: ships where drift > noise", fontsize=cm.FS_LABEL - 1, color=cm.C["signal"], va="center")

    # shallow-shelf labels next to the clusters of ships that light up (placed on land beside them)
    halo_fx = [pe.withStroke(linewidth=3.5, foreground=cm.BG)]
    shelf_lbls = [
        ax.annotate("Aegean shelves (<200 m):\n31 of 34 triggered ships sit here",
                    xy=(25.4, 37.1), xytext=(27.3, 38.2),
                    fontsize=cm.FS_LABEL, color=SHELF_COL, weight="bold", ha="left", va="center",
                    zorder=9, path_effects=halo_fx, alpha=0.0,
                    arrowprops=dict(arrowstyle="-|>", color=SHELF_COL, lw=1.8, shrinkA=2, shrinkB=2)),
        ax.text(20.1, 31.55, "Libyan shelf (<200 m)", fontsize=cm.FS_LABEL, color=SHELF_COL, weight="bold",
                ha="center", va="center", zorder=9, path_effects=halo_fx, alpha=0.0),
        ax.text(30.3, 31.2, "Nile shelf (<200 m)", fontsize=cm.FS_LABEL, color=SHELF_COL, weight="bold",
                ha="center", va="center", zorder=9, path_effects=halo_fx, alpha=0.0),
    ]

    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50)
    veil_ax.axis("off")
    veil = veil_ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG,
                                                          alpha=1.0, lw=0))

    def update(t):
        veil.set_alpha(1.0 - cm.ramp(t, 0.0, 0.5))
        m = minute_of(t, dur)
        a_lbl = cm.ramp(m, 35, 60)
        for lb in shelf_lbls:
            lb.set_alpha(a_lbl)
            if hasattr(lb, "arrow_patch") and lb.arrow_patch is not None:
                lb.arrow_patch.set_alpha(a_lbl)
        clock_big.set_text(f"{int(np.floor(m + 1e-6)):d}")
        if im is not None:
            im.set_data(wf.rgb(m))
        if src_dot is not None:
            pulse = cm.ramp(t, 0.0, 1.8)
            src_ring.set_sizes([60 + 2400 * pulse]); src_ring.set_alpha(0.9 * (1 - pulse))
        if core is not None:
            k0 = int(np.floor(m)); k1 = min(k0 + 1, 120); f = m - k0
            snr = np.clip((1 - f) * snr_t0[:, k0] + f * snr_t0[:, k1], snr_norm.vmin, None)
            core.set_array(snr)
            lvl = np.clip(np.log10(snr / snr_norm.vmin) / np.log10(snr_norm.vmax / snr_norm.vmin), 0, 1)
            core.set_sizes(14 + 70 * lvl ** 1.5)
            halo.set_sizes(30 + 900 * lvl ** 2)
            hc = cm.SNR_CMAP(snr_norm(snr))
            hc[:, 3] = 0.22 * lvl ** 1.2
            halo.set_facecolors(hc)
            mfl = np.floor(m + 1e-6)
            cnt_ships.set_text(f"{int(np.sum(first_above <= mfl)):,} / {n_total:,}")
            cnt_gauges.set_text(f"{int(np.sum(tg <= m))} / {n_gauges}")
            if rel is not None:
                s9.draw_curve(e_line, e_dot, rel, m)
            for on, flash, tt, rgba in markers:
                fired = tt <= m
                fc = np.tile(rgba, (len(tt), 1)); fc[:, 3] = fired
                on.set_facecolors(fc)
                on.set_edgecolors(np.where(fired[:, None], mcolors.to_rgba(cm.BG), (0, 0, 0, 0)))
                age = np.where(fired, m - tt, np.inf)
                x = np.clip(age / 8.0, 0, 1)
                flash.set_sizes(np.where(age < 8, 40 + 900 * x, 0))
                ec = np.tile(rgba, (len(tt), 1)); ec[:, 3] = np.where(age < 8, 1 - x, 0)
                flash.set_edgecolors(ec)

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
