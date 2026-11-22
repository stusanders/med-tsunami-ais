"""Scene 5 (Data): real inputs, phased onto the Mediterranean map.  Full cut only (15 s).

Each data source appears first as a large centre-aligned banner in the centre of the map with its
icon centred directly beneath the text, then animates/fades onto the enlarged map and into the
summary table below, one data source at a time.

Built from: data/processed/{grid.npz, receiver_pool.npz, candidates.csv}, tsunami.gauges.load_gauges,
cards.yaml (inazu2020 citation year).
"""
import re

import common as cm
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

SCENE = 5

REGION_COL = {"hellenic": cm.C["source"], "calabrian": "#FF9E6B", "cyprus": "#FFE69A", "aegean": "#E8C9A0"}


def _read_csv(rel):
    return pd.read_csv(cm.REPO / rel) if cm.exists(rel) else None


def _inazu_year(cards):
    m = re.search(r"\((\d{4})\)", (cards.get("inazu2020") or {}).get("cite", ""))
    return m.group(1) if m else None


def _icon(ax, kind):
    """Small glyph at (0.5, 0.5) of a square-ish icon axes; returns its artists."""
    if kind == "depth":
        return [ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, color="#2F7FBF", lw=0))]
    if kind == "traffic":
        return [ax.scatter([0.3, 0.55, 0.75, 0.42], [0.35, 0.6, 0.4, 0.75], s=60, color=cm.C["ship"], lw=0)]
    if kind == "faults":
        return ax.plot([0.15, 0.4, 0.6, 0.85], [0.3, 0.55, 0.6, 0.75], color=cm.C["source"], lw=3.5) + \
            [ax.scatter([0.15, 0.4, 0.6, 0.85], [0.3, 0.55, 0.6, 0.75], s=45, color=cm.C["source"], lw=0)]
    if kind == "gauges":
        return [ax.scatter([0.3], [0.5], s=140, marker="^", color=cm.C["gauge"], lw=0),
                ax.scatter([0.72], [0.5], s=120, marker="D", color=cm.C["buoy"], lw=0)]
    if kind == "ship":
        hull = np.array([[0.1, 0.5], [0.9, 0.5], [0.78, 0.3], [0.2, 0.3]])
        return [ax.add_patch(Polygon(hull, closed=True, fc="none", ec=cm.C["ship"], lw=2.8)),
                ax.add_patch(Rectangle((0.3, 0.5), 0.28, 0.2, fc="none", ec=cm.C["ship"], lw=2.2))]
    if kind == "noise":
        xx = np.linspace(0.1, 0.9, 13)
        yy = 0.5 + 0.28 * np.where(np.arange(13) % 2, 1, -1) * np.r_[.4, 1, .6, .9, .3, 1, .7, .5, 1, .4, .8, .6, .3]
        return ax.plot(xx, yy, color=cm.C["single"], lw=2.5)
    return []


def build(cut, ctx):
    cfg = ctx.cfg
    cards = cm.load_cards()
    fig, chrome = cm.new_figure(SCENE, caption="")
    for k in chrome:
        chrome[k].set_zorder(20)

    # ---------------- enlarged Mediterranean map ----------------
    ext = cm.aoi_extent(cfg)
    map_h = 0.58
    map_w = map_h * cm.H_PX / cm.W_PX * (ext[1] - ext[0]) * np.cos(np.deg2rad(0.5 * (ext[2] + ext[3]))) / \
        (ext[3] - ext[2])
    map_x0 = 0.5 - map_w / 2
    map_y0 = 0.275
    map_rect = (map_x0, map_y0, map_w, map_h)
    w_px, h_px = int(round(map_rect[2] * cm.W_PX)), int(round(map_rect[3] * cm.H_PX))

    mfig = cm.plt.figure(figsize=(w_px / cm.DPI, h_px / cm.DPI), dpi=cm.DPI)
    ax = cm.map_axes(mfig, (0, 0, 1, 1), extent=ext, depth=False, coast=True)
    ax.set_aspect("auto")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    layers = {}
    grid = ctx.grid
    d_im = ax.imshow(np.log10(np.clip(np.where(grid.wet, grid.depth, np.nan), 10, 5000)),
                     origin="lower", cmap=cm.DEPTH_CMAP, extent=(grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1]),
                     vmin=1, vmax=3.7, interpolation="bilinear", zorder=1, aspect="auto")
    layers["depth"] = [d_im]

    try:
        pool = ctx.pool()
        sh = pool.rtype == 0
        w = pool.weight[sh].astype(float)
        sz = 0.6 + 11.0 * np.sqrt(w / np.percentile(w, 99.5)).clip(0, 1.5)
        order = np.argsort(w)
        layers["traffic"] = [ax.scatter(pool.lon[sh][order], pool.lat[sh][order], s=sz[order],
                                        color=cm.C["ship"], lw=0, alpha=0.55, zorder=3)]
    except Exception as e:  # noqa: BLE001
        print("scene 05: pool unavailable:", e)

    cand = _read_csv(cfg["paths"]["candidates"])
    if cand is not None:
        fault_arts = []
        pos = cand.drop_duplicates("pos_id").sort_values("pos_id")
        for reg, df in pos.groupby("region", sort=False):
            col = REGION_COL.get(reg, cm.C["source"])
            if reg != "aegean":
                fault_arts += ax.plot(df.trench_lon_deg, df.trench_lat_deg, color=col, lw=2.6, zorder=4)
            fault_arts.append(ax.scatter(df.trench_lon_deg, df.trench_lat_deg, s=38 if reg != "aegean" else 28,
                                         marker="o" if reg != "aegean" else "D", color=col, edgecolor=cm.BG,
                                         lw=0.7, zorder=4.1))
        layers["faults"] = fault_arts

    try:
        from tsunami import gauges as gauges_mod
        gt = gauges_mod.load_gauges(ctx.grid, cfg)
        g, b = gt[gt.rtype == 1], gt[gt.rtype == 2]
        layers["gauges"] = [
            ax.scatter(g.lon_deg, g.lat_deg, s=75, marker="^", color=cm.C["gauge"], edgecolor=cm.BG, lw=0.8, zorder=5),
            ax.scatter(b.lon_deg, b.lat_deg, s=210, marker="D", color=cm.C["buoy"], edgecolor="white", lw=1.3,
                       zorder=5.1)]
    except Exception as e:  # noqa: BLE001
        print("scene 05: gauges unavailable:", e)

    for arts in layers.values():
        for ar in arts:
            ar.set_visible(False)

    # Snapshot 0: base coastline outline before GEBCO depth fades in
    mfig.canvas.draw()
    base_snap = np.asarray(mfig.canvas.buffer_rgba()).copy()
    snaps = [("base", base_snap)]
    for key in ("depth", "traffic", "faults", "gauges"):
        for ar in layers.get(key, []):
            ar.set_visible(True)
        mfig.canvas.draw()
        snaps.append((key, np.asarray(mfig.canvas.buffer_rgba()).copy()))
    cm.plt.close(mfig)

    max_ = fig.add_axes(map_rect)
    max_.set_xlim(0, 1); max_.set_ylim(0, 1); max_.set_xticks([]); max_.set_yticks([])
    for sp in max_.spines.values():
        sp.set_visible(True); sp.set_color(cm.FAINT); sp.set_alpha(0.0)
    snap_im = [(key, max_.imshow(img, extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest",
                                 alpha=0.0, zorder=i)) for i, (key, img) in enumerate(snaps)]

    # ---------------- data items & timing ----------------
    yr = _inazu_year(cards)
    inazu_name = f"Inazu et al. ({yr})" if yr else "Inazu et al."
    real_items = [
        ("depth", "GEBCO: Seafloor depth", "GEBCO", "Seafloor depth, which shapes the wave"),
        ("traffic", "EMODnet: Ship traffic", "EMODnet", "Ship traffic: where merchant ships sail"),
        ("faults", "EFSM20: Fault geometry", "EFSM20", "Fault geometry: where big quakes can happen"),
        ("gauges", "IOC + CAT-INGV: Tide gauges & buoys", "IOC + CAT-INGV", "Tide gauges and buoys to compare against"),
        ("ship", "Piraeus AIS: How real ships turn & report", "Piraeus AIS", "How real ships turn and report"),
        ("noise", f"{inazu_name}: Noise levels in ship drift", inazu_name, "Noise levels in ship drift"),
    ]
    # Each item gets a ~2.1s window:
    #   [t0, t0 + 0.35]: centre callout fades in at map centre (large text + icon centred below)
    #   [t0 + 0.35, t0 + 1.05]: holds in map centre while layer starts appearing
    #   [t0 + 1.05, t0 + 1.65]: callout moves smoothly from map centre down to its summary table slot
    T_ITEMS = [0.4 + i * 2.1 for i in range(len(real_items))]
    T_LAYER = dict(zip(("depth", "traffic", "faults", "gauges"), [t + 0.55 for t in T_ITEMS[:4]]))

    # ---------------- 2-column summary table below the map ----------------
    COL_W, ROW_H = 0.445, 0.056
    COL_X = [0.04, 0.515]
    ROW_Y = [0.185, 0.120, 0.055]
    real_arts = []
    slot_centres = []
    for i, (kind, banner, src, gives) in enumerate(real_items):
        col_idx = i // 3
        row_idx = i % 3
        x0 = COL_X[col_idx]
        y0 = ROW_Y[row_idx]
        slot_centres.append((x0 + COL_W / 2, y0 + ROW_H / 2))
        arts = []
        bg_p = fig.add_artist(FancyBboxPatch((x0, y0), COL_W, ROW_H - 0.006,
                                             boxstyle="round,pad=0,rounding_size=0.012",
                                             transform=fig.transFigure, facecolor=cm.PANEL,
                                             edgecolor=cm.PANEL_EDGE, lw=1.2, alpha=0.0, zorder=2))
        arts.append(bg_p)
        ia = fig.add_axes((x0 + 0.008, y0 + 0.008, 0.024, ROW_H - 0.022), zorder=3); ia.axis("off")
        ia.set_xlim(0, 1); ia.set_ylim(0, 1)
        arts += _icon(ia, kind)
        arts.append(fig.text(x0 + 0.038, y0 + (ROW_H - 0.006) / 2, f"{src}:", fontsize=cm.FS_SMALL,
                             weight="bold", color=cm.TEXT, va="center", alpha=0.0, zorder=4))
        arts.append(fig.text(x0 + 0.175, y0 + (ROW_H - 0.006) / 2, gives, fontsize=cm.FS_LABEL + 2,
                             color=cm.TEXT, va="center", alpha=0.0, zorder=4))
        real_arts.append(arts)

    # ---------------- animated centre-of-map callout card ----------------
    map_cx = map_x0 + map_w / 2
    map_cy = map_y0 + map_h / 2
    CW0, CH0 = 0.48, 0.21
    callout_ax = fig.add_axes((map_cx - CW0 / 2, map_cy - CH0 / 2, CW0, CH0), zorder=15)
    callout_ax.set_xlim(0, 1); callout_ax.set_ylim(0, 1); callout_ax.axis("off")
    callout_patch = callout_ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92,
                                                        boxstyle="round,pad=0,rounding_size=0.08",
                                                        facecolor="#0D1528", edgecolor=cm.C["ship"], lw=2.5, alpha=0.0))
    callout_txt = callout_ax.text(0.5, 0.72, "", fontsize=cm.FS_H2 + 2, weight="bold", color=cm.TEXT,
                                  ha="center", va="center", alpha=0.0)
    icon_cax = fig.add_axes((map_cx - 0.038, map_cy - 0.075, 0.076, 0.085), zorder=16)
    icon_cax.set_xlim(0, 1); icon_cax.set_ylim(0, 1); icon_cax.axis("off")
    callout_icons = {kind: _icon(icon_cax, kind) for kind, _, _, _ in real_items}
    for arts in callout_icons.values():
        for a in arts:
            a.set_alpha(0.0)

    def update(t):
        a_base = cm.ramp(t, 0.0, 0.4)
        for sp in max_.spines.values():
            sp.set_alpha(a_base)
        alphas = [a_base] + [cm.ramp(t, T_LAYER[key], T_LAYER[key] + 0.75) for key, _ in snap_im[1:]]
        for i, (key, im) in enumerate(snap_im):
            covered = any(a >= 1.0 for a in alphas[i + 1:])
            im.set_visible(alphas[i] > 0 and not covered)
            im.set_alpha(alphas[i])

        # Summary rows appear as each callout lands in its slot
        for i, (arts, t0) in enumerate(zip(real_arts, T_ITEMS)):
            a_row = cm.ramp(t, t0 + 1.25, t0 + 1.75)
            for ar in arts:
                ar.set_alpha(a_row)

        # Centre callout animation for the currently active data source
        active_idx = None
        for i, t0 in enumerate(T_ITEMS):
            if t0 <= t < t0 + 1.85:
                active_idx = i
                break
        for arts in callout_icons.values():
            for a in arts:
                a.set_alpha(0.0)
        if active_idx is None:
            callout_patch.set_alpha(0.0)
            callout_txt.set_alpha(0.0)
        else:
            t0 = T_ITEMS[active_idx]
            kind, banner, _, _ = real_items[active_idx]
            a_in = cm.ramp(t, t0, t0 + 0.3)
            m_prog = cm.ramp(t, t0 + 1.05, t0 + 1.65)
            a_out = 1.0 - cm.ramp(t, t0 + 1.45, t0 + 1.80)
            alpha_c = a_in * a_out
            tx, ty = slot_centres[active_idx]
            cx = cm.lerp(map_cx, tx, m_prog)
            cy = cm.lerp(map_cy, ty, m_prog)
            sc = cm.lerp(1.0, 0.55, m_prog)
            cw, ch = CW0 * sc, CH0 * sc
            callout_ax.set_position((cx - cw / 2, cy - ch / 2, cw, ch))
            iw, ih = 0.076 * sc, 0.085 * sc
            icon_cax.set_position((cx - iw / 2, cy - ch * 0.36, iw, ih))
            callout_txt.set_text(banner)
            callout_txt.set_fontsize((cm.FS_H2 + 2) * sc)
            callout_patch.set_alpha(0.92 * alpha_c)
            callout_txt.set_alpha(alpha_c)
            for a in callout_icons[kind]:
                a.set_alpha(alpha_c)

    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
