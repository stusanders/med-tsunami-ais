"""Scene 15 (Conclusion): policy recommendation and project limitations. The closing frame of both cuts.

Left: A stand-alone two-step policy recommendation map showing:
  1. Run the matched-filter network on today's live AIS ship feeds now (zero hardware cost; confirms Mw 7.5+
     in the shallow Aegean shelf and locates major Mw 8.5 arc tsunamis within ~13 km).
  2. Equip regular merchant ferries and cargo ships on key fault-crossing shipping corridors (Hellenic, Cyprus,
     Calabrian, Aegean) with live geodetic vertical GPS telemetry to confirm Mw 7.0+ tsunamis in 1 to 9 minutes.

Right: The four core limitations of the project:
  1. Precise vertical GPS is historical/logged today, not broadcast live on AIS (requires ship-to-shore telemetry).
  2. AIS drift only works where ships cross shallow shelves (<200 m); deep-ocean tsunami currents (~0.02 m/s)
     are smaller than normal ship motion, and shallow coastal zones (<50 m) are excluded.
  3. Simulated tsunami waves & simplified fault geometry (OSSE): real non-trench faults (Amorgos, Samos) were
     harder to detect than our synthetic catalogue.
  4. Real Piraeus AIS ship data was 17 to 21% less noisy than our calm simulation setting, making our drift
     thresholds conservative.
"""
import re

import common as cm
import numpy as np

SCENE = 15
OPERATOR, REGIME, LEVEL = "ais", "calm", 0.5


def verdict(cfg):
    """DataFrame[pos_id, region, lon_deg, lat_deg, add_value] or None."""
    s13 = cm.load_scene_module(13)
    ev = s13.load_map2_events(cfg)
    if ev is None:
        return None
    pos = s13.positions(ev)
    floor = s13.position_floor(ev, OPERATOR, REGIME, LEVEL)
    s = ev[(ev.operator == OPERATOR) & (ev.regime == REGIME)]
    s = s.assign(floor=s.pos_id.map(floor))
    s = s[s.detected.astype(bool) & (s.mw >= s.floor)]
    first = (s.confirm_s < s.first_instrument_s).groupby(s.pos_id).mean()
    pos["add_value"] = pos.pos_id.map(first).fillna(0).to_numpy(float) >= LEVEL
    return pos


def _piraeus_ratio():
    """(min, max) of real_over_sim_calm from the markdown table in outputs/noise_check_piraeus.md."""
    p = cm.OUTPUTS / "noise_check_piraeus.md"
    if not p.exists():
        return None
    lines = [l for l in p.read_text().splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return None
    head = [h.strip() for h in lines[0].strip("|").split("|")]
    if "real_over_sim_calm" not in head:
        return None
    k = head.index("real_over_sim_calm")
    vals = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) > k and re.match(r"^[0-9.]+$", cells[k]):
            vals.append(float(cells[k]))
    return (min(vals), max(vals)) if vals else None


def _bake(rect, extent=None, depth=True, coast=True):
    import matplotlib.pyplot as plt
    tmp = plt.figure(figsize=cm.FIGSIZE, dpi=cm.DPI)
    tmp.patch.set_alpha(0)
    a = cm.map_axes(tmp, rect, extent=extent, depth=depth, coast=coast)
    a.patch.set_alpha(0)
    tmp.canvas.draw()
    pos = a.get_position()
    buf = np.asarray(tmp.canvas.buffer_rgba())
    H, W = buf.shape[:2]
    img = buf[int(round((1 - pos.y1) * H)):int(round((1 - pos.y0) * H)), int(round(pos.x0 * W)):int(round(pos.x1 * W))].copy()
    xl, yl = a.get_xlim(), a.get_ylim()
    plt.close(tmp)
    return img, pos.bounds, xl, yl, (int(round(pos.x0 * W)), int(round(pos.y0 * H)))


def _baked_map_axes(fig, rect, extent=None, layers=((True, True),)):
    ax, ims = None, []
    for k, (depth, coast) in enumerate(layers):
        img, bounds, xl, yl, (xo, yo) = _bake(rect, extent, depth=depth, coast=coast)
        if ax is None:
            ax = fig.add_axes(bounds)
            ax.patch.set_alpha(0)
        ims.append(ax.imshow(img, extent=(*xl, *yl), aspect="auto", interpolation="antialiased", zorder=3 if k else 0))
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_xticks([]); ax.set_yticks([])
    ax.set_autoscale_on(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(cm.FAINT)
    return ax, ims


def _fade_card(ax, a):
    for p in ax.patches:
        p.set_alpha(a)
    for t in ax.texts:
        t.set_alpha(a)


def _wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > n and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    return "\n".join(lines + [cur])


def _replay_clause(cfg):
    s13 = cm.load_scene_module(13)
    names, _ = s13.historic_not_detected(cfg)
    if not names:
        return ""
    return f" ({", ".join(s13.event_label(n) for n in names)} were not confirmed by drift)"


def limit_texts(cfg):
    depth = float(cfg["traffic"]["min_ship_depth_m"])
    gps_cm = 100 * float(cfg["gps"]["noise_sd_m"])
    pr = _piraeus_ratio()
    l4 = (f"Real Greek AIS data was {100 * (1 - pr[1]):.0f} to {100 * (1 - pr[0]):.0f}% less noisy than our calm "
          "model, so our drift thresholds are cautious") if pr else None
    return [
        f"Precise vertical GPS is historical today, not broadcast live on AIS (assumes {gps_cm:g} cm height noise and live ship-to-shore telemetry)",
        f"AIS drift only works on shallow shelves (50–200 m): deep-sea currents are too weak, and shallow ports (<{depth:g} m) are excluded",
        "Simulated waves & simplified faults (OSSE): we test synthetic tsunamis on real ship routes rather than live AIS during a Mw 8+ quake" + _replay_clause(cfg),
        l4,
    ]


def build(cut, ctx):
    cfg = ctx.cfg
    D = cm.duration(SCENE, cut)
    t_map, t_leg = 0.5, 1.5
    t_lims = [0.15 * D + 0.14 * D * k for k in range(4)]
    t_next, t_cap = 0.71 * D, 0.78 * D
    fig, chrome = cm.new_figure(
        SCENE, caption="Ships already at sea can be networked in software today for shallow-shelf confirmation,\n"
                       "while equipping regular ships on key fault-crossing corridors with live precise GPS creates a fast offshore warning network.")
    chrome["caption"].set_y(0.045); chrome["caption"].set_linespacing(1.35)
    chrome["caption"].set_alpha(0)

    # ---- left: Policy recommendation map (shipping corridors + fault arcs)
    pol_head = fig.text(0.018, 0.815, "Policy recommendation: where to deploy live precise GPS corridors",
                        fontsize=cm.FS_SMALL + 2, weight="bold", color=cm.TEXT, ha="left", va="center", alpha=0)
    ax, _ = _baked_map_axes(fig, (0.015, 0.25, 0.56, 0.54))
    v = verdict(cfg)
    map_arts = []

    try:
        pool = ctx.pool()
        sh = pool.rtype == 0
        w = pool.weight[sh].astype(float)
        hi = w >= np.percentile(w, 65)
        map_arts.append(ax.scatter(pool.lon[sh][hi], pool.lat[sh][hi], s=3.5, color=cm.C["ship"],
                                   lw=0, zorder=3, alpha=0))
    except Exception as e:  # noqa: BLE001
        print("scene 15: pool unavailable:", e)

    if v is not None:
        arcs = v[v.region != "aegean"]
        aeg = v[v.region == "aegean"]
        for reg, df in arcs.groupby("region", sort=False):
            df_s = df.sort_values("pos_id")
            ln, = ax.plot(df_s.lon_deg, df_s.lat_deg, color=cm.C["gps"], lw=4.5, zorder=4.5, alpha=0,
                          solid_capstyle="round")
            pts = ax.scatter(df_s.lon_deg, df_s.lat_deg, s=55, color=cm.C["gps"], edgecolors=cm.BG,
                             linewidths=0.8, zorder=5, alpha=0)
            map_arts.extend([ln, pts])
        g_aeg = aeg[aeg.add_value]
        o_aeg = aeg[~aeg.add_value]
        map_arts.append(ax.scatter(o_aeg.lon_deg, o_aeg.lat_deg, s=48, color=cm.C["network"], edgecolors=cm.BG,
                                   linewidths=0.7, zorder=4.8, alpha=0))
        map_arts.append(ax.scatter(g_aeg.lon_deg, g_aeg.lat_deg, s=130, color=cm.C["good"], edgecolors=cm.BG,
                                   linewidths=1.2, zorder=5.5, alpha=0))
    else:
        ph_ax = fig.add_axes((0.075, 0.38, 0.45, 0.27))
        ph_ax.patch.set_visible(False)
        map_arts += cm.placeholder(ph_ax, "Policy map: Live AIS today + Precise GPS on key fault-crossing routes")

    # Two-step policy cards under the map
    x0m = ax.get_position().x0
    wm = ax.get_position().width
    leg = cm.card(fig, (x0m, 0.095, wm, 0.142), edgecolor=cm.PANEL_EDGE, lw=1.5)
    leg.scatter([0.03], [0.77], s=130, color=cm.C["good"], edgecolors=cm.BG, transform=leg.transAxes, clip_on=False)
    leg.text(0.06, 0.77, "Step 1 (Now, software only): Run matched filter on live AIS",
             fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.C["good"], va="center")
    leg.text(0.06, 0.57, "Confirms Mw 7.5+ on the Aegean shelf (beats gauges at green dots) & locates arc quakes",
             fontsize=cm.FS_LABEL - 1, color=cm.MUTED, va="center")
    leg.scatter([0.03], [0.32], s=130, color=cm.C["gps"], edgecolors=cm.BG, transform=leg.transAxes, clip_on=False)
    leg.text(0.06, 0.32, "Step 2 (Hardware): Equip corridor ships with live precise GPS",
             fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.C["gps"], va="center")
    leg.text(0.06, 0.12, "Ferries & cargo ships crossing purple fault arcs confirm Mw 7.0+ tsunamis in 1–9 min",
             fontsize=cm.FS_LABEL - 1, color=cm.MUTED, va="center")

    # ---- right: key project limitations and next step
    x0, w = 0.60, 0.375
    lim_head = fig.text(x0, 0.815, "Key project limitations", fontsize=cm.FS_H2 - 2, weight="bold",
                        color=cm.TEXT, ha="left", va="center", alpha=0)
    lim_cards = []
    line_h, pad, gap = 0.0235, 0.022, 0.008
    y_top = 0.78
    for k, txt in enumerate(limit_texts(cfg)):
        body = "result to follow" if txt is None else _wrap(txt, 56)
        h = line_h * (body.count("\n") + 1) + pad
        c = cm.card(fig, (x0, y_top - h, w, h))
        c.text(0.055, 0.5, f"{k + 1}", fontsize=cm.FS_H2 - 2, weight="bold", color=cm.MUTED, ha="center", va="center")
        c.text(0.11, 0.5, body, fontsize=cm.FS_LABEL, color=cm.MUTED if txt is None else cm.TEXT, va="center",
               linespacing=1.20)
        lim_cards.append(c)
        y_top -= h + gap
    nxt_h = max(0.105, min(0.135, y_top - 0.095 - 0.004))
    nxt = cm.card(fig, (x0, 0.095, w, nxt_h), edgecolor=cm.C["gps"], lw=2.0)
    nxt.text(0.04, 0.80, "Next step", fontsize=cm.FS_SMALL, weight="bold", color=cm.C["gps"], va="center")
    nxt.text(0.04, 0.34, _wrap("Test on live AIS feeds during future quakes, and pilot live geodetic GPS on ferries crossing the Hellenic Arc", 56),
             fontsize=cm.FS_LABEL, color=cm.TEXT, va="center", linespacing=1.22)

    for c in lim_cards + [nxt, leg]:
        _fade_card(c, 0)
    for a in leg.collections:
        a.set_alpha(0)

    def update(t):
        a_m = cm.fade(t, t_map, dur=1.0)
        pol_head.set_alpha(a_m)
        for i, a in enumerate(map_arts):
            a.set_alpha((0.42 if i == 0 else 0.95) * a_m)
        a_l = cm.fade(t, t_leg, dur=0.8)
        _fade_card(leg, a_l)
        for a in leg.collections:
            a.set_alpha(a_l)
        lim_head.set_alpha(cm.fade(t, t_lims[0] - 0.4, dur=0.6))
        for k, c in enumerate(lim_cards):
            _fade_card(c, cm.fade(t, t_lims[k], dur=0.7))
        _fade_card(nxt, cm.fade(t, t_next, dur=0.8))
        chrome["caption"].set_alpha(cm.fade(t, t_cap, dur=0.8))

    return cm.Scene(fig, update, still_t=D - 0.5)


if __name__ == "__main__":
    cm.main(__file__)
