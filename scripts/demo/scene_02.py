"""Scene 2 (Problem): how tsunamis are currently confirmed.

On screen (docs/DEMO_SCRIPT.md scene 2): the map labels, the Palermo panel and the caption, nothing else.
Three beats (the 3-minute cut drops beat A):
  A. history: AD 365 Crete, 1908 Messina, 1956 Amorgos, 2020 Samos appear on the map (cards.yaml).
  B. today's instruments: coastal tide gauges, the 2025 deep-sea buoys, the 2025 Stromboli coastal buoy,
     and the empty open sea, each labelled on the map.
  C. 23 Oct 2026, southern Tyrrhenian: the map slides left and zooms to Sicily; the Palermo panel lists
     the alert, the evacuations and the cancellation, times from cards.yaml.

Built from: grid + coastline; gauges.load_gauges(grid, cfg) (IOC gauges + data/raw/historic/
ionian_buoys_2025.json buoys; the buoy year is read from its deployed_date); data/raw/historic/
{amorgos_1956,samos_2020}_sources.json (positions); scripts/demo/cards.yaml (labels, positions, times,
the Stromboli buoy: Costanza et al., EGU26-21354).
"""
import textwrap

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import numpy as np

import common as cm

SCENE = 2

# beat start times (s) per cut
TIMING = {"full": dict(A=0.0, B=7.0, C=14.0), "3min": dict(A=None, B=0.0, C=7.0)}
ZOOM = (12.0, 18.6, 36.42, 39.588)   # Sicily / Calabria / Ionian, same lon:lat ratio as the AOI
MAP_W, MAP_H, MAP_Y = 0.71, 0.80, 0.075
MAP_X_CENTRE, MAP_X_LEFT = (1 - MAP_W) / 2, 0.01
HIST_OFFSET = {"hist_crete_365": (0.35, -0.75, "left"), "hist_messina_1908": (0.35, 0.3, "left"),
               "hist_amorgos_1956": (0.35, 0.25, "left"), "hist_samos_2020": (0.35, 0.25, "left")}  # deg, layout


def _hist_position(card):
    """Marker position: the project's historic data file when the card names one, else the card."""
    pf = card.get("position_file")
    if pf:
        d = cm.load_json(pf)
        if d and "recommended_for_replay" in d:
            r = d["recommended_for_replay"]
            return float(r["lon_deg"]), float(r["lat_deg"])
    return float(card["lon"]), float(card["lat"])


def _buoy_year(card):
    """Deployment year of the deep-sea buoys: earliest deployed_date in the card's data file."""
    d = cm.load_json(card.get("data_file", "")) if card.get("data_file") else None
    years = [int(b["deployed_date"][:4]) for b in (d or {}).get("buoys", []) if b.get("deployed_date")]
    return min(years) if years else card.get("year")


def build(cut, ctx):
    T = TIMING[cut]
    tB, tC = T["B"], T["C"]
    cards = cm.load_cards()
    fig, chrome = cm.new_figure(SCENE, caption="Warnings are inferred from the earthquake and then confirmed at "
                                               "the coast, as there are very few instruments at sea.")
    chrome["caption"].set_position((0.5, 0.04))

    ext0 = cm.aoi_extent(ctx.cfg)
    ax = cm.map_axes(fig, (MAP_X_CENTRE, MAP_Y, MAP_W, MAP_H))
    shallow_rgb = np.array(cm.DEPTH_CMAP(0.12)[:3]); bg_rgb = np.array(mcolors.to_rgb(cm.BG))
    stroke = [pe.withStroke(linewidth=4, foreground=cm.BG)]

    def label(x, y, s, color, ha="left", va="center", size=cm.FS_SMALL, weight="bold", **kw):
        return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=12, alpha=0,
                       path_effects=stroke, clip_on=True, **kw)

    # ---------------- instruments ----------------
    try:
        from tsunami import gauges as gauges_mod
        gt = gauges_mod.load_gauges(ctx.grid, ctx.cfg)
        g_lon, g_lat = gt["lon_deg"].to_numpy(float), gt["lat_deg"].to_numpy(float)
        g_type = gt["rtype"].to_numpy(int)
    except Exception as e:  # noqa: BLE001  (fall back to the gauge/buoy part of the receiver pool)
        print("scene 02: load_gauges failed, using pool:", e)
        pool = ctx.pool()
        sel = pool.rtype > 0
        g_lon, g_lat, g_type = pool.lon[sel], pool.lat[sel], pool.rtype[sel].astype(int)
    is_g, is_b = g_type == 1, g_type == 2
    gauge_sc = ax.scatter(g_lon[is_g], g_lat[is_g], marker="^", s=70, color=cm.C["gauge"], edgecolors="none",
                          zorder=6, alpha=0)
    buoy_sc = ax.scatter(g_lon[is_b], g_lat[is_b], marker="*", s=420, color=cm.C["buoy"], edgecolors="none",
                         zorder=7, alpha=0)
    inst = []   # (t_on, [artists])
    t_g, t_b, t_s, t_o = tB + 0.3, tB + 1.6, tB + 2.9, tB + 4.2
    inst.append((t_g, [gauge_sc, label(27.6, 39.9, "Coastal tide gauges", cm.C["gauge"])]))

    bc = cards.get("buoys_2025")
    if bc and is_b.any():
        yr = _buoy_year(bc)
        txt = bc.get("label", "").format(year=yr)
        bx, by = float(g_lon[is_b].min()), float(g_lat[is_b].min())   # below the southern buoy, open Ionian
        inst.append((t_b, [buoy_sc, label(bx - 1.1, by - 0.45, txt.replace(" (", "\n("), cm.C["buoy"], va="top",
                                          linespacing=1.2)]))
    else:
        inst.append((t_b, [buoy_sc]))

    sc = cards.get("stromboli_buoy_2025")
    if sc:
        s_dot = ax.scatter([sc["lon"]], [sc["lat"]], marker="o", s=140, color=cm.C["buoy"], edgecolors=cm.TEXT,
                           lw=1.5, zorder=8, alpha=0)
        s_lbl = label(sc["lon"] - 2.85, sc["lat"] + 0.65, sc["label"].format(year=sc["year"]), cm.C["buoy"])
        inst.append((t_s, [s_dot, s_lbl]))

    open_lbl = label(19.6, 33.6, "Open sea: no instruments", cm.MUTED, ha="center", size=cm.FS_BODY,
                     weight="normal", style="italic")
    inst.append((t_o, [open_lbl]))

    # Legend box explaining the two locator types (coastal tide gauges and tsunami buoys)
    leg_card = cm.card(fig, (MAP_X_CENTRE + MAP_W - 0.255, MAP_Y + 0.02, 0.24, 0.125), edgecolor=cm.PANEL_EDGE, lw=1.5)
    leg_card.set_zorder(15)
    leg_g_sc = leg_card.scatter([0.09], [0.70], marker="^", s=95, color=cm.C["gauge"], edgecolors="none",
                                transform=leg_card.transAxes, zorder=16)
    leg_g_tx = leg_card.text(0.18, 0.70, f"Tidal gauges ({int(is_g.sum())} coastal)", fontsize=cm.FS_LABEL + 1,
                             weight="bold", color=cm.C["gauge"], va="center", zorder=16)
    leg_b_sc = leg_card.scatter([0.08], [0.30], marker="*", s=240, color=cm.C["buoy"], edgecolors="none",
                                transform=leg_card.transAxes, zorder=16)
    leg_b_sc2 = leg_card.scatter([0.125], [0.30], marker="o", s=75, color=cm.C["buoy"], edgecolors=cm.TEXT, lw=1.0,
                                 transform=leg_card.transAxes, zorder=16)
    leg_b_tx = leg_card.text(0.18, 0.30, "Buoys (deep-sea & coastal)", fontsize=cm.FS_LABEL + 1,
                             weight="bold", color=cm.C["buoy"], va="center", zorder=16)
    leg_arts = [leg_card.patches[0], leg_g_sc, leg_g_tx, leg_b_sc, leg_b_sc2, leg_b_tx]
    for a in leg_arts:
        a.set_alpha(0)
    inst_labels = [a for _, arts in inst for a in arts if hasattr(a, "get_text")]

    # ---------------- beat A: historic tsunamis ----------------
    hist = []
    hist_card_arts = []
    if T["A"] is not None:
        for k, key in enumerate(["hist_crete_365", "hist_messina_1908", "hist_amorgos_1956", "hist_samos_2020"]):
            c = cards.get(key)
            if c is None:
                continue
            lon, lat = _hist_position(c)
            dx, dy, ha = HIST_OFFSET[key]
            t_on = T["A"] + 0.7 + 1.3 * k
            dot = ax.scatter([lon], [lat], s=170, color=cm.C["source"], edgecolors=cm.BG, lw=1, zorder=9, alpha=0)
            ring = ax.scatter([lon], [lat], s=[0], facecolors="none", edgecolors=cm.C["source"], lw=2.5, zorder=9)
            lab = label(lon + dx, lat + dy, f"{c['label']}, {c['place']}", cm.C["source"], ha=ha, size=cm.FS_BODY)
            hist.append(dict(t_on=t_on, dot=dot, ring=ring, lab=lab))
        rec = cards.get("med_tsunami_record")
        if rec:
            hcard = cm.card(fig, (MAP_X_CENTRE + MAP_W - 0.325, MAP_Y + 0.02, 0.31, 0.155),
                            edgecolor=cm.C["source"], lw=1.6)
            hcard.set_zorder(15)
            h1 = hcard.text(0.05, 0.79, f"{rec['rank']} tsunami record globally",
                            fontsize=cm.FS_LABEL + 1, weight="bold", color=cm.C["source"], va="center", zorder=16)
            h2 = hcard.text(0.05, 0.54, rec["count_label"],
                            fontsize=cm.FS_LABEL - 1, color=cm.TEXT, va="center", zorder=16)
            h3 = hcard.text(0.05, 0.30, rec["rate"],
                            fontsize=cm.FS_LABEL, weight="bold", color=cm.C["signal"], va="center", zorder=16)
            h4 = hcard.text(0.05, 0.10, "Source: INGV EMTC 2.0",
                            fontsize=cm.FS_LABEL - 3, color=cm.MUTED, va="center", zorder=16)
            hist_card_arts = [hcard.patches[0], h1, h2, h3, h4]
            for a in hist_card_arts:
                a.set_alpha(0)
    hist_out = tB - 0.6 if T["A"] is not None else None

    # ---------------- beat C: 23 Oct 2026, Palermo panel ----------------
    pq = cards.get("palermo_quake")
    e_star = e_ring = None
    if pq:
        e_star = ax.scatter([pq["lon"]], [pq["lat"]], marker="*", s=650, color=cm.C["alarm"], edgecolors=cm.TEXT,
                            lw=1.2, zorder=10, alpha=0)
        e_ring = ax.scatter([pq["lon"]], [pq["lat"]], s=[0], facecolors="none", edgecolors=cm.C["alarm"], lw=3,
                            zorder=9)
    rx = 0.745
    c_arts, c_rows = [], []
    keys = ("palermo_quake", "palermo_alert", "palermo_evac", "palermo_lifted")
    if all(k in cards for k in keys):
        head = pq["heading"].format(date=pq["date"], mw=float(pq["mw"]), place=pq["place"])
        c_arts.append(fig.text(rx, 0.80, head.replace(", ", ",\n", 1), fontsize=cm.FS_H2, weight="bold",
                               color=cm.TEXT, va="top", alpha=0, linespacing=1.25))
        spec = [("palermo_alert", 0.62, cm.C["alarm"]), ("palermo_evac", 0.47, cm.TEXT),
                ("palermo_lifted", 0.32, cm.C["gauge"])]
        spine = fig.add_artist(mlines.Line2D([rx + 0.0055] * 2, [0.615, 0.27],
                                                                     color=cm.FAINT, lw=2, alpha=0))
        c_arts.append(spine)
        for k, (key, y, col) in enumerate(spec):
            c = cards[key]
            dot = fig.text(rx, y + 0.002, "●", fontsize=cm.FS_SMALL, color=col, va="top", alpha=0)
            arts = [dot]
            x_txt = rx + 0.025
            if c.get("time"):
                arts.append(fig.text(x_txt, y, c["time"], fontsize=cm.FS_BODY, weight="bold", color=col, va="top",
                                     alpha=0))
                lab_y = y - 0.045
            else:
                lab_y = y
            arts.append(fig.text(x_txt, lab_y, "\n".join(textwrap.wrap(c["label"], 24)),
                                 fontsize=cm.FS_SMALL, color=cm.TEXT, va="top", alpha=0, linespacing=1.25))
            c_rows.append(dict(t_on=tC + 1.8 + 1.0 * k, arts=arts))
    else:
        c_arts += cm.placeholder(fig, "Palermo / Trapani 23 Oct 2026 timeline", needs="scripts/demo/cards.yaml",
                                 rect=(0.735, 0.2, 0.25, 0.6))

    # gentle fade-in from the background colour (scenes are joined back to back)
    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50); veil_ax.axis("off")
    veil = veil_ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG, alpha=1.0))

    def set_alpha(arts, a):
        for x in arts:
            x.set_alpha(a)

    BEAT_TITLES = ["Mediterranean: 3rd-largest tsunami record (~1 major event/century)",
                   "Where current instruments are",
                   "Warnings are inferred from the earthquake, then confirmed at the coast"]
    beat_fs = [cm.set_title(fig, chrome, s).get_fontsize() for s in BEAT_TITLES]

    def update(t):
        veil.set_alpha(1 - cm.ramp(t, 0.0, 0.7))
        chrome["caption"].set_alpha(cm.fade(t, 0.5, dur=0.8))
        beat = 0 if (T["A"] is not None and t < tB - 0.15) else (1 if t < tC - 0.15 else 2)
        chrome["title"].set_text(BEAT_TITLES[beat])
        chrome["title"].set_fontsize(beat_fs[beat])
        a_leg = cm.fade(t, t_g, dur=0.5) * (1 - cm.ramp(t, tC, tC + 0.5))
        set_alpha(leg_arts, a_leg)

        # beat A
        if hist_card_arts:
            set_alpha(hist_card_arts, cm.fade(t, T["A"] + 0.4, hist_out, dur=0.5))
        for h in hist:
            a = cm.fade(t, h["t_on"], hist_out, dur=0.5)
            h["dot"].set_alpha(a); h["lab"].set_alpha(a)
            x = (t - h["t_on"]) / 1.2
            h["ring"].set_sizes([150 + 2200 * x if 0 <= x <= 1 else 0])
            h["ring"].set_alpha(max(0.0, 1 - x) if 0 <= x <= 1 else 0)

        # beat B: instruments stay on the map; their labels give way to the Palermo panel in beat C
        for t_on, arts in inst:
            a = cm.fade(t, t_on, dur=0.5)
            for x in arts:
                x.set_alpha(a * (1 - cm.ramp(t, tC, tC + 0.5)) if x in inst_labels else a)

        # beat C: slide left, zoom to Sicily, quake, panel
        s = cm.ramp(t, tC, tC + 1.3)
        ax.set_position((cm.lerp(MAP_X_CENTRE, MAP_X_LEFT, s), MAP_Y, MAP_W, MAP_H))
        ax.set_facecolor(tuple(cm.lerp(bg_rgb, shallow_rgb, s)))
        ax.set_xlim(cm.lerp(ext0[0], ZOOM[0], s), cm.lerp(ext0[1], ZOOM[1], s))
        ax.set_ylim(cm.lerp(ext0[2], ZOOM[2], s), cm.lerp(ext0[3], ZOOM[3], s))
        if e_star is not None:
            e_star.set_alpha(cm.fade(t, tC + 1.2, dur=0.3))
            x = (t - tC - 1.2) / 1.4
            e_ring.set_sizes([300 + 6000 * x if 0 <= x <= 1 else 0])
            e_ring.set_alpha(max(0.0, 1 - x) if 0 <= x <= 1 else 0)
        set_alpha(c_arts, cm.fade(t, tC + 1.0, dur=0.5))
        for r in c_rows:
            set_alpha(r["arts"], cm.fade(t, r["t_on"], dur=0.5))

    # still: the Palermo panel fully built
    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 0.6)


if __name__ == "__main__":
    cm.main(__file__)
