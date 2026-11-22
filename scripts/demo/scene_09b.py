"""Scene 09b / 13b (Results, placed after Scene 13): Precise GPS confirms tsunamis in minutes,
beating coastal tide gauges.

Merges the contents and title takeaway of the former Scene 9b (showcase Mw 8.6 GPS confirmation in
1 minute, 100% of simulated days, before even a noise-free tide gauge) with the "Minutes gained or
lost against tide gauges and buoys, by coast" map (Map 3 from outputs/map3_table.csv), and explicitly
notes that unlike live AIS broadcasts, precise vertical GPS data is historical/logged today rather
than live.
"""
import common as cm
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import showcase_trials as st
from matplotlib.cm import ScalarMappable

SCENE = 16
BEFORE_GAUGE_FALLBACK = 1.0


def before_gauge_frac():
    """Share of Hellenic Mw 8.6 trials (truth_map2, network detector, GPS, calm) confirmed before the first tide gauge."""
    try:
        import glob
        root = cm.Path.home() / "projects" / "med-tsunami-ais"
        parts = glob.glob(str(root / "outputs" / "sweep" / "part_*_40000_gps_calm.parquet"))
        d = pd.concat(pd.read_parquet(x, columns=["run_id", "run_kind", "truth_mw", "detector", "detected", "confirm_s",
                                                  "first_gauge_s"]) for x in parts)
        reg = pd.read_csv(root / "data" / "truth" / "sources.csv").set_index("run_id")["region"]
        d = d[(d.run_kind == "truth_map2") & (d.detector == "network") & np.isclose(d.truth_mw, 8.6)]
        d = d[d.run_id.map(reg) == "hellenic"]
        return float((d.detected & (d.confirm_s < d.first_gauge_s)).mean())
    except Exception as e:  # noqa: BLE001
        print(f"scene 09b NOTE: before-gauge share from fallback ({e})")
        return BEFORE_GAUGE_FALLBACK


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    s9 = cm.load_scene_module(9)
    s13 = cm.load_scene_module(13)
    D = st.load("showcase86")
    meta, tr = D["meta"], D["traffic"]
    fig, chrome = cm.new_figure(SCENE, caption="Ships above the source rise as soon as the seafloor moves, "
                                               "so precise GPS heights confirm tsunamis ahead of tide gauges.")
    for k in chrome.values():
        k.set_zorder(20)
    if meta is None or tr is None:
        cm.placeholder(fig, "GPS network evidence and minutes gained vs tide gauges",
                       needs="showcase86_meta.json + traffic.npz + map3_table.csv",
                       rect=(0.1, 0.15, 0.8, 0.65))
        return cm.Scene(fig, lambda t: None, still_t=dur - 1)

    gps, typ = meta["gps"], meta["typical"]
    noise_m = float(ctx.cfg["gps"]["noise_sd_m"])
    # source of the GPS noise level (config.yaml gps.noise_sd_m): "Foster and Thomas (2025)" -> "Foster and Thomas, 2025"
    gps_cite = cm.load_cards().get("foster_thomas", {}).get("cite", "Foster and Thomas (2025)").replace(" (", ", ").rstrip(")")
    T_CURVE0, T_MID, T_CURVE1 = 0.8, 3.8, 5.8
    M_MID = 5.0
    T_LBL = [0.3, 3.2, 4.2, 5.5]

    def minute_of(t):
        if t < T_MID:
            return M_MID * float(np.clip((t - T_CURVE0) / (T_MID - T_CURVE0), 0, 1))
        return M_MID + (120.0 - M_MID) * cm.ramp(t, T_MID, T_CURVE1)

    # ------------------------------------------------------------------ left: GPS evidence curve & showcase stats
    head_l = fig.text(0.025, 0.825, "Hellenic Mw 8.6: GPS fleet evidence vs alarm level",
                      fontsize=cm.FS_SMALL + 2, color=cm.TEXT, weight="bold", ha="left", va="center")
    rel = np.asarray(tr["stat_gps"], float) / float(tr["thr_gps"])
    big = np.nanmax(rel) > 8
    axs, rel, line, dot, alarm, ymax = s9.evidence_axes(fig, (0.035, 0.50, 0.41, 0.28), tr["stat_gps"],
                                                        tr["thr_gps"], cm.C["gps"],
                                                        ymax=float(np.nanmax(rel)) * 2.0 if big else None)
    if big:
        lo = min(0.1, float(np.nanmin(rel[rel > 0])) if (rel > 0).any() else 0.1)
        axs.set_yscale("log"); axs.set_ylim(lo, ymax); axs.set_yticks([]); axs.minorticks_off()
        rel = np.clip(rel, lo, None)
    if gps.get("detected") and gps.get("confirm_s") is not None:
        c_min = float(gps["confirm_s"]) / 60.0
        cross_xy = (c_min, 1.0)
    else:
        c_min, cross_xy = None, None
    alarm_lbl = axs.text(8, 1.0 * (1.25 if big else 1.0) + (0 if big else 0.025 * ymax), "1% false-alarm threshold",
                         color=cm.C["alarm"], fontsize=cm.FS_SMALL, weight="bold", ha="left", va="bottom", zorder=7)
    cross = axs.scatter([], [], s=320, facecolors="none", edgecolors=cm.C["alarm"], lw=3, zorder=6, clip_on=False)

    det = fig.text(0.025, 0.39, f"Confirmed on {st.pct(typ['gps_det_frac'])} of simulated days",
                   fontsize=cm.FS_BODY + 1, color=cm.C["gps"], weight="bold", ha="left", va="center")
    bg = before_gauge_frac()
    first = fig.text(0.025, 0.315, f"Before the first tide gauge: {st.pct(bg)} of days",
                     fontsize=cm.FS_BODY + 1, color=cm.C["gps"], weight="bold", ha="left", va="center")
    gm = float(typ["gps_med_confirm_min"])
    unit = "minute" if round(gm) == 1 else "minutes"
    tmed = fig.text(0.025, 0.24, f"Typical time to confirm: {st.fmt_min(gm)} {unit}",
                    fontsize=cm.FS_BODY + 1, color=cm.C["gps"], weight="bold", ha="left", va="center")

    # ------------------------------------------------------------------ right: Map 3 (Minutes gained by coast)
    x_r = 0.48
    head_r = fig.text(x_r, 0.825, "Minutes gained against tide gauges and buoys, by coast (precise GPS)",
                      fontsize=cm.FS_SMALL + 1, color=cm.TEXT, weight="bold", ha="left", va="center")
    ext = cm.aoi_extent(ctx.cfg)
    ax3 = cm.map_axes(fig, (x_r, 0.26, 0.495, 0.52), extent=ext)
    pool = ctx.pool()
    d_hi = float(ctx.cfg["domain"]["depth_band_edges_m"][1])
    sh = ((pool.rtype == 0) if hasattr(pool, "rtype") else np.ones(len(pool.lon), bool)) & (pool.depth < d_hi)
    sh_sc = ax3.scatter(pool.lon[sh], pool.lat[sh], s=1.2, color=cm.C["ship"], lw=0, zorder=1, alpha=0.28)

    m3 = s13.load_map3(ctx.cfg)
    gnorm = mcolors.Normalize(-s13.GAIN_LIM, s13.GAIN_LIM)
    m3_sc = None
    cb_arts = []
    if m3 is not None:
        coasts = m3.groupby("gauge_code", as_index=False).agg(lon_deg=("lon_deg", "first"), lat_deg=("lat_deg", "first"))
        s_gps = m3[(m3.operator == "gps") & (m3.regime == "calm")].set_index("gauge_code")[s13.MAP3_COL]
        g_val = coasts.gauge_code.map(s_gps).to_numpy(float)
        ok = np.isfinite(g_val)
        fc = np.zeros((len(g_val), 4)); fc[ok] = s13.GAIN_CMAP(gnorm(np.clip(g_val[ok], -s13.GAIN_LIM, s13.GAIN_LIM)))
        ec = np.tile(mcolors.to_rgba(cm.BG), (len(g_val), 1)); ec[~ok] = s13.NONE_EDGE
        m3_sc = ax3.scatter(coasts.lon_deg, coasts.lat_deg, s=155, facecolors=fc, edgecolors=ec, lw=1.2, zorder=6)
        cax3 = fig.add_axes((x_r + 0.115, 0.215, 0.21, 0.015))
        cb3 = fig.colorbar(ScalarMappable(gnorm, s13.GAIN_CMAP), cax=cax3, orientation="horizontal", extend="both",
                           extendfrac=0.06)
        ticks = np.arange(-s13.GAIN_LIM, s13.GAIN_LIM + 1e-9, s13.GAIN_LIM / 2)
        cb3.set_ticks(ticks)
        cb3.set_ticklabels([f"{v:+.0f}".replace("-", "−") if v else "0" for v in ticks])
        cb3.outline.set_visible(False)
        cax3.tick_params(labelsize=cm.FS_LABEL - 2, length=3, colors=cm.MUTED, pad=2)
        lbl3 = fig.text(x_r, 0.222, "Minutes gained:", fontsize=cm.FS_LABEL - 1, color=cm.TEXT, weight="bold", va="center")
        lbl3_pos = fig.text(0.975, 0.222, "(+ = GPS beats first gauge)", fontsize=cm.FS_LABEL - 2,
                            color=cm.C["good"], ha="right", va="center")
        cb_arts = [cax3.patch, *cax3.images, *cax3.collections, *cax3.patches, *cax3.get_xticklabels(),
                   *cax3.get_xticklines(), lbl3, lbl3_pos]

    # ------------------------------------------------------------------ bottom banner: historical vs live + assumptions
    box = cm.card(fig, (0.02, 0.085, 0.955, 0.088), edgecolor=cm.C["moderate"], lw=2.0)
    box_txt = box.text(
        0.5, 0.5,
        f"Caveat: unlike live AIS broadcasts, precise vertical GPS is logged onboard today (not broadcast live).\n"
        f"Assumes instant seafloor uplift and {100 * noise_m:.0f} cm GPS height noise ({gps_cite}).",
        fontsize=cm.FS_LABEL + 1, color=cm.TEXT, weight="bold", ha="center", va="center", linespacing=1.3
    )
    box_patch = box.patches[0]

    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50)
    veil_ax.axis("off")
    veil = veil_ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG,
                                                          alpha=1.0, lw=0))

    def update(t):
        veil.set_alpha(1.0 - cm.ramp(t, 0.0, 0.5))
        m = minute_of(t)
        head_l.set_alpha(cm.fade(t, T_LBL[0]))
        head_r.set_alpha(cm.fade(t, T_LBL[0]))
        alarm.set_alpha(cm.fade(t, 0.4)); alarm_lbl.set_alpha(cm.fade(t, 0.4))
        s9.draw_curve(line, dot, rel, m)
        if cross_xy is not None and m >= c_min:
            cross.set_offsets([cross_xy]); cross.set_alpha(cm.fade(t, T_CURVE0, dur=0.3))
        else:
            cross.set_offsets(np.empty((0, 2)))
        a_map = cm.fade(t, 0.6, dur=0.8)
        if m3_sc is not None:
            m3_sc.set_alpha(a_map)
        for a in cb_arts:
            a.set_alpha(a_map)
        det.set_alpha(cm.fade(t, T_LBL[1]))
        first.set_alpha(cm.fade(t, T_LBL[1] + 0.5))
        tmed.set_alpha(cm.fade(t, T_LBL[2]))
        a_b = cm.fade(t, T_LBL[3])
        box_txt.set_alpha(a_b); box_patch.set_alpha(a_b)

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
