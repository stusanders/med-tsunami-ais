"""Scene 9c (Simulation): however, this is not always the case.

Built from T17 sweep trial Q (showcase_trials.load("showcase80")):
  meta.trial.mw/lon/lat, n_shelf_ships  heading "Mw X, a day with no ships on the Cretan shelf", true source
  meta.ais.best_lon/best_lat, loc_err_km   the network's best estimate at the end of the window (drawn only if it
                                         lies in the source area, i.e. within LOC_OK_KM of the true source)
  meta.ais.detected                      must be False for "Alarm level not reached within 2 hours"
  traffic.lon/lat, stat_ais / thr_ais    ships; evidence relative to the alarm level
Shelf shading: model-grid cells matching meta.trial.shelf.definition (depth < X m inside the Crete box).
"""
import common as cm
import matplotlib
import numpy as np
import showcase_trials as st

SCENE = 17
LOC_OK_KM = 60.0          # "points to the correct area": estimate within this distance of the true source
SHELF_M_DEFAULT = 200.0


def shelf_def(trial):
    """(max depth m, (lon0, lon1, lat0, lat1) or None) parsed from meta.trial.shelf.definition."""
    import re
    sd = trial.get("shelf") or {}
    txt = sd.get("definition", "") if isinstance(sd, dict) else str(sd)
    md = re.search(r"depth\s*<\s*([\d.]+)\s*m", txt)
    mb = re.search(r"lon\s*\[([-\d.]+),\s*([-\d.]+)\]\s*x\s*lat\s*\[([-\d.]+),\s*([-\d.]+)\]", txt)
    return (float(md.group(1)) if md else SHELF_M_DEFAULT), (tuple(float(x) for x in mb.groups()) if mb else None)


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    s9 = cm.load_scene_module(9)
    D = st.load("showcase80")
    meta, tr = D["meta"], D["traffic"]
    fig, chrome = cm.new_figure(SCENE, caption="Ship drift is only strong where the water is shallow (<200 m). With 0 ships "
                                               "on the Cretan shelf, deep-water ships (>800 m) never reach the alarm level.")
    if meta is None or tr is None:
        cm.placeholder(fig, "Trial with no ships on the Cretan shelf", needs="showcase80_meta.json + traffic.npz",
                       rect=(0.1, 0.15, 0.8, 0.65))
        return cm.Scene(fig, lambda t: None, still_t=dur - 1)

    trial, ais = meta["trial"], meta["ais"]
    if ais.get("detected"):
        print("scene 09c WARNING: trial Q is marked detected; 'Alarm level not reached within 2 hours' is not true")
    if trial.get("n_shelf_ships"):
        print(f"scene 09c WARNING: trial Q has n_shelf_ships = {trial['n_shelf_ships']}")
    T_CURVE0, T_CURVE1 = 1.0, 7.0
    T_EST, T_VERDICT = T_CURVE1 - 0.3, T_CURVE1 + 0.2

    # ------------------------------------------------------------------ left: map
    sx, sy = trial["lon"], trial["lat"]
    rect = (0.02, 0.12, 0.47, 0.67)
    hh = 2.1
    hw = hh * rect[2] * cm.W_PX / (rect[3] * cm.H_PX) / np.cos(np.deg2rad(sy))
    cx, cy = sx, sy + 0.6                              # keep Crete and its shelf in view above the source
    ext = (cx - hw, cx + hw, cy - hh, cy + hh)
    ax = cm.map_axes(fig, rect, extent=ext, frame=True)
    head = fig.text(0.02, 0.83, f"Mw {trial['mw']:.1f}: drift is only strong over shallow water (<200 m), and 0 ships were there",
                    fontsize=cm.FS_H2, color=cm.TEXT, weight="bold", ha="left", va="center")
    g = ctx.grid
    dsh, box = shelf_def(trial)
    dep = np.where(g.wet, g.depth, np.nan)
    if box is not None:                                # only the Cretan shelf (as in the trial's definition)
        LO, LA = np.meshgrid(g.lon, g.lat)
        dep = np.where((LO >= box[0]) & (LO <= box[1]) & (LA >= box[2]) & (LA <= box[3]), dep, np.nan)
    if "shelf" in tr and np.any(tr["shelf"]):
        print(f"scene 09c WARNING: {int(np.sum(tr['shelf']))} ships flagged on the shelf in the traffic file")
    SHELF_C = cm.C["moderate"]
    shelf_cs = ax.contourf(g.lon, g.lat, dep, levels=[0.0, dsh], colors=[SHELF_C], zorder=0.6, antialiased=True)
    shelf_ln = ax.contour(g.lon, g.lat, np.nan_to_num(dep, nan=1e4), levels=[dsh], colors=[SHELF_C],
                          linewidths=1.6, zorder=0.7)
    ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); ax.set_aspect(1 / np.cos(np.deg2rad(cy)))
    # point the label at the western tip of the Cretan shelf (lat > 34.95 excludes Gavdos); text sits in open water to the west
    LOg, LAg = np.meshgrid(g.lon, g.lat)
    on = (np.isfinite(dep) & (dep < dsh) & (LOg > ext[0]) & (LOg < ext[1]) & (LAg > max(ext[2], 34.95))
          & (LAg < ext[3]))
    if on.any():
        cand = np.flatnonzero(on.ravel())
        k = cand[np.argmin(LOg.ravel()[cand])]
        px, py = float(LOg.ravel()[k]), float(LAg.ravel()[k])
    else:
        px, py = cx, cy + 0.9
    shelf_lbl = ax.annotate(f"Cretan shelf (<{dsh:.0f} m):\n0 ships today\n(where drift reaches ~25 cm/s)", xy=(px, py),
                            xytext=(ext[0] + 0.08 * hw, py - 0.05 * hh), color=SHELF_C, fontsize=cm.FS_SMALL,
                            weight="bold", ha="left", va="center", zorder=8, linespacing=1.3,
                            arrowprops=dict(arrowstyle="-", color=SHELF_C, lw=1.8, shrinkA=4, shrinkB=2),
                            path_effects=st.stroke())
    ships = ax.scatter(tr["lon"], tr["lat"], s=34, color=cm.C["ship"], edgecolors=cm.BG, lw=0.8, zorder=4)
    true_pt = ax.scatter([sx], [sy], marker="*", s=600, color=cm.C["source"], edgecolors=cm.BG, lw=1.2, zorder=6)
    est = est_tag = None

    # ------------------------------------------------------------------ right: evidence vs alarm level + depth comparison card
    axs, rel, line, dot, alarm, ymax = s9.evidence_axes(fig, (0.55, 0.42, 0.42, 0.36), tr["stat_ais"],
                                                        tr["thr_ais"], cm.C["network"])
    if np.nanmax(rel) >= 1:
        print(f"scene 09c WARNING: stat_ais reaches {np.nanmax(rel):.2f} x the alarm level")
    alarm_lbl = axs.text(2, 1.0 + 0.025 * ymax, "1% false-alarm threshold", color=cm.C["alarm"],
                         fontsize=cm.FS_SMALL, weight="bold", ha="left", va="bottom", zorder=7)
    verdict = fig.text(0.55, 0.315, "Alarm level not reached within 2 hours", fontsize=cm.FS_H2 - 2,
                       color=cm.C["alarm"], weight="bold", ha="left", va="center")

    # Compute deep-water ship statistics in Crete box from trial metadata / traffic
    sd = trial.get("shelf") or {}
    box_ships = sd.get("box_ships") or []
    n_shelf = int(sd.get("n_shelf_ships", 0))
    n_deep = len(box_ships)
    med_depth = int(np.median([s["depth_m"] for s in box_ships])) if box_ships else 980
    max_snr_deep = max([s["peak_snr_ais"] for s in box_ships]) if box_ships else 0.17
    med_sigma_cm = 100.0 * float(np.median(tr["sigma_total"]))

    why_ax = cm.card(fig, (0.55, 0.12, 0.42, 0.165))
    why_ax.patches[0].set_alpha(0)
    w_t0 = why_ax.text(0.035, 0.80, "Why shallow-shelf ships (<200 m) are required:",
                       fontsize=cm.FS_SMALL - 1, weight="bold", color=cm.TEXT, va="center", alpha=0)
    w_t1 = why_ax.text(0.035, 0.48, f"• Shallow (<200 m): {n_shelf} ships today (currents ~25 cm/s > {med_sigma_cm:.0f} cm/s noise)",
                       fontsize=cm.FS_LABEL, color=SHELF_C, va="center", alpha=0)
    w_t2 = why_ax.text(0.035, 0.18, f"• Deep (~{med_depth:,} m): {n_deep} nearby ships (currents <3 cm/s, max {max_snr_deep:.2f}× noise)",
                       fontsize=cm.FS_LABEL, color=cm.MUTED, va="center", alpha=0)

    veil_ax = fig.add_axes((0, 0, 1, 1), zorder=50)
    veil_ax.axis("off")
    veil = veil_ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=veil_ax.transAxes, color=cm.BG,
                                                          alpha=1.0, lw=0))

    def update(t):
        veil.set_alpha(1.0 - cm.ramp(t, 0.0, 0.5))
        a0 = cm.fade(t, 0.2)
        head.set_alpha(a0); true_pt.set_alpha(a0); ships.set_alpha(a0)
        a_sh = cm.fade(t, 0.6, dur=0.8)
        shelf_cs.set_alpha(0.55 * a_sh); shelf_ln.set_alpha(a_sh)
        shelf_lbl.set_alpha(a_sh); shelf_lbl.arrow_patch.set_alpha(a_sh)
        alarm.set_alpha(cm.fade(t, 0.4)); alarm_lbl.set_alpha(cm.fade(t, 0.4))
        m = 120.0 * float(np.clip((t - T_CURVE0) / (T_CURVE1 - T_CURVE0), 0, 1))
        s9.draw_curve(line, dot, rel, m)
        if est is not None:
            est.set_alpha(cm.fade(t, T_EST, dur=0.6))
            est_tag.set_alpha(cm.fade(t, T_EST, dur=0.6)); est_tag.arrow_patch.set_alpha(cm.fade(t, T_EST, dur=0.6))
        a_v = cm.fade(t, T_VERDICT, dur=0.6)
        verdict.set_alpha(a_v)
        why_ax.patches[0].set_alpha(a_v)
        w_t0.set_alpha(a_v); w_t1.set_alpha(a_v); w_t2.set_alpha(a_v)

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
