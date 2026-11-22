"""Scene 14 (Results): replaying the 2020 Samos earthquake with today's ships.

On-screen text is docs/DEMO_SCRIPT.md scene 14, word for word; numbers are read at render time.

Inputs:
  * outputs/demo_data/samos_snapshots.npz, samos_gauges.npz, samos_ships.npz, samos_meta.json
    (scripts/demo/make_samos_data.py: run_id 0 of data/historic_runs/sources.csv, NumPy solver, slip x sin(dip)
    uplift, traffic_draw index 0, calm AIS noise; IOC records de-tided). Wave field, ships, Kos time series.
  * outputs/historic_samos_gauges.csv (Stream B, T26), row kos1 + samos_meta.json recorded_series_max_abs_0_120min_m:
    Within the 2-h simulation window (0-120 min), recorded Kos peak is 6.7 cm and simulated (x Green's-law factor)
    is 6.4 cm (late harbour resonance reached 11.9 cm at ~273 min, after the 2-h simulation window).
  * Ship lines: outputs/historic_replays_summary.csv (Stream B, T26, scripts/historic_replays.py), rows samos_2020 /
    catalogue arcs_plus_aegean / network / calm / both range caps (40 and 100 km), one line per operator; the
    range is min to max of detected_frac over the two caps.
"""
import common as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCENE = 14
EXTENT = (23.0, 29.6, 34.9, 40.6)           # Aegean view (lon0, lon1, lat0, lat1)
WAVE_LIM_M = 0.10                            # colour range +-10 cm
SNR_MAX = 3.0                                # ship colour bar 0..3 sigma
T_CLOCK = (1.0, 9.5)                         # scene seconds over which the clock runs 0 -> 120 min
T_TAIL = (9.5, 11.0)                         # recorded Kos series continues past the simulation
T_NUMBERS, T_NETWORK, T_CAPTION = 11.0, 12.0, 1.2       # GPS line at T_NETWORK + 1
DD = cm.DEMO_DATA
EVENT, CATALOGUE, CONFIRM_LEVEL = "samos_2020", "arcs_plus_aegean", 0.5


def _load():
    try:
        meta = cm.load_json(DD / "samos_meta.json")
        if meta is None:
            return None
        return dict(meta=meta, snaps=np.load(DD / "samos_snapshots.npz"), gauges=np.load(DD / "samos_gauges.npz"),
                    ships=np.load(DD / "samos_ships.npz"))
    except Exception as e:  # noqa: BLE001
        print("scene 14: demo data unavailable:", e)
        return None


def kos_row(station):
    p = cm.OUTPUTS / "historic_samos_gauges.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    t = t[t.station == station]
    return None if t.empty else t.iloc[0]


def _pct_range(v):
    lo, hi = 100 * float(np.min(v)), 100 * float(np.max(v))
    return f"{lo:.0f}%" if round(lo) == round(hi) else f"{lo:.0f} to {hi:.0f}%"


def network_lines(cfg):
    """('Ship drift network: ...', 'Ships with precise GPS: ...') from T26 (see the module docstring)."""
    s13 = cm.load_scene_module(13)
    out = []
    for op, who in (("ais", "Ship drift network"), ("gps", "Ships with precise GPS")):
        t = s13.historic_rows(cfg, op, regimes=("calm",), catalogue=CATALOGUE)
        t = None if t is None else t[t["name"] == EVENT]
        if t is None or t.empty:
            out.append(f"{who}: result to follow")
        elif t["at_far"].all():
            out.append(f"{who}: NO detection — wave too small for ship drift ({_pct_range(t.detected_frac)} of days = false-alarm floor)")
        elif (t["detected_frac"] >= CONFIRM_LEVEL).any():
            out.append(f"{who}: confirmed on {_pct_range(t.detected_frac)} of days")
        else:
            out.append(f"{who}: partial ({_pct_range(t.detected_frac)} of days — rarely detected)")
    return out


def build(cut, ctx):
    cfg = ctx.cfg
    fig, chrome = cm.new_figure(
        SCENE, caption="Physics verified at Kos (6.7 cm vs 6.4 cm in 0–120 min) — AIS fails because Mw 7.0 offshore currents (< 2 cm/s) sit below ship noise (~10 cm/s).",
    )
    D = _load()
    if D is None:
        cm.placeholder(fig, "2020 Samos replay: wave field, ships, and Kos tide gauge",
                       needs="outputs/demo_data/samos_*.npz + samos_meta.json (run scripts/demo/make_samos_data.py)")
        return cm.Scene(fig, lambda t: None, still_t=1.0)

    meta, S, G, SH = D["meta"], D["snaps"], D["gauges"], D["ships"]
    kos = meta["kos"]

    # ---- left: Aegean map
    ax = cm.map_axes(fig, (0.025, 0.13, 0.50, 0.72), extent=EXTENT)
    lon, lat = S["lon"], S["lat"]
    wet, eta = S["wet"].astype(bool), S["eta"]              # (121, ny, nx) in float16, metres
    t_snap = S["t_min"].astype(float)
    n_min = float(t_snap[-1])
    norm = mcolors.Normalize(-WAVE_LIM_M, WAVE_LIM_M)

    def field(tm):
        k = int(np.clip(np.floor(tm), 0, len(t_snap) - 2))
        f = float(np.clip(tm - k, 0, 1))
        e = (1 - f) * eta[k].astype(np.float32) + f * eta[k + 1].astype(np.float32)
        return np.ma.masked_where(~wet, e)

    im = ax.imshow(field(0.0), origin="lower", extent=(lon[0], lon[-1], lat[0], lat[-1]), cmap=cm.WAVE_CMAP, norm=norm,
                   interpolation="bilinear", zorder=1, alpha=0.0, aspect="auto")   # keep the cos(lat) map aspect

    slon, slat = SH["lon"], SH["lat"]
    inview = (slon >= EXTENT[0]) & (slon <= EXTENT[1]) & (slat >= EXTENT[2]) & (slat <= EXTENT[3])
    slon, slat = slon[inview], slat[inview]
    snr_t = SH["snr_t"][inview].astype(np.float32)   # (S, 120), minute m ends at m+1 min
    snr_norm = mcolors.Normalize(0, SNR_MAX)
    ships = ax.scatter(slon, slat, c=np.zeros(len(slon)), cmap=cm.SNR_CMAP, norm=snr_norm, s=16,
                       edgecolors=cm.C["ship"], linewidths=0.5, zorder=4, alpha=0.0)

    row = meta["source"]["row"]
    epi = ax.scatter([row["lon_deg"]], [row["lat_deg"]], marker="*", s=520, color=cm.C["source"], edgecolors=cm.BG,
                     linewidths=1.2, zorder=6, alpha=0)
    epi_lbl = ax.text(row["lon_deg"] + 0.12, row["lat_deg"] + 0.18, f"Samos, Mw {row['mw']:.1f}", color=cm.C["source"],
                      fontsize=cm.FS_SMALL, weight="bold", zorder=7, alpha=0,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor=cm.BG, edgecolor="none", alpha=0))
    codes = list(G["codes"])
    ki = codes.index(kos["station"])
    st = ax.scatter(G["lon"], G["lat"], marker="^", s=90, color=cm.C["gauge"], edgecolors=cm.BG, linewidths=0.8,
                    zorder=6, alpha=0)
    kos_mk = ax.scatter([G["lon"][ki]], [G["lat"][ki]], marker="^", s=330, facecolor="none", edgecolors=cm.C["gauge"],
                        linewidths=2.4, zorder=7, alpha=0)
    kos_lbl = ax.text(G["lon"][ki] + 0.25, G["lat"][ki] - 0.05, "Kos", color=cm.C["gauge"], fontsize=cm.FS_SMALL,
                      weight="bold", zorder=7, alpha=0)
    clock = ax.text(0.03, 0.965, "", transform=ax.transAxes, ha="left", va="top", fontsize=cm.FS_H2, weight="bold",
                    color=cm.TEXT, zorder=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=cm.BG, edgecolor="none", alpha=0.75))

    # colour bars under the right column
    from matplotlib.cm import ScalarMappable
    cax_w = fig.add_axes((0.565, 0.155, 0.18, 0.018))
    cb_w = fig.colorbar(ScalarMappable(norm, cm.WAVE_CMAP), cax=cax_w, orientation="horizontal", extend="both")
    cb_w.set_ticks([-WAVE_LIM_M, 0, WAVE_LIM_M])
    cb_w.set_ticklabels([f"≤ −{100 * WAVE_LIM_M:.0f}", "0", f"≥ +{100 * WAVE_LIM_M:.0f}"])
    cb_w.outline.set_edgecolor(cm.FAINT)
    cax_w.set_title("Open-sea wave (cm, clipped; 2 m bay)", fontsize=cm.FS_LABEL - 1, color=cm.TEXT, pad=6)
    cax_s = fig.add_axes((0.785, 0.155, 0.18, 0.018))
    cb_s = fig.colorbar(ScalarMappable(snr_norm, cm.SNR_CMAP), cax=cax_s, orientation="horizontal")
    cb_s.set_ticks([0, 1, 2, 3]); cb_s.set_ticklabels(["0", "1σ", "2σ", "3σ"])
    cb_s.outline.set_edgecolor(cm.FAINT)
    cax_s.set_title("Ship signal relative to its noise", fontsize=cm.FS_LABEL, color=cm.TEXT, pad=6)
    for c in (cax_w, cax_s):
        c.tick_params(labelsize=cm.FS_LABEL, colors=cm.MUTED)

    # ---- right: Kos time series (recorded; simulated x Green's-law factor from historic_samos_gauges.csv)
    kr = kos_row(kos["station"])
    green = float(kr.green_factor) if kr is not None else float(kos["green_factor"])
    sim_raw_max = float(np.abs(G["sim_eta"][:, ki]).max())
    if kr is not None and not np.isclose(sim_raw_max, float(kr.simulated_max_abs_m), rtol=0.02):
        print(f"scene 14: WARNING samos_gauges.npz Kos max {sim_raw_max:.4f} m != CSV {kr.simulated_max_abs_m:.4f} m")
    tax = fig.add_axes((0.585, 0.54, 0.385, 0.26))
    tax.set_facecolor(cm.BG)
    st_t = G["sim_t_min"].astype(float)
    st_e = 100 * green * G["sim_eta"][:, ki].astype(float)
    rk = f"rec_t_min_{kos['station']}"
    have_rec = rk in G.files
    rt = G[rk].astype(float) if have_rec else np.array([])
    re = 100 * G[f"rec_eta_{kos['station']}"].astype(float) if have_rec else np.array([])
    t_peak = float(kos.get("recorded_series_max_t_min") or n_min)
    x_max = 60.0 * np.ceil(max(n_min, t_peak + 15) / 60.0)
    w = (rt >= 0) & (rt <= x_max)
    rt, re = rt[w], re[w]
    ylim = 1.2 * max(np.abs(st_e).max(), np.abs(re).max() if len(re) else 0)
    tax.set_xlim(0, x_max); tax.set_ylim(-ylim, ylim)
    tax.axvspan(0, n_min, color=cm.FAINT, alpha=0.14, zorder=0)
    tax.axvline(n_min, color=cm.MUTED, lw=1.0, ls="--", alpha=0.6)
    tax.text(n_min - 3, -0.88 * ylim, "2-h simulated window", color=cm.MUTED, fontsize=cm.FS_LABEL - 1,
             ha="right", va="bottom")
    tax.axhline(0, color=cm.FAINT, lw=1)
    tax.set_xlabel("Minutes after the earthquake", fontsize=cm.FS_LABEL)
    tax.set_ylabel("cm", fontsize=cm.FS_LABEL)
    tax.set_xticks(np.arange(0, x_max + 1, 60))
    tax.set_title("Kos tide gauge: recorded vs simulated (0–120 min)", fontsize=cm.FS_SMALL,
                  color=cm.TEXT, loc="left", weight="bold", pad=10)
    l_rec, = tax.plot([], [], color=cm.C["gauge"], lw=2.4)
    l_sim, = tax.plot([], [], color=cm.TEXT, lw=2.2)
    tax.legend(handles=[Line2D([], [], color=cm.C["gauge"], lw=3, label="recorded"),
                        Line2D([], [], color=cm.TEXT, lw=3, label="simulated (0–120 min)")],
               loc="upper left", fontsize=cm.FS_LABEL, frameon=False, ncol=2)
    cursor = tax.axvline(0, color=cm.MUTED, lw=1, ls=":")

    # ---- right: key numbers
    rec_120_cm = 100 * float(kos.get("recorded_series_max_abs_0_120min_m") or (kr.recorded_max_abs_m if kr is not None else 0.067))
    rec_late_cm = 100 * float(kr.recorded_max_abs_m if kr is not None else kos.get("recorded_max_abs_m", 0.119))
    sim_120_cm = 100 * float((kr.simulated_max_abs_m * kr.green_factor) if kr is not None else (kos["sim_max_abs_m"] * green))
    l_wave = (f"First 2 hours at Kos: recorded {rec_120_cm:.1f} cm vs simulated {sim_120_cm:.1f} cm\n"
              f"(late bay resonance {rec_late_cm:.1f} cm at 4.5 h; offshore currents < 2 cm/s)")
    num_ax = cm.card(fig, (0.55, 0.225, 0.425, 0.235))
    l_ais, l_gps = (x.replace(" (", "\n(", 1) if len(x) > 60 else x for x in network_lines(cfg))
    t1 = num_ax.text(0.03, 0.80, l_wave, fontsize=cm.FS_LABEL, color=cm.TEXT, va="center", alpha=0,
                     linespacing=1.25)
    t2 = num_ax.text(0.03, 0.45, l_ais, fontsize=cm.FS_LABEL + 1, color=cm.C["network"], va="center", alpha=0,
                     linespacing=1.25)
    t3 = num_ax.text(0.03, 0.15, l_gps, fontsize=cm.FS_LABEL + 1, color=cm.C["gps"], va="center", alpha=0,
                     linespacing=1.25)
    num_ax.patches[0].set_alpha(0)

    chrome["caption"].set_position((0.025, 0.045)); chrome["caption"].set_ha("left")
    chrome["caption"].set_linespacing(1.3)
    chrome["caption"].set_alpha(0)

    def update(t):
        a_in = cm.fade(t, 0.0, dur=0.8)
        chrome["caption"].set_alpha(cm.fade(t, T_CAPTION, dur=0.8))
        tm = n_min * np.clip((t - T_CLOCK[0]) / (T_CLOCK[1] - T_CLOCK[0]), 0, 1)   # linear clock (map)
        tx = tm + (x_max - n_min) * cm.ramp(t, *T_TAIL)                             # time-series cursor
        im.set_data(field(tm))
        im.set_alpha(0.95 * a_in)
        epi.set_alpha(a_in); epi_lbl.set_alpha(a_in); epi_lbl.get_bbox_patch().set_alpha(0.7 * a_in)
        st.set_alpha(0.8 * a_in); kos_mk.set_alpha(a_in); kos_lbl.set_alpha(a_in)
        m = int(np.clip(np.floor(tm) - 1, 0, snr_t.shape[1] - 1))
        s_now = snr_t[:, m] if tm >= 1 else np.zeros(len(slon))
        ships.set_array(s_now)
        ships.set_sizes(16 + 60 * np.clip(s_now / SNR_MAX, 0, 1))
        ships.set_alpha(cm.fade(t, 0.6, dur=0.8))
        clock.set_text(f"{tm:3.0f} min"); clock.set_alpha(a_in); clock.get_bbox_patch().set_alpha(0.75 * a_in)
        ws = st_t <= tx
        l_sim.set_data(st_t[ws], st_e[ws])
        wr = rt <= tx
        l_rec.set_data(rt[wr], re[wr])
        cursor.set_xdata([tx, tx]); cursor.set_alpha(0.8 * (1 - cm.ramp(t, T_TAIL[1], T_TAIL[1] + 0.6)))
        num_ax.patches[0].set_alpha(cm.fade(t, T_NUMBERS, dur=0.7))
        t1.set_alpha(cm.fade(t, T_NUMBERS + 0.15, dur=0.6))
        t2.set_alpha(cm.fade(t, T_NETWORK, dur=0.6))
        t3.set_alpha(cm.fade(t, T_NETWORK + 1.0, dur=0.6))

    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 0.5)


if __name__ == "__main__":
    cm.main(__file__)
