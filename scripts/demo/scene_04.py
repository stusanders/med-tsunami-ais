"""Scene 4 (Solution): how a ship measures the sea.

On-screen text is DEMO_SCRIPT.md scene 4 only (title, five labels, panel heading, caption).

1. One traffic draw over GEBCO bathymetry, ships coloured by water depth (bands from config
   domain.depth_band_edges_m), the receiver pool shown faintly.
2. Smooth zoom to one real ship of the draw on a shelf (50 to 200 m band, west of Crete);
   schematic heading / track / sideways-drift arrows using the heading-normal convention of
   docs/interfaces.md section 1 (drift = SOG sin(COG - HDG) = u . n). Not to scale (no number shown).
3. Zoom out to the two T24 showcase ships (shelf and deep water) and the same wave's peak
   heading-normal current at each.

Numbers on screen (all read at render time from outputs/demo_data/cleaning_traces.json,
written by scripts/demo/make_cleaning_traces.py, Hellenic Mw 8.0, cand 186, calm):
  * depths: ships.deep.depth_m (2,766 m) and ships.shelf.depth_m (61 m), rounded to 1 m;
  * currents: ships.deep.peak_true_filtered_m_s and ships.shelf.peak_true_filtered_m_s (peak true
    heading-normal current after the 1-minute average and the causal baseline filter), rounded to 0.01 m/s.
The earlier version of this scene computed the current as an illustrative 0.5 m wave x sqrt(g / depth)
(0.20 m/s on the shelf); that formula is gone.

Built from: data/processed/grid.npz, data/processed/receiver_pool.npz,
ships.traffic_draw(pool, cfg, index=0, purpose=1), config.yaml (domain.depth_band_edges_m),
outputs/demo_data/cleaning_traces.json.
"""
import common as cm
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

SCENE = 4

# On-screen text: DEMO_SCRIPT.md scene 4 (numbers filled in from cleaning_traces.json).
CAPTION = ("As the water becomes shallower, the same wave pushes ships sideways far more strongly, "
           "which makes depth the deciding factor.")
LABELS = ["Heading: where the ship points", "Actual course: where the ship actually goes",
          "Sideways drift: the gap between the two"]
PANEL_HEADING = "The same wave, at two depths"
POINT_NAME = {"deep": "Deep water", "shelf": "Shelf"}

# Diagram-only constants (not results; no number from them appears on screen).
DRIFT_ANGLE_SCHEMATIC_DEG = 18.0  # exaggerated COG - HDG angle
TARGET_BOX = {"lon": (22.0, 24.6), "lat": (34.3, 36.3)}  # where to look for the close-up ship (west of Crete)
KM_PER_DEG = np.pi * 6371.0 / 180.0               # drawing only (km per degree of latitude)


def _showcase_ships():
    """{'shelf'|'deep': dict(pool_idx, lon, lat, depth_m, u)} from cleaning_traces.json, or None if missing."""
    trc = cm.load_json("outputs/demo_data/cleaning_traces.json")
    try:
        return {k: dict(pool_idx=int(v["pool_idx"]), lon=float(v["lon"]), lat=float(v["lat"]),
                        depth_m=float(v["depth_m"]), u=float(v["peak_true_filtered_m_s"]))
                for k, v in ((k, trc["ships"][k]) for k in ("shelf", "deep"))}
    except (TypeError, KeyError, ValueError):
        print("scene_04 WARNING: outputs/demo_data/cleaning_traces.json missing or incomplete")
        return None


def point_label(key, p):
    return f"{POINT_NAME[key]} ({p['depth_m']:,.0f} m): {p['u']:.2f} m/s"


def _ease(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (x * (6 * x - 15) + 10)   # smootherstep


class View:
    """Animated map camera with a fixed axes box: lon span = lat span x box ratio / cos(lat)."""

    def __init__(self, ax, fig):
        self.ax = ax
        pos = ax.get_position()
        w_in, h_in = fig.get_size_inches()
        self.ratio = (pos.width * w_in) / (pos.height * h_in)
        ax.set_aspect("auto")

    def set(self, cx, cy, half_h):
        half_w = half_h * self.ratio / np.cos(np.deg2rad(cy))
        self.ax.set_xlim(cx - half_w, cx + half_w)
        self.ax.set_ylim(cy - half_h, cy + half_h)

    @staticmethod
    def interp(a, b, x):
        """a, b = (cx, cy, half_h): zoom geometric, centre follows so the zoom feels uniform."""
        e = _ease(x)
        hh = np.exp(cm.lerp(np.log(a[2]), np.log(b[2]), e))
        # centre weight tracks the zoom level, so the target stays put on screen while zooming
        w = e if np.isclose(a[2], b[2]) else (a[2] - hh) / (a[2] - b[2])
        return cm.lerp(a[0], b[0], w), cm.lerp(a[1], b[1], w), hh


def km_to_deg(lat, de_km, dn_km):
    return de_km / (KM_PER_DEG * np.cos(np.deg2rad(lat))), dn_km / KM_PER_DEG


def coastal_bathymetry(ax, grid, ext):
    """Bathymetry layer drawn over map_axes' one, with the few land cells next to the coast filled by the
    nearest wet depth (display only), so bilinear shading has no dark NaN blocks when zoomed in."""
    from scipy.ndimage import distance_transform_edt
    dist, (jj, ii) = distance_transform_edt(~grid.wet, return_indices=True)
    d = grid.depth[jj, ii].astype(float)
    d[dist > 4] = np.nan
    ax.imshow(np.log10(np.clip(d, 10, 5000)), origin="lower", cmap=cm.DEPTH_CMAP, extent=ext,
              vmin=1, vmax=3.7, interpolation="bilinear", zorder=0.1, aspect="auto")


def build(cut, ctx):
    cfg = ctx.cfg
    edges = cfg["domain"]["depth_band_edges_m"]
    fig, chrome = cm.new_figure(SCENE, caption=CAPTION)
    for k in chrome:
        chrome[k].set_zorder(20)

    # ---------------- map ----------------
    grid = ctx.grid
    full_ext = (grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1])
    ax = cm.map_axes(fig, (0.02, 0.085, 0.635, 0.78), extent=full_ext)
    ax.images[0].set_visible(False)          # map_axes' bathymetry: fully covered by the gap-filled layer below
    coastal_bathymetry(ax, grid, full_ext)
    shallow = np.zeros(grid.depth.shape + (4,), dtype=float)
    msk = grid.wet & (grid.depth < edges[0])
    shallow[msk, :3] = cm.mcolors.to_rgb(cm.C["depth_shallow"])
    shallow[msk, 3] = 1.0
    shallow_im = ax.imshow(shallow, origin="lower", extent=full_ext, interpolation="nearest",
                           zorder=0.45, alpha=0.0, aspect="auto")
    view = View(ax, fig)   # after every imshow (imshow resets the aspect)
    veil = ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=cm.BG, alpha=0.0, zorder=0.4))
    intro = ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=cm.BG, alpha=1.0, zorder=2.6))
    V0 = (24.5, 36.2, 6.6)             # eastern Mediterranean

    try:
        from tsunami import ships as ships_mod
        pool = ctx.pool()
        draw = ships_mod.traffic_draw(pool, cfg, index=0, purpose=1)
    except Exception as e:  # noqa: BLE001
        print("scene 04: pool / traffic draw unavailable:", e)
        cm.placeholder(fig, "Ships as sensors: one traffic draw over bathymetry",
                       needs="data/processed/receiver_pool.npz (T04)", rect=(0.1, 0.15, 0.8, 0.65))
        return cm.Scene(fig, lambda t: None)

    is_ship = pool.rtype == 0
    pool_sc = ax.scatter(pool.lon[is_ship], pool.lat[is_ship], s=1.6, color=cm.C["ship"], lw=0,
                         alpha=0.0, zorder=2.7)
    lon, lat = pool.lon[draw.pool_idx], pool.lat[draw.pool_idx]
    dep = pool.depth[draw.pool_idx].astype(float)
    band = np.digitize(dep, edges)                     # 0 shallow, 1 mid, 2 deep
    band_cols = [cm.C["depth_shallow"], cm.C["depth_mid"], cm.C["depth_deep"]]
    ship_sc = []
    for b in (2, 1, 0):                                # deep first so shelf ships sit on top
        m = band == b
        if m.any():
            ship_sc.append(ax.scatter(lon[m], lat[m], s=26, color=band_cols[b], edgecolor="white",
                                      lw=0.35, alpha=0.0, zorder=3 + b * 0.1))

    # ---- close-up ship: a 50-200 m draw ship west of Crete (else the shelf ship nearest the T24 point)
    sc_pts = _showcase_ships()
    have_sc = sc_pts is not None
    in_box = ((lon > TARGET_BOX["lon"][0]) & (lon < TARGET_BOX["lon"][1]) &
              (lat > TARGET_BOX["lat"][0]) & (lat < TARGET_BOX["lat"][1]))
    ref = (sc_pts["shelf"]["lon"], sc_pts["shelf"]["lat"]) if have_sc else (24.7, 35.0)
    dist = np.hypot((lon - ref[0]) * np.cos(np.deg2rad(ref[1])), lat - ref[1])
    shelf_ok = band == 1
    pick = shelf_ok & in_box if (shelf_ok & in_box).any() else shelf_ok
    k_t = int(np.where(pick)[0][np.argmin(dist[pick])])
    tlon, tlat = float(lon[k_t]), float(lat[k_t])
    theta = float(draw.heading[k_t])
    HH1 = 0.36                                          # close-up half-height, deg lat

    ring = ax.scatter([tlon], [tlat], s=0, facecolor="none", edgecolor=cm.TEXT, lw=2.5, alpha=0.0, zorder=5)

    # ship glyph and arrows in a local (east, north) km frame
    h_vec = np.array([np.sin(theta), np.cos(theta)])        # heading unit vector
    n_vec = np.array([np.cos(theta), -np.sin(theta)])       # starboard normal (interfaces.md section 1)

    def to_ll(p):
        dlo, dla = km_to_deg(tlat, p[0], p[1])
        return tlon + dlo, tlat + dla

    L = 0.95 * HH1 * KM_PER_DEG                             # arrow length, km (schematic)
    hs = 0.085 * L                                           # hull half-length, km (not to scale)
    hull_local = np.array([[0, 1.0], [0.3, 0.45], [0.3, -0.9], [-0.3, -0.9], [-0.3, 0.45]]) * hs
    def make_hull(offset_km=np.zeros(2)):
        return [to_ll(offset_km + p[0] * n_vec + p[1] * h_vec) for p in hull_local]
    glyph = ax.add_patch(Polygon(make_hull(), closed=True, fc=cm.C["depth_mid"], ec="white", lw=1.5,
                                 alpha=0.0, zorder=9.5))
    d_ang = np.deg2rad(DRIFT_ANGLE_SCHEMATIC_DEG)
    c_vec = np.cos(d_ang) * h_vec + np.sin(d_ang) * n_vec   # course over ground direction
    p0 = 1.1 * hs * h_vec
    tip_h = p0 + L * h_vec
    tip_c = p0 + L * c_vec
    foot = p0 + L * np.cos(d_ang) * h_vec                   # drift = component of track on n
    mid = to_ll(0.5 * (p0 + tip_c))
    V1 = (mid[0], mid[1], HH1)                              # frame the arrow triangle
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=30, lw=4.0, zorder=8, alpha=0.0, shrinkA=0, shrinkB=0)
    a_head = ax.add_patch(FancyArrowPatch(to_ll(p0), to_ll(tip_h), color=cm.TEXT, **arrow_kw))
    a_cog = ax.add_patch(FancyArrowPatch(to_ll(p0), to_ll(tip_c), color=cm.C["ship"], **arrow_kw))
    a_drift = ax.add_patch(FancyArrowPatch(to_ll(foot), to_ll(foot), color=cm.C["signal"],
                                           **{**arrow_kw, "lw": 5.0, "mutation_scale": 34}))
    _dl = to_ll(foot + 0.45 * (tip_c - foot) + 0.14 * L * h_vec)
    drift_lbl = ax.text(_dl[0], _dl[1], "Drift", color=cm.C["signal"], fontsize=cm.FS_BODY, weight="bold",
                        ha="right", va="center", zorder=9, alpha=0.0)
    cur_lbls = {}

    # ---- stage 3: the two showcase ships and the peak current at each (arrow length proportional to it)
    cur_arrows, sc_art = {}, []
    V2 = V1
    if have_sc:
        u_max = max(p["u"] for p in sc_pts.values())
        L_max = 85.0                                     # km of map for the strongest arrow
        lons = [p["lon"] for p in sc_pts.values()] + [tlon]
        lats = [p["lat"] for p in sc_pts.values()] + [tlat]
        V2 = (0.5 * (min(lons) + max(lons)) - 0.15, 0.5 * (min(lats) + max(lats)),
              0.5 * (max(lats) - min(lats)) + 0.38)
        for key, p in sc_pts.items():
            mk = ax.scatter([p["lon"]], [p["lat"]], s=170, color=band_cols[int(np.digitize(p["depth_m"], edges))],
                            edgecolor="white", lw=1.8, zorder=7.5, alpha=0.0)
            dlo, _ = km_to_deg(p["lat"], -L_max * p["u"] / u_max, 0.0)
            ar = ax.add_patch(FancyArrowPatch((p["lon"], p["lat"]), (p["lon"], p["lat"]), color=cm.C["signal"],
                                              arrowstyle="-|>", mutation_scale=30, lw=5, zorder=8,
                                              alpha=0.0, shrinkA=5, shrinkB=0))
            cur_arrows[key] = (ar, (p["lon"], p["lat"]), (p["lon"] + dlo, p["lat"]))
            cur_lbls[key] = ax.text(p["lon"] + 0.5 * dlo, p["lat"] + 0.07, "Drift", color=cm.C["signal"],
                                    fontsize=cm.FS_SMALL + 2, weight="bold", ha="center", va="bottom",
                                    zorder=9, alpha=0.0)
            sc_art.append(mk)

    # ---------------- right column cards ----------------
    X0, CW = 0.675, 0.30
    cB = cm.card(fig, (X0, 0.585, CW, 0.28))
    b_rows = []
    for i, (col, txt) in enumerate(zip((cm.TEXT, cm.C["ship"], cm.C["signal"]), LABELS)):
        y = 0.78 - i * 0.28
        ln = cB.add_patch(FancyArrowPatch((0.05, y), (0.15, y), arrowstyle="-|>", mutation_scale=18,
                                          lw=3, color=col, alpha=0.0, transform=cB.transAxes))
        head, rest = txt.split(":", 1)
        t1 = cB.text(0.19, y + 0.075, head + ":", fontsize=cm.FS_SMALL, color=col, va="center", alpha=0.0,
                     weight="bold")
        t2 = cB.text(0.19, y - 0.075, rest.strip(), fontsize=cm.FS_SMALL, color=cm.TEXT, va="center", alpha=0.0)
        b_rows.append((ln, t1, t2))

    cB.set_zorder(15)

    def update(t):
        veil.set_alpha(0.5)                       # steady dimming of the bathymetry so dots read clearly
        a_in = 1.0 - cm.ramp(t, 0.0, 1.2)        # scene fade-in from BG (covers land and coast too)
        intro.set_alpha(a_in); intro.set_visible(a_in > 0)
        a_sh = 0.75 * cm.ramp(t, 1.8, 2.8) * (1 - cm.ramp(t, 3.0, 4.4))
        shallow_im.set_alpha(a_sh); shallow_im.set_visible(a_sh > 0)
        a_pool = 0.22 * cm.ramp(t, 0.6, 2.0) * (1 - cm.ramp(t, 3.0, 4.2))
        pool_sc.set_alpha(a_pool); pool_sc.set_visible(a_pool > 0)
        for sc in ship_sc:
            sc.set_alpha(0.95 * cm.ramp(t, 0.9, 2.4))

        # camera: zoom smoothly into the shelf ship and stay focused on the drift geometry
        if t < 3.0:
            view.set(*V0)
        elif t < 5.8:
            view.set(*View.interp(V0, V1, (t - 3.0) / 2.8))
        else:
            view.set(*V1)

        ring.set_alpha(cm.fade(t, 2.3, 4.6, dur=0.5))
        ring.set_sizes([(34 + 8 * np.sin(2 * np.pi * 1.2 * t)) ** 2])

        glyph.set_alpha(cm.ramp(t, 5.0, 5.6))
        ship_prog = _ease(cm.ramp(t, 7.0, 9.4))
        glyph.set_xy(make_hull(ship_prog * (tip_c - p0)))
        a_head.set_alpha(cm.ramp(t, 5.8, 6.3))
        a_cog.set_alpha(cm.ramp(t, 6.8, 7.3))
        g = max(cm.ramp(t, 8.0, 9.0), 1e-3)
        a_drift.set_positions(to_ll(foot), to_ll(foot + g * (tip_c - foot)))
        a_drift.set_alpha(cm.ramp(t, 8.0, 8.3))
        drift_lbl.set_alpha(cm.ramp(t, 8.8, 9.2))

        aB = cm.ramp(t, 5.4, 6.0)
        cB.patches[0].set_alpha(aB)
        for (ln, t1, t2), t0 in zip(b_rows, (5.8, 6.8, 8.0)):
            a = cm.ramp(t, t0, t0 + 0.5)
            ln.set_alpha(a); t1.set_alpha(a); t2.set_alpha(a)
        chrome["caption"].set_alpha(cm.fade(t, 0.8, dur=0.6))

    return cm.Scene(fig, update, still_t=18.5, meta={"target_pool_idx": int(draw.pool_idx[k_t])})


if __name__ == "__main__":
    cm.main(__file__)
