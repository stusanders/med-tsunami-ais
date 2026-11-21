"""Earthquake tsunami sources: EFSM20 arcs, Aegean crustal grid, Gaussian dipole (T03, Stream C).

Contract (docs/interfaces.md section 7):
    sources.candidate_table(cfg) -> pandas.DataFrame
    sources.source_eta0(grid, row, cfg, kind=None, polarity=None) -> (ny, nx) float32
    sources.perturb(row, along_strike_m, dmw, cfg) -> dict
    sources.custom_source(lon_deg, lat_deg, strike_deg, dip_deg, mw, fault, cfg,
                          length_m=None, width_m=None, slip_m=None) -> dict
    sources.dipole_volume(eta0, grid) -> (uplift_m3, subsidence_m3, net_m3)

Magnitude scaling:
    Strasser, Arango & Bommer (2010), Seismological Research Letters 81(6), 941-950,
    doi:10.1785/gssrl.81.6.941 (Table 1, subduction interface OLS regressions):
        log10(L_km) = -2.477 + 0.585 * Mw
        log10(W_km) = -0.882 + 0.351 * Mw
    Seismic moment (Hanks & Kanamori 1979):
        log10(M0_Nm) = 1.5 * Mw + 9.05
        slip_m = M0_Nm / (rigidity_pa * L_m * W_m)
"""
from collections import defaultdict
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import xarray as xr

from .config import magnitudes, repo_path
from .geometry import haversine_m, heading_normal, heading_unit
from .units import deg2rad, km_to_m, rad2deg

# Strasser et al. (2010, SRL 81(6):941-950, doi:10.1785/gssrl.81.6.941) subduction interface
STRASSER_L_A = -2.477
STRASSER_L_B = 0.585
STRASSER_W_A = -0.882
STRASSER_W_B = 0.351
# Hanks & Kanamori (1979) moment-magnitude relation in SI (N m): log10(M0) = 1.5 * Mw + 9.05
HK_M0_A = 9.05
HK_M0_B = 1.5

# EFSM20 subduction-interface identifiers -> region names (Basili et al. 2024, doi:10.13127/efsm20/meshes)
ARC_IDFS = (
    ("hellenic", "GRSS001"),
    ("calabrian", "ITSS001"),
    ("cyprus", "CYSS001"),
)

# Default Aegean crustal box & regional fault orientation (overridable in config.yaml)
AEGEAN_DEFAULT_LON = (23.5, 27.2)
AEGEAN_DEFAULT_LAT = (36.0, 40.2)
AEGEAN_DEFAULT_STRIKE_DEG = 270.0   # E-W extensional trend in the Aegean back-arc
AEGEAN_DEFAULT_DIP_DEG = 45.0       # typical crustal normal-fault dip
AEGEAN_MIN_DEPTH_M = 50.0           # offshore sea cells only

# Fallback historic epicentres (used only if data/raw/historic/*_sources.json is absent)
# Samos 2020: USGS us7000c7y0 (26.784 E, 37.897 N); Amorgos 1956: Okal et al. 2009 (25.76 E, 36.72 N)
HISTORIC_EXCLUSION_FALLBACK = (
    (26.784, 37.897),
    (25.760, 36.720),
)

CANDIDATE_COLUMNS = [
    "cand_id",
    "pos_id",
    "region",
    "lon_deg",
    "lat_deg",
    "strike_deg",
    "dip_deg",
    "mw",
    "length_m",
    "width_m",
    "slip_m",
    "polarity",
    "flip_allowed",
]


def _get(row, key):
    if isinstance(row, dict):
        return row[key]
    return row[key] if hasattr(row, "__getitem__") else getattr(row, key)


def _in_area_of_interest(lon_deg, lat_deg, cfg):
    dom = cfg["domain"]
    if "area_of_interest" in dom:
        aoi = dom["area_of_interest"]
        lon0, lon1 = float(aoi["lon_deg"][0]), float(aoi["lon_deg"][1])
        lat0, lat1 = float(aoi["lat_deg"][0]), float(aoi["lat_deg"][1])
    else:
        lon0, lon1 = float(dom["lon_min_deg"]), float(dom["lon_max_deg"])
        lat0, lat1 = float(dom["lat_min_deg"]), float(dom["lat_max_deg"])
    return (lon_deg >= lon0) & (lon_deg <= lon1) & (lat_deg >= lat0) & (lat_deg <= lat1)


def seismic_moment_nm(mw):
    """Hanks & Kanamori (1979) scalar seismic moment M0 in N m."""
    return float(10.0 ** (HK_M0_B * float(mw) + HK_M0_A))


def scaling_dimensions(mw, cfg):
    """Compute (length_m, width_m, slip_m) from Strasser et al. (2010) interface scaling."""
    sc = cfg["sources"]
    l_a = float(sc.get("strasser_interface_L_a", STRASSER_L_A))
    l_b = float(sc.get("strasser_interface_L_b", STRASSER_L_B))
    w_a = float(sc.get("strasser_interface_W_a", STRASSER_W_A))
    w_b = float(sc.get("strasser_interface_W_b", STRASSER_W_B))
    mu = float(sc["rigidity_pa"])

    mw_f = float(mw)
    length_m = float(km_to_m(10.0 ** (l_a + l_b * mw_f)))
    width_m = float(km_to_m(10.0 ** (w_a + w_b * mw_f)))
    m0 = seismic_moment_nm(mw_f)
    slip_m = float(m0 / (mu * length_m * width_m))
    return length_m, width_m, slip_m


@lru_cache(maxsize=2)
def _load_efsm20_geojson(path_str):
    with open(path_str) as f:
        return json.load(f)


def extract_trench_chain(tris, radius_m):
    """Extract the ordered shallow (trench) boundary polyline of an EFSM20 subduction mesh.

    EFSM20 depths are negative metres down; converted to positive metres down on read.
    Returns (lons, lats, s_m, min_depth_m).
    """
    edge_tris = defaultdict(list)
    verts = {}
    for p in tris:
        vs = []
        for k in (1, 2, 3):
            vid = p[f"idvertex{k}"]
            verts[vid] = (float(p[f"lon{k}"]), float(p[f"lat{k}"]), -float(p[f"depth{k}"]))
            vs.append(vid)
        for a, b in ((vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])):
            e = (a, b) if a < b else (b, a)
            edge_tris[e].append(p)

    b_edges = [e for e, ts in edge_tris.items() if len(ts) == 1]
    adj = defaultdict(list)
    for a, b in b_edges:
        adj[a].append(b)
        adj[b].append(a)

    min_d = min(verts[v][2] for v in adj)
    shallow_vs = {v for v in adj if abs(verts[v][2] - min_d) < 1.0}
    ends = sorted(v for v in shallow_vs if sum(1 for n in adj[v] if n in shallow_vs) == 1)
    if len(ends) != 2:
        raise RuntimeError(f"Expected 2 shallow-edge endpoints, got {len(ends)}")

    chain = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxts = [n for n in adj[cur] if n in shallow_vs and n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        chain.append(cur)

    lons = np.array([verts[v][0] for v in chain], dtype=np.float64)
    lats = np.array([verts[v][1] for v in chain], dtype=np.float64)
    ds = haversine_m(lons[:-1], lats[:-1], lons[1:], lats[1:], radius_m)
    s_m = np.concatenate([[0.0], np.cumsum(ds)])
    return lons, lats, s_m, float(min_d)


def _sample_arc_positions(cfg):
    """Sample candidate positions every `sources.arc_spacing_m` along the three EFSM20 arcs."""
    radius_m = float(cfg["physics"]["earth_radius_m"])
    spacing_m = float(cfg["sources"]["arc_spacing_m"])
    mesh_path = repo_path(cfg["sources"]["arc_geometry"])
    gj = _load_efsm20_geojson(str(mesh_path))

    rows = []
    for region, idfs in ARC_IDFS:
        tris = [ft["properties"] for ft in gj["features"] if ft["properties"]["idfs"] == idfs]
        lons, lats, s_m, min_d = extract_trench_chain(tris, radius_m)

        tri_lon = np.array([float(p["lon_c"]) for p in tris], dtype=np.float64)
        tri_lat = np.array([float(p["lat_c"]) for p in tris], dtype=np.float64)
        tri_dep = np.array([-float(p["depth_c"]) for p in tris], dtype=np.float64)
        tri_str = np.array([float(p["strike"]) for p in tris], dtype=np.float64)
        tri_dip = np.array([float(p["dip"]) for p in tris], dtype=np.float64)
        deep_mask = tri_dep > (min_d + 5000.0)

        # Centred spacing along arc so consecutive points are exactly `spacing_m` apart
        n_pts = max(1, int(np.floor(s_m[-1] / spacing_m)) + 1)
        rem = s_m[-1] - (n_pts - 1) * spacing_m
        s_pts = 0.5 * rem + np.arange(n_pts, dtype=np.float64) * spacing_m

        # Orient chain consistently west-to-east (increasing longitude) for neat ordering
        if lons[-1] < lons[0]:
            lons = lons[::-1]
            lats = lats[::-1]
            s_m = s_m[-1] - s_m[::-1]

        for sp in s_pts:
            lo = float(np.interp(sp, s_m, lons))
            la = float(np.interp(sp, s_m, lats))
            if not _in_area_of_interest(lo, la, cfg):
                continue

            # Take strike and dip from nearby non-flat mesh triangles (within 1.2 * spacing)
            d_tri = haversine_m(lo, la, tri_lon, tri_lat, radius_m)
            near = (d_tri <= 1.2 * spacing_m) & (tri_dip > 0.5)
            if not near.any():
                idx = np.argsort(d_tri)[:10]
                near = np.zeros(len(tris), dtype=bool)
                near[idx] = True

            # Circular mean of mesh triangle strikes; verify Aki-Richards down-dip (strike+90) points deeper
            str_rad = deg2rad(tri_str[near])
            strike_deg = float(rad2deg(np.arctan2(np.mean(np.sin(str_rad)), np.mean(np.cos(str_rad)))) % 360.0)
            dip_deg = float(np.median(tri_dip[near]))

            # Guard against any 180-deg flip by checking strike+90 against deeper mesh direction
            d_deep = haversine_m(lo, la, tri_lon[deep_mask], tri_lat[deep_mask], radius_m)
            k_deep = np.argsort(d_deep)[:20]
            vx = float(np.mean((tri_lon[deep_mask][k_deep] - lo) * np.cos(deg2rad(la))))
            vy = float(np.mean(tri_lat[deep_mask][k_deep] - la))
            nx, ny = heading_normal(deg2rad(strike_deg))
            if vx * nx + vy * ny < 0.0:
                strike_deg = (strike_deg + 180.0) % 360.0

            rows.append(
                {
                    "region": region,
                    "lon_deg": lo,
                    "lat_deg": la,
                    "strike_deg": strike_deg,
                    "dip_deg": dip_deg,
                    "flip_allowed": False,
                }
            )
    return rows


def _historic_exclusion_points(cfg):
    """Read Samos 2020 and Amorgos 1956 epicentres from data/raw/historic/ (with verified fallback)."""
    hdir = repo_path(cfg["paths"]["historic_dir"])
    pts = []
    for fn in ("samos_2020_sources.json", "amorgos_1956_sources.json"):
        p = hdir / fn
        if p.exists():
            rec = json.loads(p.read_text()).get("recommended_for_replay", {})
            if "lon_deg" in rec and "lat_deg" in rec:
                pts.append((float(rec["lon_deg"]), float(rec["lat_deg"])))
    if len(pts) == 2:
        return tuple(pts)
    return HISTORIC_EXCLUSION_FALLBACK


def _sample_aegean_positions(cfg):
    """Build a coarse Aegean crustal grid (~50 km) on wet sea cells away from Samos and Amorgos."""
    radius_m = float(cfg["physics"]["earth_radius_m"])
    sc = cfg["sources"]
    spacing_m = float(sc["aegean_spacing_m"])
    excl_m = float(sc["aegean_exclusion_radius_m"])
    lon0, lon1 = sc.get("aegean_lon_deg", AEGEAN_DEFAULT_LON)
    lat0, lat1 = sc.get("aegean_lat_deg", AEGEAN_DEFAULT_LAT)
    strike_deg = float(sc.get("aegean_strike_deg", AEGEAN_DEFAULT_STRIKE_DEG))
    dip_deg = float(sc.get("aegean_dip_deg", AEGEAN_DEFAULT_DIP_DEG))
    min_depth_m = float(cfg["traffic"].get("min_ship_depth_m", AEGEAN_MIN_DEPTH_M))

    ref_lat = 0.5 * (float(lat0) + float(lat1))
    dlat_deg = float(rad2deg(spacing_m / radius_m))
    dlon_deg = float(rad2deg(spacing_m / (radius_m * np.cos(deg2rad(ref_lat)))))
    lats = np.arange(float(lat0), float(lat1) + 1e-9, dlat_deg)
    lons = np.arange(float(lon0), float(lon1) + 1e-9, dlon_deg)

    excl_pts = _historic_exclusion_points(cfg)
    gebco_path = repo_path(cfg["paths"]["gebco"])
    rows = []
    with xr.open_dataset(gebco_path) as ds:
        elev = ds["elevation"]
        for la in lats:
            for lo in lons:
                if not _in_area_of_interest(lo, la, cfg):
                    continue
                # Exclude Sea of Marmara (NE corner outside the Aegean)
                if lo > 26.5 and la > 40.0:
                    continue
                # Exclude nodes within exclusion radius of Samos 2020 and Amorgos 1956 epicentres
                if any(haversine_m(lo, la, ex_lo, ex_la, radius_m) < excl_m for ex_lo, ex_la in excl_pts):
                    continue
                z = float(elev.sel(lat=la, lon=lo, method="nearest").values)
                if z <= -min_depth_m:
                    rows.append(
                        {
                            "region": "aegean",
                            "lon_deg": float(lo),
                            "lat_deg": float(la),
                            "strike_deg": strike_deg,
                            "dip_deg": dip_deg,
                            "flip_allowed": True,
                        }
                    )
    return rows


def candidate_table(cfg, *, save=False):
    """Return the full candidate source table (arcs + Aegean crustal grid).

    Row order IS the library order (`cand_id = 0..C-1`).
    If `save=True`, writes `data/processed/candidates.csv`.
    """
    mws_arc = magnitudes(cfg)
    aegean_mw_max = float(cfg["sources"]["aegean_mw_max"])
    mws_aegean = mws_arc[mws_arc <= aegean_mw_max + 1e-9]

    positions = _sample_arc_positions(cfg) + _sample_aegean_positions(cfg)
    records = []
    cand_id = 0
    for pos_id, pos in enumerate(positions):
        mws = mws_aegean if pos["region"] == "aegean" else mws_arc
        for mw in mws:
            length_m, width_m, slip_m = scaling_dimensions(float(mw), cfg)
            records.append(
                {
                    "cand_id": cand_id,
                    "pos_id": pos_id,
                    "region": pos["region"],
                    "lon_deg": round(float(pos["lon_deg"]), 5),
                    "lat_deg": round(float(pos["lat_deg"]), 5),
                    "strike_deg": round(float(pos["strike_deg"]), 2),
                    "dip_deg": round(float(pos["dip_deg"]), 2),
                    "mw": float(mw),
                    "length_m": round(length_m, 1),
                    "width_m": round(width_m, 1),
                    "slip_m": round(slip_m, 4),
                    "polarity": 1,
                    "flip_allowed": bool(pos["flip_allowed"]),
                }
            )
            cand_id += 1

    df = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    if save:
        out_path = repo_path(cfg["paths"]["candidates"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def source_eta0(grid, row, cfg, kind=None, polarity=None):
    """Build initial sea-surface displacement `eta0` (ny, nx) float32 on `grid`.

    Gaussian uplift centred at `(lon_deg, lat_deg)` elongated along `strike_deg`
    with `sigma_s = length_m / 2`, `sigma_d = width_m / 2`. For `kind == "dipole"`,
    adds a landward (down-dip = `strike_deg + 90 deg`) Gaussian subsidence lobe
    offset by `landward_offset_widths * width_m` with amplitude `subsidence_ratio`
    times the uplift peak. Multiplied by `polarity` (`+1` uplift-first / thrust,
    `-1` subsidence-first / normal). Land cells are masked to 0.
    """
    sc = cfg["sources"]
    radius_m = float(cfg["physics"]["earth_radius_m"])
    kind_use = (kind or sc["kind"]).lower()
    if kind_use not in ("dipole", "monopole"):
        raise ValueError(f"Unknown source kind: {kind_use!r}; expected 'dipole' or 'monopole'")

    pol = int(polarity) if polarity is not None else int(_get(row, "polarity"))
    if pol not in (-1, 1):
        raise ValueError(f"polarity must be +1 or -1, got {pol}")

    lon0 = float(_get(row, "lon_deg"))
    lat0 = float(_get(row, "lat_deg"))
    strike_rad = float(deg2rad(float(_get(row, "strike_deg"))))
    length_m = float(_get(row, "length_m"))
    width_m = float(_get(row, "width_m"))
    slip_m = float(_get(row, "slip_m"))

    peak_up = float(sc["uplift_fraction_of_slip"]) * slip_m
    sub_ratio = float(sc["subsidence_ratio"])
    offset_d = float(sc["landward_offset_widths"]) * width_m

    # Local tangent-plane coordinates (east, north) in metres relative to (lon0, lat0)
    cos_lat0 = float(np.cos(deg2rad(lat0)))
    dx_m = radius_m * cos_lat0 * deg2rad(grid.lon[None, :] - lon0)
    dy_m = radius_m * deg2rad(grid.lat[:, None] - lat0)

    # Strike unit vector and Aki-Richards down-dip (landward = strike + 90 deg) normal
    sx, sy = heading_unit(strike_rad)
    nx, ny = heading_normal(strike_rad)
    s_coord = dx_m * sx + dy_m * sy
    d_coord = dx_m * nx + dy_m * ny

    sigma_s = 0.5 * length_m
    sigma_d = 0.5 * width_m

    up = peak_up * np.exp(-0.5 * ((s_coord / sigma_s) ** 2 + (d_coord / sigma_d) ** 2))
    if kind_use == "dipole":
        sub = (sub_ratio * peak_up) * np.exp(
            -0.5 * ((s_coord / sigma_s) ** 2 + ((d_coord - offset_d) / sigma_d) ** 2)
        )
        eta = pol * (up - sub)
    else:
        eta = pol * up

    eta = np.where(grid.wet, eta, 0.0)
    return eta.astype(np.float32)


def perturb(row, along_strike_m, dmw, cfg):
    """Return a new source dict shifted `along_strike_m` along strike with `mw + dmw` rescaled."""
    radius_m = float(cfg["physics"]["earth_radius_m"])
    lon0 = float(_get(row, "lon_deg"))
    lat0 = float(_get(row, "lat_deg"))
    strike_deg = float(_get(row, "strike_deg"))
    dip_deg = float(_get(row, "dip_deg"))
    new_mw = round(float(_get(row, "mw")) + float(dmw), 4)

    sx, sy = heading_unit(float(deg2rad(strike_deg)))
    dlat_deg = float(rad2deg((float(along_strike_m) * sy) / radius_m))
    dlon_deg = float(rad2deg((float(along_strike_m) * sx) / (radius_m * np.cos(deg2rad(lat0)))))

    length_m, width_m, slip_m = scaling_dimensions(new_mw, cfg)
    out = dict(row) if isinstance(row, dict) else row.to_dict()
    out.update(
        {
            "lon_deg": round(lon0 + dlon_deg, 5),
            "lat_deg": round(lat0 + dlat_deg, 5),
            "strike_deg": strike_deg,
            "dip_deg": dip_deg,
            "mw": new_mw,
            "length_m": round(length_m, 1),
            "width_m": round(width_m, 1),
            "slip_m": round(slip_m, 4),
        }
    )
    return out


def custom_source(
    lon_deg,
    lat_deg,
    strike_deg,
    dip_deg,
    mw,
    fault,
    cfg,
    length_m=None,
    width_m=None,
    slip_m=None,
):
    """Build a custom earthquake source dict from published parameters (used for T26).

    `fault`: 'thrust' (polarity +1, uplift-first) or 'normal' (polarity -1, subsidence-first).
    Any omitted dimensions default to Strasser et al. (2010) scaling; if custom length_m or
    width_m is supplied without slip_m, slip_m is computed from M0 = mu * L * W * D.
    """
    fault_norm = str(fault).strip().lower()
    if fault_norm == "thrust":
        polarity = 1
    elif fault_norm == "normal":
        polarity = -1
    else:
        raise ValueError(f"fault must be 'thrust' or 'normal', got {fault!r}")

    def_l, def_w, _ = scaling_dimensions(float(mw), cfg)
    l_use = float(length_m) if length_m is not None else def_l
    w_use = float(width_m) if width_m is not None else def_w
    if slip_m is not None:
        s_use = float(slip_m)
    else:
        mu = float(cfg["sources"]["rigidity_pa"])
        s_use = float(seismic_moment_nm(float(mw)) / (mu * l_use * w_use))

    return {
        "cand_id": -1,
        "pos_id": -1,
        "region": "custom",
        "lon_deg": float(lon_deg),
        "lat_deg": float(lat_deg),
        "strike_deg": float(strike_deg) % 360.0,
        "dip_deg": float(dip_deg),
        "mw": float(mw),
        "length_m": round(l_use, 1),
        "width_m": round(w_use, 1),
        "slip_m": round(s_use, 4),
        "polarity": polarity,
        "flip_allowed": False,
        "fault": fault_norm,
    }


def dipole_volume(eta0, grid):
    """Integrate positive, negative, and net displaced water volumes (m^3) over wet cells.

    Returns `(uplift_m3, subsidence_m3, net_m3)` where `uplift_m3 >= 0` is the positive
    volume, `subsidence_m3 >= 0` is the magnitude of the negative volume, and
    `net_m3 = uplift_m3 - subsidence_m3` is the signed net volume integral.
    """
    eta = np.asarray(eta0, dtype=np.float64)
    cell_area = (grid.dx[:, None].astype(np.float64)) * float(grid.dy)
    wet = np.asarray(grid.wet, dtype=bool)
    dv = np.where(wet, eta * cell_area, 0.0)
    uplift_m3 = float(np.sum(np.where(dv > 0.0, dv, 0.0)))
    subsidence_m3 = float(np.sum(np.where(dv < 0.0, -dv, 0.0)))
    net_m3 = float(uplift_m3 - subsidence_m3)
    return uplift_m3, subsidence_m3, net_m3
