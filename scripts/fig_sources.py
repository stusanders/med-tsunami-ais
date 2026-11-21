"""T03 figures and report: candidate positions, EFSM20 geometry cross-check, Mw 8 source sizes.

Outputs: outputs/sources_candidates.png, outputs/geometry_check.png, outputs/sources_check.md,
data/processed/candidates.csv.
Run: PYTHONPATH=src python scripts/fig_sources.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tsunami import sources
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

CFG = load_config()
R = CFG["physics"]["earth_radius_m"]
OUT = repo_path(CFG["paths"]["outputs"])
RAW = repo_path("data/raw/neamthm18")
COLORS = {"hellenic": "tab:red", "calabrian": "tab:purple", "cyprus": "tab:orange", "aegean": "tab:green"}
FLAG_KM = 25.0     # report arcs whose geometries differ by more than this (Prompt C)


def coast(ax, grid):
    ax.contour(grid.lon, grid.lat, grid.wet.astype(float), levels=[0.5], colors="k", linewidths=0.5)


def nearest_km(lon_a, lat_a, lon_b, lat_b):
    """Distance (km) from each point a to the nearest point b (local equirectangular)."""
    lat0 = np.mean(lat_a)
    ax = R * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon_a)[:, None]
    bx = R * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon_b)[None, :]
    ay, by = R * np.deg2rad(lat_a)[:, None], R * np.deg2rad(lat_b)[None, :]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2).min(axis=1) / 1e3


def efsm_depth_line(idfs, depth_m):
    """EFSM20 mesh vertices within 500 m of a given depth (positive down)."""
    v, _ = sources.arc_mesh(idfs, CFG)
    sel = np.abs(v[:, 2] - depth_m) <= 500.0
    return v[sel, 0], v[sel, 1]


def main():
    grid = load_grid(CFG)
    table = sources.candidate_table(CFG, grid, save=True)
    pos = table.drop_duplicates("pos_id")

    # ---------------- candidate positions
    fig, ax = plt.subplots(figsize=(11, 6))
    coast(ax, grid)
    for region, sub in pos.groupby("region"):
        ax.plot(sub.trench_lon_deg, sub.trench_lat_deg, "o", ms=3.5, color=COLORS[region],
                label=f"{region} ({len(sub)} positions)")
    for name, ep in CFG["sources"]["exclusion_epicentres"].items():
        ax.plot(ep["lon_deg"], ep["lat_deg"], "k*", ms=10)
        ax.annotate(name, (ep["lon_deg"], ep["lat_deg"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    a = CFG["domain"]["area_of_interest"]
    ax.set_xlim(*a["lon_deg"])
    ax.set_ylim(*a["lat_deg"])
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Library candidate positions: {len(pos)} positions, {len(table)} candidates")
    fig.text(0.01, 0.01, "Arc points every 25 km on the EFSM20 trench edge; Aegean crustal grid ~50 km, kept "
             f"{CFG['sources']['aegean_exclusion_radius_m'] / 1e3:.0f} km from the Samos and Amorgos epicentres (stars).",
             fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "sources_candidates.png", dpi=130)
    plt.close(fig)

    # ---------------- geometry cross-check
    slab = xr.open_dataset(RAW / "slab2_hellenic/hel_slab2_dep_02.24.18.grd")
    sx, sy = np.meshgrid(slab.x.values, slab.y.values)
    sz = -slab.z.values * 1e3                                  # km negative -> m positive down
    mae = pd.read_excel(RAW / "maesano2017/41598_2017_9074_MOESM4_ESM.xls")
    mae_depth = -mae["Z"].to_numpy()                           # negative metres -> positive down
    checks, rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (region, idfs, other_name) in zip(axes, [("hellenic", "GRSS001", "Slab2"), ("calabrian", "ITSS001", "Maesano 2017")]):
        coast(ax, grid)
        edge = sources.trench_edge(idfs, CFG)
        v, _ = sources.arc_mesh(idfs, CFG)
        ax.plot(v[:, 0], v[:, 1], ",", color="0.75")
        ax.plot(edge[:, 0], edge[:, 1], "-", color=COLORS[region], lw=2, label=f"EFSM20 top edge ({v[:, 2].min() / 1e3:.0f} km)")
        for depth_km in (10.0, 20.0):
            elon, elat = efsm_depth_line(idfs, depth_km * 1e3)
            if other_name == "Slab2":
                sel = np.isfinite(sz) & (np.abs(sz - depth_km * 1e3) <= 500.0)
                olon, olat = sx[sel], sy[sel]
            else:
                sel = np.abs(mae_depth - depth_km * 1e3) <= 500.0
                olon, olat = mae["X"].to_numpy()[sel], mae["Y"].to_numpy()[sel]
            ax.plot(olon, olat, ".", ms=2, color="tab:blue", alpha=0.6,
                    label=f"{other_name} 10 and 20 km" if depth_km == 10.0 else None)
            ax.plot(elon, elat, ".", ms=2.5, color=COLORS[region], zorder=3,
                    label="EFSM20 10 and 20 km" if depth_km == 10.0 else None)
            if len(elon) and len(olon):
                # distance from every EFSM20 iso-depth point to the nearest point of the other model
                d = nearest_km(elon, elat, olon, olat)
                if len(d):
                    rows.append(dict(arc=region, compared_with=other_name, depth_km=depth_km, n=len(d),
                                     median_km=float(np.median(d)), p90_km=float(np.percentile(d, 90))))
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude (deg N)")
        ax.set_title(f"{region.capitalize()} Arc: EFSM20 vs {other_name}")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(edge[:, 1]))))
    fig.suptitle("Geometry check: EFSM20 subduction interface against independent models")
    fig.text(0.01, 0.01, "Dots: iso-depth points at 10 and 20 km of the interface (EFSM20 in colour, other model blue); "
             "line: EFSM20 trench edge used for candidate positions.", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "geometry_check.png", dpi=130)
    plt.close(fig)
    comp = pd.DataFrame(rows)

    # ---------------- source sizes
    size_rows = []
    for region in ("hellenic", "calabrian", "cyprus", "aegean"):
        mw = 8.0 if region != "aegean" else 7.0
        sub = table[(table.region == region) & np.isclose(table.mw, mw)]
        r = sub.iloc[len(sub) // 2]
        e = sources.source_eta0(grid, r, CFG)
        up, down, net = sources.dipole_volume(e, grid)
        size_rows.append(dict(region=region, mw=mw, length_km=r.length_m / 1e3, width_km=r.width_m / 1e3,
                              slip_m=r.slip_m, peak_uplift_m=float(e.max()), peak_subsidence_m=float(e.min()),
                              uplift_km3=up / 1e9, subsidence_km3=down / 1e9, net_km3=net / 1e9))
    sizes = pd.DataFrame(size_rows)

    counts = table.groupby("region").agg(positions=("pos_id", "nunique"), candidates=("cand_id", "size"),
                                         mw_min=("mw", "min"), mw_max=("mw", "max"),
                                         strike_median_deg=("strike_deg", "median"), dip_median_deg=("dip_deg", "median"))
    flagged = comp[comp.median_km > FLAG_KM] if len(comp) else comp
    gb = len(table) * CFG["library"]["n_minutes"] * 20061 * 3 * 4 / 1e9
    md = ["# Earthquake sources (T03)", "",
          f"{len(table)} library candidates at {table.pos_id.nunique()} positions; library size about **{gb:.0f} GB** "
          "float32 on the Colab runtime (20,061 receivers).", "",
          counts.round(1).to_markdown(), "",
          "![](sources_candidates.png)", "",
          "## Mw 8 (arcs) and Mw 7 (Aegean) source sizes", "",
          sizes.round(2).to_markdown(index=False), "",
          "Peak uplift is `uplift_fraction_of_slip x slip` (PLACEHOLDER 0.3) and subsidence `subsidence_ratio x uplift` "
          "(PLACEHOLDER 0.3). A Mw 8 interface event gives about 1 m uplift and 0.3 m subsidence; published Hellenic "
          "Arc Mw 8 scenarios report seafloor uplift of order 1 to 2 m (citation to add in the report).", "",
          "## Geometry cross-check", "",
          (comp.round(1).to_markdown(index=False) if len(comp) else "no overlap found"), "",
          (f"**Arcs differing by more than {FLAG_KM:.0f} km (median): "
           + ", ".join(f"{r.arc} vs {r.compared_with} at {r.depth_km:.0f} km" for r in flagged.itertuples()) + "**"
           if len(flagged) else f"No arc differs by more than {FLAG_KM:.0f} km (median nearest-point distance)."),
          "", "Hellenic at 10 km: Slab2 barely reaches 10 km depth along this arc (an isolated patch near 33.35 N and "
          "the Cyprus segment), so that row is not meaningful; the 20 km row is the real comparison. There the two "
          "models are about 150 km apart (Slab2 places the interface further seaward), far beyond 25 km. The choice "
          "of geometry therefore moves Hellenic candidates by up to ~150 km; this is listed as a limitation.",
          "", "EFSM20's Calabrian interface is built from Maesano et al. (2017) by the same group, so close agreement "
          "there is expected and is not an independent test. Slab2 is derived mainly from seismicity and is poorly "
          "constrained at shallow depth; EFSM20 follows the EDSF geometry used by the European hazard models. "
          "Candidates use EFSM20 as instructed.",
          "", "Cyprus Arc has no independent model here (Slab2 Cyprus not downloaded), so it is not cross-checked.", "",
          "![](geometry_check.png)", "",
          "## To verify before the report", "",
          "- Scaling coefficients in `config.yaml` (`sources.interface_scaling`, `sources.crustal_scaling`) were "
          "transcribed from Strasser et al. (2010) Table 2 and Wells & Coppersmith (1994) Table 2A without the PDFs at hand.",
          "- Aegean strike, dip and box, uplift fraction, subsidence ratio, Gaussian widths and the Samos/Amorgos "
          "the Samos and Amorgos exclusion epicentres now come from T26a (USGS; Okal et al. 2009)."]
    (OUT / "sources_check.md").write_text("\n".join(md))
    print(counts)
    print(sizes.round(2).to_string(index=False))
    print(comp.round(1).to_string(index=False) if len(comp) else "no comparison")


if __name__ == "__main__":
    main()
