"""T10 / Map 1: share of ship traffic in three depth bands, overall and within each AIS range cap.

Traffic = expected merchant vessels present (EMODnet 2024 passenger + cargo + tanker density / hours per month x cell area), summed over wet cells of the
area of interest, excluding strait cells (no ships are drawn there). Bands: < 50 m, 50 to 200 m, > 200 m.
Writes outputs/map1.png, outputs/map1_table.csv and outputs/map1.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import ships
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

try:                                     # Stream C's helpers (T07); a minimal local stand-in keeps this script runnable
    from tsunami.viz import COLORS, map_axes, save_figure
except ImportError:
    COLORS = {}

    def map_axes(cfg, ax=None):
        a = cfg["domain"]["area_of_interest"]
        fig, ax = (plt.subplots(figsize=(9, 5.5)) if ax is None else (ax.figure, ax))
        ax.set_xlim(*a["lon_deg"]); ax.set_ylim(*a["lat_deg"])
        ax.set_aspect(1 / np.cos(np.deg2rad(cfg["scoring"]["cell_ref_lat_deg"])))
        ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
        return fig, ax

    def save_figure(fig, path, title, caption):
        fig.suptitle(title); fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
        fig.savefig(path, dpi=130, bbox_inches="tight")

cfg = load_config()
g = load_grid(cfg)
out = repo_path(cfg["paths"]["outputs"])
aoi = cfg["domain"]["area_of_interest"]
inside = ships.in_area_of_interest(g.lon[None, :], g.lat[:, None], cfg)
dens = ships.regrid_density(cfg, g)
E = ships.expected_ships(dens, g, cfg) * inside * ~g.strait

caps = cfg["traffic"]["range_caps_m"]
band_names = ["< 50 m", "50 to 200 m", "> 200 m"]
rows = []
for label, mask in [("all traffic in area of interest", np.ones_like(g.wet)),
                    *[(f"within {c/1e3:.0f} km of coast", g.coast_dist <= c) for c in caps[::-1]]]:
    tot = E[mask].sum()
    share = [E[mask & (g.depth_band == b)].sum() / tot for b in range(3)]
    rows.append({"subset": label, "expected_ships": tot, **{f"share_{n}": s for n, s in zip(band_names, share)}})
tab = pd.DataFrame(rows)
tab.to_csv(out / "map1_table.csv", index=False)
print(tab.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.2, 1]})
_, ax1 = map_axes(cfg, ax1)
ext = [g.lon[0] - g.dlon_deg / 2, g.lon[-1] + g.dlon_deg / 2, g.lat[0] - g.dlat_deg / 2, g.lat[-1] + g.dlat_deg / 2]
# traffic aggregated to blocks of 8 cells (about 25 km) so it reads on a map
k = 8
ny, nx = (E.shape[0] // k) * k, (E.shape[1] // k) * k
blk = E[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))
bext = [g.lon[0] - g.dlon_deg / 2, g.lon[0] - g.dlon_deg / 2 + nx * g.dlon_deg, g.lat[0] - g.dlat_deg / 2, g.lat[0] - g.dlat_deg / 2 + ny * g.dlat_deg]
ax1.imshow(np.where(g.wet, 0.0, 1.0), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2, aspect="auto", zorder=0)
im = ax1.imshow(np.ma.masked_less_equal(blk, 0), origin="lower", extent=bext, cmap="YlOrRd", aspect="auto",
                norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=blk.max()), zorder=1)
plt.colorbar(im, ax=ax1, shrink=0.7, label="expected ships present per ~25 km block")
band_colors = ["#1b9e77", "#7570b3", "#08519c"]
ax1.contour(g.lon, g.lat, g.depth_band >= 1, levels=[0.5], colors=band_colors[0], linewidths=0.8, zorder=2)
ax1.contour(g.lon, g.lat, g.depth_band >= 2, levels=[0.5], colors=band_colors[2], linewidths=0.8, zorder=2)
for c, ls in zip(caps, [":", "--"]):
    ax1.contour(g.lon, g.lat, g.coast_dist, levels=[c], colors="k", linewidths=0.6, linestyles=ls, zorder=2)
ax1.set_xlim(*aoi["lon_deg"]); ax1.set_ylim(*aoi["lat_deg"])
ax1.set_title("Where the ships are (green line: 50 m depth, blue line: 200 m; black dotted 40 km, dashed 100 km from coast)", fontsize=9)

bottom = np.zeros(len(tab))
for b, (n, col) in enumerate(zip(band_names, band_colors)):
    vals = 100 * tab[f"share_{n}"].values
    ax2.barh(range(len(tab)), vals, left=bottom, color=col, label=n, height=0.6)
    for y, (v0, v) in enumerate(zip(bottom, vals)):
        if v > 4:
            ax2.text(v0 + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
    bottom += vals
ax2.set_yticks(range(len(tab)))
ax2.set_yticklabels([f"{r.subset}\n({r.expected_ships:.0f} ships)" for r in tab.itertuples()], fontsize=8)
ax2.invert_yaxis(); ax2.set_xlim(0, 100)
ax2.set_xlabel("share of traffic (%)"); ax2.legend(title="depth band", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
ax2.set_title("Traffic by depth band")
save_figure(fig, out / "map1.png", "Map 1: share of Mediterranean ship traffic by water depth",
            "Caption: share of expected ship presence (EMODnet 2024 merchant density: passenger, cargo, tanker) in depth bands, overall and within the two AIS range caps; ships in water under 50 m are masked from the observing system.")

md = ["# Map 1: traffic share by depth band", "",
      "Traffic = expected vessels present (EMODnet 2024 merchant vessel density, types 8 + 9 + 10 / 730 h per month x cell area), wet cells inside the area of interest "
      f"({aoi['lon_deg'][0]} to {aoi['lon_deg'][1]} E, {aoi['lat_deg'][0]} to {aoi['lat_deg'][1]} N), strait cells removed. "
      "EMODnet 2024 lost satellite AIS from June 2024, so far-offshore density is under-counted (2023 merchant gives 6% more ships and the same shares to within 1.5 points); the density includes anchored and moored ships. Fishing, pleasure craft, tugs and other types are excluded because the ship network is merchant ships only.", "",
      "| Subset | Expected ships | " + " | ".join(f"share {n}" for n in band_names) + " |", "|---|---|---|---|---|"]
for r in tab.itertuples(index=False):
    md.append(f"| {r[0]} | {r[1]:.0f} | " + " | ".join(f"{100*x:.1f}%" for x in r[2:]) + " |")
(out / "map1.md").write_text("\n".join(md) + "\n")
