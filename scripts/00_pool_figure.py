"""T04 check figures: lane directions over vessel density, and one traffic draw -> outputs/pool_lanes.png, traffic_draw.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami import ships
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

cfg = load_config()
g = load_grid(cfg)
pool = ships.load_pool(cfg)
out = repo_path(cfg["paths"]["outputs"])
ext = [g.lon[0] - g.dlon_deg / 2, g.lon[-1] + g.dlon_deg / 2, g.lat[0] - g.dlat_deg / 2, g.lat[-1] + g.dlat_deg / 2]
asp = 1 / np.cos(np.deg2rad(cfg["scoring"]["cell_ref_lat_deg"]))
dens = ships.regrid_density(cfg, g)
lanes, coh = ships.lane_axis(dens, g, cfg)
elig = ships._eligible(g, cfg)

# lanes
fig, ax = plt.subplots(figsize=(15, 7))
im = ax.imshow(np.log10(1 + np.where(g.wet, dens, np.nan)), origin="lower", extent=ext, cmap="magma_r", aspect=asp)
plt.colorbar(im, ax=ax, shrink=0.7, label="log10(1 + vessel density, h / km2 / month)")
step = 6
jj, ii = np.mgrid[0:len(g.lat):step, 0:len(g.lon):step]
m = elig[jj, ii] & (coh[jj, ii] > 0.3)
th = lanes[jj, ii][m]
ax.quiver(g.lon[ii][m], g.lat[jj][m], np.sin(th), np.cos(th), color="tab:cyan", headlength=0, headaxislength=0, pivot="mid",
          scale=60, width=0.0016)
ax.contour(g.lon, g.lat, g.coast_dist / 1e3, levels=[100], colors="grey", linewidths=0.5)
ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
ax.set_title("Lane axis from the density structure tensor (cyan, where coherence > 0.3), eligible cells")
fig.text(0.5, 0.01, "Caption: lane directions at every 6th eligible cell over EMODnet vessel density; grey line is 100 km from the coast.",
         ha="center", fontsize=9)
fig.savefig(out / "pool_lanes.png", dpi=110, bbox_inches="tight")

# one draw
draw = ships.traffic_draw(pool, cfg, 0)
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5))
for ax, cap in zip(axs, [None, cfg["traffic"]["range_caps_m"][0]]):
    d = draw if cap is None else ships.apply_range_cap(draw, pool, cap)
    ax.imshow(np.where(g.wet, 0, 1), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2, aspect=asp)
    ax.scatter(pool.lon[d.pool_idx], pool.lat[d.pool_idx], c=np.rad2deg(d.heading), cmap="twilight", s=6, vmin=0, vmax=360)
    gb = pool.rtype > 0
    ax.scatter(pool.lon[gb & (pool.rtype == 1)], pool.lat[gb & (pool.rtype == 1)], marker="^", c="tab:green", s=22, label="gauges")
    ax.scatter(pool.lon[pool.rtype == 2], pool.lat[pool.rtype == 2], marker="*", c="red", s=80, label="buoys")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"One traffic draw, {len(d.pool_idx)} ships" + (f", within {cap/1e3:.0f} km of coast" if cap else ", within 100 km"))
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)"); ax.legend(loc="lower left", fontsize=8)
fig.text(0.5, 0.01, "Caption: ships (coloured by heading, deg) drawn from the receiver pool; the 40 km fleet is a subset of the 100 km fleet.",
         ha="center", fontsize=9)
fig.savefig(out / "traffic_draw.png", dpi=110, bbox_inches="tight")
print("ok")
