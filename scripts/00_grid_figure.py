"""T01 check figure: bathymetry, land, masks (and, when available, vessel density, gauges, buoys) -> outputs/grid.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

cfg = load_config()
g = load_grid(cfg)
ext = [g.lon[0] - g.dlon_deg / 2, g.lon[-1] + g.dlon_deg / 2, g.lat[0] - g.dlat_deg / 2, g.lat[-1] + g.dlat_deg / 2]
fig, axs = plt.subplots(1, 2, figsize=(18, 6.5))
ax = axs[0]
d = np.ma.masked_where(~g.wet, g.depth)
im = ax.imshow(d, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=4000, aspect=1 / np.cos(np.deg2rad(36)))
ax.imshow(np.ma.masked_where(g.wet, np.ones_like(g.depth)), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2,
          aspect=1 / np.cos(np.deg2rad(36)))
ax.contour(g.lon, g.lat, g.strait, levels=[0.5], colors="red", linewidths=1)
for name, b in {**cfg["domain"]["straits"], **cfg["domain"]["exclude_boxes"]}.items():
    ax.text(b["lon_deg"][0], b["lat_deg"][1], name, color="red", fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.7, label="depth (m)")
ax.set_title("Model grid: depth (m), land grey, strait masks red")
ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
ax = axs[1]
cd = np.ma.masked_where(~g.wet, g.coast_dist / 1e3)
im = ax.imshow(cd, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=300, aspect=1 / np.cos(np.deg2rad(36)))
ax.contour(g.lon, g.lat, g.coast_dist / 1e3, levels=[40, 100], colors=["white", "orange"], linewidths=0.6)
plt.colorbar(im, ax=ax, shrink=0.7, label="distance to coast (km)")
ax.set_title("Distance to coast (white 40 km, orange 100 km)")
ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
fig.suptitle("Grid check: 2 arc-min GEBCO, min wet depth 10 m")
fig.text(0.5, 0.01, "Caption: the model grid with land mask, strait and excluded-basin boxes and distance to coast (T01).", ha="center", fontsize=9)
out = repo_path(cfg["paths"]["outputs"]) / "grid.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(out)
