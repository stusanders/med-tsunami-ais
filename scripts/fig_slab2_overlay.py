"""Item 4 of the report review: overlay EFSM20 and Slab2 Hellenic interface depth contours to check
sign and depth conventions before reporting their offset. Output: outputs/slab2_overlay.png
Run: PYTHONPATH=src python scripts/fig_slab2_overlay.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from tsunami import sources
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

CFG = load_config()
LEVELS_KM = [10, 20, 30, 40, 60, 100]

grid = load_grid(CFG)
slab = xr.open_dataset(repo_path("data/raw/neamthm18/slab2_hellenic/hel_slab2_dep_02.24.18.grd"))
print("Slab2 z units/sign: min", float(slab.z.min()), "max", float(slab.z.max()), "(km, negative down)")
sz_km = -slab.z.values                                    # positive down, km
v, _ = sources.arc_mesh("GRSS001", CFG)                   # lon, lat, depth_m positive down
lon = np.linspace(v[:, 0].min(), v[:, 0].max(), 300)
lat = np.linspace(v[:, 1].min(), v[:, 1].max(), 200)
LON, LAT = np.meshgrid(lon, lat)
ez_km = griddata(v[:, :2], v[:, 2] / 1e3, (LON, LAT), method="linear")
print("EFSM20 depth range km:", np.nanmin(ez_km), np.nanmax(ez_km))

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.contour(grid.lon, grid.lat, grid.wet.astype(float), levels=[0.5], colors="0.4", linewidths=0.5)
c1 = ax.contour(LON, LAT, ez_km, levels=LEVELS_KM, colors="tab:red", linewidths=1.5)
c2 = ax.contour(slab.x, slab.y, sz_km, levels=LEVELS_KM, colors="tab:blue", linewidths=1.5, linestyles="--")
ax.clabel(c1, fmt="%d", fontsize=8)
ax.clabel(c2, fmt="%d", fontsize=8)
ax.plot([], [], "tab:red", label="EFSM20 GRSS001 interface depth (km)")
ax.plot([], [], "tab:blue", ls="--", label="Slab2 hel_slab2_dep (km, sign flipped to positive down)")
ax.set_xlim(18, 30)
ax.set_ylim(32, 39)
ax.set_xlabel("longitude (deg E)")
ax.set_ylabel("latitude (deg N)")
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Hellenic Arc interface: EFSM20 vs Slab2 depth contours")
fig.text(0.01, 0.01, "Same depth levels, both positive down in km; Slab2 grid converted from its negative-km convention.",
         fontsize=8, style="italic")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(repo_path("outputs/slab2_overlay.png"), dpi=130)
