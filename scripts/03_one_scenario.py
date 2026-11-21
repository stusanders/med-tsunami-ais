"""T13 / G2 check: one Hellenic Arc Mw 8 event. Ships coloured by per-ship signal-to-noise, gauge and buoy trigger times.

Library candidate = Hellenic, Mw 8.0, nearest to (23.35 E, 34.08 N) (the T03 check position). One traffic draw (index 0),
100 km range cap, calm weather, AIS operator. Per-ship SNR = peak over 120 min of |filtered heading-normal current| divided by the
total filtered noise sd of that ship (independent wobble + shared field).
Outputs: outputs/first_detection.png and outputs/first_detection.json (numbers behind the figure).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from tsunami import backend, gauges, noise, observe, ships
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler
from tsunami.sources import source_eta0
from tsunami.viz import COLORS, map_axes, save_figure

cfg = load_config()
grid, pool = load_grid(cfg), ships.load_pool(cfg)
cand = pd.read_csv(repo_path(cfg["paths"]["candidates"]))
c = cand[(cand.region == "hellenic") & (cand.mw == 8.0)]
d2 = (c.lon_deg - 23.35) ** 2 + (c.lat_deg - 34.08) ** 2
row = c.loc[d2.idxmin()].to_dict()
eta0 = source_eta0(grid, row, cfg)
solver = Solver(grid, make_sampler(grid, pool), cfg)
gb_idx = np.nonzero(pool.rtype > 0)[0]
recf, gb30 = solver.run_library(backend.to_xp(eta0[None]), gb_idx)
recf, gb30 = backend.to_numpy(recf)[0], backend.to_numpy(gb30)            # (120, R, 3), (1, 240, R_gb)
snap_t = [900.0, 1800.0, 3600.0]
snap = backend.to_numpy(solver.snapshots(backend.to_xp(eta0[None]), snap_t))[0]

regime, operator = "calm", "ais"
draw = ships.apply_range_cap(ships.traffic_draw(pool, cfg, 0), pool, cfg["traffic"]["range_caps_m"][-1])
sig = observe.ais_signal(observe.gather_truth(recf, draw), draw)                       # (S, 120) m/s
sigma = np.sqrt(noise.ship_sigma(draw, regime, operator, cfg) ** 2 + noise.shared_sigma(regime, operator, cfg) ** 2)
snr_t = np.abs(sig) / sigma[:, None]
peak_snr = snr_t.max(axis=1)
k = cfg["single_ship"]["literal_k_sigma"]
first_min = np.where((snr_t > k).any(axis=1), (snr_t > k).argmax(axis=1), -1)
gt = gauges.load_gauges(grid, cfg)
trig = gauges.trigger_times(gb30, gt, eta0_gb=gauges.eta0_at_receivers(eta0, gt)[None])[0]    # s, inf if never; eta(t) - eta(0)
trig_raw = gauges.trigger_times(gb30, gt)[0]                                           # old behaviour, for the comparison only

lon, lat = pool.lon[draw.pool_idx], pool.lat[draw.pool_idx]
res = dict(source=dict(cand_id=int(row["cand_id"]), lon_deg=row["lon_deg"], lat_deg=row["lat_deg"], mw=row["mw"]),
           n_ships=int(len(lon)), regime=regime, sigma_total_m_s_median=float(np.median(sigma)),
           n_ships_peak_snr_gt3=int((peak_snr > k).sum()), n_ships_peak_snr_gt1=int((peak_snr > 1).sum()), max_peak_snr=float(peak_snr.max()),
           first_ship_over_3sigma_s=float(60 * (first_min[first_min >= 0].min() + 1)) if (first_min >= 0).any() else None,
           first_ship_over_3sigma_minute_end_note="confirmation = end of the minute in which one ship first exceeds 3 sigma (single-ship, uncalibrated)",
           n_gauges_triggered=int(np.isfinite(trig[gt.rtype == 1]).sum()), n_gauges=int((gt.rtype == 1).sum()),
           n_buoys_triggered=int(np.isfinite(trig[gt.rtype == 2]).sum()), n_buoys=int((gt.rtype == 2).sum()),
           first_gauge_s=float(trig[gt.rtype == 1].min()), first_buoy_s=float(trig[gt.rtype == 2].min()),
           first_gauge_s_raw_eta=float(trig_raw[gt.rtype == 1].min()), n_gauges_triggered_raw_eta=int(np.isfinite(trig_raw[gt.rtype == 1]).sum()),
           eta0_at_first_trigger_gauge_m=float(gauges.eta0_at_receivers(eta0, gt)[np.argmin(np.where(gt.rtype == 1, trig, np.inf))]),
           gauge_triggers={c_: float(t) for c_, t in zip(gt.code, trig) if np.isfinite(t)})
(repo_path(cfg["paths"]["outputs"]) / "first_detection.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k_: v for k_, v in res.items() if k_ != "gauge_triggers"}, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(17, 6.3), gridspec_kw={"width_ratios": [1.5, 1]})
_, ax = map_axes(cfg, axs[0], grid=False)
ext = [grid.lon[0] - grid.dlon_deg / 2, grid.lon[-1] + grid.dlon_deg / 2, grid.lat[0] - grid.dlat_deg / 2, grid.lat[-1] + grid.dlat_deg / 2]
j = 1                                                                                   # 30 min snapshot as background
ax.imshow(np.ma.masked_where(~grid.wet | (np.abs(snap[j]) < 0.01), snap[j]), origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
          aspect="auto", alpha=0.6, zorder=1)
ax.set_xlim(*cfg["domain"]["area_of_interest"]["lon_deg"]); ax.set_ylim(*cfg["domain"]["area_of_interest"]["lat_deg"])
order = np.argsort(peak_snr)
sc = ax.scatter(lon[order], lat[order], c=np.maximum(peak_snr[order], 0.01), s=14, cmap="viridis", norm=LogNorm(0.01, 10), zorder=3, edgecolors="none")
plt.colorbar(sc, ax=ax, shrink=0.65, label="peak per-ship SNR over 2 h (log; 3 = Inazu threshold)")
is_g = gt.rtype.values == 1
tm = np.where(np.isfinite(trig), trig / 60, np.nan)
for sel, mk, size, lab in [(is_g, "^", 70, "gauge"), (~is_g, "*", 220, "buoy")]:
    ok = sel & np.isfinite(tm)
    ax.scatter(gt.lon_deg[sel & ~ok], gt.lat_deg[sel & ~ok], marker=mk, s=size * 0.5, facecolors="none", edgecolors="grey", linewidths=0.8, zorder=4)
    s2 = ax.scatter(gt.lon_deg[ok], gt.lat_deg[ok], marker=mk, s=size, c=tm[ok], cmap="autumn_r", vmin=0, vmax=120, edgecolors="k", linewidths=0.7, zorder=5)
ax.plot(row["lon_deg"], row["lat_deg"], marker="X", color="k", ms=11, zorder=6, linestyle="none")
plt.colorbar(s2, ax=ax, shrink=0.65, label="gauge / buoy trigger time (min after origin, incl. telemetry delay)")
ax.set_title(f"Hellenic Arc Mw {row['mw']:.1f} (X): {len(lon)} ships; background = sea surface at 30 min (m); grey open symbols never trigger", fontsize=9)

ax = axs[1]
best = np.argsort(-peak_snr)[:3]
t_min = np.arange(1, 121)
for b in best:
    ax.plot(t_min, 100 * sig[b], label=f"ship {b}: peak SNR {peak_snr[b]:.1f}, sd {100*sigma[b]:.1f} cm/s")
    ax.axhline(100 * k * sigma[b], color="grey", ls=":", lw=0.6); ax.axhline(-100 * k * sigma[b], color="grey", ls=":", lw=0.6)
ax.set_xlabel("minutes after origin"); ax.set_ylabel("filtered heading-normal current (cm/s)")
ax.set_title("Three strongest ships (dotted = their own 3 sigma)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
txt = (f"ships > 3 sigma at some time: {res['n_ships_peak_snr_gt3']} of {res['n_ships']} (> 1 sigma: {res['n_ships_peak_snr_gt1']})\n"
       f"first gauge trigger: {res['first_gauge_s']/60:.0f} min ({res['n_gauges_triggered']}/{res['n_gauges']} gauges)\n"
       f"first buoy trigger: {res['first_buoy_s']/60:.0f} min ({res['n_buoys_triggered']}/{res['n_buoys']} buoys)")
ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="grey"))
save_figure(fig, repo_path(cfg["paths"]["outputs"]) / "first_detection.png", "First detection check: Hellenic Arc Mw 8, AIS ships, calm weather",
            "Caption: per-ship signal-to-noise (peak over 2 h) for one traffic draw within 100 km of the coast, with gauge and buoy trigger times; single event, not a detection statistic.")
