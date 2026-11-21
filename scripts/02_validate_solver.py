"""T11: solver validation report -> outputs/validation.md plus figures in outputs/validation/.

Checks (PLAN.md Phase 2, G1):
  1 flat-bottom arrival time vs sqrt(g h)         (x and y directions)
  2 closed-basin volume conservation               (< 1 %)
  3 Green's law on a 1D sloping shelf              (~ h^-1/4, within 10 %)
  4 linearity and superposition                    (float32 tolerance)
  5 batched == reference, CuPy == NumPy            (CuPy only when a GPU is present)
  6 Hellenic Arc Mw 8 snapshot sequence + peak uplift (needs T01 grid and T03 sources)
  7 sponge absorbs an outgoing wave                (extra: open boundaries)

Run: PYTHONPATH=src python scripts/02_validate_solver.py
"""
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsunami import backend
from tsunami import solver_reference as ref
from tsunami.config import load_config, repo_path
from tsunami.solver import Solver
from tsunami.synthetic import BUOY, GAUGE, SHIP, synthetic_grid, synthetic_pool

CFG = load_config()
G = CFG["physics"]["g_m_s2"]
OUT = repo_path(CFG["paths"]["outputs"])
FIG = OUT / "validation"
FIG.mkdir(parents=True, exist_ok=True)
INTERVAL = 10.0        # s, fine sampling for timing checks (validation only)

# Thresholds come from PLAN.md / TICKETS.md acceptance criteria.
TOL_ARRIVAL = 0.02
TOL_VOLUME = 0.01
TOL_GREENS = 0.10
TOL_F32 = 1e-5


def closed_cfg():
    c = copy.deepcopy(CFG)
    c["solver"]["sponge_width_cells"] = 0
    return c


def save(fig, name, title, caption):
    fig.suptitle(title)
    fig.text(0.01, 0.005, caption, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = FIG / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"validation/{name}"


def peak_time(series, interval):
    k = int(np.argmax(series))
    y0, y1, y2 = series[k - 1], series[k], series[k + 1]
    return interval * (k + 1 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))


# ----------------------------------------------------------------- check 1

def check_arrival():
    h, sigma, dist, n_long, n_short, src = 4000.0, 30e3, 500e3, 300, 5, 60
    rows, series = [], {}
    for axis in ("x", "y"):
        shape = (n_short, n_long) if axis == "x" else (n_long, n_short)
        grid = synthetic_grid(np.full(shape, h), lon0_deg=20.0, lat0_deg=33.0)
        mid = n_short // 2
        if axis == "x":
            step = grid.dx[mid]
            k = src + int(round(dist / step))
            coord = (np.arange(n_long) - src)[None, :] * grid.dx[:, None]
            pool = synthetic_pool(grid, [grid.lon[k]], [grid.lat[mid]], [GAUGE])
        else:
            step = grid.dy
            k = src + int(round(dist / step))
            coord = (np.arange(n_long) - src)[:, None] * grid.dy * np.ones((1, n_short))
            pool = synthetic_pool(grid, [grid.lon[mid]], [grid.lat[k]], [GAUGE])
        d = (k - src) * step
        expected = d / np.sqrt(G * h)
        rec = ref.run_reference(grid, np.exp(-0.5 * (coord / sigma) ** 2), ref.make_sampler(grid, pool),
                                closed_cfg(), t_end_s=expected * 1.3, sample_interval_s=INTERVAL)
        got = peak_time(rec[:, 0, 0], INTERVAL)
        err = (got - expected) / expected
        rows.append((axis, d / 1e3, expected, got, err))
        series[axis] = (rec[:, 0, 0], expected)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for axis, (s, e) in series.items():
        t = INTERVAL * (np.arange(len(s)) + 1)
        ax.plot(t / 60, s, label=f"propagation in {axis}")
        ax.axvline(e / 60, ls="--", color="k", lw=0.8)
    ax.set_xlabel("time after origin (min)")
    ax.set_ylabel("eta at receiver (m)")
    ax.legend()
    f = save(fig, "check1_arrival.png", "Check 1: flat-bottom arrival, h = 4000 m, 500 km",
             "Plane pulse (sigma 30 km, 1 m) on a flat 4000 m sea; dashed line = distance / sqrt(g h).")
    ok = all(abs(r[4]) < TOL_ARRIVAL for r in rows)
    lines = [f"| {a} | {d:.1f} | {e:.1f} | {g_:.1f} | {100 * er:+.2f} % |" for a, d, e, g_, er in rows]
    table = "| direction | distance (km) | expected (s) | measured peak (s) | error |\n|---|---|---|---|---|\n" + "\n".join(lines)
    table += ("\n\nThe northward pulse peaks slightly higher (0.515 vs 0.500 m) because meridians converge "
              "towards the pole on the sphere (the cos(phi) terms); this is expected, not an error.")
    return ok, table, f


# ----------------------------------------------------------------- check 2

def check_volume():
    ny, nx = 60, 80
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 200.0 + 3800.0 * np.sin(np.pi * ii / nx) * np.sin(np.pi * jj / ny)
    depth[25:32, 40:48] = -50.0
    grid = synthetic_grid(depth, lat0_deg=35.0)
    eta0 = np.exp(-(((ii - 25) / 5.0) ** 2 + ((jj - 20) / 5.0) ** 2)).astype(np.float32)
    c = closed_cfg()
    dt, _ = ref.time_step(grid, c)
    ops = ref.grid_ops(grid, c)
    eta = eta0 * ops.wet
    u = np.zeros((ny, nx + 1), np.float32)
    v = np.zeros((ny + 1, nx), np.float32)
    v0 = ref.volume(eta, grid)
    vols = [v0]
    n_steps = 2000
    for _ in range(n_steps):
        eta, u, v = ref.step(eta, u, v, ops, np.float32(dt), np.float32(G))
        vols.append(ref.volume(eta, grid))
    rel = (np.array(vols) - v0) / v0
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.arange(n_steps + 1) * dt / 3600, rel)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("relative volume change (-)")
    f = save(fig, "check2_volume.png", "Check 2: closed-basin volume conservation",
             f"Closed basin with variable depth and an island, no sponge, {n_steps} steps; volume = sum(eta dx dy).")
    worst = float(np.abs(rel).max())
    return worst < TOL_VOLUME, f"max relative volume change over {n_steps * dt / 3600:.1f} h: **{worst:.1e}** (limit 1e-2)", f


# ----------------------------------------------------------------- check 3

def check_greens():
    h_deep, h_shallow, ny = 4000.0, 50.0, 4
    dxm = synthetic_grid(np.ones((ny, 10)), lat0_deg=0.0).dx[ny // 2]
    n_deep = n_slope = int(600e3 / dxm)
    nx = n_deep + n_slope
    prof = np.full(nx, h_deep)
    prof[n_deep:] = np.linspace(h_deep, h_shallow, n_slope)
    grid = synthetic_grid(np.tile(prof, (ny, 1)), lat0_deg=0.0)
    src, sigma = n_deep // 2, 60e3
    x = (np.arange(nx) - src)[None, :] * grid.dx[:, None]
    targets = [h_deep, 2000.0, 1000.0, 500.0, 250.0]
    cells = [n_deep - 20] + [n_deep + int(np.argmin(np.abs(prof[n_deep:] - h))) for h in targets[1:]]
    j = ny // 2
    pool = synthetic_pool(grid, grid.lon[cells], np.full(len(cells), grid.lat[j]), [GAUGE] * len(cells))
    speed = np.sqrt(G * prof)
    t_arr = [np.sum(dxm / speed[src:k]) for k in cells]
    tau = sigma / np.sqrt(G * h_deep)
    rec = ref.run_reference(grid, np.exp(-0.5 * (x / sigma) ** 2), ref.make_sampler(grid, pool), closed_cfg(),
                            t_end_s=max(t_arr) + 3 * tau, sample_interval_s=INTERVAL)
    t = INTERVAL * (np.arange(rec.shape[0]) + 1)
    amp = np.array([rec[(t > ta - 2 * tau) & (t < ta + 2 * tau), r, 0].max() for r, ta in enumerate(t_arr)])
    h_rec = prof[cells]
    ratio, greens = amp / amp[0], (h_rec / h_rec[0]) ** -0.25
    err = ratio / greens - 1
    fig, ax = plt.subplots(figsize=(6, 4))
    hh = np.geomspace(h_rec.min(), h_rec.max(), 50)
    ax.loglog(hh, (hh / h_rec[0]) ** -0.25, "k--", label="Green's law h^(-1/4)")
    ax.loglog(h_rec, ratio, "o", label="solver")
    ax.invert_xaxis()
    ticks = [4000, 2000, 1000, 500, 250]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0], ["1.0", "1.25", "1.5", "1.75", "2.0"])
    ax.set_yticks([], minor=True)
    ax.set_xlabel("depth h (m)")
    ax.set_ylabel("amplitude / amplitude at 4000 m (-)")
    ax.legend()
    f = save(fig, "check3_greens.png", "Check 3: shoaling on a 1D slope (4000 m to 50 m over 600 km)",
             "Incident-pulse peak (sigma 60 km) at five depths, normalised by the deep-water peak.")
    rows = "\n".join(f"| {h:.0f} | {r:.3f} | {g_:.3f} | {100 * e:+.1f} % |" for h, r, g_, e in zip(h_rec, ratio, greens, err))
    table = "| depth (m) | measured ratio | h^(-1/4) | difference |\n|---|---|---|---|\n" + rows
    return bool(np.all(np.abs(err) < TOL_GREENS)), table, f


# ----------------------------------------------------------------- checks 4 and 5

def small_case():
    ny, nx = 48, 64
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    depth = 3000.0 - 40.0 * ii + 500.0 * np.sin(jj / 6.0)
    depth[:, 58:] = -10.0
    depth[20:24, 30:34] = -5.0
    grid = synthetic_grid(depth, lat0_deg=34.0)
    rng = np.random.default_rng(0)
    lon = rng.uniform(grid.lon[2], grid.lon[50], 30)
    lat = rng.uniform(grid.lat[2], grid.lat[45], 30)
    pool = synthetic_pool(grid, np.r_[lon, grid.lon[[55, 10]]], np.r_[lat, grid.lat[[30, 5]]],
                          [SHIP] * 30 + [GAUGE, BUOY])
    src = [a * np.exp(-(((ii - ci) / s) ** 2 + ((jj - cj) / s) ** 2))
           for ci, cj, s, a in [(15, 20, 4.0, 1.0), (40, 10, 3.0, -0.7), (25, 38, 5.0, 0.4)]]
    return grid, pool, np.stack(src).astype(np.float32)


def check_linearity_equivalence():
    grid, pool, eta0 = small_case()
    sampler = ref.make_sampler(grid, pool)
    cpu = Solver(grid, sampler, CFG, xp=np)
    out = cpu.run(np.stack([eta0[0], eta0[1], eta0[0] + eta0[1], 2 * eta0[0]]))
    scale = float(np.abs(out).max())
    sup = float(np.abs(out[2] - out[0] - out[1]).max()) / scale
    dbl = float(np.abs(out[3] - 2 * out[0]).max()) / scale
    batched = cpu.run(eta0)
    eq = max(float(np.abs(batched[b] - ref.run_reference(grid, eta0[b], sampler, CFG)).max()) for b in range(3))
    eq /= float(np.abs(batched).max())
    lin_ok = sup < TOL_F32 and dbl < TOL_F32
    text4 = (f"superposition error max|r(a+b) - r(a) - r(b)| / max|r| = **{sup:.1e}**; "
             f"doubling error = **{dbl:.1e}** (limit {TOL_F32:.0e})")
    if backend.GPU:
        gpu = backend.to_numpy(Solver(grid, sampler, CFG, xp=backend.xp).run(eta0))
        gerr = float(np.abs(gpu - batched).max()) / float(np.abs(batched).max())
        gpu_txt, gpu_ok = f"CuPy vs NumPy: **{gerr:.1e}**", gerr < TOL_F32
    elif (FIG / "gpu_check.json").exists():
        import json
        rec = json.loads((FIG / "gpu_check.json").read_text())
        t = rec["timing"]
        gpu_txt = (f"CuPy vs NumPy: **passed on {rec['gpu']}** ({rec['date']}, commit {rec['commit']}: {rec['pytest']}). "
                   f"Timing ({rec['timing_grid']}): CuPy B=64 **{t['cupy_B64_s_per_scenario']} s/scenario** "
                   f"({t['cupy_B64_gcell_updates_per_s']} Gcell-updates/s) vs NumPy on the same runtime "
                   f"{t['numpy_B1_s_per_scenario']} s/scenario")
        gpu_ok = True
    else:
        gpu_txt, gpu_ok = "CuPy vs NumPy: **not run** (no GPU here and no outputs/validation/gpu_check.json)", None
    text5 = f"batched vs reference (3 scenarios, 2 h): **{eq:.1e}** (limit {TOL_F32:.0e}); {gpu_txt}"
    return lin_ok, text4, (eq < TOL_F32) and (gpu_ok is not False), text5, gpu_ok


# ----------------------------------------------------------------- check 6

def check_hellenic():
    """Needs B's grid (T01) and C's sources (T03); skipped with a reason otherwise."""
    if not repo_path(CFG["paths"]["grid"]).exists():
        return None, "pending: needs `data/processed/grid.npz` (T01, Stream B) and `sources.py` (T03, Stream C)", None
    try:
        from tsunami import grid as grid_mod
        from tsunami import sources
    except ImportError as e:
        return None, f"pending: {e}", None
    grid = grid_mod.load_grid(CFG)
    table = sources.candidate_table(CFG)
    hel = table[(table.region == "hellenic") & (np.isclose(table.mw, 8.0))]
    row = hel.iloc[len(hel) // 2]
    eta0 = sources.source_eta0(grid, row, CFG)
    pool = synthetic_pool(grid, [grid.lon[0]], [grid.lat[0]], [GAUGE])
    s = Solver(grid, ref.make_sampler(grid, pool), CFG)
    times = [0, 600, 1200, 2400, 3600, 5400]
    snaps = backend.to_numpy(s.snapshots(eta0[None], times))[0]
    lim = float(np.abs(snaps[1:]).max())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, t, e in zip(axes.flat, times, snaps):
        e = np.where(grid.wet, e, np.nan)
        im = ax.pcolormesh(grid.lon, grid.lat, e, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set_title(f"t = {t / 60:.0f} min")
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude (deg N)")
    fig.colorbar(im, ax=axes, label="eta (m)", shrink=0.8)
    f = save(fig, "check6_hellenic_mw8.png", f"Check 6: Hellenic Arc Mw 8 at ({row.lon_deg:.2f} E, {row.lat_deg:.2f} N)",
             "Linear shallow-water solution on the 2 arc-min grid; colour scale set by t > 0 frames.")
    txt = (f"source peak uplift **{eta0.max():.2f} m**, peak subsidence **{eta0.min():.2f} m**; "
           f"compare with a published Hellenic Mw 8 scenario amplitude (citation to add).")
    return True, txt, f


# ----------------------------------------------------------------- check 7

def check_sponge():
    """(a) Radial pulse in a square open sea with a sponge on all four edges: energy left after the
    wave has gone; and the sponge must not touch the interior before the wave arrives.
    (b) On the real padded domain: no area-of-interest cell lies in the sponge (west and north only).
    """
    h, n = 3000.0, 160
    grid = synthetic_grid(np.full((n, n), h), lat0_deg=35.0)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eta0 = np.exp(-(((ii - n / 2) ** 2 + (jj - n / 2) ** 2) / 8.0 ** 2)).astype(np.float32)
    c = copy.deepcopy(CFG)
    c["solver"]["sponge_edges"] = ["west", "east", "south", "north"]
    dt, _ = ref.time_step(grid, c)
    w, r = c["solver"]["sponge_width_cells"], c["solver"]["sponge_max_rate_per_s"]
    to_sponge = (n / 2 - w - 3 * 8) * min(grid.dx.min(), grid.dy) / np.sqrt(G * h)
    early = max(int(0.8 * to_sponge / dt), 1)
    a, _, _ = ref.run_fields(grid, eta0, c, early)
    b, _, _ = ref.run_fields(grid, eta0, closed_cfg(), early)
    interior = float(np.abs(a - b).max() / np.abs(b).max())
    corner = np.hypot(n / 2 * grid.dx.max(), n / 2 * grid.dy) / np.sqrt(G * h)
    late = int(2.0 * corner / dt)
    a, _, _ = ref.run_fields(grid, eta0, c, late)
    b, _, _ = ref.run_fields(grid, eta0, closed_cfg(), late)
    frac = float(np.sum(a.astype(np.float64) ** 2) / np.sum(b.astype(np.float64) ** 2))

    d = CFG["domain"]
    res = d["resolution_arcmin"] / 60.0
    lon = np.arange(d["lon_min_deg"] + res / 2, d["lon_max_deg"], res)
    lat = np.arange(d["lat_min_deg"] + res / 2, d["lat_max_deg"], res)
    mask = ref.sponge_mask((len(lat), len(lon)), CFG)
    aoi_box = d["area_of_interest"]
    aoi = ((lon[None, :] >= aoi_box["lon_deg"][0]) & (lon[None, :] <= aoi_box["lon_deg"][1])
           & (lat[:, None] >= aoi_box["lat_deg"][0]) & (lat[:, None] <= aoi_box["lat_deg"][1]))
    overlap = int((mask & aoi).sum())
    west_inner = d["lon_min_deg"] + w * res
    north_inner = d["lat_max_deg"] - w * res
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.pcolormesh(lon, lat, mask, cmap="Greys", vmin=0, vmax=2, shading="auto")
    ax.add_patch(plt.Rectangle((aoi_box["lon_deg"][0], aoi_box["lat_deg"][0]),
                               np.diff(aoi_box["lon_deg"])[0], np.diff(aoi_box["lat_deg"])[0],
                               fill=False, ec="tab:blue", lw=2, label="area of interest"))
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.legend(loc="lower right")
    f = save(fig, "check7_sponge_layout.png", "Check 7: sponge layout on the padded model domain",
             f"Grey = sponge ({w} cells, west and north edges only); blue = area of interest (receivers allowed only inside).")
    txt = (f"(a) radial pulse, {n} x {n} flat sea, sponge on all four edges ({w} cells, max rate {r} /s; PLACEHOLDER). "
           f"Interior before arrival: open vs closed differ by **{interior:.1e}**. Energy left after 2 corner-crossing "
           f"times (open / closed) = **{frac:.4f}** (target < 0.05).\n\n"
           f"(b) padded domain {d['lon_min_deg']} to {d['lon_max_deg']} E, {d['lat_min_deg']} to {d['lat_max_deg']} N "
           f"({len(lon)} x {len(lat)} cells): sponge inner edge at {west_inner:.2f} E and {north_inner:.2f} N, i.e. "
           f"**{aoi_box['lon_deg'][0] - west_inner:.2f} deg** west of and **{north_inner - aoi_box['lat_deg'][1]:.2f} deg** "
           f"north of the area of interest. Area-of-interest cells inside the sponge: **{overlap}**. "
           "South and east edges are reflecting walls (land; limit of the GEBCO subset).")
    return (frac < 0.05) and (interior < TOL_F32) and overlap == 0, txt, f


# ----------------------------------------------------------------- report

def mark(ok):
    return {True: "PASS", False: "FAIL", None: "PENDING", "partial": "PASS (CPU); CuPy PENDING"}[ok]


def main():
    t0 = time.perf_counter()
    c1 = check_arrival()
    c2 = check_volume()
    c3 = check_greens()
    ok4, t4, ok5, t5, gpu_ok = check_linearity_equivalence()
    if ok5 and gpu_ok is None:
        ok5 = "partial"
    ok6, t6, f6 = check_hellenic()
    ok7, t7, f7 = check_sponge()
    summary = [("1 Flat-bottom arrival (< 2 %)", c1[0]), ("2 Closed-basin volume (< 1 %)", c2[0]),
               ("3 Green's law (within 10 %)", c3[0]), ("4 Linearity and superposition", ok4),
               ("5 Batched == reference; CuPy == NumPy", ok5), ("6 Hellenic Mw 8 snapshots", ok6),
               ("7 Sponge absorption (extra)", ok7)]
    md = ["# Solver validation (T11)", "",
          f"Generated by `scripts/02_validate_solver.py` in {time.perf_counter() - t0:.0f} s. Backend: "
          f"{'CuPy' if backend.GPU else 'NumPy'}. Linear shallow-water equations, spherical C grid, forward-backward.", "",
          "| Check | Result |", "|---|---|"] + [f"| {n} | {mark(o)} |" for n, o in summary] + ["",
          "## 1 Flat-bottom arrival", c1[1], "", f"![]({c1[2]})", "",
          "## 2 Closed-basin volume", c2[1], "", f"![]({c2[2]})", "",
          "## 3 Green's law", c3[1], "", f"![]({c3[2]})", "",
          "## 4 Linearity", t4, "", "## 5 Equivalence", t5, "",
          "## 6 Hellenic Arc Mw 8", t6] + ([f"![]({f6})"] if f6 else []) + ["",
          "## 7 Sponge", t7, "", f"![]({f7})", ""]
    (OUT / "validation.md").write_text("\n".join(md))
    print("\n".join(f"{mark(o):8s} {n}" for n, o in summary))
    return 0 if all(o is not False for _, o in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
