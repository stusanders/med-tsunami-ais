"""GPU check for Stream A (runs on the Colab runtime): CuPy-vs-NumPy equivalence tests plus full-size solver timing.

Usage on the runtime:  cd /content/mts && PYTHONPATH=src python scripts/gpu_check.py
Writes outputs/gpu_check.json (GPU name, test result, timings per batch size).
"""
import json
import subprocess
import sys
import time

import numpy as np

from tsunami import backend
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.ships import load_pool
from tsunami.solver import Solver, make_sampler

cfg = load_config()
res = {"gpu": None, "backend_gpu": backend.GPU}
if backend.GPU:
    res["gpu"] = backend.xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
t = subprocess.run([sys.executable, "-m", "pytest", "tests/test_solver_equivalence.py", "-q"], capture_output=True, text=True)
res["pytest_returncode"], res["pytest_tail"] = t.returncode, t.stdout[-800:]
print(t.stdout[-800:], flush=True)

grid, pool = load_grid(cfg), load_pool(cfg)
sampler = make_sampler(grid, pool)
solver = Solver(grid, sampler, cfg)
ny, nx = grid.depth.shape
res.update(ny=ny, nx=nx, R=len(pool), dt_s=float(solver.dt), n_steps=int(solver.n_steps))
gb_idx = np.nonzero(pool.rtype > 0)[0]
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:ny, 0:nx]
timings = []
for B in (8, 32, 64):
    cj, ci = rng.integers(60, ny - 60, B), rng.integers(60, nx - 60, B)
    eta0 = np.stack([np.exp(-(((yy - a) / 4.0) ** 2 + ((xx - b) / 4.0) ** 2)) for a, b in zip(cj, ci)]).astype(np.float32)
    backend.sync()
    t0 = time.perf_counter()
    recf, gb30 = solver.run_library(eta0, gb_idx)
    backend.sync()
    dt = time.perf_counter() - t0
    rec = dict(B=B, seconds=dt, s_per_scenario=dt / B, cell_updates_per_s=B * ny * nx * solver.n_steps / dt,
               finite=bool(np.isfinite(backend.to_numpy(recf)).all()), peak_eta_m=float(np.abs(backend.to_numpy(gb30)).max()))
    timings.append(rec)
    print(rec, flush=True)
    del recf, gb30
res["timings"] = timings
out = repo_path("outputs") / "gpu_check.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
