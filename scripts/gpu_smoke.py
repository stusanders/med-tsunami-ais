"""Smoke test of the library queue on the real grid and pool with synthetic Gaussian sources (no sources.py needed).

Run on the runtime from the repo root:  PYTHONPATH=src python scripts/gpu_smoke.py [out_dir] [n_runs]
Checks: GPU name, memory, time per chunk, disk write speed, interrupt + resume, checksum verification.
Writes <out_dir>/smoke_report.json (small; download this, not the arrays).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsunami import backend, library_build as LB
from tsunami.config import load_config
from tsunami.grid import load_grid
from tsunami.ships import load_pool
from tsunami.solver import Solver, make_sampler

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/smoke")
n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 130
gpu_batch = int(sys.argv[3]) if len(sys.argv) > 3 else 16
cfg = load_config()
grid, pool = load_grid(cfg), load_pool(cfg)
solver = Solver(grid, make_sampler(grid, pool), cfg)
gb_idx = np.nonzero(pool.rtype > 0)[0]
ny, nx = grid.depth.shape
rng = np.random.default_rng(0)
jj, ii = np.nonzero(grid.wet & (grid.depth > 500))
pick = rng.choice(len(jj), n_runs, replace=False)
table = pd.DataFrame({"run_id": np.arange(n_runs), "cj": jj[pick], "ci": ii[pick]})
YY, XX = np.mgrid[0:ny, 0:nx]


def source_fn(g, row, cfg, kind=None):
    return np.exp(-(((YY - row["cj"]) / 4.0) ** 2 + ((XX - row["ci"]) / 8.0) ** 2)).astype(np.float32)


if out.exists():
    shutil.rmtree(out)
rep = {"gpu_batch": gpu_batch, "device": LB.gpu_name(), "R": len(pool), "R_gb": len(gb_idx), "n_runs": n_runs, "dt_s": float(solver.dt), "n_steps": solver.n_steps}
if backend.GPU:
    free, total = backend.xp.cuda.runtime.memGetInfo()
    rep["gpu_mem_total_gb"] = total / 1e9
rep["disk_free_gb_before"] = shutil.disk_usage(out.parent).free / 1e9
t0 = time.perf_counter()
n1 = LB.build_stage("smoke", table, out, solver, grid, cfg, gb_idx, source_fn, max_chunks=1, gpu_batch=gpu_batch)     # "killed" after one chunk
rep["first_chunk_wall_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
n2 = LB.build_stage("smoke", table, out, solver, grid, cfg, gb_idx, source_fn, gpu_batch=gpu_batch)                    # resume
rep["resume_wall_s"] = time.perf_counter() - t0
n3 = LB.build_stage("smoke", table, out, solver, grid, cfg, gb_idx, source_fn, gpu_batch=gpu_batch)                    # nothing left
rep.update(chunks_first=n1, chunks_resume=n2, chunks_third=n3)
log = LB.read_log(out)
rep["log"] = log.to_dict("records")
rep["write_MB_per_s"] = float((log.rec_bytes + log.gb_bytes).sum() / 1e6 / log.write_s.sum())
t0 = time.perf_counter()
n_ok, bad = LB.verify_stage(out, len(pool), len(gb_idx), cfg, solver.n_samples)
rep.update(checksum_ok=n_ok, checksum_bad=bad, verify_s=time.perf_counter() - t0, disk_free_gb_after=shutil.disk_usage(out.parent).free / 1e9)
rec0 = np.load(out / "rec_0000.npy", mmap_mode="r")
rep["rec0_shape"] = list(rec0.shape)
rep["rec0_peak_abs"] = float(np.abs(rec0[:4]).max())
(out / "smoke_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "log"}, indent=1))
