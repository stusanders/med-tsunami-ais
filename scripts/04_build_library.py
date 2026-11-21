"""T14: the overnight GPU queue. Library -> truth events -> historic runs, resumable and chunked.

Run on the Colab runtime from the repo root (inside tmux or nohup):
    PYTHONPATH=src python scripts/04_build_library.py [--stages library,truth,historic] [--batch 64] [--max-chunks N]
Every finished chunk is on disk and listed in run_log.csv, so re-running the same command resumes where it stopped.
Progress for the morning:  cat data/build_progress.json;  tail -3 data/library/run_log.csv;  tail -20 data/build.log
"""
import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

from tsunami import backend, library_build as LB
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.ships import BUOY, load_pool
from tsunami.solver import Solver, make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="library,truth,historic")
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--max-chunks", type=int, default=None, help="per stage; for smoke tests")
ap.add_argument("--allow-no-buoys", action="store_true")
ap.add_argument("--root", default=None, help="data root override (smoke tests)")
args = ap.parse_args()

cfg = load_config()
grid, pool = load_grid(cfg), load_pool(cfg)
if not (pool.rtype == BUOY).any() and not args.allow_no_buoys:
    sys.exit("receiver pool has no buoys yet (T26a). The library must be built on the FINAL pool; use --allow-no-buoys to override.")
from tsunami.sources import source_eta0                      # Stream C (T03)

paths = {"library": repo_path(cfg["paths"]["library_dir"]), "truth": repo_path(cfg["paths"]["truth_dir"]),
         "historic": repo_path(cfg["paths"]["historic_runs_dir"])}
tables = {"library": repo_path(cfg["paths"]["candidates"]), "truth": paths["truth"] / "sources.csv",
          "historic": paths["historic"] / "sources.csv"}
if args.root:
    from pathlib import Path
    paths = {k: Path(args.root) / k for k in paths}

sampler = make_sampler(grid, pool)
solver = Solver(grid, sampler, cfg)
gb_idx = np.nonzero(pool.rtype > 0)[0]
batch = args.batch or cfg["solver"]["batch_size"]
R, R_gb = len(pool), len(gb_idx)

stages = []
for name in args.stages.split(","):
    t = pd.read_csv(tables[name])
    if name == "library":
        t = t.assign(run_id=t["cand_id"])
    stages.append((name, t))
per_run = (int(cfg["library"]["n_minutes"]) * R * 3 + solver.n_samples * R_gb) * 4
need = sum(len(t) for _, t in stages) * per_run
free = shutil.disk_usage(paths["library"].parent if paths["library"].parent.exists() else ".").free
print("runs:", {n: len(t) for n, t in stages})
print(f"backend gpu={backend.GPU} device={LB.gpu_name()} R={R} R_gb={R_gb} batch={batch} dt={float(solver.dt):.3f}s n_steps={solver.n_steps}")
print(f"disk: need {need/1e9:.1f} GB (float32), free {free/1e9:.1f} GB")
if free < 1.05 * need:
    sys.exit("not enough disk for the float32 library on this runtime")

prog_path = repo_path("data") / "build_progress.json"
t_all = time.perf_counter()
summary = {"device": LB.gpu_name(), "stages": {}}


def progress(d):
    d["total_elapsed_s"] = time.perf_counter() - t_all
    LB.write_progress(prog_path, d)
    print(json.dumps(d), flush=True)


for name, table in stages:
    n = LB.build_stage(name, table, paths[name], solver, grid, cfg, gb_idx, source_eta0, batch=batch,
                       max_chunks=args.max_chunks, progress=progress)
    log = LB.read_log(paths[name])
    n_ok, bad = LB.verify_stage(paths[name], R, R_gb, cfg, solver.n_samples) if (paths[name] / "checksums.txt").exists() else (0, [])
    summary["stages"][name] = {"runs": len(table), "chunks_written_this_session": n, "chunks_logged": len(log),
                               "gpu_solve_s_total": float(log["solve_s"].sum()), "wall_s_total": float(log["wall_s"].sum()),
                               "files_checksum_ok": n_ok, "files_bad": bad, "gpu_names": sorted(set(log["gpu"]))}
    print(name, summary["stages"][name], flush=True)
summary["total_elapsed_s_this_session"] = time.perf_counter() - t_all
(repo_path("data") / "build_summary.json").write_text(json.dumps(summary, indent=1))
progress({"state": "DONE", **summary})
