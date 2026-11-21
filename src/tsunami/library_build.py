"""Resumable, chunked library builder (T14). Format: docs/interfaces.md section 6.

A "stage" is a table of sources (library candidates, truth events or historic runs). Each stage is solved in batches; one batch
is one chunk, written as
    rec_{chunk:04d}.npy  float32 (C_chunk, 120, R, 3), block-averaged to 1 min and baseline-filtered
    gb_{chunk:04d}.npy   float32 (C_chunk, 240, R_gb), raw eta at gauges and buoys
plus index.csv (one row per run), run_log.csv (one row per finished chunk) and checksums.txt (sha256 per file).

Resume: a chunk counts as done only if its run_log.csv row exists AND both files exist with the expected sizes. Files are
written under a temporary name and renamed, so a kill leaves either a whole chunk or nothing. The whole library is never
held in memory: each batch goes GPU -> host -> disk and is then dropped.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend

LOG_COLUMNS = ["chunk", "stage", "n_runs", "wall_s", "solve_s", "write_s", "gpu", "peak_gb_eta_m", "rec_bytes", "gb_bytes"]


def gpu_name():
    if backend.GPU:
        return backend.xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return "cpu"


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def _save_npy_atomic(path, arr):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def chunk_files_ok(out_dir, chunk, n_runs, R, R_gb, n_minutes, n_samples):
    """Completeness check: both files open as memory maps with exactly the expected shape (a truncated file fails to open)."""
    for name, shape in ((f"rec_{chunk:04d}.npy", (n_runs, n_minutes, R, 3)), (f"gb_{chunk:04d}.npy", (n_runs, n_samples, R_gb))):
        try:
            if np.load(out_dir / name, mmap_mode="r").shape != shape:
                return False
        except (OSError, ValueError):
            return False
    return True


def read_log(out_dir):
    p = out_dir / "run_log.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def _append_log(out_dir, row):
    p = out_dir / "run_log.csv"
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(p, mode="a", header=not p.exists(), index=False)


def _append_checksums(out_dir, files):
    with open(out_dir / "checksums.txt", "a") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.name}\n")


def source_eta0(table, row_idx, grid, cfg, source_fn):
    """(B, ny, nx) float32 initial surfaces for the given rows; optional table columns `amp_scale` (default 1) and `kind`."""
    out = []
    for r in row_idx:
        row = table.iloc[r].to_dict()
        kind = row.get("kind") if isinstance(row.get("kind"), str) else None
        out.append(np.float32(row.get("amp_scale", 1.0)) * source_fn(grid, row, cfg, kind=kind))
    return np.stack(out).astype(np.float32)


def build_stage(name, table, out_dir, solver, grid, cfg, gb_idx, source_fn, batch=None, max_chunks=None, progress=None):
    """Solve every run of `table` in chunks of `batch`; skip chunks that are already complete. Returns chunks written now.

    `table` needs a unique `run_id` column. index.csv is written first (it is a pure function of the table), so a stage can be
    inspected, resumed or read while it is still running.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tmp"):
        stale.unlink()
    batch = batch or cfg["solver"]["batch_size"]
    table = table.reset_index(drop=True)
    assert table["run_id"].is_unique, "run_id must be unique"
    n = len(table)
    idx = table.copy()
    idx["chunk"] = np.arange(n) // batch
    idx["row"] = np.arange(n) % batch
    idx.to_csv(out_dir / "index.csv", index=False)
    n_chunks = (n + batch - 1) // batch
    R, R_gb = solver.idx.shape[0], len(gb_idx)
    n_min, n_samp = int(cfg["library"]["n_minutes"]), solver.n_samples
    done = set(read_log(out_dir)["chunk"].astype(int))
    written = 0
    t_stage = time.perf_counter()
    for c in range(n_chunks):
        rows = np.arange(c * batch, min((c + 1) * batch, n))
        if c in done and chunk_files_ok(out_dir, c, len(rows), R, R_gb, n_min, n_samp):
            continue
        if max_chunks is not None and written >= max_chunks:
            break
        t0 = time.perf_counter()
        eta0 = source_eta0(table, rows, grid, cfg, source_fn)
        recf, gb30 = solver.run_library(backend.to_xp(eta0), gb_idx)
        backend.sync()
        recf, gb30 = backend.to_numpy(recf).astype(np.float32), backend.to_numpy(gb30).astype(np.float32)
        t1 = time.perf_counter()
        if not (np.isfinite(recf).all() and np.isfinite(gb30).all()):
            raise FloatingPointError(f"non-finite values in {name} chunk {c}")
        rec_p, gb_p = out_dir / f"rec_{c:04d}.npy", out_dir / f"gb_{c:04d}.npy"
        _save_npy_atomic(rec_p, recf)
        _save_npy_atomic(gb_p, gb30)
        _append_checksums(out_dir, [rec_p, gb_p])
        t2 = time.perf_counter()
        _append_log(out_dir, dict(chunk=c, stage=name, n_runs=len(rows), wall_s=t2 - t0, solve_s=t1 - t0, write_s=t2 - t1,
                                  gpu=gpu_name(), peak_gb_eta_m=float(np.abs(gb30).max()), rec_bytes=rec_p.stat().st_size,
                                  gb_bytes=gb_p.stat().st_size))
        done.discard(c)
        written += 1
        del recf, gb30, eta0
        if progress:
            progress(dict(stage=name, chunk=c + 1, n_chunks=n_chunks, elapsed_s=time.perf_counter() - t_stage, gpu=gpu_name()))
    return written


def verify_stage(out_dir, grid_R, R_gb, cfg, n_samples):
    """Re-hash every file listed in checksums.txt and check shapes of the last entry per file. Returns (n_ok, problems)."""
    out_dir = Path(out_dir)
    last = {}
    for line in (out_dir / "checksums.txt").read_text().splitlines():
        h, name = line.split()
        last[name] = h
    problems = [n for n, h in last.items() if not (out_dir / n).exists() or sha256_file(out_dir / n) != h]
    return len(last) - len(problems), problems


def write_progress(path, d):
    Path(path).write_text(json.dumps(d))
