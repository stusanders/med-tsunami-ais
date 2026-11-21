"""T16: write the truth-event list, quiet-day list and historic source list.

Outputs: data/truth/sources.csv, data/truth/quiet_days.csv, data/historic_runs/sources.csv.
Needs Stream C's sources.py (candidate_table, perturb, custom_source).
"""
import sys

import pandas as pd

from tsunami import sources, truth_events as TE
from tsunami.config import load_config, repo_path

cfg = load_config()
cand_path = repo_path(cfg["paths"]["candidates"])
cand = pd.read_csv(cand_path) if cand_path.exists() else sources.candidate_table(cfg)
truth = TE.truth_table(cand, sources, cfg)
bad = TE.coincidences(truth, cand, cfg)
if len(bad):
    print(bad)
    sys.exit("truth sources coincide with library sources")
tdir = repo_path(cfg["paths"]["truth_dir"])
tdir.mkdir(parents=True, exist_ok=True)
truth.to_csv(tdir / "sources.csv", index=False)
q = TE.quiet_days(cfg)
q.to_csv(tdir / "quiet_days.csv", index=False)
hist = TE.historic_table(sources, cfg)
hdir = repo_path(cfg["paths"]["historic_runs_dir"])
hdir.mkdir(parents=True, exist_ok=True)
if len(hist):
    hist.to_csv(hdir / "sources.csv", index=False)
else:
    print("WARNING: no `historic_sources` in config yet (T26a); historic stage cannot run")
print(truth.run_kind.value_counts().to_dict(), q.split.value_counts().to_dict(), len(hist), "historic")
