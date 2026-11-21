"""Per-run uplift scale factors for T17/T26 (no library rebuild).

The T14 library, truth and historic runs were built with uplift = 0.3 x slip. The current rule is
uplift = slip x sin(mean dip). eta0 is linear in uplift, so every receiver series of a run scales by
uplift_scale = sin(mean dip) / 0.3 exactly. Multiply rec_*.npy and gb_*.npy rows (and the observations
derived from truth runs) by this factor.

Writes: data/processed/uplift_scale.csv (cand_id), data/truth/uplift_scale.csv (run_id),
        data/historic_runs/uplift_scale.csv (run_id); also refreshes data/processed/candidates.csv.
Run: PYTHONPATH=src python scripts/write_uplift_scales.py
"""
import pandas as pd

from tsunami import sources
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid

CFG = load_config()


def scales(table, key):
    rows = []
    for r in table.to_dict("records"):
        udip = sources.rupture_mean_dip(r["region"], r["dip_deg"], CFG)
        rows.append({key: int(r[key]), "region": r["region"], "uplift_dip_deg": udip,
                     "uplift_scale": sources.uplift_scale(dict(r, uplift_dip_deg=udip), CFG)})
    return pd.DataFrame(rows)


def main():
    cand = sources.candidate_table(CFG, load_grid(CFG), save=True)
    out = {"data/processed/uplift_scale.csv": scales(cand, "cand_id")}
    for folder in ("data/truth", "data/historic_runs"):
        p = repo_path(folder) / "sources.csv"
        if p.exists():
            out[f"{folder}/uplift_scale.csv"] = scales(pd.read_csv(p), "run_id")
    for path, df in out.items():
        df.to_csv(repo_path(path), index=False)
        print(path, len(df), df.groupby("region").uplift_scale.agg(["min", "max"]).round(3).to_dict("index"))


if __name__ == "__main__":
    main()
