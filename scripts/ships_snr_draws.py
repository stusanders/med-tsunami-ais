"""How many ships feel the showcase Hellenic Mw 8 above 1 and 3 sigma, across many traffic draws?

Reconciles the single-draw T13 result (0 of 682 ships > 1 sigma) with the T24 figure (a Cretan-shelf
pool point at 4.9 sigma): solve candidate 186 once for every pool receiver, then count, per traffic
draw, ships whose peak |heading-normal current| exceeds 1 and 3 times their total calm noise.

Outputs: outputs/ships_snr_draws.md, outputs/ships_snr_draws.png, outputs/ships_snr_draws.json
Run: PYTHONPATH=src python scripts/ships_snr_draws.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsunami import backend, noise, observe, ships, sources
from tsunami.config import load_config, repo_path
from tsunami.grid import load_grid
from tsunami.solver import Solver, make_sampler

CFG = load_config()
OUT = repo_path(CFG["paths"]["outputs"])
CAND = 186            # showcase Hellenic Mw 8 used by T13 and T24
N_DRAWS = 200
REGIME = "calm"


def main():
    grid, pool = load_grid(CFG), ships.load_pool(CFG)
    table = pd.read_csv(repo_path(CFG["paths"]["candidates"]))
    row = table.iloc[CAND].to_dict()
    row.update(uplift_dip_deg=sources.rupture_mean_dip(row["region"], row["dip_deg"], CFG))
    eta0 = sources.source_eta0(grid, row, CFG)
    solver = Solver(grid, make_sampler(grid, pool), CFG)
    recf, _ = solver.run_library(eta0[None], np.nonzero(pool.rtype > 0)[0])
    recf = backend.to_numpy(recf)[0]                                   # (120, R, 3)

    # pool-wide: peak SNR at every ship pool point for both headings along its lane (same |value|)
    shipi = np.flatnonzero(pool.rtype == ships.SHIP)
    lane = np.asarray(pool.lane_dir)[shipi]
    sig_u, sig_v = recf[:, shipi, 1], recf[:, shipi, 2]
    peak_cur = np.abs(sig_u * np.cos(lane) - sig_v * np.sin(lane)).max(axis=0)

    rows, best = [], []
    for k in range(N_DRAWS):
        draw = ships.traffic_draw(pool, CFG, k)
        sig = observe.ais_signal(observe.gather_truth(recf, draw), draw)          # (S, 120)
        tot = np.sqrt(noise.ship_sigma(draw, REGIME, "ais", CFG).astype(float) ** 2
                      + noise.shared_sigma(REGIME, "ais", CFG) ** 2)
        snr = np.abs(sig).max(axis=1) / tot
        depth = np.asarray(pool.depth)[draw.pool_idx]
        rows.append(dict(draw=k, n_ships=len(draw.pool_idx), gt1=int((snr > 1).sum()), gt3=int((snr > 3).sum()),
                         max_snr=float(snr.max()), n_shelf_lt200m=int((depth < 200).sum())))
        best.append(snr.max())
    df = pd.DataFrame(rows)
    sog_med = float(np.median(ships.traffic_draw(pool, CFG, 0).sog))
    tot_med = float(np.sqrt(noise.ship_sigma(type("D", (), {"sog": np.array([sog_med])}), REGIME, "ais", CFG)[0] ** 2
                            + noise.shared_sigma(REGIME, "ais", CFG) ** 2))
    pool_gt1 = int((peak_cur / tot_med > 1).sum())
    pool_gt3 = int((peak_cur / tot_med > 3).sum())
    w = np.asarray(pool.weight)[shipi]
    summ = dict(candidate=CAND, regime=REGIME, n_draws=N_DRAWS, ships_per_draw_mean=float(df.n_ships.mean()),
                frac_draws_any_gt1=float((df.gt1 > 0).mean()), frac_draws_any_gt3=float((df.gt3 > 0).mean()),
                gt1_mean=float(df.gt1.mean()), gt1_max=int(df.gt1.max()), gt3_mean=float(df.gt3.mean()),
                gt3_max=int(df.gt3.max()), max_snr_median=float(df.max_snr.median()),
                max_snr_p90=float(df.max_snr.quantile(0.9)), max_snr_max=float(df.max_snr.max()),
                draw0=rows[0], pool_points_gt1_at_median_speed=pool_gt1, pool_points_gt3_at_median_speed=pool_gt3,
                expected_ships_at_pool_points_gt1=float(w[peak_cur / tot_med > 1].sum()),
                expected_ships_at_pool_points_gt3=float(w[peak_cur / tot_med > 3].sum()),
                pool_ship_points=int(len(shipi)))
    (OUT / "ships_snr_draws.json").write_text(json.dumps(summ, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(df.max_snr, bins=30, color="tab:blue")
    axes[0].axvline(1, color="k", ls="--")
    axes[0].axvline(3, color="k", ls=":")
    axes[0].set_xlabel("best single-ship peak SNR in the draw (-)")
    axes[0].set_ylabel("traffic draws")
    axes[0].set_title(f"Best ship per draw ({N_DRAWS} draws)")
    bins = np.arange(-0.5, max(df.gt1.max(), 3) + 1.5)
    axes[1].hist(df.gt1, bins=bins, alpha=0.7, label="ships > 1 sigma")
    axes[1].hist(df.gt3, bins=bins, alpha=0.7, label="ships > 3 sigma")
    axes[1].set_xlabel("ships above threshold in one draw")
    axes[1].set_ylabel("traffic draws")
    axes[1].legend()
    axes[1].set_title("Ships above 1 and 3 sigma per draw")
    fig.suptitle(f"Hellenic Arc Mw 8 (cand {CAND}), calm AIS noise: single ships across traffic draws")
    fig.text(0.01, 0.01, "Peak |heading-normal current| over 2 h divided by each ship's total filtered noise "
             "(independent + shared). Signal only; no noise realisation.", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "ships_snr_draws.png", dpi=130)
    plt.close(fig)

    md = ["# Single ships above 1 and 3 sigma across traffic draws (Hellenic Mw 8, cand 186, calm)", "",
          f"- {N_DRAWS} traffic draws, mean {summ['ships_per_draw_mean']:.0f} ships per draw.",
          f"- Draws with at least one ship > 1 sigma: **{100 * summ['frac_draws_any_gt1']:.0f}%** "
          f"(mean {summ['gt1_mean']:.2f}, max {summ['gt1_max']} ships); > 3 sigma: "
          f"**{100 * summ['frac_draws_any_gt3']:.0f}%** (mean {summ['gt3_mean']:.2f}, max {summ['gt3_max']}).",
          f"- Best ship per draw: median SNR {summ['max_snr_median']:.2f}, 90th percentile {summ['max_snr_p90']:.2f}, "
          f"max {summ['max_snr_max']:.2f}.",
          f"- The T13 showcase draw (draw 0): {rows[0]['n_ships']} ships, {rows[0]['gt1']} > 1 sigma, max SNR "
          f"{rows[0]['max_snr']:.2f}.",
          f"- Whole pool ({summ['pool_ship_points']} ship points, at the median speed): {pool_gt1} points > 1 sigma, "
          f"{pool_gt3} > 3 sigma; expected ships present at those points per draw: "
          f"{summ['expected_ships_at_pool_points_gt1']:.2f} and {summ['expected_ships_at_pool_points_gt3']:.3f}.", "",
          "The T24 shelf ship (4.9 sigma) is one of these rare high-SNR pool points on the Cretan shelf, chosen "
          "deliberately as the strongest-current point for illustration; it is not a ship from the T13 draw.", "",
          "![](ships_snr_draws.png)"]
    (OUT / "ships_snr_draws.md").write_text("\n".join(md))
    print("\n".join(md[2:7]))


if __name__ == "__main__":
    main()
