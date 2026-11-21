"""Compare real AIS drift noise (Piraeus, steady ships) with the simulated AIS noise (T12).

Real: SOG * sin(COG - HDG) on steady underway ships, averaged to 1 minute per vessel, then the
same causal 45-min trailing baseline as the simulation, inside continuous steady runs. What is
left is everything a quiet ocean gives a ship: heading/COG noise, real currents, leeway, crab.
Simulated: total filtered per-ship noise sqrt(ship_sigma^2 + shared_sigma^2) at the same speeds,
for each weather regime.

Outputs: outputs/noise_check_piraeus.md, outputs/noise_check_piraeus.png
Run: PYTHONPATH=src python scripts/noise_check_piraeus.py
"""
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import textwrap

import numpy as np
import pandas as pd

from tsunami import noise
from tsunami.config import load_config, repo_path
from tsunami.units import knots_to_ms

CFG = load_config()
CAL = CFG["calibration"]
W = int(CFG["library"]["baseline_window_min"])
WINDOWS_MIN = [10, 15, 20]       # Saronic Gulf steady runs never last 45 min (max 34), so compare at shorter W
OUT = repo_path(CFG["paths"]["outputs"])
SPEED_BINS_KN = [8.0, 11.0, 14.0, 17.0, 20.0, 30.0]
MIN_REPORTS_PER_MIN = 2          # a minute needs at least this many reports to be averaged
STEADY_DHDG_DEG_PER_MIN = CAL["yaw_steady_rot_deg_per_min"]   # same steady rule as T23
MIN_RUN_EXTRA_MIN = 5            # a run must give at least this many filtered minutes


def robust_sd(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def minute_series():
    df = pd.read_csv(repo_path(CFG["paths"]["piraeus_ais"]), usecols=["t", "vessel_id", "heading", "speed", "course"])
    df = df.drop_duplicates(["vessel_id", "t"]).dropna(subset=["heading", "course", "speed"])
    df = df[(df.speed >= CAL["steady_speed_kn"]) & (df.speed <= CAL["max_valid_speed_kn"])]
    dang = (df.course - df.heading + 180.0) % 360.0 - 180.0
    df = df[np.abs(dang) <= CAL["yaw_max_cog_hdg_deg"]].copy()
    dang = dang.loc[df.index]
    df["d"] = knots_to_ms(df.speed) * np.sin(np.deg2rad(dang))
    df["minute"] = (df.t // 60000).astype(np.int64)
    h = np.deg2rad(df.heading)
    df["hs"], df["hc"] = np.sin(h), np.cos(h)
    g = df.groupby(["vessel_id", "minute"]).agg(d=("d", "mean"), sog=("speed", "mean"), hs=("hs", "mean"),
                                                hc=("hc", "mean"), n=("d", "size")).reset_index()
    g = g[g.n >= MIN_REPORTS_PER_MIN]
    g["hdg"] = np.rad2deg(np.arctan2(g.hs, g.hc))
    return g.sort_values(["vessel_id", "minute"])


def filtered_residuals(g, W=W):
    """Causal 45-min baseline inside continuous steady runs (consecutive minutes, small heading change)."""
    out = []
    for vid, v in g.groupby("vessel_id"):
        m = v.minute.to_numpy()
        dh = np.abs((np.diff(v.hdg.to_numpy()) + 180.0) % 360.0 - 180.0)
        brk = np.r_[True, (np.diff(m) != 1) | (dh > STEADY_DHDG_DEG_PER_MIN)]
        run_id = np.cumsum(brk)
        d, s = v.d.to_numpy(), v.sog.to_numpy()
        for r in np.unique(run_id):
            idx = np.flatnonzero(run_id == r)
            if len(idx) < W + MIN_RUN_EXTRA_MIN:
                continue
            x = d[idx]
            c = np.r_[0.0, np.cumsum(x)]
            k = np.arange(W, len(x))
            y = x[k] - (c[k] - c[k - W]) / W          # same operator as timeseries.baseline_filter
            out.append(pd.DataFrame(dict(vessel_id=vid, y=y, sog=s[idx][k], raw=x[k] - x.mean())))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["vessel_id", "y", "sog", "raw"])


def simulated_sigma(sog_kn, regime, W=W):
    cfg = copy.deepcopy(CFG)
    cfg["library"]["baseline_window_min"] = W
    draw = SimpleNamespace(sog=np.atleast_1d(knots_to_ms(np.asarray(sog_kn, float))))
    ind = noise.ship_sigma(draw, regime, "ais", cfg)
    return np.sqrt(ind.astype(float) ** 2 + noise.shared_sigma(regime, "ais", cfg) ** 2)


def main():
    g = minute_series()
    rows = []
    for w in WINDOWS_MIN:
        res = filtered_residuals(g, w)
        if len(res) < 30:
            continue
        sk = float(res.sog.median())
        row = dict(window_min=w, n_minutes=len(res), n_vessels=res.vessel_id.nunique(), median_kn=round(sk, 1),
                   real_sd=float(res.y.std()), real_robust_sd=robust_sd(res.y))
        for regime in CFG["weather_regimes"]:
            row[f"sim_{regime}"] = float(simulated_sigma(sk, regime, w)[0])
        row["real_over_sim_calm"] = row["real_robust_sd"] / row["sim_calm"]
        rows.append(row)
    tab = pd.DataFrame(rows)
    sim45 = {r: float(simulated_sigma(14.0, r, W)[0]) for r in CFG["weather_regimes"]}

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = tab.window_min.to_numpy()
    for regime, ls in zip(CFG["weather_regimes"], ["-", "--", ":"]):
        ax.plot(x, tab[f"sim_{regime}"], "k", ls=ls, marker=".", label=f"simulated, {regime}")
    ax.plot(x, tab.real_robust_sd, "o-", color="tab:red", label="Piraeus steady ships (robust sd)")
    ax.plot(x, tab.real_sd, "x--", color="tab:red", label="Piraeus steady ships (plain sd)")
    ax.set_xlabel("causal baseline window W (min)")
    ax.set_ylabel("filtered drift noise per ship-minute (m/s)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.set_title("AIS drift noise after 1-min mean and causal baseline: Piraeus vs simulation")
    fig.text(0.01, 0.01, textwrap.fill(f"Piraeus Oct 2017 steady underway ships (>= {CAL['steady_speed_kn']:.0f} kn, heading change <= "
             f"{STEADY_DHDG_DEG_PER_MIN:.0f} deg/min); simulation at the same median speed and the same W. "
             f"The simulation uses W = {W} min, which no Saronic Gulf ship sustains.", 150), fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "noise_check_piraeus.png", dpi=130)
    plt.close(fig)
    md = ["# AIS noise check: Piraeus steady ships vs simulated noise", "",
          "Same processing as the simulation: SOG*sin(COG-HDG) per report, 1-minute means per vessel, then a causal "
          "trailing-mean baseline inside continuous steady runs. No Saronic Gulf ship stays steady for 45 min "
          f"(longest steady run 34 min), so the comparison is made at W = {WINDOWS_MIN} min with the simulated noise "
          "computed at the same W and speed.", "",
          tab.round(3).to_markdown(index=False), "",
          f"Simulated at W = {W} min and 14 kn: calm {sim45['calm']:.3f}, moderate {sim45['moderate']:.3f}, rough {sim45['rough']:.3f} m/s.", "",
          "![](noise_check_piraeus.png)"]
    (OUT / "noise_check_piraeus.md").write_text("\n".join(md))
    print(tab.round(3).to_string(index=False))
    print("sim W=45 at 14 kn", {k: round(v, 3) for k, v in sim45.items()})


if __name__ == "__main__":
    main()
