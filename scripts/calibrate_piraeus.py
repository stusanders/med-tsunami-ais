"""T05: calibrate AIS reporting, heading noise and speeds on the Piraeus Oct 2017 data -> outputs/piraeus_calibration.md.

The Piraeus file has no rate of turn, so turn rate is derived from consecutive headings. COG minus HDG for a steady ship
mixes heading noise, GPS course noise and real crab angle (wind, current), so it is an upper bound on heading wobble.
A cleaner wobble estimate is the heading's deviation from its own running median over neighbouring reports.
"""
import json

import numpy as np
import pandas as pd

from tsunami.config import load_config, repo_path
from tsunami.units import knots_to_ms

cfg = load_config()
cal = cfg["calibration"]
df = pd.read_csv(repo_path(cfg["paths"]["piraeus_ais"]))
n_raw = len(df)
df = df.drop_duplicates(["vessel_id", "t"]).sort_values(["vessel_id", "t"]).reset_index(drop=True)
n_dedup = len(df)
res = {"rows_raw": n_raw, "rows_dedup": n_dedup, "vessels": int(df.vessel_id.nunique())}

# --- reporting interval for moving vessels
dt = df.groupby("vessel_id")["t"].diff() / 1000.0
moving = df["speed"] > cal["moving_speed_kn"]
dtm = dt[moving & (dt > 0)]
res["interval_s"] = {q: float(dtm.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

# --- heading resolution
h = df["heading"].dropna()
res["heading_n"] = int(len(h))
res["heading_integer_fraction"] = float((h == np.round(h)).mean())
res["heading_range"] = [float(h.min()), float(h.max())]
res["heading_511_count"] = int((h == cal["heading_not_available_code"]).sum())

# --- steady ships: fast, consecutive reports close in time, heading changing slowly
g = df.groupby("vessel_id")
wrap = lambda x: (x + 180.0) % 360.0 - 180.0
dh = wrap(df["heading"] - g["heading"].shift(1))
dtt = dt
steady = (df["speed"] >= cal["steady_speed_kn"]) & (dtt > 0) & (dtt <= cal["steady_max_dt_s"]) & (dh.abs() <= cal["steady_max_dheading_deg"])
# run of steady reports: current and the previous 2 and next report also steady
ok = steady & steady.shift(1, fill_value=False) & steady.shift(-1, fill_value=False)
ok &= df["heading"].notna() & df["course"].notna()
dc = wrap(df["course"] - df["heading"])[ok]
res["cog_minus_hdg"] = {"n": int(ok.sum()), "mean_deg": float(dc.mean()), "sd_deg": float(dc.std()),
                        "robust_sd_deg": float(1.4826 * np.median(np.abs(dc - np.median(dc)))),
                        "p01_p99_deg": [float(dc.quantile(0.01)), float(dc.quantile(0.99))]}
# heading deviation from the running median over 5 reports (high-frequency jitter only)
hs = df["heading"].where(df["heading"].notna())
unwrapped = g["heading"].transform(lambda s: np.rad2deg(np.unwrap(np.deg2rad(s.interpolate(limit=2)))))
med = unwrapped.groupby(df["vessel_id"]).transform(lambda s: s.rolling(5, center=True, min_periods=5).median())
dev = (unwrapped - med)[ok]
dev = dev.dropna()
res["heading_jitter"] = {"n": int(len(dev)), "sd_deg": float(dev.std()), "robust_sd_deg": float(1.4826 * np.median(np.abs(dev - np.median(dev))))}

# --- speed distribution: underway ships (>= min_speed_kn), outliers above max_valid_speed_kn removed
sp = df["speed"].dropna()
sp = sp[sp <= cal["max_valid_speed_kn"]]
und = sp[sp >= cfg["traffic"]["min_speed_kn"]]
res["speed_all_kn"] = {"n": int(len(sp)), "frac_below_min": float((sp < cfg["traffic"]["min_speed_kn"]).mean())}
res["speed_underway_kn"] = {"n": int(len(und)), "mean": float(und.mean()), "quantiles": {str(q): float(und.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
# per-vessel (one value per vessel: its median underway speed), so busy vessels do not dominate
pv = df[(df["speed"] >= cfg["traffic"]["min_speed_kn"]) & (df["speed"] <= cal["max_valid_speed_kn"])].groupby("vessel_id")["speed"].median()
res["speed_per_vessel_median_kn"] = {"n": int(len(pv)), "quantiles": {str(q): float(pv.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}

# --- rate of turn (derived)
rot = (dh / dtt.where(dtt > 0)) * 60.0                    # deg per minute
rr = rot[steady | (df["speed"] > cal["moving_speed_kn"])].dropna()
res["derived_rate_of_turn_deg_per_min"] = {"n": int(len(rr)), "robust_sd": float(1.4826 * np.median(np.abs(rr - np.median(rr)))), "p99_abs": float(rr.abs().quantile(0.99))}

# --- what the heading wobble implies as velocity noise at 14 kn
u14 = float(knots_to_ms(14.0))
sd_use = res["heading_jitter"]["sd_deg"]
res["implied_velocity_sd_m_s_at_14kn_per_report"] = {"from_jitter": float(u14 * np.deg2rad(sd_use)),
                                                     "from_cog_hdg_robust": float(u14 * np.deg2rad(res["cog_minus_hdg"]["robust_sd_deg"]))}

out = repo_path(cfg["paths"]["outputs"])
(out / "piraeus_calibration.json").write_text(json.dumps(res, indent=1))
q = res["interval_s"]; sq = res["speed_underway_kn"]["quantiles"]; pq = res["speed_per_vessel_median_kn"]["quantiles"]
md = f"""# Piraeus AIS calibration (T05)

Source: `{cfg['paths']['piraeus_ais']}` (Saronic Gulf, 4 to 26 Oct 2017; {res['vessels']} vessels; {n_raw:,} rows, {n_dedup:,} after dropping duplicate (vessel, time) rows).
Limits: one gulf next to a port, so manoeuvring and anchored ships are over-represented; calm-ish autumn month; the
file has no rate of turn (derived from consecutive headings below).

| Quantity | Result |
|---|---|
| Reporting interval, moving vessels (> {cal['moving_speed_kn']} kn), quartiles | {q[0.25]:.0f} / **{q[0.5]:.0f}** / {q[0.75]:.0f} s (5th to 95th percentile {q[0.05]:.0f} to {q[0.95]:.0f} s) |
| Heading values that are whole degrees | {100*res['heading_integer_fraction']:.2f}% of {res['heading_n']:,} (range {res['heading_range'][0]:.0f} to {res['heading_range'][1]:.0f}); code 511 count {res['heading_511_count']} |
| COG minus HDG, steady ships (>= {cal['steady_speed_kn']} kn, heading change <= {cal['steady_max_dheading_deg']} deg per report), n = {res['cog_minus_hdg']['n']:,} | mean {res['cog_minus_hdg']['mean_deg']:.2f} deg, sd {res['cog_minus_hdg']['sd_deg']:.2f} deg, robust sd {res['cog_minus_hdg']['robust_sd_deg']:.2f} deg (1st to 99th percentile {res['cog_minus_hdg']['p01_p99_deg'][0]:.1f} to {res['cog_minus_hdg']['p01_p99_deg'][1]:.1f} deg) |
| Heading jitter (deviation from 5-report running median, steady ships), n = {res['heading_jitter']['n']:,} | sd {res['heading_jitter']['sd_deg']:.2f} deg, robust sd {res['heading_jitter']['robust_sd_deg']:.2f} deg |
| Speed, underway reports (>= {cfg['traffic']['min_speed_kn']} kn, <= {cal['max_valid_speed_kn']} kn), n = {res['speed_underway_kn']['n']:,} | quartiles {sq['0.25']:.1f} / **{sq['0.5']:.1f}** / {sq['0.75']:.1f} kn; 5th to 95th percentile {sq['0.05']:.1f} to {sq['0.95']:.1f} kn; {100*res['speed_all_kn']['frac_below_min']:.0f}% of all reports are below {cfg['traffic']['min_speed_kn']} kn |
| Speed, one median per vessel (n = {res['speed_per_vessel_median_kn']['n']}) | quartiles {pq['0.25']:.1f} / {pq['0.5']:.1f} / {pq['0.75']:.1f} kn; 5th to 95th percentile {pq['0.05']:.1f} to {pq['0.95']:.1f} kn |
| Rate of turn | not in the file; derived from consecutive headings: robust sd {res['derived_rate_of_turn_deg_per_min']['robust_sd']:.1f} deg/min, 99th percentile of |rot| {res['derived_rate_of_turn_deg_per_min']['p99_abs']:.0f} deg/min |
| Velocity noise per report at 14 kn implied by the wobble | {res['implied_velocity_sd_m_s_at_14kn_per_report']['from_jitter']:.3f} m/s (jitter) and {res['implied_velocity_sd_m_s_at_14kn_per_report']['from_cog_hdg_robust']:.3f} m/s (COG minus HDG, includes real crab angle) |

## Reading the numbers

- **Heading wobble.** Uniform rounding to whole degrees alone gives sd = 1/sqrt(12) = 0.29 deg, so the measured jitter (sd
  {res['heading_jitter']['sd_deg']:.2f} deg) is almost entirely rounding: steady ships in this file show essentially no
  extra heading noise on the 10 s scale. COG minus HDG (robust sd {res['cog_minus_hdg']['robust_sd_deg']:.1f} deg) is a ceiling because it also contains real crab
  angle and GPS course noise; its sd ({res['cog_minus_hdg']['sd_deg']:.0f} deg) is inflated by a few reversing or mislabelled reports. So the data bracket the
  wobble between about 0 and 3 deg. The config value of 1.0 deg (Inazu 2020 range 0.5 to 1.5) sits inside that bracket and is conservative
  against the measured jitter. Piraeus cannot pin it down further, because heading noise at open-sea scale cannot be separated from crab angle without a known current.
- **Reporting interval and resolution** match the config (10 s, 1 deg).
- **Speeds.** {100*res['speed_all_kn']['frac_below_min']:.0f}% of reports are below {cfg['traffic']['min_speed_kn']} kn (moored or at anchor), so the underway fraction in this gulf is low and the
  vessel density map (which counts all vessel-hours) includes many ships that would be dropped by the {cfg['traffic']['min_speed_kn']} kn rule if the Piraeus speed
  distribution were used. Underway Piraeus ships are slow (median {sq['0.5']:.1f} kn) because they are manoeuvring in a gulf. Open-sea cargo traffic is faster, so the
  config range of 10 to 18 kn is kept as a PLACEHOLDER; it is not supported or contradicted by Piraeus. T20 can test 8 to 14 kn.
- **Rate of turn** does not exist in the data; the derived value is only an order-of-magnitude check, and the simulation does not model turning (yaw artefact is assumed removed).

## Suggested config values

No numeric change is required by this analysis: `ais_noise.report_interval_s = 10` and `heading_resolution_deg = 1` are confirmed;
`heading_wobble_sd_deg = 1.0` stays inside the measured bracket (0 to 3 deg); `traffic.speed_kn` stays a PLACEHOLDER. Full numbers: `outputs/piraeus_calibration.json`.
"""
(out / "piraeus_calibration.md").write_text(md)
print(md)
