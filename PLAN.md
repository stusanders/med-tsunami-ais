# PLAN.md: Ships as a Tsunami Network (hackathon build plan)

## How to use this file

Read the whole file first. Work is split across three AI instances (streams A, B and C); see TICKETS.md for who owns what, the gates and how to coordinate. Within your stream, build one ticket at a time. At the end of each phase, run its checks, show me the outputs (figures and numbers) and wait for my go-ahead. Keep the code simple and readable, because I have to explain it to judges. Every physical constant lives in `config.yaml`; mark any value that is a guess as `PLACEHOLDER` with a one-line note. Never invent a number and present it as sourced.

## 0. Judging criteria and how this plan meets them

| Criterion | What we do |
|---|---|
| 1. Correctness and assumptions | Physics checks (travel time, volume conservation, Green's law, linearity); GPU output matches the CPU reference within float32 tolerance; batched output matches one-at-a-time output; every assumption listed with its justification and a sensitivity test; false-alarm rates checked on a held-out set of quiet days, not the set used to calibrate them. |
| 2. Fundamental reshaping | The serial way runs one tsunami model per hypothesis and checks it against observations one at a time. We reshape it in three steps: (a) batch scenarios as an extra array dimension so one stencil update advances hundreds of tsunamis; (b) because the equations are linear, each candidate source (position and magnitude) is simulated once and reused for every traffic draw, noise draw and weather regime, so detection becomes a matrix multiply (candidates x receivers x time) instead of a simulation per event; (c) compute scores for every time step at once with cumulative sums (a parallel prefix scan) instead of looping over minutes. |
| 3. Meaningful speed-up | Three-tier baseline, not a strawman: naive serial NumPy (one scenario, loop over time), vectorised NumPy batched on the MacBook CPU, then batched CuPy on the GPU. Same for scoring: loops vs matrix multiply plus prefix sums. Report wall-clock time, cell updates per second, and achieved memory bandwidth against the GPU's peak. Quote Tsunami-HySEA's published speeds for context. |
| 4. Scientific value | Baselines on the same simulated events: Inazu's literal single-ship three-sigma rule (with its measured fleet-wide false-alarm rate), a calibrated single-ship rule at the same false-alarm rate, coastal tide gauges and the two 2025 Ionian deep-sea buoys; historic event replays (Samos 2020, 1956 Amorgos, optionally AD 365 Crete) with today's traffic, including a check against Samos's recorded wave sizes; three maps answering the question; twin experiment with planted off-grid sources; measured false-alarm rate; uncertainty bands across traffic and noise draws; sensitivity runs on the main assumptions; honest limitations. |

## 1. The question

Where, and for which earthquakes, could the ships already at sea confirm a Mediterranean tsunami faster than tide gauges?

Every large ship broadcasts AIS, including heading (HDG), course over ground (COG) and speed over ground (SOG). A tsunami current pushes ships sideways, so the heading-normal drift `SOG * sin(COG - HDG)` acts as a crude current meter (Inazu et al. 2018, 2020). Mediterranean warnings are currently inferred from the earthquake and confirmed by coastal tide gauges and, since 2025, two deep-sea buoys off eastern Sicily and Calabria. This project is an observing system simulation experiment: simulate many tsunamis, simulate what ships would report, and measure detection skill and timing against tide gauges and the buoys.

The Palermo and Trapani evacuation of 23 October 2026 (Mw 5.5, alert lifted after about an hour once gauges showed nothing) illustrates the cost of relying on seismic inference alone. It is context, not the target case: the ship network targets early offshore confirmation of large tsunamis in the central and eastern Mediterranean, and ships staying quiet cannot cancel an alert for a moderate earthquake.

### Why we are doing this

Inazu et al. (2020) showed, with real AIS data from 13 ships during the 2011 Japanese tsunami, that sideways drift tracks tsunami currents. Their detection limit, however, applies to one ship at one moment (three times its normal noise, about 0.1 to 0.3 m/s), from one very large event in one sea. They could not test smaller tsunamis, other seas, whole fleets, or false-alarm rates. Large tsunamis are too rare to wait for, so we simulate thousands and test a detector that combines many ships, many candidate earthquakes and every minute at once, with a measured false-alarm rate.

What the simulation gives us that real data cannot:
1. Where ships can and cannot sense a tsunami (shallow busy corridors versus deep water).
2. For which earthquakes: the smallest one ships would reliably detect at each source, measured across hundreds of Mediterranean scenarios rather than estimated from one Japanese event.
3. How much warning time ships add over tide gauges and the buoys, coast by coast.
4. How often the detector would raise false alarms.
5. How robust the answer is across many traffic arrangements and noise conditions.
6. What upgrading ships to precise GPS would buy.

What is new: opportunistic AIS heading-normal drift alone, as a matched-filter network in the Mediterranean, with measured false-alarm rates and a tide-gauge and buoy comparison. The closest studies differ: Hossen et al. (2021) assimilated synthetic ship GNSS elevation and velocity for Cascadia with optimal interpolation and found that ship velocity data alone could not recover the reference tsunami, while Foster and Thomas (2024) assessed ship GNSS height for warning in Europe and the Mediterranean. Hossen's velocity result is a warning we must take seriously, so a null result for AIS drift in deep water is an expected and useful outcome.

## 2. Deliverables

Must have:
1. **Map 1:** share of ship traffic in three depth bands (under 50 m, 50 to 200 m, over 200 m). Needs only data, so build it in Phase 1.
2. **Map 2:** smallest magnitude ships can confirm, by source location, for two operators (AIS drift and ship GPS height, 6 to 10 cm threshold, Foster and Thomas 2024) and three weather regimes (calm, moderate, rough). Definition: at each source position, the smallest magnitude detected within the 2-hour window in at least 90% of traffic and noise draws (also show the 50% level).
3. **Map 3:** warning time ships add, by coast, at two AIS range caps (40 km and 100 km), against three benchmarks: the coast's nearest gauge, the first gauge anywhere, and the first instrument anywhere including the two Ionian buoys.
4. **Benchmark:** two GPU workloads timed separately: (A) the batched solver, (B) the observation and scoring tensor. Same code on my MacBook (NumPy) and on the Colab GPU (CuPy).

Also for the pitch: a cleaning-steps figure. The simulation does not model turning, so this figure is a standalone illustration: the figure script synthesises one ship's drift with a made-up turn artefact (apparent sway = rate of turn x ship length), then shows raw, after averaging and yaw correction, and after baseline subtraction, with the true tsunami current overlaid.

Stretch, in order: animation of the wave crossing the sea with ships lighting up; Okada (1985) dislocation sources; yaw correction (Inazu 2020) calibrated on Piraeus AIS; GLS scoring with full covariance; whole-Mediterranean grid; 1 arc-minute grid.

## 3. Environment

- One codebase everywhere: `try: import cupy as xp` / `except ImportError: import numpy as xp`. float32 throughout.
- Develop on this Mac with NumPy on a small or coarse grid. Run large ensembles on Colab via SETUP.md (`colab new -s hack --gpu T4`, or L4 for heavy runs; ask before A100). Copy results back with scp or `colab download`. Delete runtimes when idle.
- Write all results to files (`.npz`, `.parquet`, `.png`). Never rely on printed output from the runtime.
- Commit after each working step. `data/`, `outputs/` and `status/` stay out of git and live in one shared folder linked into every worktree (see TICKETS.md).

Repo layout:

```
config.yaml
src/tsunami/
  backend.py            # xp NumPy/CuPy switch
  grid.py               # bathymetry load, regrid, masks, metrics
  sources.py            # magnitude -> uplift (Gaussian dipole)
  solver_reference.py   # slow single-scenario solver for tests and benchmark
  solver.py             # batched linear shallow-water solver, receiver sampling
  ships.py              # receiver pool, ship sampling
  observe.py, noise.py  # observation operators and noise
  gauges.py             # tide-gauge and buoy receivers and triggers
  scoring.py            # matched-filter scoring, thresholds, confirmation times
  baseline_detector.py  # single-ship rules
  bench.py              # benchmark timers
  viz.py                # all figures
scripts/
  01_map1.py, 02_validate_solver.py, 03_one_scenario.py, 04_build_library.py,
  05_sweep.py, 06_maps.py, 06_benchmark.py, 07_sensitivity.py,
  make_truth_events.py, historic_replays.py, calibrate_piraeus.py,
  fig_cleaning_steps.py, make_report.py
data/ outputs/ status/   (gitignored; shared with the worktrees by symlink)
```

If a file name here and in TICKETS.md ever disagree, TICKETS.md wins.

## 4. Key settings (all in config.yaml)

| Setting | Value | Note |
|---|---|---|
| Domain | lon 12 to 37 E, lat 30 to 42 N | eastern Med incl. Sicily and Calabria |
| Grid | 2 arc-minutes (about 750 x 360 cells) | 1 arc-minute only as stretch |
| Min wet depth | 10 m | so waves reach gauges |
| Ship observation mask | depth >= 50 m | linear model unreliable shallower |
| Simulation length | 2 h, receivers sampled every 30 s, block-averaged to 1 min before saving | |
| Time step | CFL factor 0.5 on min(dx, dy) / sqrt(g h_max) | about 6 to 9 s |
| Open boundaries | sponge layer at domain edges | |
| Magnitudes | Mw 6.5 to 8.5, step 0.25 | |
| Candidate sources | every 25 km along Hellenic, Calabrian and Cyprus arcs (from NEAMTHM18 geometry), plus a coarse Aegean crustal grid (about 50 km spacing) | Aegean grid deliberately offset from the Samos and Amorgos sources |
| Source shape | Gaussian uplift plus a landward Gaussian subsidence lobe (dipole) | subsidence ratio PLACEHOLDER; monopole tested in T20 |
| Template polarity | arc candidates: uplift-first only; Aegean candidates: both uplift-first and subsidence-first (flipped = template x -1, no extra simulation) | Aegean earthquakes are often normal faults (seafloor drops first) |
| Units | SI inside the code; knots x 0.514444 = m/s; degrees converted to radians before any trig | |
| AIS heading | rounded to 1 degree, plus wobble sd 0.5 to 1.5 degrees | Inazu 2020 |
| Averaging window | 1 min | |
| Baseline window | causal trailing mean over the previous 30 to 60 min, subtracted per ship | removes slow drift; same operator applied to templates |
| Shared noise | 2D current field, 5 to 15 cm/s, correlation length 20 to 50 km | PLACEHOLDER |
| Weather regimes | calm, moderate, rough: noise multipliers x1, x1.5, x2 | applied to heading wobble and shared noise; PLACEHOLDER (no published calibration of AIS noise against sea state) |
| AIS range caps | 40 km and 100 km from coast | terrestrial AIS only |
| GPS height noise | 5 cm, threshold 6 to 10 cm | Foster and Thomas 2024 |
| Gauge trigger | abs(eta) > 5 to 10 cm, plus 2 to 5 min telemetry delay | |
| Buoy trigger | 2 Ionian buoys, threshold and delay PLACEHOLDER until sourced | positions from T26a |
| False-alarm rate | 1% per 2-hour window | |
| Quiet-day ensembles | at least 500 calibration and 500 held-out draws per stratum | strata = operator x range cap x weather regime |

## 5. Phases

### Phase 1 (Sat to 22:55, gate G1): data, grid and Map 1

- Load GEBCO for the domain, block-average to 2 arc-minutes. Land where depth <= 0. Clamp wet depths to >= 10 m.
- Spherical metrics per row, with latitude phi and longitude lambda in radians: `dy = R * dphi`, `dx = R * cos(phi) * dlambda`. Keep `cos(phi)` at cell centres and at v faces for the solver.
- Distance to coast for every cell (distance transform), for the AIS range masks.
- Load EMODnet vessel density, regrid to the model grid. Load tide gauges and snap each to its nearest wet cell centre. Add the two Ionian buoys as receivers once T26a supplies their positions.
- Mask narrow straits with strong natural currents and internal waves (at least the Strait of Messina; list any others in config) so no ships are drawn there.
- Build the **receiver pool**: about 20,000 candidate ship positions sampled from vessel density (depth >= 50 m, within 100 km of the coast), plus all gauges and buoys. Store the local lane direction at each pool point (from the density field's principal axis). The library is recorded at these points only, and each traffic draw is a random subset of the pool, so one library serves every traffic draw.
- **Map 1:** traffic share in the three depth bands, overall and within each AIS range cap.
- **Check:** plot bathymetry, land mask, vessel density, gauges and buoys. Map 1 numbers make sense.

### Phase 2 (Sat to 22:55, gate G1): solver and sources

Linear shallow-water equations in spherical coordinates on a staggered Arakawa C grid, no Coriolis, no dispersion:

```
d(eta)/dt + 1/(R cos(phi)) * [ d(h u)/d(lambda) + d(h v cos(phi))/d(phi) ] = 0
du/dt = -g / (R cos(phi)) * d(eta)/d(lambda)
dv/dt = -g / R * d(eta)/d(phi)
```

- C-grid layout: eta and h at cell centres; u on east and west faces; v on north and south faces; face depths are the mean of the two neighbouring cells. Velocity zero on faces touching land.
- Forward-backward stepping (update eta, then u and v from the new eta).
- Batched: arrays shaped (B, ny, nx), so B scenarios step together. Batch size about 50 to 100.
- Receivers: average u and v from faces to cell centres, then precompute bilinear interpolation weights for the receiver pool. Gauges and buoys read eta at their snapped cell. Inside the time loop, store only eta, u and v at receivers every 30 s, giving output (B, T, R, 3).
- Sources: a Gaussian uplift elongated along strike, with a smaller Gaussian subsidence lobe on the landward side, so the source has the dipole shape of a real thrust rupture (horizontal currents follow the gradient of eta, so the missing trough would bias them). Rupture length and width from a published magnitude scaling relation (e.g. Strasser et al. 2010 for subduction interfaces), slip from M0 = mu L W D, peak uplift a fraction of slip and subsidence a fraction of uplift (both PLACEHOLDER). Note in config which relation is used.

**G1 checks (Sat 22:55):**
1. Flat-bottom test: arrival time matches sqrt(g h).
2. Closed basin: total water volume conserved to within 1%.
3. Sloping shelf (1D test): wave height grows roughly as h^(-1/4) (Green's law).
4. Linearity: doubling the uplift doubles every receiver output, and the sum of two sources' outputs equals the output of both sources together (this justifies the library approach).
5. Equivalence: batched output equals one-at-a-time output, and CuPy output equals NumPy output to float32 tolerance. Keep a slow, simple reference solver (`solver_reference.py`) for these tests and the benchmark baseline.
6. A snapshot sequence of one Hellenic Arc Mw 8 scenario, plus peak uplift compared with a published scenario amplitude.

**If the checks fail at G1:** fall back to travel times from sqrt(g h) (fast marching), wave direction from the travel-time gradient, heights from Green's law with r^(-1/2) spreading, and a 10 to 20 minute wavelet for the time series. Everything downstream stays the same.

### Phase 3 (Sun to 01:25, gate G2): ships, gauges and the first detection map

- Ship positions: each traffic draw selects a random subset of the receiver pool, with the number of ships per area set by Poisson draws from vessel density (expected count per cell = monthly vessel-hours / hours in month). Heading along the pool point's lane direction, with random direction of travel. Speed from the Piraeus distribution, or 10 to 18 knots as a placeholder (converted to m/s). Drop ships under 2 knots.
- Keep ships with depth >= 50 m and within the AIS range cap.
- AIS operator: heading-normal current `u . n_hat`, plus noise. Heading error (in radians) becomes velocity error `SOG * delta_theta` (SOG in m/s). Add the shared 2D noise field projected on each ship's heading normal, varying slowly in time. Then 1-minute averaging and causal baseline subtraction (mean over the previous 30 to 60 minutes only, never future data).
- Yaw correction: a turning ship pivots near its bow while its GPS antenna sits near the stern, so the stern swings out and the ship appears to slide sideways. Inazu et al. (2020) subtract this apparent slide (rate of turn x ship length). We do not simulate turning; instead the noise levels in config are Inazu's post-correction values (three-sigma floor about 0.1 to 0.3 m/s), which assumes a real system applies the correction. State this assumption in the report.
- GPS operator: eta at the ship plus 5 cm noise.
- Gauges and buoys: eta at each receiver; trigger time from the threshold plus delay.
- **G2 check (Sun 01:25):** for one Hellenic Mw 8 scenario, a map of ships coloured by per-ship signal-to-noise ratio, plus gauge and buoy trigger times.

**If late at G2:** one source region (Hellenic Arc) and one traffic draw only.

### Phase 4 (overnight to Sun 10:30, gate G3): library and scoring

**Overnight GPU job (one queue, run by Stream B):**
- Library: every candidate source (about 64 arc positions x 9 magnitudes, plus the coarse Aegean grid). Before saving, block-average to 1 minute (T = 120) and apply the same causal baseline filter used on the data to u, v and eta (both operations are linear, so filtering the library once is identical to filtering each ship's projected template later). Keep raw eta at gauges and buoys for their triggers. Save in chunks by candidate, resumable, float32 on the runtime disk (about 576 x 120 x 20,000 x 3 x 4 bytes, roughly 16.6 GB), plus a float16 copy downloaded in the background. Never load the whole library onto the GPU.
- Truth events: **separate** runs, never library entries. One truth event per library candidate, offset by half a grid spacing along strike with magnitude perturbed by +0.1 (about 576 runs, these populate Map 2), plus 50 to 100 random off-grid events for the twin check. These run in the same overnight job, since a batch of 100 scenarios takes minutes on the GPU.
- Historic sources: the Samos and Amorgos truth runs (from T26a parameters) also run overnight.

**Scoring (one traffic draw at a time):**
1. Gather the active ships' pool points from the library on the CPU, project each ship's u and v onto that ship's own heading normal, then move only this slice to the GPU (well under 1 GB).
2. Sign-align before averaging: within each 25 km cell, multiply each ship's observation and projected template by `sign(n_i . n_cell)`, where `n_cell` is the cell's lane normal, so ships travelling in opposite directions along a lane add their signals instead of cancelling. Where a cell holds crossing lanes, group ships by cell and by folded normal direction (45 degree bins) and treat each group as its own cell.
3. Average aligned observations and aligned templates within each cell group (identical operations on both). Weight each cell by `w_c = 1 / sigma_c^2`, with `sigma_c^2 = sigma_shared^2 + sum_i(sigma_wobble_i^2) / N_c^2`, since averaging N_c ships reduces only the independent part of the noise. Empty cells are masked.
4. For every candidate k and minute t, accumulate with cumulative sums over time: `gd_t = cumsum(w * g * d)` and `gg_t = cumsum(w * g * g)` (sums over cells, then over minutes up to t). The cell sums for every candidate, event and noise draw are one batched matrix multiply.
5. Detection score: `S = max(0, gd_t)^2 / (gg_t + eps)`. Clamping before squaring means an anti-correlated waveform scores zero rather than a large positive value, and `eps` (set in config) prevents 0/0 before the wave reaches any ship. The detection statistic is the maximum of S over all candidates.
   - Because clamping would otherwise make a subsidence-first wave invisible, every Aegean candidate is also scored in flipped form (`-g`). Linearity means this needs no extra simulation or storage: the flipped version's `gd_t` is just `-gd_t`, and `gg_t` is unchanged. Flipped candidates count as separate candidates when setting the quiet-day threshold.
6. Location and magnitude estimate: the normalised score ignores template size, so a small event could fit a large template scaled down. To estimate magnitude, use the fixed-amplitude log-likelihood `L = 2 * gd_t - gg_t` across the 9 magnitude bins at each position, and report the best (position, magnitude) and the top three by L as the uncertainty region.
7. Keep a loop-based reference version for the equivalence test and benchmark.

**Thresholds:** for each stratum (operator x range cap x weather regime), the 99th percentile of the maximum score over all candidates and times on at least 500 quiet-day calibration draws. Quiet days need no solver runs (eta = u = v = 0), so they are cheap. Check the false-alarm rate on a separate set of at least 500 held-out quiet days.

**Confirmation time:** first minute the detection statistic clears the threshold.

**Single-ship baselines (run on every event alongside the network detector):**
- (i) Inazu's literal rule: alarm when any ship's cleaned drift exceeds three times its own noise. Report its measured false-alarm rate on quiet days; with thousands of ships and 120 minutes it will be close to 100%, which shows why a one-ship rule does not scale to a fleet.
- (ii) Calibrated single-ship rule: alarm when the maximum over ships and minutes of `abs(d_i) / sigma_i` exceeds a threshold k set on the quiet days of the same stratum, so it has the same 1% false-alarm rate as the network detector.
- (iii) Optional, if time: a multi-ship coincidence rule (at least N ships within 50 km over a calibrated threshold within 5 minutes; N PLACEHOLDER), in the spirit of Foster and Thomas (2024).

### Phase 5 (Sun to 13:30, gate G4): full sweep on the GPU

Sweep truth events x at least 5 traffic draws x at least 20 noise draws x 3 weather regimes x 2 range caps x 2 operators. Thresholds come from each (operator, range cap, weather regime) stratum's own quiet-day calibration set. Save per-event results to parquet.

**G3/G4 check:** twin experiment recovers planted sources (location and magnitude errors), and held-out quiet days stay at or under the 1% false-alarm rate.

**If scoring fails by G3 (Sun 10:30):** report per-ship detection only.

### Phase 6 (Sun to 15:30, gate G5): maps, benchmark and report

- Maps 1 to 3 as PNGs, plus a single `report.html` with captions.
- Benchmark A (solver): naive serial NumPy, batched NumPy on the MacBook CPU, batched CuPy on the GPU, at several batch sizes. Report wall-clock time per scenario, cell updates per second, and achieved memory bandwidth as a share of the GPU's peak.
- Benchmark B (scoring): loop-based reference vs matrix multiply plus prefix sums, on CPU and GPU.
- Report the two benchmarks separately, and state which part of each speed-up comes from reshaping (algorithm) and which from hardware.
- "What the speed makes possible": for a fixed budget (e.g. one hour), the size of experiment affordable on CPU versus GPU (traffic draws, noise draws, weather regimes, grid resolution), and what that does to the uncertainty in the maps.
- Historic event replays (core, not stretch): replay historic earthquakes with today's ship traffic and report what the network would have detected, how fast and how accurately. A null result is still a result. The custom Samos and Amorgos runs only generate the synthetic observations; they are never added to the library (that would be the inverse crime). Each replay is scored twice: against the arc-only library (showing the blind spot of an arc-restricted catalogue, compared with the source-agnostic single-ship rule) and against the library with the coarse Aegean grid (separating what the ships can sense from what the library covers). Priority order:
  1. Samos-like, about Mw 7.0, 30 October 2020: the only one with modern recorded wave heights, so also compare simulated gauge amplitudes with recorded ones (about 10 cm at Kos). The 2 arc-minute grid stops at 10 m depth and has no harbour, so apply a Green's law correction `(h_grid / h_gauge)^(1/4)` and accept agreement within a factor of 2.
  2. 1956 Amorgos-like (Aegean): the largest Mediterranean tsunami of the 20th century.
  3. If time: AD 365 Crete-like, about Mw 8.3 to 8.5 on the Hellenic Arc, inside the library's source region.
  Keep Samos and Amorgos source parameters in config with sources cited, including fault type (both are believed to be normal faults, i.e. subsidence-first; T26a confirms). Model their truth sources with the correct polarity.
- Comparison table: network detector vs single-ship rules (literal and calibrated) vs tide gauges vs buoys, by smallest detectable magnitude, detection time and false-alarm rate.
- Sensitivity runs (small, on the most informative sources): shared-noise correlation length, uplift fraction (scaled on truth events only, library fixed, so it tests a source-size mismatch), monopole versus dipole source, AIS range cap, ship response factor (50%), and AIS data gaps (10% dropped; receivers near the source switched off). Noise level is already covered by the weather regimes. Show how Map 2 shifts.
- **If no GPU:** small ensemble on CPU, GPU speed-up as a projection from a timed sample.

## 6. Assumptions and limitations (put these in the report)

Those marked (S) are tested in the sensitivity runs (T20).

1. **Noise levels assume the yaw correction is applied.** "Simulated noise levels are taken from Inazu et al. (2020), measured after their yaw correction. We therefore assume a real system would apply that correction; without it, noise would be higher and detection harder."
2. Noise calibrated on calm conditions off Japan. Results are reported for three weather regimes (noise x1, x1.5, x2); the multipliers are placeholders, as no published study measures how AIS heading noise grows with sea state.
3. Ships move with the water (S: ships feel 50% of the current).
4. AIS data complete (S: 10% of reports dropped; receivers near the source switched off).
5. Linear shallow-water equations, no Coriolis or dispersion; ship readings shallower than 50 m masked.
6. Gaussian dipole sources sized by magnitude, not a full Okada dislocation (S: uplift size varied; monopole versus dipole).
7. Synthetic traffic from density maps, lane headings, Piraeus speeds.
8. The library covers the three arcs and a coarse Aegean grid only; earthquakes elsewhere are detected, if at all, by templates that do not match them.
9. Cell weights use a diagonal approximation to the noise; correlated noise between neighbouring cells is not modelled (full covariance is a stretch goal).
10. Not modelled: captains reacting to the earthquake, GPS jamming in the eastern Mediterranean, internal waves in narrow straits, landslide and volcanic sources.
11. Ships confirm large waves only; their silence cannot cancel an alert.

## 7. Guardrails

- No magic numbers in code.
- Every figure has a title, labelled axes with units, and a one-line caption.
- Delete GPU runtimes when idle. Ask before using A100.
- If something fails twice, stop and tell me rather than piling on workarounds.
