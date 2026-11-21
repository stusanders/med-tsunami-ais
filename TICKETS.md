# TICKETS.md: three parallel workstreams

Read PLAN.md first. Three AI instances work in parallel, each owning one stream. They cannot talk to each other directly: they coordinate through git, `docs/interfaces.md`, and their status files. The human relays anything urgent.

## The three instances

| Stream | Instance | Owns | Merges |
|---|---|---|---|
| **A: Physics core** (lead) | Claude, Pro subscription | interfaces, solver, validation, observation operators, scoring, single-ship baselines | **all merges to `main`** |
| **B: Data, ships and GPU runs** | Claude, second instance | grid, receiver pool, ship sampling, calibration, gauges and buoys, Map 1, library build, truth events, sweep, historic replays, sensitivity | its own branch only |
| **C: Sources, figures, benchmark and report** | Antigravity (Gemini) | earthquake sources, plotting, benchmark harness and runs, Maps 2 and 3, figures, report, pitch materials, historic data gathering | its own branch only |

Each instance may use its own sub-agents for small jobs. On a Pro subscription, use Sonnet as the default model and save Opus for interfaces, review and the hardest debugging, so usage limits are not hit mid-hackathon.

## Rules

1. **Separate working copies.** Stream A works in the main folder `~/projects/med-tsunami-ais` on branch `stream-a` (switching to `main` only to merge). B and C each get their own worktree and branch:
   ```
   git worktree add ../mts-B -b stream-b
   git worktree add ../mts-C -b stream-c
   ```
2. **Shared, untracked folders.** `data/`, `outputs/` and `status/` are gitignored, so a new worktree would not see them. They live only in the main folder, and each worktree links to them:
   ```
   cd ../mts-B && ln -s ../med-tsunami-ais/data data && ln -s ../med-tsunami-ais/outputs outputs && ln -s ../med-tsunami-ais/status status
   cd ../mts-C && ln -s ../med-tsunami-ais/data data && ln -s ../med-tsunami-ais/outputs outputs && ln -s ../med-tsunami-ais/status status
   ```
   Every stream therefore reads the same data and sees every status file immediately, without a merge. Each stream writes only its own output files (named in its tickets), so nothing collides.
3. **Interfaces first.** Nobody writes module code until T00 (`docs/interfaces.md`) is merged to `main`. Code against the contract, not against another stream's unfinished code. If a stream needs another stream's output before it exists, use a small synthetic stand-in that matches the contract.
4. **File ownership.** Each ticket lists its files. Never edit another stream's files; write a request in your status file and the human will relay it.
5. **Integration.** When a ticket passes its checks, push the branch and note it in your status file. Stream A reviews and merges to `main`. Streams B and C pull `main` into their branch at least every two hours, and immediately after each gate.
6. **Status files.** `status/A.md`, `status/B.md`, `status/C.md` (ticket, state, blocker, what it needs from whom, last update), in the shared folder. Each stream writes only its own file and reads the others.
7. **One GPU user at a time:** Stream B runs the GPU queue. A and C ask B (via status file and the human) for GPU time. T4 or L4 only; A100 needs the human's approval. Colab runtime disks are lost when a runtime is deleted, so download every result file before deleting; the library can be rebuilt if lost. Delete runtimes when idle.
8. Constants in `config.yaml`; guesses marked `PLACEHOLDER`; results to files under `outputs/`; tests ship with code; if something fails twice, stop and report.

## Re-timed gates (from Saturday 18:55; Saturday gates shifted, Sunday gates unchanged because the overnight GPU run absorbs the delay)

| Gate | Time | Must be merged |
|---|---|---|
| G0 Interfaces | Sat 19:40 | T00 |
| G1 Solver validated, Map 1 | Sat 22:55 | T01, T02, T04, T07, T09, T10, T11 |
| G2 First detection map, overnight GPU job running | Sun 01:25 | T03, T06, T12, T13, T16, T26a (source parameters); T14 started on GPU |
| G3 Twin check, sweep running | Sun 10:30 | T15, T25; T17 started |
| G4 Maps, benchmark, Samos replay | Sun 13:30 | T17, T18, T19, T26 (Samos) |
| G5 Report and pitch locked | Sun 15:30 | T21, T24, pitch slides |

Confirm the presentation time in the Discord and move G5 if needed.

## Dependency graph

```
G0:  T00 (A)
Then in parallel:
  A: T02 -> T09 -> T11 -> T12 -> T15 -> T25
  B: T01 -> T04 -> T10 (needs C's T07) ; T05 ; T06 ; then T13 (needs T03, T12), T16 (needs T03, T04, T09),
     T14 overnight job (needs T03, T09, T16, T26a), T17, T26, T20
  C: T26a ; T07 (by G1) ; T03 (by G2) ; T08 ; T24 ; then T19, T18, T21, pitch
```

## Priority if time runs short (cut from the bottom)

1. Solver passing its checks; Maps 1 to 3; three-tier benchmark.
2. Single-ship baselines (T25); Samos replay (T26 part 1).
3. Weather regimes; cleaning-steps figure (T24); top-three candidates.
4. Amorgos replay; sensitivity runs (T20); "what the speed makes possible".
5. Crete replay; multi-ship coincidence rule; animation (T22); yaw correction (T23).

---

## Stream A: Physics core (lead)

### T00: Skeleton, config and interface contract (G0)
- **Files:** repo layout from PLAN.md, `config.yaml`, `src/tsunami/backend.py` (`xp` NumPy/CuPy switch), `docs/interfaces.md`, `pyproject.toml`, `.gitignore` (including `data/`, `outputs/`, `status/`), empty `status/A.md`, `status/B.md`, `status/C.md`.
- **Contract must fix:**
  - Grid fields: lon, lat, depth, wet mask, strait mask, `dx` per row, `dy`, `cos(phi)` at centres and v faces, distance to coast. Latitude and longitude stored in degrees, converted to radians inside any trig.
  - C-grid layout: eta and h at centres; u on east-west faces, v on north-south faces; face depth = mean of neighbouring cells; spherical continuity and momentum equations as written in PLAN.md Phase 2.
  - Receiver pool format: positions, depth, lane direction, receiver type (ship, gauge, buoy). Ship receivers read u and v averaged from faces to centres then bilinearly interpolated; gauges and buoys read eta at their snapped cell.
  - Shapes: state `(B, ny, nx)`; solver receiver output `(B, T, R, 3)` for eta, u, v at 30 s; saved library `(C, T=120, R, 3)` at 1 minute, already baseline-filtered, plus raw eta at gauges and buoys.
  - Units: SI throughout (m, s, m/s, radians internally). One helper for knots to m/s (x 0.514444) and one for degrees to radians; no inline conversions.
  - Baseline filter: causal trailing mean over the previous W minutes (W in config), applied identically to observations and templates.
  - Scoring signature (see T15), threshold strata (operator, range cap, weather regime), and signatures for every function in later tickets.
- **Done when:** imports work, `pytest` runs, the human has approved `docs/interfaces.md`.

### T02: Reference solver and analytic tests
- **Files:** `src/tsunami/solver_reference.py`, `tests/test_reference_solver.py`.
- Slow, readable, single-scenario linear shallow-water solver (spherical C grid, forward-backward). Tests: flat bottom, closed basin, 1D sloping shelf.
- **Done when:** arrival time within 2% of sqrt(g h); volume conserved within 1%; shoaling roughly h^(-1/4).

### T09: Batched solver
- **Files:** `src/tsunami/solver.py`, `tests/test_solver_equivalence.py`. **Needs:** T02; T01 and T04 (use synthetic stand-ins until merged).
- `xp` backend; `(B, ny, nx)`; sponge boundaries; CFL step; face-to-centre averaging and bilinear weights for the receiver pool, plus gauge and buoy cells; records eta, u, v every 30 s; helper to block-average to 1 minute and apply the causal baseline filter.
- **Done when:** batched equals reference per scenario (float32 tolerance); superposition test passes; runs on Colab with CuPy.

### T11: Solver validation report (G1)
- **Files:** `scripts/02_validate_solver.py`. **Needs:** T02, T09.
- All checks from PLAN.md Phase 2 plus a Hellenic Mw 8 snapshot sequence. If checks fail at G1, apply the PLAN.md fallback.
- **Done when:** `outputs/validation.md` with pass or fail per check and figures.

### T12: Observation operators and noise
- **Files:** `src/tsunami/observe.py`, `src/tsunami/noise.py`, `tests/test_observation.py`. **Needs:** T05 (values), T09.
- Heading-normal current; heading rounding and wobble (radians) scaled by SOG (m/s); shared 2D noise field projected on heading normal; weather multipliers; 1-minute averaging; causal baseline subtraction; GPS height operator; data-gap and response-factor options (for T20).
- **Done when:** zero noise returns the true current; noise statistics match config; baseline removes a constant drift; a unit test shows the baseline never uses future samples.

### T15: Scoring
- **Files:** `src/tsunami/scoring.py`, `tests/test_scoring.py`. **Needs:** T12.
- Follow PLAN.md Phase 4 "Scoring" exactly: gather active ships from the library on the CPU and project onto each ship's normal; sign-align to the cell lane normal (split crossing lanes into 45 degree bins); average aligned observations and templates identically per cell group; weights `1 / (sigma_shared^2 + sum(sigma_wobble_i^2) / N_c^2)`, empty cells masked; cumulative `gd_t` and `gg_t`; detection score `max(0, gd_t)^2 / (gg_t + eps)`; Aegean candidates also scored flipped (`-gd_t`, same `gg_t`, no extra storage) and counted as extra candidates for the threshold; magnitude and location from the fixed-amplitude log-likelihood `2 gd_t - gg_t` across magnitude bins; top three by log-likelihood; thresholds per (operator, range cap, weather regime) from at least 500 quiet-day calibration draws; false-alarm check on at least 500 held-out draws; confirmation time. Plus a loop-based reference version.
- **Done when:** tensor equals loop version; a noiseless planted source taken from the library is recovered exactly (position and magnitude); a test with two opposite-heading ships in one cell shows the signals add rather than cancel; a sign-flipped template scores zero; a subsidence-first planted Aegean event is detected by its flipped candidate; no NaN at early minutes.

### T25: Single-ship baselines
- **Files:** `src/tsunami/baseline_detector.py`, `tests/test_baseline_detector.py`. **Needs:** T12, T15.
- (i) Inazu's literal rule (any ship over three times its own noise), reporting its measured quiet-day false-alarm rate; (ii) calibrated single-ship rule, threshold k on the maximum of `abs(d_i) / sigma_i` over ships and minutes, set per stratum on the same quiet days so it shares the 1% false-alarm rate; (iii) optional multi-ship coincidence rule if time.
- **Done when:** runs on the same events as T15; a large planted single-ship signal triggers both rules; the literal rule's false-alarm rate is reported.

### T23 (stretch): Yaw correction on Piraeus data.

---

## Stream B: Data, ships and GPU runs

### T01: Grid and masks
- **Files:** `src/tsunami/grid.py`, `tests/test_grid.py`.
- GEBCO subset to 2 arc-minutes; land mask; wet depth clamped to 10 m; spherical metric spacing; distance to coast; depth bands; strait mask (at least Messina, list in config).
- **Done when:** shapes match the contract; `outputs/grid.png`.

### T04: Receiver pool and ship sampling
- **Files:** `src/tsunami/ships.py`, `tests/test_ships.py`.
- Regrid EMODnet density. Build the receiver pool: about 20,000 candidate ship positions (depth >= 50 m, within 100 km of coast, outside masked straits), each with its lane direction, plus gauges and buoys, saved to file. Traffic draws pick subsets with Poisson counts per area; headings along lanes with random direction of travel; speeds from config (converted to m/s); drop ships under 2 knots; AIS range caps.
- **Done when:** pool file saved; a figure of one traffic draw; expected and drawn counts agree; seeded and reproducible.

### T05: Piraeus calibration
- **Files:** `scripts/calibrate_piraeus.py`, `outputs/piraeus_calibration.md`.
- Reporting interval, heading resolution, COG minus HDG spread for steady ships, speed distribution, whether rate of turn exists. Suggested config values.

### T06: Tide gauges and buoys
- **Files:** `src/tsunami/gauges.py`, `tests/test_gauges.py`.
- IOC station list, snapped to wet cells; the two Ionian buoys from T26a; trigger function (threshold plus telemetry delay, separate settings for gauges and buoys).

### T10: Map 1 (G1)
- **Files:** `scripts/01_map1.py`. **Needs:** T01, T04, T07.
- Traffic share by depth band, overall and within each AIS range cap. `outputs/map1.png` plus table.

### T13: First detection map (G2)
- **Files:** `scripts/03_one_scenario.py`. **Needs:** T03, T06, T09, T12.
- One Hellenic Mw 8 event: ships coloured by signal-to-noise, gauge and buoy trigger times. `outputs/first_detection.png`.

### T16: Truth events and quiet days (before G2)
- **Files:** `scripts/make_truth_events.py`. **Needs:** T03, T04, T09.
- Truth source list for the overnight job: one per library candidate, offset half a grid spacing along strike with magnitude +0.1 (about 576, used for Map 2), plus 50 to 100 random off-grid events for the twin check; never identical to library entries. Quiet-day definitions: at least 500 calibration and 500 held-out draws per stratum (no solver runs needed).
- **Done when:** source lists saved; a test confirms no truth source coincides with a library source.

### T14: Overnight GPU job (start before G2)
- **Files:** `scripts/04_build_library.py`. **Needs:** T03, T09, T16, T26a.
- One resumable queue, batched: (1) the library (arc candidates plus the coarse Aegean grid); (2) the truth events from T16; (3) the Samos and Amorgos historic truth runs. Save receiver series block-averaged to 1 minute and baseline-filtered (plus raw eta at gauges and buoys), chunked by candidate, float32 on the runtime disk with a float16 copy downloaded in the background. Never load the whole library onto the GPU. Run log with timings (feeds T19).
- **Done when:** every chunk saved and checksummed; the log shows total GPU time.

### T17: Full sweep (GPU)
- **Files:** `scripts/05_sweep.py`. **Needs:** T14, T15, T16, T25.
- All detectors on every trial: truth events x traffic draws x noise draws x weather regimes x range caps x operators. Thresholds per stratum from the calibration quiet days. Parquet results; twin summary; false-alarm rates on held-out quiet days.

### T26: Historic event replays
- **Files:** `scripts/historic_replays.py`, `outputs/historic_replays.md`. **Needs:** T26a (from C), T14, T15, T25.
- Today's traffic, all detectors, all weather regimes. The historic runs only provide the observations; score each twice, against the arc-only library and against the library with the Aegean grid. (1) Samos 2020, comparing simulated and recorded gauge amplitudes after a Green's law correction `(h_grid / h_gauge)^(1/4)`, accepting a factor of 2; (2) 1956 Amorgos; (3) AD 365 Crete if time. A null result is valid.

### T20: Sensitivity runs
- **Files:** `scripts/07_sensitivity.py`. **Needs:** T17.
- Shared-noise correlation length; uplift fraction (truth events only, library fixed); monopole versus dipole source (truth only); AIS range cap; ship response 50%; data gaps (10% dropped; receivers near the source off). A few informative sources only.

---

## Stream C: Sources, figures, benchmark and report

### T26a: Historic data gathering (start immediately; source parameters needed by G2)
- **Files:** `data/raw/historic/`, update `data/README.md`.
- Published source parameters for the 30 October 2020 Samos earthquake (USGS plus a peer-reviewed model) and the 9 July 1956 Amorgos earthquake (e.g. Okal et al. 2009), including fault type (normal or thrust) and therefore whether the seafloor rose or dropped first; IOC Sea Level Station Monitoring Facility recordings for Aegean gauges including Kos, 30 to 31 October 2020; positions and published detection specifications for the two Ionian deep-sea buoys deployed in 2025. Ask before using any value you cannot source.

### T07: Plotting helpers (by G1, needed for Map 1)
- **Files:** `src/tsunami/viz.py`. Map base with coastline, consistent colours, titles, units, captions.

### T03: Earthquake sources (by G2)
- **Files:** `src/tsunami/sources.py`, `tests/test_sources.py`.
- Candidate positions every 25 km along Hellenic, Calabrian and Cyprus arcs from NEAMTHM18, plus a coarse Aegean crustal grid (about 50 km, kept away from the Samos and Amorgos epicentres); magnitude scaling to a Gaussian dipole (uplift plus landward subsidence, ratio in config, with a monopole option for T20; cite the scaling relation); off-grid perturbation function; a function for custom sources from published parameters, with a polarity option for normal faults (for T26).
- **Done when:** a figure of candidate positions; Mw 8 uplift and subsidence have plausible sizes; the volume of the dipole is reported.

### T08: Benchmark harness
- **Files:** `src/tsunami/bench.py`. GPU-synchronised timer; wall-clock, cell updates per second, achieved bandwidth; JSON output.

### T24: Cleaning-steps figure
- **Files:** `scripts/fig_cleaning_steps.py`. **Needs:** T07 only.
- Self-contained illustration, not taken from the simulation: synthesise one ship's drift locally (a tsunami current, noise, and a turn artefact `v_yaw(t) = rate_of_turn(t) x ship_length`), then three stacked panels: raw; after averaging and subtracting the yaw artefact; after baseline subtraction, with the true current overlaid. The caption states that the main simulation does not model turning and uses post-correction noise levels.

### T19: Benchmarks
- **Files:** `scripts/06_benchmark.py`. **Needs:** T08, T09, T15 (GPU time via B).
- Solver: naive serial, batched CPU, batched GPU. Scoring: loops vs tensor. Separate algorithmic from hardware speed-up. Second layer: what a one-hour budget buys on CPU vs GPU. Outputs: `benchmark.json`, `benchmark.png`, `what_speed_buys.png`.

### T18: Maps 2 and 3
- **Files:** `scripts/06_maps.py`. **Needs:** T06, T07, T17.
- Map 2: smallest magnitude detected in at least 90% of draws (and the 50% level), by source position; one panel per weather regime; AIS and GPS operators; uncertainty across draws. Map 3: warning time ships add by coast, at 40 km and 100 km range caps, against the nearest gauge, the first gauge anywhere, and the first instrument including the Ionian buoys.

### T21: Report (G5)
- **Files:** `scripts/make_report.py`, `outputs/report.html`. **Needs:** T10, T11, T18, T19, T20, T24, T26.
- Question and why; method; assumptions; validation; three maps; detector comparison; historic replays; benchmarks; limitations; one section per judging criterion.

### Pitch slides
- Draft the slide outline from PLAN.md and the plan doc's Demo and pitch section; fill figures as they land.

### T22 (stretch): Animation of the wave crossing the sea with ships lighting up.
