# docs/interfaces.md: the interface contract (T00)

Every stream codes against this file, not against another stream's unfinished code. Only Stream A changes it. A stream that needs a change writes the request in its status file. If this file and PLAN.md disagree on an interface, this file wins. If it disagrees with TICKETS.md on file names, TICKETS.md wins.

## 0. Environment and repo rules

- Python 3.13, using the shared venv `~/projects/med-tsunami-ais/.venv`. Do **not** `pip install -e .`: the shared venv would then import main's `src/` from inside a worktree. Run from your own folder:
  `PYTHONPATH=src ../med-tsunami-ais/.venv/bin/python scripts/x.py`, and `../med-tsunami-ais/.venv/bin/python -m pytest` (pytest picks up `src/` from `pyproject.toml`).
- Backend: `from tsunami.backend import xp, to_numpy, to_xp, sync, array_module`. Never import cupy anywhere else. Library functions take arrays and use `array_module(a)`, so they work with either NumPy or CuPy. `TSUNAMI_BACKEND=numpy` forces NumPy.
- Default dtype is float32 for all fields and series. Coordinates, metrics and accumulators are float64.
- Config: `from tsunami.config import load_config, magnitudes`. Every key carries its unit as a suffix. No numeric constant appears in code without a config key, apart from 0, 1, 2 and pi.
- Seeds: `np.random.default_rng([cfg["meta"]["seed"], PURPOSE, index])`, with PURPOSE codes `1` traffic, `2` AIS noise, `3` GPS noise, `4` truth perturbation, `5` quiet-day traffic, `6` quiet-day noise, `7` gaps.
- Shared folders (gitignored, symlinked into the worktrees): `data/` (inputs and generated arrays), `outputs/` (figures, reports, tables), `status/`.

### Module ownership

| Module | Owner | Ticket |
|---|---|---|
| `backend.py`, `units.py`, `geometry.py`, `timeseries.py`, `config.py` | A | T00 (implemented now) |
| `solver_reference.py`, `solver.py` | A | T02, T09 |
| `synthetic.py` (stand-in grid and pool matching sections 2 and 4, for tests only) | A | T02 |
| `observe.py`, `noise.py` | A | T12 |
| `scoring.py` | A | T15 |
| `baseline_detector.py` | A | T25 |
| `grid.py`, `ships.py`, `gauges.py` | B | T01, T04, T06 |
| `sources.py`, `viz.py`, `bench.py` | C | T03, T07, T08 |

## 1. Units and sign conventions

| Quantity | Unit | Sign or convention |
|---|---|---|
| lon, lat | **degrees** (stored) | east, north; converted to radians inside trig only |
| every other angle (heading, lane direction, strike, wobble) | **radians** inside the code | compass bearing, clockwise from true north |
| depth `h` | m | **positive down**; land = 0 |
| `eta` | m | positive up |
| `u`, `v` | m/s | `u` eastward, `v` northward |
| time | s after origin time; minute index `m` = minute ending at 60(m+1) s | |
| speeds from config | knots, converted via `units.knots_to_ms` (x 0.514444) | |
| config angles | degrees (`_deg`), converted via `units.deg2rad` | |

These are the only conversion helpers (`tsunami/units.py`): `knots_to_ms`, `ms_to_knots`, `deg2rad`, `rad2deg`, `km_to_m`. No inline factors anywhere.

**Heading normal** (`tsunami/geometry.py`): for heading `theta`, the heading unit vector is `(sin theta, cos theta)` (east, north). The starboard normal is `n = (cos theta, -sin theta)`. The observed drift `SOG * sin(COG - HDG)` equals `u . n = u cos theta - v sin theta` (`project_normal`), which is positive when the ship is pushed to starboard. `fold_axis(theta)` maps a direction to an axis in [0, pi).

## 2. Grid (T01, B): `data/processed/grid.npz`

`grid.load_grid(cfg) -> Grid` and `grid.build_grid(cfg) -> Grid` (which also saves the file). `Grid` is a dataclass with these fields:

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `lon` | (nx,) | f64 | cell-centre longitude, deg, ascending |
| `lat` | (ny,) | f64 | cell-centre latitude, deg, **ascending (row 0 = south)** |
| `dlon_deg`, `dlat_deg` | scalar | f64 | 2 arc-min = 1/30 deg |
| `depth` | (ny, nx) | f32 | m, positive down, wet cells >= `min_wet_depth_m`, land 0 |
| `wet` | (ny, nx) | bool | `depth > 0` after clamping and exclusion boxes |
| `strait` | (ny, nx) | bool | cells inside `domain.straits` (no ships drawn there) |
| `dx` | (ny,) | f64 | m, `R cos(phi_j) dlambda` at centre rows |
| `dy` | scalar | f64 | m, `R dphi` |
| `cos_c` | (ny,) | f64 | `cos(phi)` at cell centres (also the u-face rows) |
| `cos_v` | (ny+1,) | f64 | `cos(phi)` at v faces; `cos_v[j]` is the south face of row j |
| `coast_dist` | (ny, nx) | f32 | m from the cell centre to the nearest land cell centre (0 on land) |
| `depth_band` | (ny, nx) | int8 | 0: h < 50 m, 1: 50 to 200 m, 2: > 200 m, -1: land |

Indexing is always `[j, i] = [lat, lon]`. `R` is `physics.earth_radius_m`. The grid covers the padded model domain (about 795 x 390 cells at 2 arc-min). `wet` keeps only cells connected to the open Mediterranean: land-locked below-sea-level depressions (Qattara, Dead Sea) are land.

## 3. C-grid layout and equations (T02, T09, A)

- `eta`, `h` at centres: `(B, ny, nx)`.
- `u` on east-west faces: `(B, ny, nx+1)`. `u[:, j, i]` sits on the **west** face of cell `(j, i)`, between `i-1` and `i`.
- `v` on north-south faces: `(B, ny+1, nx)`. `v[:, j, i]` sits on the **south** face of cell `(j, i)`, between `j-1` and `j`.
- Face depths: `hu[j, i] = (h[j, i-1] + h[j, i]) / 2` when both cells are wet, otherwise 0; `hv` likewise. Faces touching land, and the outer domain faces, have `u = v = 0` at all times. Face masks are `mu` and `mv`.
- Forward-backward step (eta first, then u and v from the new eta):
  ```
  eta[j,i] -= dt * ( (hu*u)[j,i+1] - (hu*u)[j,i] ) / dx[j]
            + dt * ( (hv*v*cos_v)[j+1,i] - (hv*v*cos_v)[j,i] ) / (dy * cos_c[j])
  u[j,i]   -= dt * g * (eta[j,i] - eta[j,i-1]) / dx[j]        # interior wet faces
  v[j,i]   -= dt * g * (eta[j,i] - eta[j-1,i]) / dy            # interior wet faces
  ```
  These are the spherical equations in PLAN.md Phase 2, with `dx = R cos(phi) dlambda` and `dy = R dphi`.
- Time step: `dt_cfl = cfl * min(min(dx), dy) / sqrt(g * max(depth))`. Then `n_sub = ceil(30 / dt_cfl)` and `dt = 30 / n_sub`, so every 30 s sample falls exactly on a step.
- Sponge: within `sponge_width_cells` of an outer edge listed in `solver.sponge_edges` (`[west, north]`), multiply eta, u and v by `exp(-gamma * dt)` after each step. `gamma` rises quadratically from 0 at the inner edge of the band to `sponge_max_rate_per_s` at the outer edge. South and east edges are reflecting walls (land, and the limit of the GEBCO subset).
- **Domain padding:** the model domain (`domain.lon/lat_min/max_deg`: 10.5 to 37 E, 30 to 43 N) is the area of interest (`domain.area_of_interest`: 12 to 37 E, 30 to 42 N) padded by 1.5 deg west and 1 deg north, so the sponge lies entirely outside the area of interest. `solver_reference.sponge_mask(shape, cfg) -> (ny, nx) bool` marks damped cells.
- Initial condition: `eta(t=0) = eta0` (the source, instantaneous uplift) and `u = v = 0`.

## 4. Receiver pool (T04, B): `data/processed/receiver_pool.npz`

`ships.load_pool(cfg) -> ReceiverPool`. This is a dataclass of length-R arrays. Ships come first, then gauges, then buoys.

| Field | dtype | Meaning |
|---|---|---|
| `lon`, `lat` | f64 | deg. Ships: exact sampled position. Gauges and buoys: their **snapped wet cell centre** |
| `depth` | f32 | m at the position (bilinear for ships, cell value for gauges and buoys) |
| `lane_dir` | f32 | rad, lane axis folded to [0, pi). NaN for gauges and buoys |
| `rtype` | int8 | `0` ship, `1` gauge, `2` buoy (constants `SHIP, GAUGE, BUOY` in `ships.py`) |
| `code` | `<U24` | station code for gauges and buoys, `""` for ships |
| `cell_j`, `cell_i` | int32 | model cell containing the point (the snapped cell for gauges and buoys) |
| `coast_dist` | f32 | m |
| `weight` | f32 | ships: expected vessels present at this pool point (for Poisson draws); 0 otherwise |
| `area_id` | int32 | traffic Poisson area (the scoring cell via `geometry.cell_index`); -1 otherwise |

Ship pool points satisfy depth >= `min_ship_depth_m`, coast distance <= `pool_max_coast_dist_m` and not in a strait. **Every receiver (ship, gauge, buoy) must lie inside `domain.area_of_interest` and outside `sponge_mask`**; gauges or buoys that fail this are dropped (and listed in the T06 output), never snapped into the sponge. `R_gb` means the gauge and buoy receivers, `pool.rtype > 0`.

**Sampling at receivers** (the solver precomputes this once): `solver.make_sampler(grid, pool) -> Sampler(idx: int64 (R, 4), w: float32 (R, 4))`, flat indices into `ny*nx`.
- Ships: first average u and v from faces to centres (`uc[j,i] = (u[j,i] + u[j,i+1]) / 2`, `vc[j,i] = (v[j,i] + v[j+1,i]) / 2`), then interpolate eta, uc and vc **bilinearly** between the 4 surrounding centres.
- Gauges and buoys: the snapped cell, with weights `[1, 0, 0, 0]`. The output format is therefore identical for every receiver. Only the eta channel is used for gauges and buoys.

## 5. Traffic draw (T04, B)

`ships.sample_traffic(pool, cfg, rng) -> TrafficDraw` draws ships within 100 km of the coast. `ships.apply_range_cap(draw, pool, cap_m) -> TrafficDraw` returns the subset with `coast_dist <= cap_m`, so the 40 km fleet is always a subset of the 100 km fleet.

| `TrafficDraw` field | Shape | Meaning |
|---|---|---|
| `pool_idx` | (S,) int64 | indices into the pool (ships only, no repeats) |
| `heading` | (S,) f32 | rad, `lane_dir` or `lane_dir + pi` with equal probability |
| `sog` | (S,) f32 | m/s, every value >= `knots_to_ms(min_speed_kn)` |
| `seed` | int | for reproducibility |

Count per area: `N_a ~ Poisson(sum of pool weight in area a)`, capped at the number of pool points in the area. The points are then chosen without replacement, in proportion to their weight.

## 6. Shapes, time axes and saved arrays

| Object | Shape | Notes |
|---|---|---|
| solver state | `eta (B, ny, nx)`, `u (B, ny, nx+1)`, `v (B, ny+1, nx)` | float32 |
| solver receiver output `rec30` | `(B, T30=240, R, 3)` | channels `[eta, u, v]`; sample k at t = 30 (k+1) s |
| per-minute `rec` | `(B, 120, R, 3)` | `timeseries.block_average(rec30, 2, axis=1)` |
| filtered `recf` | `(B, 120, R, 3)` | `timeseries.baseline_filter(rec, W, axis=1)`, with zero history |
| raw gauge and buoy eta `gb30` | `(B, 240, R_gb)` | unfiltered eta at gauges and buoys, at 30 s (for triggers) |

**Saved library** (T14, B; the same format is used for truth events and historic runs, in `data/library/`, `data/truth/` and `data/historic_runs/`):
- `rec_{chunk:04d}.npy`: float32 `(C_chunk, 120, R, 3)`, **already block-averaged and baseline-filtered**. Saved as .npy so it can be opened with `np.load(..., mmap_mode="r")`.
- `gb_{chunk:04d}.npy`: float32 `(C_chunk, 240, R_gb)`, raw eta at gauges and buoys.
- `index.csv`: one row per run, with `run_id, chunk, row, ` plus every source column from section 7.
- `checksums.txt` (sha256 per file) and `run_log.csv` (chunk, B, wall seconds, GPU name).
- **Storage:** the library lives only on the Colab runtime disk (`library.runtime_dir` in config), float32. It is never downloaded to the Mac (no float16 copy). Library-based scoring (T17, T26) runs on Colab; only results (`outputs/...`) come back. If the runtime is lost, B rebuilds it. Local development uses a small library built on the Mac (a few candidates, coarse grid) in the same format under `data/library_dev/`.
- Never load a whole library onto the GPU. Gather the active ships on the CPU (section 9).

**Baseline filter** (`timeseries.baseline_filter`, implemented in T00): `y[m] = x[m] - mean(x[m-W .. m-1])`, with `W = library.baseline_window_min` = 45 (fixed; not varied in T20, because changing it means re-filtering the whole library). It never uses sample m or later. Before index 0 the history is zero for templates and truth runs (there is no signal before origin). For noise, the noise model generates W extra minutes of pre-origin history and passes it as `history`. The filter is linear, so `filter(truth + noise) = filter(truth) + filter(noise)`, and filtering the library once equals filtering each projected template.

## 7. Sources (T03, C)

`sources.candidate_table(cfg) -> pandas.DataFrame`, saved to `data/processed/candidates.csv`. Arc positions are taken every 25 km along the trench-side edge of the EFSM20 subduction-interface meshes (`sources.arc_geometry` in config; Hellenic GRSS001, Calabrian ITSS001, Cyprus CYSS001), with strike and dip from the mesh. The row order **is** the library order, so `cand_id = 0..C-1`.

| Column | Meaning |
|---|---|
| `cand_id`, `pos_id` | int; the same `pos_id` for all magnitudes at a position |
| `region` | `hellenic`, `calabrian`, `cyprus` or `aegean` |
| `lon_deg`, `lat_deg` | centre of the rupture (uplift) |
| `strike_deg`, `dip_deg` | Aki-Richards: dip direction = strike + 90 deg (down-dip = landward for arcs) |
| `mw` | magnitude: arcs 6.5 to 8.5 (9 bins, `magnitudes(cfg)`); Aegean 6.5 to `sources.aegean_mw_max` = 7.75 (6 bins) |
| `length_m`, `width_m`, `slip_m` | from the scaling relation in config |
| `polarity` | +1 uplift-first (all library entries) |
| `flip_allowed` | True for `aegean` (scored as both +g and -g), False for arcs |

- `sources.source_eta0(grid, row, cfg, kind=None, polarity=None) -> (ny, nx) float32`. `row` is a table row or dict with the columns above. `kind` is `dipole` or `monopole` (default from config); `polarity` is +1 or -1 (default from the row).
- `sources.perturb(row, along_strike_m, dmw, cfg) -> dict`, a new source moved along strike with rescaled dimensions.
- `sources.custom_source(lon_deg, lat_deg, strike_deg, dip_deg, mw, fault, cfg, length_m=None, width_m=None, slip_m=None) -> dict`, where `fault` is `thrust` (polarity +1) or `normal` (polarity -1, subsidence-first). Used for Samos and Amorgos.
- `sources.dipole_volume(eta0, grid) -> (uplift_m3, subsidence_m3, net_m3)`.
- Truth and historic runs use the same columns, plus `run_kind` (`truth_map2`, `truth_twin` or `historic`), `parent_cand_id` (-1 if none) and `name`.

## 8. Solver API (T02, T09, A)

```python
solver_reference.run_reference(grid, eta0, sampler, cfg) -> rec30 (240, R, 3)    # one scenario, explicit loops over time
solver_reference.step(eta, u, v, grid_ops, dt, g) -> (eta, u, v)                 # one forward-backward step

solver.make_sampler(grid, pool) -> Sampler     # implemented in solver_reference, re-exported by solver
solver.Solver(grid, sampler, cfg)            # moves the masks and metrics to xp once
solver.Solver.run(eta0) -> rec30 (B, 240, R, 3) xp array; eta0 is (B, ny, nx)
solver.Solver.snapshots(eta0, times_s) -> (B, len(times_s), ny, nx) eta   # for figures
solver.Solver.dt, .n_steps
solver.to_library(rec30, gb_idx, cfg) -> (recf (B, 120, R, 3), gb30 (B, 240, R_gb))   # block average + filter
```

## 9. Observations and noise (T12, A)

```python
observe.gather_truth(recf_run (120, R, 3), draw) -> (120, S, 3)       # numpy, CPU
observe.ais_signal(truth_s (120, S, 3), draw, response=1.0) -> (S, 120)   # response * project_normal(u, v, heading)
observe.gps_signal(truth_s, draw) -> (S, 120)                          # eta
noise.ais_noise(draw, pool, regime, cfg, rng, n_draws) -> (E, S, 120)  # filtered: generated with W min of history, then baseline_filter
noise.gps_noise(draw, regime, cfg, rng, n_draws) -> (E, S, 120)
noise.ship_sigma(draw, regime, operator, cfg) -> (S,)                  # std of FILTERED independent per-ship noise per minute (m/s or m)
noise.shared_sigma(regime, operator, cfg) -> float                     # std of FILTERED shared-field noise per minute
observe.gaps(draw, cfg, rng, drop_frac=0.0, off_mask=None) -> valid (S, 120) bool   # one mask per traffic draw, shared by its noise draws
observe.observe(signal (S,120), noise (E,S,120)) -> d (E, S, 120)      # signal + noise
```

- AIS per-ship noise: heading error (rounding to `heading_resolution_deg` plus wobble with sd `heading_wobble_sd_deg x regime multiplier`, independent per report) becomes velocity error `sog * delta_theta`. It is averaged over `60 / report_interval_s` reports per minute.
- Shared noise: a 2D vector current field with sd `shared_sd_m_s x multiplier`, spatial correlation `shared_corr_length_m` and temporal AR(1) with `shared_corr_time_s`, sampled at the ships and projected on each ship's normal.
- Gaps are applied after filtering. This approximates a real system that computes its baseline from the reports that did arrive.

## 10. Scoring (T15, A)

Processing is one traffic draw at a time. Here K = library candidates, M = cell groups and E = events x noise draws.

```python
scoring.build_groups(draw, pool, cfg, operator) -> Groups
    # Groups.group (S,) int, Groups.sign (S,) +1/-1, Groups.n_groups M, Groups.normal (M,) rad
    # cell = geometry.cell_index(...) at scoring.cell_m; AIS: bin = floor(fold_axis(theta + pi/2) / normal_bin)
    # group = (cell, bin); group normal = bin centre; sign_i = sign(n_i . n_group). GPS: bin 0, sign +1.
    # The bin centre ONLY decides sign_i. Every ship's observation and template are projected onto that
    # ship's OWN heading normal (project_normal with its own heading), never onto the bin centre
    # (projecting onto the bin centre would lose up to 1 - cos(22.5 deg) ~ 8% of the signal at bin edges).
scoring.aggregate(x (..., S, T), groups, valid (S, T)) -> (..., M, T)
    # mean over valid ships of sign_i * x_i; zero where a group has no valid ship. Used identically on d and on g.
scoring.aggregate_templates(lib_recf (C,120,R,3) memmap, cand_idx (K,), draw, groups, valid, operator, chunk=32)
    -> G (K, M, 120) numpy f32          # CPU gather + project + aggregate, chunked over K
scoring.cell_weights(groups, sigma_i (S,), sigma_shared, valid) -> w (M, T)
    # w = 1 / (sigma_shared^2 + sum_valid sigma_i^2 / N_gt^2); 0 where N_gt = 0
scoring.cumulative_stats(G (K,M,T), D (E,M,T), w (M,T)) -> gd (E,K,T), gg (K,T)
    # gd = cumsum_t sum_m w G D ; gg = cumsum_t sum_m w G^2   (batched matmul + prefix sum, on xp)
scoring.expand_flips(gd, gg, flip_allowed (K,)) -> gd2 (E,K2,T), gg2 (K2,T), parent (K2,), polarity (K2,)
    # appends -gd for flip_allowed candidates; gg unchanged. K2 = K + n_flip
scoring.detection_scores(gd2, gg2, eps) -> S (E,K2,T) = max(0, gd2)^2 / (gg2 + eps)
scoring.detection_statistic(S) -> stat (E,T) = max_k S, argk (E,T)
scoring.log_likelihood(gd2, gg2) -> L (E,K2,T) = 2 gd2 - gg2
scoring.estimate_source(L, t (E,) minute, cand_table, parent, polarity, top_n) -> DataFrame (best + top 3)
scoring.calibrate_threshold(stat_quiet (N,T), far) -> float      # (1 - far) quantile of max_t stat
scoring.confirmation_minute(stat (E,T), thr) -> (E,) int, -1 if never   # confirmation time = 60 (m+1) s
scoring.score_loop_reference(...)  # the same outputs, using explicit Python loops (equivalence test and benchmark)
```

- **Strata** = (operator in `{ais, gps}`) x (range cap in `{40 km, 100 km}`) x (regime in `{calm, moderate, rough}`), 12 in total. Each stratum has its own threshold for each detector.
- **Quiet days** need no solver run (d = noise only). For each stratum: calibration uses `n_quiet_traffic_draws` traffic draws (PURPOSE 5, indices 0..) x enough noise draws (PURPOSE 6) to reach >= 500. Held-out uses **disjoint** traffic and noise seeds, also >= 500. T16 writes the list to `data/truth/quiet_days.csv`: `quiet_id, split (cal|hold), traffic_seed, noise_seed`.
- **Thresholds** go to `outputs/thresholds.csv`: `operator, range_cap_m, regime, detector (network|single_literal|single_calibrated), threshold, n_cal, far_cal, n_hold, far_hold`.

## 11. Single-ship baselines (T25, A)

```python
baseline_detector.single_ship_stat(d (E,S,T), sigma_i (S,), valid) -> (E,T)   # running max over ships and minutes up to t of |d| / sigma
baseline_detector.literal_alarm_minute(d, sigma_i, valid, k=3.0) -> (E,) int, -1 if never
baseline_detector.calibrated_threshold(stat_quiet, far) -> k
baseline_detector.alarm_minute(stat, k) -> (E,) int
```

## 12. Gauges and buoys (T06, B)

```python
gauges.load_gauges(grid, cfg) -> DataFrame   # code, name, lon_deg, lat_deg, cell_j, cell_i, rtype (1|2), threshold_m, delay_s
gauges.trigger_times(gb30 (E, 240, R_gb), table) -> (E, R_gb) seconds, inf if never
    # first sample with |eta| > threshold_m, at t = 30 (k+1) s, plus delay_s
```

The gauges and buoys in `load_gauges` are the same, in the same order, as the gauge and buoy section of the pool.

## 13. Truth events (T16, B)

`data/truth/` holds runs in the library format, with the `index.csv` columns from section 7 plus `run_kind`. `map2` runs: one per library candidate, moved half an along-strike spacing and `mw + 0.1`. `twin` runs: 50 to 100 random off-grid events. A test confirms that no truth source has the same (lon, lat, strike, mw) as any library entry.

## 14. Sweep results (T17, B): `outputs/sweep/*.parquet`

One row per (event, traffic draw, noise draw, regime, range cap, operator, detector), with columns `run_id, traffic_seed, noise_seed, regime, range_cap_m, operator, detector, detected, confirm_s, best_cand_id, best_polarity, best_lon_deg, best_lat_deg, best_mw, loc_err_m, mw_err, top3_cand_ids, n_ships, first_gauge_s, nearest_gauge_code, nearest_gauge_s, first_instrument_s`.

## 15. Figures and benchmark (T07, T08, C)

```python
viz.map_axes(cfg, ax=None) -> (fig, ax)        # domain extent, lon/lat labels in deg, Natural Earth coastline via pyshp
viz.save_figure(fig, path, title, caption)     # title and a one-line caption are required
viz.COLORS                                     # dict of consistent colours: ships, gauges, buoys, sources, ...
bench.Timer(name, sync=backend.sync)           # context manager; .seconds
bench.record(name, seconds, cell_updates=None, bytes_moved=None, **meta) -> dict
bench.write_json(records, path)
```

## 16. Decisions (human, 2026-11-21, G0)

1. Aegean candidates are limited to Mw 6.5 to 7.75 (6 bins). Arcs keep 6.5 to 8.5 (9 bins). The library stays at about 25 GB or less (laptop has about 26 GB free). Update: the library is never downloaded to the Mac; it stays on the Colab runtime and only results are downloaded.
2. Arc geometry comes from the EFSM20 subduction meshes (downloaded by the human; see data/README.md). Slab2 Hellenic and Maesano et al. 2017 (Calabrian) are kept as cross-checks.
3. The baseline window is 45 min and fixed. The 45 degree bins decide sign alignment only; projection always uses each ship's own normal.
4. Strait and Black Sea boxes in config are still PLACEHOLDERs for B to check on `grid.png`.
