"""Observation noise (T12). docs/interfaces.md section 9.

All series are per minute, time on the last axis, shape (E, S, 120). Every generator draws W
extra minutes of pre-origin history and applies the causal baseline filter with that history,
so the returned noise is exactly what the detector sees.

AIS (heading-normal drift, m/s), per ship i and minute m:
  independent part: sog_i * dtheta, with dtheta the minute mean of n_rep = 60 / report_interval
      heading errors, each = wobble (sd x weather multiplier) + rounding to the AIS resolution
      (uniform, variance res^2 / 12, independent per report because wobble dithers it)
  constant part:    sog_i * bias_i (HDG misalignment, removed exactly by the filter)
  shared part:      2D current field (u, v) projected on the ship's starboard normal; Gaussian
      spatial covariance (length scale L) via random Fourier features, AR(1) in time
GPS height (m): independent white noise per minute; optional shared part (0 by default).
"""
import numpy as np

from .geometry import project_normal
from .timeseries import baseline_filter
from .units import deg2rad


def regime_multiplier(regime, cfg):
    return float(cfg["weather_regimes"][regime])


def window(cfg):
    return int(cfg["library"]["baseline_window_min"])


def n_minutes(cfg):
    return int(cfg["library"]["n_minutes"])


def filter_with_history(raw, cfg):
    """raw (..., W + T): first W minutes are pre-origin history. Returns filtered (..., T)."""
    W = window(cfg)
    return baseline_filter(raw[..., W:], W, axis=-1, history=raw[..., :W])


# ------------------------------------------------------------ analytic sigmas

def _filter_gain_white(W):
    """std(filtered) / std(raw) for white noise: sqrt(1 + 1/W)."""
    return np.sqrt(1.0 + 1.0 / W)


def _filter_gain_ar1(rho, W):
    """std(filtered) / std(raw) for a unit AR(1) series with lag-1 correlation rho."""
    k = np.arange(1, W + 1)
    cov_x_mean = np.mean(rho ** k)
    lags = np.abs(np.subtract.outer(k, k))
    var_mean = np.mean(rho ** lags)
    return np.sqrt(max(1.0 + var_mean - 2.0 * cov_x_mean, 0.0))


def heading_sd_per_minute(regime, cfg):
    """sd (rad) of the minute-mean heading error, independent part."""
    a = cfg["ais_noise"]
    n_rep = cfg["library"]["minute_s"] / a["report_interval_s"]
    wob = deg2rad(a["heading_wobble_sd_deg"]) * regime_multiplier(regime, cfg)
    res = deg2rad(a["heading_resolution_deg"])
    return np.sqrt(wob**2 + res**2 / 12.0) / np.sqrt(n_rep)


def ar1_rho(cfg):
    tau = cfg["ais_noise"]["shared_corr_time_s"]
    return float(np.exp(-cfg["library"]["minute_s"] / tau)) if tau > 0 else 0.0


def ship_sigma(draw, regime, operator, cfg):
    """(S,) std of the FILTERED independent per-ship noise per minute (m/s for AIS, m for GPS)."""
    gain = _filter_gain_white(window(cfg))
    if operator == "ais":
        return (np.asarray(draw.sog, np.float64) * heading_sd_per_minute(regime, cfg) * gain).astype(np.float32)
    if operator == "gps":
        return np.full(len(draw.sog), cfg["gps"]["noise_sd_m"] * gain, np.float32)
    raise ValueError(operator)


def shared_sigma(regime, operator, cfg):
    """std of the FILTERED shared-field noise per minute, as seen by one ship."""
    if operator == "ais":
        sd = cfg["ais_noise"]["shared_sd_m_s"] * regime_multiplier(regime, cfg)
        return float(sd * _filter_gain_ar1(ar1_rho(cfg), window(cfg)))
    if operator == "gps":
        return float(cfg["gps"]["shared_sd_m"] * _filter_gain_white(window(cfg)))
    raise ValueError(operator)


# ------------------------------------------------------------ generators

def ship_xy_m(pool, draw, cfg):
    """Ship positions in metres on the equirectangular plane used for the shared field."""
    R = cfg["physics"]["earth_radius_m"]
    lon = np.asarray(pool.lon)[draw.pool_idx]
    lat = np.asarray(pool.lat)[draw.pool_idx]
    ref = deg2rad(cfg["scoring"]["cell_ref_lat_deg"])
    x = R * np.cos(ref) * deg2rad(lon - cfg["domain"]["lon_min_deg"])
    y = R * deg2rad(lat - cfg["domain"]["lat_min_deg"])
    return np.stack([x, y], axis=1)


def shared_field(xy, sd, cfg, rng, n_draws, n_times):
    """(E, 2, S, n_times) vector current at the points xy (S, 2) in metres.

    Gaussian covariance sd^2 exp(-r^2 / (2 L^2)) per component via random Fourier features;
    feature amplitudes follow AR(1) in time with rho = exp(-60 s / tau).
    """
    a = cfg["ais_noise"]
    K, L = int(a["shared_n_features"]), a["shared_corr_length_m"]
    rho = ar1_rho(cfg)
    S = xy.shape[0]
    out = np.empty((n_draws, 2, S, n_times), np.float32)
    for e in range(n_draws):
        for comp in range(2):
            k = rng.normal(0.0, 1.0 / L, size=(K, 2))
            phase = xy @ k.T                                          # (S, K)
            basis = np.concatenate([np.cos(phase), np.sin(phase)], axis=1) * np.sqrt(1.0 / K)  # (S, 2K)
            amp = np.empty((2 * K, n_times))
            amp[:, 0] = rng.standard_normal(2 * K)
            innov = rng.standard_normal((2 * K, n_times)) * np.sqrt(1.0 - rho**2)
            for t in range(1, n_times):
                amp[:, t] = rho * amp[:, t - 1] + innov[:, t]
            out[e, comp] = (sd * basis @ amp).astype(np.float32)
    return out


def ais_noise(draw, pool, regime, cfg, rng, n_draws, include_shared=True, include_bias=True):
    """(E, S, 120) filtered AIS drift noise in m/s."""
    W, T = window(cfg), n_minutes(cfg)
    S = len(draw.sog)
    sog = np.asarray(draw.sog, np.float64)[None, :, None]
    sd_head = heading_sd_per_minute(regime, cfg)
    raw = sog * sd_head * rng.standard_normal((n_draws, S, W + T))
    if include_bias:
        bias = deg2rad(cfg["ais_noise"]["heading_bias_sd_deg"]) * rng.standard_normal((n_draws, S, 1))
        raw = raw + sog * bias
    if include_shared:
        sd = cfg["ais_noise"]["shared_sd_m_s"] * regime_multiplier(regime, cfg)
        f = shared_field(ship_xy_m(pool, draw, cfg), sd, cfg, rng, n_draws, W + T)   # (E, 2, S, W+T)
        theta = np.asarray(draw.heading, np.float64)[None, :, None]
        raw = raw + project_normal(f[:, 0], f[:, 1], theta)
    return filter_with_history(raw.astype(np.float32), cfg)


def gps_noise(draw, regime, cfg, rng, n_draws):
    """(E, S, 120) filtered GPS height noise in m (no weather multiplier: PLAN applies it to AIS only)."""
    W, T = window(cfg), n_minutes(cfg)
    S = len(draw.sog)
    raw = cfg["gps"]["noise_sd_m"] * rng.standard_normal((n_draws, S, W + T))
    if cfg["gps"]["shared_sd_m"] > 0:
        raw = raw + cfg["gps"]["shared_sd_m"] * rng.standard_normal((n_draws, 1, W + T))
    return filter_with_history(raw.astype(np.float32), cfg)
