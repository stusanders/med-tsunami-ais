"""Batched linear shallow-water solver (T09).

Same equations and layout as solver_reference (docs/interfaces.md section 3), with a leading
batch dimension B so one stencil update advances B independent scenarios. Runs on NumPy or
CuPy through the `xp` backend. Receiver output is (B, T30, R, 3) = [eta, u, v] every 30 s.
"""
import numpy as np

from . import backend
from .solver_reference import Sampler, centre_velocities, grid_ops, make_sampler, time_step  # noqa: F401
from .timeseries import baseline_filter, block_average


class Solver:
    def __init__(self, grid, sampler, cfg, xp=None):
        self.xp = xp if xp is not None else backend.xp
        self.cfg = cfg
        self.g = np.float32(cfg["physics"]["g_m_s2"])
        dt, self.n_sub = time_step(grid, cfg)
        self.dt = np.float32(dt)
        self.sample_interval_s = cfg["solver"]["sample_interval_s"]
        self.n_samples = int(round(cfg["solver"]["t_end_s"] / self.sample_interval_s))
        self.n_steps = self.n_samples * self.n_sub
        self.shape = grid.depth.shape
        ops = grid_ops(grid, cfg)
        a = lambda arr: self.xp.asarray(arr, dtype=np.float32)
        self.hu, self.hv, self.mu, self.mv = a(ops.hu), a(ops.hv), a(ops.mu), a(ops.mv)
        self.wet, self.dx, self.dy = a(ops.wet), a(ops.dx), np.float32(ops.dy)
        self.cos_c, self.cos_v = a(ops.cos_c), a(ops.cos_v)
        self.dy_cos_c = a(ops.dy * ops.cos_c)
        self.hv_cos_v = a(ops.hv * ops.cos_v)
        self.damp_c, self.damp_u, self.damp_v = a(ops.damp_c), a(ops.damp_u), a(ops.damp_v)
        self.idx = self.xp.asarray(sampler.idx)
        self.w = a(sampler.w)

    # ------------------------------------------------------------ core

    def initial_state(self, eta0):
        xp = self.xp
        eta = xp.asarray(eta0, dtype=np.float32)
        if eta.ndim == 2:
            eta = eta[None]
        B, (ny, nx) = eta.shape[0], self.shape
        eta = eta * self.wet
        u = xp.zeros((B, ny, nx + 1), np.float32)
        v = xp.zeros((B, ny + 1, nx), np.float32)
        return eta, u, v

    def step(self, eta, u, v):
        """One forward-backward step on (B, ...) arrays (u and v updated in place)."""
        dt, g = self.dt, self.g
        flux_u = self.hu * u
        flux_v = self.hv_cos_v * v
        div = (flux_u[:, :, 1:] - flux_u[:, :, :-1]) / self.dx \
            + (flux_v[:, 1:, :] - flux_v[:, :-1, :]) / self.dy_cos_c
        eta = (eta - dt * div) * self.wet
        u[:, :, 1:-1] -= (dt * g) * (eta[:, :, 1:] - eta[:, :, :-1]) / self.dx
        v[:, 1:-1, :] -= (dt * g / self.dy) * (eta[:, 1:, :] - eta[:, :-1, :])
        u *= self.mu * self.damp_u
        v *= self.mv * self.damp_v
        return eta * self.damp_c, u, v

    def sample(self, eta, u, v):
        """(B, R, 3) at every receiver: bilinear on centre-averaged velocities."""
        xp = self.xp
        B = eta.shape[0]
        uc, vc = centre_velocities(u, v)
        fields = xp.stack([eta.reshape(B, -1), uc.reshape(B, -1), vc.reshape(B, -1)], axis=1)  # (B, 3, N)
        vals = (fields[:, :, self.idx] * self.w).sum(axis=-1)                                   # (B, 3, R)
        return vals.transpose(0, 2, 1)

    # ------------------------------------------------------------ drivers

    def run(self, eta0):
        """rec30 (B, T30, R, 3) on the xp backend; sample k at t = 30 (k+1) s."""
        eta, u, v = self.initial_state(eta0)
        B, R = eta.shape[0], self.idx.shape[0]
        out = self.xp.empty((B, self.n_samples, R, 3), np.float32)
        for k in range(self.n_samples):
            for _ in range(self.n_sub):
                eta, u, v = self.step(eta, u, v)
            out[:, k] = self.sample(eta, u, v)
        return out

    def run_library(self, eta0, gb_idx):
        """Library output without holding rec30: minute means accumulated on the fly, then filtered.

        Returns (recf (B, 120, R, 3), gb30 (B, T30, R_gb)) on the xp backend. Identical to
        to_library(run(eta0), gb_idx, cfg) up to float32 rounding.
        """
        xp = self.xp
        per_min = int(round(self.cfg["library"]["minute_s"] / self.sample_interval_s))
        n_min = self.n_samples // per_min
        eta, u, v = self.initial_state(eta0)
        B, R = eta.shape[0], self.idx.shape[0]
        gb_idx = xp.asarray(gb_idx)
        rec = xp.zeros((B, n_min, R, 3), np.float32)
        gb30 = xp.empty((B, self.n_samples, len(gb_idx)), np.float32)
        for k in range(self.n_samples):
            for _ in range(self.n_sub):
                eta, u, v = self.step(eta, u, v)
            s = self.sample(eta, u, v)
            rec[:, k // per_min] += s / per_min
            gb30[:, k] = s[:, gb_idx, 0]
        recf = baseline_filter(rec, self.cfg["library"]["baseline_window_min"], axis=1)
        return recf, gb30

    def snapshots(self, eta0, times_s):
        """eta fields (B, len(times_s), ny, nx) at the given times (rounded to whole steps)."""
        eta, u, v = self.initial_state(eta0)
        steps = [int(round(t / float(self.dt))) for t in times_s]
        out = self.xp.empty((eta.shape[0], len(steps)) + self.shape, np.float32)
        n = 0
        for m, target in enumerate(steps):
            while n < target:
                eta, u, v = self.step(eta, u, v)
                n += 1
            out[:, m] = eta
        return out


def to_library(rec30, gb_idx, cfg):
    """(recf (B, 120, R, 3), gb30 (B, T30, R_gb)): block-average to 1 min, then causal baseline filter."""
    per_min = int(round(cfg["library"]["minute_s"] / cfg["solver"]["sample_interval_s"]))
    rec = block_average(rec30, per_min, axis=1)
    recf = baseline_filter(rec, cfg["library"]["baseline_window_min"], axis=1)
    return recf, rec30[:, :, gb_idx, 0]
