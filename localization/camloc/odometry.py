"""Causal access to the car state (allowed fields only, see camloc.allowed)."""
import numpy as np


class OdometryStream:
    """advance(ekf, t) integrates every car-state sample with stamp <= t into the filter and then
    extrapolates the last sample to t. Samples with stamp > t are never read: `max_read` records the
    largest stamp touched, and tests/test_rules.py checks it against t."""

    def __init__(self, state, v_still=0.05, sigma_zero_rate=0.0005):
        self.t = state['stamp_ns']
        self.vx, self.vy, self.wz, self.v = state['vx_mps'], state['vy_mps'], state['wz_radps'], state['v_mps']
        self.v_still = v_still
        self.sigma_zero_rate = sigma_zero_rate
        self.i = 0              # next sample to consume
        self.last = None        # last consumed sample
        self.max_read = None

    def _touch(self, k):
        self.max_read = self.t[k] if self.max_read is None else max(self.max_read, self.t[k])

    def seek(self, t_ns):
        """Skip to the first sample after t_ns without integrating (the initial pose is at t_ns)."""
        self.i = int(np.searchsorted(self.t, t_ns, side='right'))
        self.last = self.i - 1 if self.i > 0 else None
        if self.last is not None:
            self._touch(self.last)

    def stationary(self, k):
        return self.v[k] < self.v_still

    def advance(self, ekf, t_ns):
        assert t_ns >= ekf.t, 'time must not run backwards'
        while self.i < len(self.t) and self.t[self.i] <= t_ns:
            k = self.i
            self._touch(k)
            j = self.last if self.last is not None else k
            if self.t[k] > ekf.t:
                ekf.predict((self.t[k] - ekf.t) / 1e9, 0.5 * (self.vx[j] + self.vx[k]),
                            0.5 * (self.vy[j] + self.vy[k]), 0.5 * (self.wz[j] + self.wz[k]),
                            stationary=self.stationary(k) and self.stationary(j))
                ekf.t = int(self.t[k])
            if self.stationary(k):
                ekf.update_zero_rate(self.wz[k], self.sigma_zero_rate)
            self.last = k
            self.i += 1
        if t_ns > ekf.t and self.last is not None:
            k = self.last
            ekf.predict((t_ns - ekf.t) / 1e9, self.vx[k], self.vy[k], self.wz[k], self.stationary(k))
            ekf.t = int(t_ns)

    def standstill_bias(self, t_ns, window_s=5.0):
        """Mean yaw rate over the standstill samples in the first window_s seconds before t_ns
        (the run starts with the car standing: the recording begins 5 s before first motion)."""
        m = (self.t <= t_ns) & (self.t >= self.t[0]) & (self.t < self.t[0] + window_s * 1e9) & (self.v < self.v_still)
        return float(self.wz[m].mean()) if m.any() else 0.0
