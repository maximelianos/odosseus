"""Pose filter: extended Kalman filter on x = [px, py, yaw, b], b = yaw-rate (gyro) bias (D-006).

Prediction integrates the body-frame velocities and the bias-corrected yaw rate of the car state.
Updates: absolute pose fixes from the cameras, a zero-rate update of b at standstill, and a one-sided
track constraint (D-008). Every update can be gated on its normalized innovation squared (NIS).
"""
import numpy as np

from .geometry import wrap

H_POSE = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])


class PoseEKF:
    def __init__(self, stamp_ns, x, y, yaw, bias, sigma0, noise):
        """sigma0: initial std of (xy [m], yaw [rad], bias [rad/s]). noise: spectral densities
        v_long, v_lat [m/s/sqrt(Hz)], v_scale [-], wz [rad/s/sqrt(Hz)], bias_rw [rad/s/sqrt(s)]."""
        self.t = int(stamp_ns)
        self.x = np.array([x, y, yaw, bias], float)
        sxy, syaw, sb = sigma0
        self.P = np.diag([sxy ** 2, sxy ** 2, syaw ** 2, sb ** 2])
        self.q = noise

    @property
    def pose(self):
        return float(self.x[0]), float(self.x[1]), float(wrap(self.x[2]))

    def predict(self, dt, vx, vy, wz, stationary=False):
        if dt <= 0:
            return
        px, py, th, b = self.x
        w = 0.0 if stationary else wz - b          # a car at standstill does not turn
        thm = th + 0.5 * w * dt
        c, s = np.cos(thm), np.sin(thm)
        dx = (c * vx - s * vy) * dt
        dy = (s * vx + c * vy) * dt
        self.x = np.array([px + dx, py + dy, th + w * dt, b])

        F = np.eye(4)
        F[0, 2], F[1, 2] = -dy, dx
        if not stationary:
            F[2, 3] = -dt
            F[0, 3], F[1, 3] = 0.5 * dy * dt, -0.5 * dx * dt
        q = self.q
        var_long = q['v_long'] ** 2 + (q['v_scale'] * vx) ** 2
        Rb = np.array([[c, -s], [s, c]])
        Q = np.zeros((4, 4))
        Q[:2, :2] = Rb @ np.diag([var_long, q['v_lat'] ** 2]) @ Rb.T * dt
        Q[2, 2] = 0.0 if stationary else q['wz'] ** 2 * dt
        Q[3, 3] = q['bias_rw'] ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R, yaw_row=None, gate=None):
        """Generic update. Returns (accepted, nis)."""
        z, H, R = np.atleast_1d(z), np.atleast_2d(H), np.atleast_2d(R)
        y = z - H @ self.x
        if yaw_row is not None:
            y[yaw_row] = wrap(y[yaw_row])
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        nis = float(y @ Si @ y)
        if gate is not None and nis > gate:
            return False, nis
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return True, nis

    def update_pose(self, x, y, yaw, sigma_xy, sigma_yaw, gate=None):
        R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        return self.update(np.array([x, y, yaw]), H_POSE, R, yaw_row=2, gate=gate)

    def update_zero_rate(self, wz, sigma):
        """At standstill the measured yaw rate is the bias."""
        return self.update(np.array([wz]), np.array([[0, 0, 0, 1.0]]), np.array([[sigma ** 2]]))

    def update_inside(self, normal, offset, sigma):
        """Move the position `offset` metres along the unit `normal` (one-sided track constraint)."""
        H = np.array([[normal[0], normal[1], 0, 0]])
        z = H[0, :2] @ self.x[:2] + offset
        return self.update(np.array([z]), H, np.array([[sigma ** 2]]))
