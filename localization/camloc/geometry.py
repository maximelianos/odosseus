"""SE(3) helpers. Rotations are 3x3 arrays, poses 4x4 homogeneous matrices mapping child -> parent."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_to_rot(q):
    """q = (x, y, z, w) -> 3x3 rotation."""
    return Rotation.from_quat(q).as_matrix()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def roll_pitch_of(R):
    """ZYX Euler roll and pitch of a rotation (radians)."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class PoseInterpolator:
    """Linear position + slerp rotation over a stamped pose sequence (stamps in ns, sorted)."""

    def __init__(self, stamp_ns, pos, quat):
        self.t = np.asarray(stamp_ns, dtype=np.int64)
        self.pos = np.asarray(pos, dtype=float)
        self.slerp = Slerp((self.t - self.t[0]) / 1e9, Rotation.from_quat(quat))

    def covers(self, t_ns):
        return self.t[0] <= t_ns <= self.t[-1]

    def __call__(self, t_ns):
        """Pose (4x4) at t_ns."""
        s = (np.asarray(t_ns, dtype=np.int64) - self.t[0]) / 1e9
        i = np.clip(np.searchsorted(self.t, t_ns), 1, len(self.t) - 1)
        a = (t_ns - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        p = self.pos[i - 1] + a * (self.pos[i] - self.pos[i - 1])
        return make_T(self.slerp(s).as_matrix(), p)
