"""The causal loop: one pose per image set, using only car state and images with stamps <= its stamp.

For every image set (stamp t):
    1. integrate the car state up to t                       (odometry.OdometryStream)
    2. relocalize the images of t against the map, gate, fuse (reloc.Relocalizer, optional)
    3. apply the one-sided track constraint                   (D-008)
    4. emit (t, x, y, yaw)
"""
import time

import numpy as np

from . import allowed
from .bag import iter_image_sets
from .ekf import PoseEKF
from .odometry import OdometryStream


def track_constraint(ekf, track, margin, sigma):
    """If the estimate is less than `margin` inside the drivable area, push it back (soft)."""
    corridor, m = track.corridor_of(ekf.x[:2])
    if m >= margin:
        return False
    _, inward = track.nearest_border_point(ekf.x[:2], corridor)
    ekf.update_inside(inward, margin - m, sigma)
    return True


class Recovery:
    """Gate lock-out guard (D-014): if camera fixes keep arriving but every one is rejected for
    `after_s`, the filter is probably wrong rather than the fixes, so its uncertainty is inflated
    to let the gate reopen."""

    def __init__(self, rc):
        self.rc = rc
        self.t_accept = None
        self.rejected = 0
        self.count = 0

    def step(self, ekf, t_ns, n_fix, n_acc):
        if self.t_accept is None or n_acc:
            self.t_accept, self.rejected = t_ns, 0
            return False
        self.rejected += n_fix
        if (t_ns - self.t_accept) / 1e9 < self.rc['after_s'] or self.rejected < self.rc['min_rejected']:
            return False
        ekf.P[0, 0] += self.rc['inflate_xy_m'] ** 2
        ekf.P[1, 1] += self.rc['inflate_xy_m'] ** 2
        ekf.P[2, 2] += np.radians(self.rc['inflate_yaw_deg']) ** 2
        self.t_accept, self.rejected = t_ns, 0
        self.count += 1
        return True


def run(cfg, bag_path, state, track, init, end_ns=None, reloc=None, log_every=0):
    """init = (stamp_ns, x, y, yaw): the pose the run starts from (first pose of the run, or the GT
    pose at the first image of a validation lap). Returns a dict of per-image-set arrays."""
    fc = cfg['filter']
    odo = OdometryStream(state, fc['v_still'], fc['sigma_zero_rate'])
    t0_ns, x0, y0, yaw0 = init
    bias0 = odo.standstill_bias(t0_ns) if fc['init_bias_from_standstill'] else 0.0
    ekf = PoseEKF(t0_ns, x0, y0, yaw0, bias0,
                  (fc['sigma0_xy'], np.radians(fc['sigma0_yaw_deg']), fc['sigma0_bias']), fc['noise'])
    odo.seek(t0_ns)
    tc = cfg['track_constraint']

    # images stay JPEG bytes here; the relocalizer decodes them in its per-camera threads, so the
    # measured time per set includes decoding
    cams = reloc.cameras if reloc else allowed.CAMERAS
    recovery = Recovery(cfg['reloc']['recovery']) if reloc else None
    rows = []
    for s in iter_image_sets(bag_path, cams, t0_ns, end_ns):
        t_start = time.perf_counter()
        odo.advance(ekf, s.stamp_ns)
        assert odo.max_read is None or odo.max_read <= s.stamp_ns
        fixes = reloc.localize(s, ekf) if reloc else []
        n_acc = 0
        for f in fixes:
            acc, nis = ekf.update_pose(f['x'], f['y'], f['yaw'], f['sigma_xy'], f['sigma_yaw'],
                                       gate=cfg['reloc']['gate_nis'])
            f['accepted'], f['nis'] = acc, nis
            n_acc += acc
        if reloc:
            reloc.feedback(s, fixes, ekf)
            recovery.step(ekf, s.stamp_ns, len(fixes), n_acc)
        pushed = track_constraint(ekf, track, tc['margin_m'], tc['sigma_m']) if tc['enabled'] else False
        x, y, yaw = ekf.pose
        rows.append((s.stamp_ns, x, y, yaw, np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]), np.sqrt(ekf.P[2, 2]),
                     ekf.x[3], len(fixes), n_acc, pushed, time.perf_counter() - t_start))
        if log_every and len(rows) % log_every == 0:
            print(f'  {len(rows):5d} sets  fixes {sum(r[8] for r in rows[-log_every:])}/{sum(r[7] for r in rows[-log_every:])}'
                  f'  sigma_xy {rows[-1][4]:.2f} m  bias {rows[-1][6]:+.2e}', flush=True)
    cols = ('stamp_ns', 'x', 'y', 'yaw', 'sigma_xy', 'sigma_yaw', 'bias', 'n_fix', 'n_accepted', 'pushed', 'proc_s')
    out = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    out['stamp_ns'] = out['stamp_ns'].astype(np.int64)
    out['recoveries'] = np.array(recovery.count if recovery else 0)
    return out


def write_csv(res, path):
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for t, x, y, a in zip(res['stamp_ns'], res['x'], res['y'], res['yaw']):
            f.write(f'{int(t)},{x:.4f},{y:.4f},{a:.6f}\n')
