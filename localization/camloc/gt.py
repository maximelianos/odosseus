"""Ground truth: map -> constructor0/base_link from /tf (full 6-DoF).

Used ONLY for building the relocalization map from training laps, defining laps, initializing a
validation lap at its first pose, and scoring. The inference pipeline never imports this module
(tests/test_rules.py checks that).
"""
import csv

import numpy as np

from .bag import _reader, stamp_ns
from .geometry import PoseInterpolator, quat_to_rot, yaw_of

GT_TOPIC = '/tf'


def read_gt(path):
    """{'stamp_ns', 'pos' (N,3), 'quat' (N,4) x y z w}, sorted by stamp."""
    t, pos, quat = [], [], []
    with open(path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[GT_TOPIC]):
            for tr in msg.transforms:
                if tr.header.frame_id != 'map' or tr.child_frame_id != 'constructor0/base_link':
                    continue
                p, q = tr.transform.translation, tr.transform.rotation
                t.append(stamp_ns(tr.header))
                pos.append((p.x, p.y, p.z))
                quat.append((q.x, q.y, q.z, q.w))
    t = np.asarray(t, dtype=np.int64)
    o = np.argsort(t, kind='stable')
    return {'stamp_ns': t[o], 'pos': np.asarray(pos)[o], 'quat': np.asarray(quat)[o]}


def save_gt(gt, path):
    np.savez(path, **gt)


def load_gt(path):
    d = np.load(path)
    return {k: d[k] for k in ('stamp_ns', 'pos', 'quat')}


def yaws(gt):
    return np.array([yaw_of(quat_to_rot(q)) for q in gt['quat']])


def interpolator(gt):
    return PoseInterpolator(gt['stamp_ns'], gt['pos'], gt['quat'])


def write_eval_csv(gt, path, start_ns=None, end_ns=None):
    """GT in the format of the official tools/extract_gt.py (timestamp_ns,x,y,z,yaw), optionally cut
    to [start_ns, end_ns]. evaluate.py measures coverage over the whole span of this file, so a
    per-lap file is needed to score one lap (D-004)."""
    t = gt['stamp_ns']
    m = np.ones(len(t), bool)
    if start_ns is not None:
        m &= t >= start_ns
    if end_ns is not None:
        m &= t <= end_ns
    yw = yaws({'quat': gt['quat'][m]})
    with open(path, 'w', newline='') as o:
        w = csv.writer(o)
        w.writerow(['timestamp_ns', 'x', 'y', 'z', 'yaw'])
        for ti, p, a in zip(t[m], gt['pos'][m], yw):
            w.writerow([int(ti), f'{p[0]:.4f}', f'{p[1]:.4f}', f'{p[2]:.4f}', f'{a:.6f}'])
