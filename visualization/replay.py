"""Night replay: a continuous run of the training bag (localization/scripts/run.py output and its
_diag.npz), cut to a chunk of one lap. No pipeline compute: the frames are shown at their own 10 Hz.

Validation numbers must come from a continuous run from the bag's first pose (localization D-004),
so the night view replays one instead of restarting the pipeline at the lap.
"""
import os

import cv2
import numpy as np

import reference as ref
from camloc import bag


def _decode_quarter_rgb(cam, jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)[:, :, ::-1]


def read_thumbs(bag_path, ts, cam='fr', tol_ns=2_000_000):
    """Quarter-resolution RGB images of one camera at the frame stamps ts; a missing image repeats the
    previous one."""
    imgs = {}
    for s in bag.iter_image_sets(bag_path, [cam], int(ts[0]) - tol_ns, int(ts[-1]) + tol_ns, decode=_decode_quarter_rgb):
        imgs[s.stamp_ns] = s.images[cam]
    keys = np.array(sorted(imgs), dtype=np.int64)
    out, last = [], None
    for t in ts:
        i = int(np.argmin(np.abs(keys - t)))
        if abs(keys[i] - t) <= tol_ns:
            last = imgs[keys[i]]
        out.append(last if last is not None else imgs[keys[0]])
    return out


def load(run_csv, lap='L7', t_from=0.0, t_to=30.0, thumbs=True):
    """Per-frame arrays of the frames with stamps in [lap start + t_from, lap start + t_to)."""
    sub = ref.metrics.load_csv(run_csv)
    diag = dict(np.load(os.path.splitext(run_csv)[0] + '_diag.npz'))
    start = ref.laps()[lap]['start_ns']
    t0, t1 = start + int(t_from * 1e9), start + int(t_to * 1e9)
    m = (sub['timestamp_ns'] >= t0) & (sub['timestamp_ns'] < t1)
    md = (diag['stamp_ns'] >= t0) & (diag['stamp_ns'] < t1)
    ts = sub['timestamp_ns'][m]
    assert np.array_equal(ts, diag['stamp_ns'][md]), 'submission CSV and _diag.npz disagree'
    state = np.load(os.path.join(ref.ART, 'state.npz'))         # cached allowed car-state fields
    d = {'stamp_ns': ts, 'x': sub['x'][m], 'y': sub['y'][m], 'yaw': sub['yaw'][m],
         'sigma_xy': diag['sigma_xy'][md], 'sigma_yaw': diag['sigma_yaw'][md],
         'n_fix': diag['n_fix'][md].astype(int), 'n_acc': diag['n_accepted'][md].astype(int),
         'proc_ms': 1e3 * diag['proc_s'][md],
         'speed_kmh': 3.6 * np.interp(ts, state['stamp_ns'], state['vx_mps'])}
    d['thumbs'] = read_thumbs(ref.TRAIN_BAG, ts) if thumbs else None
    return d
