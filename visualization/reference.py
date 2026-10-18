"""Reference data for the visualizer, read-only from localization/: track borders, laps, ground truth
(scoring and display only, never given to the pipeline) and the driven line of a training lap.

localization/camloc is used as a library; nothing in localization/ is changed by the visualizer.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(ROOT, 'localization')
if LOC not in sys.path:
    sys.path.insert(0, LOC)
from camloc import metrics  # noqa: E402

ART = os.path.join(LOC, 'artifacts')
TRAIN_BAG = os.path.join(ROOT, 'hackFiles', 'c0_2026-10-13_22-53-42_camloc', 'c0_2026-10-13_22-53-42_camloc_0.mcap')
DAY_BAG = os.path.join(ROOT, 'hackFiles', 'DAYTIME_TEST_rosbag_2026-10-13_16-08-46_cameras_lap',
                       'rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap')
OFFICIAL_EVAL = os.path.join(ROOT, 'hackFiles', 'c0_2026-10-13_22-53-42_camloc', 'tools', 'evaluate.py')
UPDATE = os.path.join(ROOT, 'hackFiles', '23-00-38_cut_120m_update')     # organizers' track-frame update (D-019)
GATE = (-96.6, 697.6)           # start/finish line (localization/configs/iter1.yaml, D-002)


def racing_line():
    """The organizers' racing line (update map.mcap, /constructor0/map_ar/racing_line) as their own
    tools/track_frame.Line: xy_to_sn gives (s, n), n = signed distance from the line, + = left."""
    tools = os.path.join(UPDATE, 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from track_frame import load_lines
    return load_lines(os.path.join(UPDATE, 'map.mcap'))['track']


def track_bounds():
    """{name: (N, 3) points in map}: the training bag's borders (same map frame in every run)."""
    return dict(np.load(os.path.join(ART, 'track_bounds.npz')))


def laps():
    return {lap['name']: lap for lap in json.load(open(os.path.join(ART, 'laps.json')))['laps']}


_GT = None


def ground_truth():
    """Ground truth of the whole training bag in the official format (timestamp_ns, x, y, z, yaw)."""
    global _GT
    if _GT is None:
        _GT = metrics.load_csv(os.path.join(ART, 'gt_laps', 'full.csv'))
    return _GT


def cut(gt, t0_ns, t1_ns):
    m = (gt['timestamp_ns'] >= t0_ns) & (gt['timestamp_ns'] <= t1_ns)
    return {k: v[m] for k, v in gt.items()}


def interp_yaw(gt, ts):
    """GT yaw at ts, interpolated like camloc.metrics.errors (on the unwrapped angle)."""
    o = np.argsort(gt['timestamp_ns'])
    return np.interp(ts, gt['timestamp_ns'][o], np.unwrap(gt['yaw'][o]))


def write_gt_csv(gt, path):
    """GT in the format of the official tools/extract_gt.py, for running tools/evaluate.py on a chunk."""
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,z,yaw\n')
        for t, x, y, z, a in zip(gt['timestamp_ns'], gt['x'], gt['y'], gt['z'], gt['yaw']):
            f.write(f'{int(t)},{x:.4f},{y:.4f},{z:.4f},{a:.6f}\n')


def lap_line(name='L7'):
    """(x, y, yaw) of a training lap's GT path. L7's is the reference driven line of the day view."""
    lap = laps()[name]
    g = cut(ground_truth(), lap['start_ns'], lap['end_ns'])
    return g['x'], g['y'], np.unwrap(g['yaw'])
