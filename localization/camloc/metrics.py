"""The official score (same math as c0_.../tools/evaluate.py), returned as numbers, plus per-pose errors.
scripts/validate.py also runs the official script on every result as a cross-check."""
import subprocess
import sys

import numpy as np

MAX_GT_GAP_NS = 50_000_000
BIN_NS = 1_000_000_000


def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    return {c: np.asarray(d[c], dtype=np.int64 if c == 'timestamp_ns' else float) for c in d.dtype.names}


def errors(gt, ts, sx, sy, syaw):
    """Per-pose errors vs. GT interpolated at ts: dict of ok-mask, pos, lat, lon, dyaw_deg."""
    o = np.argsort(gt['timestamp_ns'])
    tg = gt['timestamp_ns'][o]
    gx, gy, gyaw = gt['x'][o], gt['y'][o], np.unwrap(gt['yaw'][o])
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    ix = gx[i - 1] + a * (gx[i] - gx[i - 1])
    iy = gy[i - 1] + a * (gy[i] - gy[i - 1])
    iyaw = gyaw[i - 1] + a * (gyaw[i] - gyaw[i - 1])
    dx, dy = sx - ix, sy - iy
    return {'ok': ok, 'pos': np.hypot(dx, dy),
            'lon': np.cos(iyaw) * dx + np.sin(iyaw) * dy,
            'lat': -np.sin(iyaw) * dx + np.cos(iyaw) * dy,
            'dyaw_deg': np.degrees(np.angle(np.exp(1j * (syaw - iyaw)))),
            'gt_x': ix, 'gt_y': iy}


def score(gt, ts, sx, sy, syaw):
    ts = np.asarray(ts, dtype=np.int64)
    e = errors(gt, ts, sx, sy, syaw)
    ok = e['ok']
    tg = np.sort(gt['timestamp_ns'])
    bins = np.arange(tg[0], tg[-1], BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts[ok], side='right') - 1)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')  # noqa: E731
    pos = e['pos'][ok]
    return {
        'poses': int(len(ts)), 'evaluated': int(ok.sum()),
        'coverage_pct': 100.0 * len(covered) / max(len(bins), 1),
        'pos_rmse_m': rms(pos), 'pos_median_m': float(np.median(pos)), 'pos_p95_m': float(np.percentile(pos, 95)),
        'lat_rmse_m': rms(e['lat'][ok]), 'lon_rmse_m': rms(e['lon'][ok]), 'yaw_rmse_deg': rms(e['dyaw_deg'][ok]),
        'within_0.25_pct': 100.0 * float(np.mean(pos < 0.25)), 'within_0.5_pct': 100.0 * float(np.mean(pos < 0.5)),
        'within_1.0_pct': 100.0 * float(np.mean(pos < 1.0)),
    }


def run_official(evaluate_py, gt_csv, sub_csv):
    r = subprocess.run([sys.executable, evaluate_py, gt_csv, sub_csv], capture_output=True, text=True)
    return r.stdout + r.stderr
