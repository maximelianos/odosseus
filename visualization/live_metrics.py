"""Per-frame errors and running metrics with the official math (camloc.metrics = tools/evaluate.py).

The GT is cut to the chunk, like the per-lap GT files of localization D-004, so coverage is measured
over the chunk. At the end, the running values must equal camloc.metrics.score on the same poses.
"""
import numpy as np

from reference import cut, interp_yaw, metrics

PAD_NS = 20_000_000             # GT kept 20 ms beyond the first and last pose, so both can be interpolated


def _rms(v):
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')


class RunningMetrics:
    def __init__(self, gt, ts, x, y, yaw, proc_ms, n_acc):
        self.ts, self.x, self.y, self.yaw = np.asarray(ts, np.int64), x, y, yaw
        self.gt = cut(gt, self.ts[0] - PAD_NS, self.ts[-1] + PAD_NS)
        self.e = metrics.errors(self.gt, self.ts, x, y, yaw)
        self.ok = self.e['ok']
        self.gt_yaw = interp_yaw(self.gt, self.ts)
        tg = np.sort(self.gt['timestamp_ns'])
        self.bins = np.arange(tg[0], tg[-1], metrics.BIN_NS)
        self.bin_of = np.searchsorted(self.bins, self.ts, side='right') - 1
        self.proc_ms = np.asarray(proc_ms, float)
        self.n_acc = np.asarray(n_acc)

    def at(self, k):
        """Metrics over frames 0..k, i.e. what is known when frame k is shown."""
        s = slice(0, k + 1)
        ok = self.ok[s]
        pos = self.e['pos'][s][ok]
        covered = len(np.unique(self.bin_of[s][ok]))
        started = int(self.bin_of[k]) + 1                    # 1 s bins begun up to this frame
        proc = self.proc_ms[s]
        return {
            'pos_rmse_m': _rms(pos), 'lat_rmse_m': _rms(self.e['lat'][s][ok]),
            'lon_rmse_m': _rms(self.e['lon'][s][ok]), 'yaw_rmse_deg': _rms(self.e['dyaw_deg'][s][ok]),
            'within_1.0_pct': 100.0 * float(np.mean(pos < 1.0)) if len(pos) else float('nan'),
            'pos_median_m': float(np.median(pos)) if len(pos) else float('nan'),
            'pos_p95_m': float(np.percentile(pos, 95)) if len(pos) else float('nan'),
            'pos_max_m': float(pos.max()) if len(pos) else float('nan'),
            'coverage_pct': 100.0 * covered / max(started, 1),
            'ms_mean': float(proc.mean()), 'ms_p95': float(np.percentile(proc, 95)),
            'fix_rate_pct': 100.0 * float(np.mean(self.n_acc[s] > 0)),
        }

    def final(self):
        """camloc.metrics.score on the whole chunk; asserts the running values at the last frame agree."""
        sc = metrics.score(self.gt, self.ts, self.x, self.y, self.yaw)
        run = self.at(len(self.ts) - 1)
        for k in ('pos_rmse_m', 'lat_rmse_m', 'lon_rmse_m', 'yaw_rmse_deg', 'within_1.0_pct', 'pos_median_m',
                  'pos_p95_m', 'coverage_pct'):
            assert abs(sc[k] - run[k]) < 1e-9, (k, sc[k], run[k])
        return sc
