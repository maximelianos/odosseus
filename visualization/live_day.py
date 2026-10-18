"""The daytime run, live: camloc's unchanged pipeline.run in a worker process, observed through a wrapped
image-set reader; the window (another process) draws what arrives.

Nothing in localization/ is changed. In the worker only, at run time:
- camloc.pipeline.iter_image_sets is wrapped. Each time pipeline.run asks for the next image set, the
  filter holds the previous frame's final pose (after the camera updates and the track constraint):
  the wrapper publishes it, waits for the next frame's 10 Hz slot and hands the set over. pipeline.run
  starts its per-frame timer after the hand-over, so publishing, waiting and decoding the thumbnail are
  not in its measured ms per frame.
- the relocalizer's localize / feedback / localize_camera are wrapped on the instance to observe the
  filter, the fixes and each camera's match counts; they return what the originals return.
At the end every published pose is compared with what pipeline.run returned (`guard`).
"""
import os
import sys
import time

import numpy as np

FAIL_TEXT = {'no_keyframe': 'no map keyframe near', 'few_features': 'too few features',
             'few_matches': 'too few matches', 'few_inliers': 'too few inliers', 'height': 'failed height check',
             'attitude': 'failed attitude check', 'inlier_ratio': 'failed inlier-ratio check'}


class _Hook:
    def __init__(self, q, reloc, pace, speed):
        from camloc.bag import iter_image_sets
        self.orig, self.q, self.pace, self.speed = iter_image_sets, q, pace, speed
        self.cam_order = list(reloc.cameras)
        self.ekf, self.fixes, self.after_fix, self.cams = None, [], None, {}
        self.published = []
        lc, loc, fb = reloc.localize_camera, reloc.localize, reloc.feedback

        def localize_camera(cam, gray, *a, **kw):
            f = lc(cam, gray, *a, **kw)
            self.cams[cam] = (bool(f.get('ok')), f.get('fail', ''), int(f.get('matches') or 0), int(f.get('inliers') or 0))
            return f

        def localize(s, ekf):
            self.ekf = ekf
            return loc(s, ekf)

        def feedback(s, fixes, ekf):
            self.fixes, self.after_fix = list(fixes), ekf.pose
            return fb(s, fixes, ekf)

        reloc.localize_camera, reloc.localize, reloc.feedback = localize_camera, localize, feedback

    def iter_image_sets(self, bag_path, cams, start_ns=None, end_ns=None, decode=None, tol_ns=2_000_000):
        import cv2
        first = t_wall0 = None
        for s in self.orig(bag_path, cams, start_ns, end_ns, decode=decode, tol_ns=tol_ns):
            if first is None:
                first, t_wall0 = s.stamp_ns, time.perf_counter()
            late = 0.0
            if self.pace:
                due = (s.stamp_ns - first) / 1e9 / self.speed
                wait = due - (time.perf_counter() - t_wall0)
                if wait > 0:
                    time.sleep(wait)
                late = max(0.0, -wait)
            jpeg = s.images.get('fr')
            thumb = None if jpeg is None else \
                cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)[:, :, ::-1].copy()
            self.cams, self.fixes, self.after_fix = {}, [], None
            t_hand = time.perf_counter()
            yield s
            self._publish(s.stamp_ns, thumb, time.perf_counter() - t_hand, late)    # frame done: its final pose

    def _publish(self, stamp, thumb, proc_s, late_s):
        x, y, yaw = self.ekf.pose
        P = self.ekf.P
        acc = {f['cam']: bool(f.get('accepted')) for f in self.fixes}
        cams = []
        for c in self.cam_order:
            ok, fail, matches, inliers = self.cams.get(c, (False, 'not tried', 0, 0))
            state = (2 if acc.get(c) else 1) if ok else 0
            text = (f'fix {"accepted" if state == 2 else "rejected"} · {inliers} inliers' if ok
                    else FAIL_TEXT.get(fail, fail) + (f' ({inliers})' if fail == 'few_inliers' else ''))
            cams.append((matches, inliers, state, text))
        self.published.append((int(stamp), x, y, yaw))
        self.q.put(('frame', {'stamp_ns': int(stamp), 'x': x, 'y': y, 'yaw': yaw,
                              'sigma_xy': float(np.sqrt(P[0, 0] + P[1, 1])), 'sigma_yaw': float(np.sqrt(P[2, 2])),
                              'n_fix': len(self.fixes), 'n_acc': int(sum(acc.values())),
                              'pushed': self.after_fix is not None and self.after_fix != (x, y, yaw),
                              'proc_ms': 1e3 * proc_s, 'late_ms': 1e3 * late_s, 'cams': cams, 'thumb': thumb}))

    def check(self, res):
        """The published poses must be exactly what pipeline.run returned."""
        pub = self.published
        ok = len(pub) == len(res['stamp_ns']) and all(
            p[0] == int(t) and p[1] == x and p[2] == y and p[3] == a
            for p, t, x, y, a in zip(pub, res['stamp_ns'], res['x'], res['y'], res['yaw']))
        return {'identical_to_pipeline_run': bool(ok), 'frames': len(pub)}


def worker(q, p):
    """Run pipeline.run on the daytime chunk; frames, then ('done', ...) or ('error', traceback) on q."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import reference as ref
        import daybag
        from camloc import config, pipeline
        from camloc.reloc.query import Relocalizer
        from camloc.trackmap import TrackMap
        cfg = config.apply_overrides(config.load(p['config']), p['overrides'])
        track = TrackMap(ref.track_bounds())
        reloc = Relocalizer(cfg, track=track)
        if p['calib'] == 'bag':
            daybag.use_bag_calibration(reloc, p['bag'], ('rr', 'rl'))
        hook = _Hook(q, reloc, p['pace'], p['speed'])
        pipeline.iter_image_sets = hook.iter_image_sets           # this worker process only
        q.put(('ready', {'cameras': reloc.cameras, 'map': cfg['reloc']['map_name']}))
        res = pipeline.run(cfg, p['bag'], p['state'], track, p['init'], end_ns=p['end_ns'], reloc=reloc)
        q.put(('done', {'guard': hook.check(res), 'proc_ms': (1e3 * res['proc_s']).tolist(),
                        'recoveries': int(res['recoveries']), 'per_camera': reloc.stats}))
    except Exception:
        import traceback
        q.put(('error', traceback.format_exc()))


class Log:
    """Per-frame arrays for the HealthDashboard, filled as frames arrive (capacity n)."""

    def __init__(self, n, cams, state, bounds, line, start_note, dt_end):
        """line: the reference line, a track_frame.Line (the organizers' racing line)."""
        self.k = -1
        self.state, self.cams, self.line = state, list(cams), line
        f = lambda dtype=float: np.zeros(n, dtype)  # noqa: E731
        self.d = {'t': f(), 'stamp_ns': f(np.int64), 'x': f(), 'y': f(), 'yaw': f(), 'sigma_xy': f(), 'sigma_yaw': f(),
                  'n_fix': f(int), 'n_acc': f(int), 'pushed': f(bool), 'proc_ms': f(), 'late_ms': f(), 'speed_kmh': f(),
                  'ref_off': f(), 'ref_s': f(), 'cam_matches': np.zeros((n, len(cams)), int),
                  'cam_state': np.zeros((n, len(cams)), int), 'cam_text': [[''] * len(cams) for _ in range(n)],
                  'thumbs': [None] * n, 'health': [None] * n, 'cams': self.cams, 'dt_end': dt_end,
                  'start_note': start_note, 'bounds': bounds, 'ref_xy': (line.p[:, 0], line.p[:, 1])}

    def add(self, r):
        d = self.d
        k = self.k = self.k + 1
        if k == 0:
            self.t0 = r['stamp_ns']
        d['t'][k] = (r['stamp_ns'] - self.t0) / 1e9
        for key in ('stamp_ns', 'x', 'y', 'yaw', 'sigma_xy', 'sigma_yaw', 'n_fix', 'n_acc', 'pushed', 'proc_ms', 'late_ms'):
            d[key][k] = r[key]
        d['speed_kmh'][k] = 3.6 * np.interp(r['stamp_ns'], self.state['stamp_ns'], self.state['vx_mps'])
        s, n, _, _ = self.line.xy_to_sn(r['x'], r['y'])             # organizers' track frame: n + = left
        d['ref_s'][k], d['ref_off'][k] = s[0], n[0]
        for j, (m, _, st, txt) in enumerate(r['cams']):
            d['cam_matches'][k, j], d['cam_state'][k, j], d['cam_text'][k][j] = m, st, txt
        d['thumbs'][k] = r['thumb']
        d['health'][k] = health_at(d, k)
        return k

    def trimmed(self, proc_ms=None):
        """The arrays cut to the frames received; proc_ms replaced by pipeline.run's own if given."""
        n = self.k + 1
        d = {k: (v[:n] if isinstance(v, (np.ndarray, list)) and k not in ('cams', 'ref_xy') else v)
             for k, v in self.d.items()}
        if proc_ms is not None:
            d['proc_ms'] = np.asarray(proc_ms[:n], float)
            d['health'] = [health_at(d, k) for k in range(n)]
        return d


def health_at(d, k):
    """What can be measured without ground truth, over frames 0..k."""
    s = slice(0, k + 1)
    proc, st = d['proc_ms'][s], d['cam_state'][s]
    bins = np.unique(((d['stamp_ns'][s] - d['stamp_ns'][0]) // 1_000_000_000))
    started = int((d['stamp_ns'][k] - d['stamp_ns'][0]) // 1_000_000_000) + 1
    return {'fix_frames_pct': 100.0 * float(np.mean(d['n_acc'][s] > 0)), 'accepted': int(d['n_acc'][s].sum()),
            'produced': int(d['n_fix'][s].sum()), 'tried': int(st.size),
            'median_matches': float(np.median(d['cam_matches'][s])), 'sigma_now': float(d['sigma_xy'][k]),
            'sigma_max': float(d['sigma_xy'][s].max()), 'pushed_pct': 100.0 * float(np.mean(d['pushed'][s])),
            'ms_mean': float(proc.mean()), 'ms_p95': float(np.percentile(proc, 95)),
            'coverage_pct': 100.0 * len(bins) / started, 'off_rms': float(np.sqrt(np.mean(d['ref_off'][s] ** 2)))}
