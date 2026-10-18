#!/usr/bin/env python3
"""Run the pipeline on laps of the training bag and score them (D-004).

Each lap is run as if it were a separate run: it starts from the GT pose at its first image set
(the rules allow the first pose of a run) and is scored with the official math on that lap's GT only.

    python scripts/validate.py --laps L5 --tag odom_only
    python scripts/validate.py --laps L5 --tag reloc --reloc
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import bag, config, gt as gtm, metrics, pipeline  # noqa: E402
from camloc.geometry import yaw_of  # noqa: E402
from camloc.trackmap import TrackMap  # noqa: E402


def first_image_stamp(bag_path, start_ns):
    return next(bag.iter_image_sets(bag_path, ('fl',), start_ns, start_ns + 2_000_000_000)).stamp_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--laps', nargs='+', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--reloc', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], help='override config values, e.g. filter.noise.wz=0.002')
    args = ap.parse_args()
    cfg = config.apply_overrides(config.load(args.config), args.set)

    A = lambda *p: config.artifact(cfg, *p)  # noqa: E731
    state = dict(np.load(A('state.npz')))
    track = TrackMap(dict(np.load(A('track_bounds.npz'))))
    gt = gtm.load_gt(A('gt.npz'))
    gti = gtm.interpolator(gt)
    laps = {lap['name']: lap for lap in json.load(open(A('laps.json')))['laps']}
    reloc = None
    if args.reloc:
        from camloc.reloc.query import Relocalizer
        reloc = Relocalizer(cfg)

    out_dir = os.path.join(cfg['results'], args.tag)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name in args.laps:
        lap = laps[name]
        t0 = first_image_stamp(cfg['bag'], lap['start_ns'])
        T = gti(t0)
        init = (t0, T[0, 3], T[1, 3], yaw_of(T[:3, :3]))
        print(f'{name}: {lap["duration_s"]:.0f} s, start pose ({init[1]:.1f}, {init[2]:.1f}, {np.degrees(init[3]):.1f} deg)')
        res = pipeline.run(cfg, cfg['bag'], state, track, init, end_ns=lap['end_ns'], reloc=reloc, log_every=200)
        sub = os.path.join(out_dir, f'{name}.csv')
        pipeline.write_csv(res, sub)
        np.savez(os.path.join(out_dir, f'{name}_diag.npz'), **res)
        gt_csv = A('gt_laps', f'{name}.csv')
        m = metrics.score(metrics.load_csv(gt_csv), res['stamp_ns'], res['x'], res['y'], res['yaw'])
        m['proc_ms_mean'] = 1e3 * float(np.mean(res['proc_s']))
        m['fix_rate_pct'] = 100.0 * float(np.mean(res['n_accepted'] > 0))
        m['fixes_accepted'] = int(res['n_accepted'].sum())
        m['fixes_rejected'] = int((res['n_fix'] - res['n_accepted']).sum())
        m['recoveries'] = int(res['recoveries'])
        if reloc:
            m['per_camera'] = {c: dict(v) for c, v in reloc.stats.items()}
            for v in reloc.stats.values():
                v.update(tried=0, fix=0, accepted=0)
        summary[name] = m
        with open(os.path.join(out_dir, f'{name}_official.txt'), 'w') as f:
            f.write(metrics.run_official(cfg['official_eval'], gt_csv, sub))
        print(f'  pos RMSE {m["pos_rmse_m"]:.3f} m  lat {m["lat_rmse_m"]:.3f}  lon {m["lon_rmse_m"]:.3f}  '
              f'yaw {m["yaw_rmse_deg"]:.3f} deg  <1m {m["within_1.0_pct"]:.1f}%  cov {m["coverage_pct"]:.1f}%  '
              f'fix {m["fix_rate_pct"]:.0f}%  {m["proc_ms_mean"]:.1f} ms/set')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'config': {k: cfg[k] for k in ('filter', 'track_constraint', 'reloc') if k in cfg},
                   'laps': summary}, f, indent=1)


if __name__ == '__main__':
    main()
