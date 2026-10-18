#!/usr/bin/env python3
"""Per-camera relocalization diagnostics on a lap (D-013, D-014).

The GT pose at each image stamp is used as the search prior, optionally perturbed. This is an
offline diagnostic to compare cameras and set thresholds; the pipeline uses the filter estimate.

    python scripts/reloc_diag.py --map map_diag --cams fl fr cl cr r rl rr --lap L5 --step 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import bag, config, gt as gtm  # noqa: E402
from camloc.geometry import wrap, yaw_of  # noqa: E402
from camloc.reloc.query import Relocalizer  # noqa: E402


def summarize(recs):
    ok = [r for r in recs if r.get('ok')]
    out = {'tried': len(recs), 'fix_pct': 100.0 * len(ok) / max(len(recs), 1),
           'fail': {k: sum(r.get('fail') == k for r in recs) for k in sorted({r.get('fail') for r in recs} - {None})},
           'ms': float(np.mean([r['ms'] for r in recs]))}
    if ok:
        pos = np.array([r['err_pos'] for r in ok])
        out.update(pos_median=float(np.median(pos)), pos_p90=float(np.percentile(pos, 90)),
                   pos_p99=float(np.percentile(pos, 99)), over_1m_pct=100.0 * float(np.mean(pos > 1)),
                   over_3m_pct=100.0 * float(np.mean(pos > 3)),
                   lat_rmse_robust=float(np.sqrt(np.mean(np.clip(np.array([r['err_lat'] for r in ok]), -3, 3) ** 2))),
                   lon_median_abs=float(np.median(np.abs([r['err_lon'] for r in ok]))),
                   yaw_median_abs_deg=float(np.median(np.abs([r['err_yaw_deg'] for r in ok]))),
                   inliers_median=float(np.median([r['inliers'] for r in ok])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--map', default='map_diag')
    ap.add_argument('--cams', nargs='+', required=True)
    ap.add_argument('--lap', default='L5')
    ap.add_argument('--step', type=int, default=3, help='use every n-th image set')
    ap.add_argument('--prior-noise', type=float, nargs=2, default=(0.0, 0.0), metavar=('XY_M', 'YAW_DEG'))
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = config.apply_overrides(config.load(args.config), args.set)
    gti = gtm.interpolator(gtm.load_gt(config.artifact(cfg, 'gt.npz')))
    lap = {x['name']: x for x in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}[args.lap]
    rel = Relocalizer(cfg, map_name=args.map, cameras=args.cams)
    rng = np.random.default_rng(0)
    recs = {c: [] for c in args.cams}
    n = 0
    for s in bag.iter_image_sets(cfg['bag'], args.cams, lap['start_ns'], lap['end_ns'], decode=rel.decode):
        n += 1
        if n % args.step:
            continue
        for c, img in s.images.items():
            T = gti(s.stamps[c])
            gx, gy, gyaw = T[0, 3], T[1, 3], yaw_of(T[:3, :3])
            px = gx + rng.normal(0, args.prior_noise[0])
            py = gy + rng.normal(0, args.prior_noise[0])
            pyaw = gyaw + np.radians(rng.normal(0, args.prior_noise[1]))
            t = time.perf_counter()
            # the prior's stated uncertainty: the injected noise, but never less than a typical filter sigma
            f = rel.localize_camera(c, img, px, py, pyaw, max(args.prior_noise[0], 0.5),
                                    np.radians(max(args.prior_noise[1], 0.3)))
            f['ms'] = 1e3 * (time.perf_counter() - t)
            if 'x' in f:
                dx, dy = f['x'] - gx, f['y'] - gy
                f.update(err_pos=float(np.hypot(dx, dy)), err_lon=float(np.cos(gyaw) * dx + np.sin(gyaw) * dy),
                         err_lat=float(-np.sin(gyaw) * dx + np.cos(gyaw) * dy),
                         err_yaw_deg=float(np.degrees(wrap(f['yaw'] - gyaw))))
            f['stamp_ns'] = int(s.stamps[c])
            recs[c].append(f)
    summary = {c: summarize(r) for c, r in recs.items()}
    tag = f'reloc_diag_{args.map}_{args.lap}' + (f'_noise{args.prior_noise[0]:g}' if args.prior_noise[0] else '')
    with open(os.path.join(cfg['results'], f'{tag}.json'), 'w') as f:
        json.dump({'summary': summary, 'records': recs}, f)
    print(f'{"cam":3} {"fix%":>5} {"med m":>6} {"p90 m":>6} {"p99 m":>6} {">1m%":>5} {">3m%":>5} {"lat":>5} {"lon":>5} '
          f'{"yaw°":>5} {"inl":>4} {"ms":>5}  failures')
    for c, s in summary.items():
        if 'pos_median' in s:
            print(f'{c:3} {s["fix_pct"]:5.1f} {s["pos_median"]:6.2f} {s["pos_p90"]:6.2f} {s["pos_p99"]:6.2f} '
                  f'{s["over_1m_pct"]:5.1f} {s["over_3m_pct"]:5.1f} {s["lat_rmse_robust"]:5.2f} {s["lon_median_abs"]:5.2f} '
                  f'{s["yaw_median_abs_deg"]:5.2f} {s["inliers_median"]:4.0f} {s["ms"]:5.1f}  {s["fail"]}')
        else:
            print(f'{c:3} {s["fix_pct"]:5.1f}  no fixes  {s["ms"]:5.1f} ms  {s["fail"]}')


if __name__ == '__main__':
    main()
