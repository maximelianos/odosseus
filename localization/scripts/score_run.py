#!/usr/bin/env python3
"""Score a continuous run of the whole training bag (scripts/run.py output) overall and per lap, and
as the challenge's evaluation scheme (one line per weighted metric).

Laps that are in the map (split.train) are relocalized against their own images, so only the
held-out laps (tune, val) are honest numbers; the full-run line checks format, coverage, the
standstill start and the pit-lane passes.

    python scripts/score_run.py results/iter1/fullrun/submission.csv
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import config, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--config')
    args = ap.parse_args()
    cfg = config.load(args.config)
    sub = metrics.load_csv(args.submission)
    full_gt = config.artifact(cfg, 'gt_laps', 'full.csv')
    official = metrics.run_official(cfg['official_eval'], full_gt, args.submission)
    print(official)
    out = {'official_full_run': official, 'laps': {}}
    role = {n: r for r in ('train', 'tune', 'val') for n in cfg['split'][r]}
    print(f'{"lap":4} {"role":5} {"in map":6} {"pos":>6} {"lat":>6} {"lon":>6} {"yaw°":>6} {"<1m %":>6}')
    for lap in json.load(open(config.artifact(cfg, 'laps.json')))['laps']:
        m = (sub['timestamp_ns'] >= lap['start_ns']) & (sub['timestamp_ns'] < lap['end_ns'])
        gt = metrics.load_csv(config.artifact(cfg, 'gt_laps', f'{lap["name"]}.csv'))
        s = metrics.score(gt, sub['timestamp_ns'][m], sub['x'][m], sub['y'][m], sub['yaw'][m])
        r = role.get(lap['name'], 'unused')
        out['laps'][lap['name']] = dict(s, role=r)
        print(f'{lap["name"]:4} {r:5} {"yes" if r == "train" else "no":6} {s["pos_rmse_m"]:6.3f} {s["lat_rmse_m"]:6.3f} '
              f'{s["lon_rmse_m"]:6.3f} {s["yaw_rmse_deg"]:6.3f} {s["within_1.0_pct"]:6.1f}')
    out['scheme'] = scheme(cfg, sub, role, full_gt, os.path.splitext(args.submission)[0] + '_runtime.json')
    with open(os.path.splitext(args.submission)[0] + '_score.json', 'w') as f:
        json.dump(out, f, indent=1)


# quantitative part of the evaluation scheme (challenge brief): metric, weight
SCHEME = (('position RMSE [m]', 25), ('lateral RMSE [m]', 20), ('heading RMSE [deg]', 10),
          ('poses with position error < 1 m [%]', 10), ('runtime per image set, mean [ms]', 5))


def scheme(cfg, sub, role, full_gt, runtime_json):
    """The weighted metrics for the validation laps, the tuning lap, the in-map laps and all poses."""
    gt = metrics.load_csv(full_gt)
    e = metrics.errors(gt, sub['timestamp_ns'], sub['x'], sub['y'], sub['yaw'])
    laps = json.load(open(config.artifact(cfg, 'laps.json')))['laps']

    def mask(r):
        m = np.zeros(len(sub['timestamp_ns']), bool)
        names = sorted(lap['name'] for lap in laps if role.get(lap['name']) == r)
        for lap in laps:
            if lap['name'] in names:
                m |= (sub['timestamp_ns'] >= lap['start_ns']) & (sub['timestamp_ns'] < lap['end_ns'])
        return m & e['ok'], '+'.join(names) or 'none'

    (mv, nv), (mt, nt), (mm, _) = mask('val'), mask('tune'), mask('train')
    groups = {f'validation ({nv})': mv, f'tuning ({nt})': mt, 'laps in the map': mm, 'all poses': e['ok']}
    rt = json.load(open(runtime_json)) if os.path.exists(runtime_json) else {}
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')  # noqa: E731
    res = {}
    for g, m in groups.items():
        p = e['pos'][m]
        res[g] = {'poses': int(m.sum()),
                  SCHEME[0][0]: rms(p), SCHEME[1][0]: rms(e['lat'][m]), SCHEME[2][0]: rms(e['dyaw_deg'][m]),
                  SCHEME[3][0]: 100.0 * float(np.mean(p < 1.0)) if m.any() else float('nan'),
                  'longitudinal RMSE [m]': rms(e['lon'][m]),
                  'position median [m]': float(np.median(p)) if m.any() else float('nan'),
                  'position p95 [m]': float(np.percentile(p, 95)) if m.any() else float('nan'),
                  'position max [m]': float(p.max()) if m.any() else float('nan')}
    print(f'\n{"evaluation metric":38} {"weight":>6}  ' + '  '.join(f'{g:>18}' for g in groups))
    for name, w in SCHEME[:4]:
        print(f'{name:38} {w:5d} %  ' + '  '.join(f'{res[g][name]:18.3f}' for g in groups))
    ms = rt.get('ms_per_set_mean', float('nan'))
    print(f'{SCHEME[4][0]:38} {SCHEME[4][1]:5d} %  {"(whole run)":>18}  {ms:18.1f}   (p95 '
          f'{rt.get("ms_per_set_p95", float("nan")):.1f} ms, {rt.get("hardware", {}).get("cpu", "?")})')
    print(f'{"poses":38} {"":>6}   ' + '  '.join(f'{res[g]["poses"]:18d}' for g in groups))
    for name in ('longitudinal RMSE [m]', 'position median [m]', 'position p95 [m]', 'position max [m]'):
        print(f'{name:38} {"":>6}   ' + '  '.join(f'{res[g][name]:18.3f}' for g in groups))
    res['runtime_ms_mean'], res['runtime_ms_p95'] = ms, rt.get('ms_per_set_p95')
    return res


if __name__ == '__main__':
    main()
