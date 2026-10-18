#!/usr/bin/env python3
"""Pool the cross-validation folds (D-021): every fold is one continuous run of the whole bag
(scripts/run.py) with its own map; only its validation laps are scored. Each of L1-L4, L6, L7 is
validated in exactly one fold, so pooling the validation poses of all folds gives one number per
weighted metric, plus the per-fold and per-lap spread.

    python scripts/cv_score.py configs/cv_f1.yaml configs/cv_f2.yaml configs/cv_f3.yaml [--name submission]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import config, metrics  # noqa: E402

rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')  # noqa: E731


def summary(e):
    p = e['pos']
    return {'poses': int(len(p)), 'pos_rmse_m': rms(p), 'lat_rmse_m': rms(e['lat']), 'lon_rmse_m': rms(e['lon']),
            'yaw_rmse_deg': rms(e['dyaw_deg']), 'within_1.0_pct': 100.0 * float(np.mean(p < 1.0)),
            'pos_median_m': float(np.median(p)), 'pos_p95_m': float(np.percentile(p, 95)), 'pos_max_m': float(p.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--name', default='submission', help='CSV name inside <results>/fullrun/')
    ap.add_argument('--out', default='results/cv/cv_score.json')
    args = ap.parse_args()
    keys = ('pos', 'lat', 'lon', 'dyaw_deg')
    pooled = {k: [] for k in keys}
    res = {'folds': {}, 'laps': {}}
    seen = set()
    for path in args.configs:
        cfg = config.load(path)
        run_csv = os.path.join(cfg['results'], 'fullrun', f'{args.name}.csv')
        sub = metrics.load_csv(run_csv)
        gt = metrics.load_csv(config.artifact(cfg, 'gt_laps', 'full.csv'))
        e = metrics.errors(gt, sub['timestamp_ns'], sub['x'], sub['y'], sub['yaw'])
        laps = {lap['name']: lap for lap in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}
        fold_mask = np.zeros(len(sub['timestamp_ns']), bool)
        for name in cfg['split']['val']:
            assert name not in seen, f'{name} is validated in two folds'
            seen.add(name)
            w = laps[name]
            m = (sub['timestamp_ns'] >= w['start_ns']) & (sub['timestamp_ns'] < w['end_ns']) & e['ok']
            res['laps'][name] = dict(summary({k: e[k][m] for k in keys}), fold=os.path.basename(cfg['results']))
            fold_mask |= m
        rt_path = os.path.join(cfg['results'], 'fullrun', f'{args.name}_runtime.json')
        rt = json.load(open(rt_path)) if os.path.exists(rt_path) else {}
        res['folds'][os.path.basename(cfg['results'])] = dict(
            summary({k: e[k][fold_mask] for k in keys}), val='+'.join(cfg['split']['val']),
            map=cfg['reloc']['map_name'], runtime_ms_mean=rt.get('ms_per_set_mean'), runtime_ms_p95=rt.get('ms_per_set_p95'))
        for k in keys:
            pooled[k].append(e[k][fold_mask])
    res['pooled'] = summary({k: np.concatenate(v) for k, v in pooled.items()})
    res['pooled']['laps'] = '+'.join(sorted(seen))

    cols = ('pos_rmse_m', 'lat_rmse_m', 'lon_rmse_m', 'yaw_rmse_deg', 'within_1.0_pct',
            'pos_median_m', 'pos_p95_m', 'pos_max_m')
    head = f'{"":22} {"pos":>6} {"lat":>6} {"lon":>6} {"yaw°":>6} {"<1m%":>6} {"med":>6} {"p95":>6} {"max":>6}'
    row = lambda label, s: f'{label:22} ' + ' '.join(f'{s[c]:6.3f}' if c != 'within_1.0_pct' else f'{s[c]:6.1f}' for c in cols)  # noqa: E731
    print(head)
    for name in sorted(res['laps']):
        print(row(f'{name} ({res["laps"][name]["fold"]})', res['laps'][name]))
    print()
    for f, s in res['folds'].items():
        rt = f'  {s["runtime_ms_mean"]:.1f} ms' if s['runtime_ms_mean'] else ''
        print(row(f'{f}: {s["val"]}', s) + rt)
    print(row('pooled (6 laps)', res['pooled']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
