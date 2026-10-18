#!/usr/bin/env python3
"""Prepare the training bag: caches (calibration, track, car state, GT), lap table, split, per-lap GT
files for the official scorer, and the data checks behind D-001..D-004.

    python scripts/prepare.py [--config configs/iter1.yaml]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import bag, config, gt as gtm, laps  # noqa: E402
from camloc.trackmap import TrackMap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    cfg = config.load(ap.parse_args().config)
    A = lambda *p: config.artifact(cfg, *p)  # noqa: E731

    print('calibration ...')
    cams = bag.read_calibration(cfg['bag'])
    with open(A('calibration.json'), 'w') as f:
        json.dump({c: cam.to_dict() for c, cam in cams.items()}, f, indent=1)

    print('track bounds ...')
    bounds = bag.read_track_bounds(cfg['bag'])
    np.savez(A('track_bounds.npz'), **bounds)
    track = TrackMap(bounds)

    print('car state (allowed fields only) ...')
    state = bag.read_state(cfg['bag'])
    np.savez(A('state.npz'), **state)

    print('ground truth (training/scoring only) ...')
    g = gtm.read_gt(cfg['bag'])
    gtm.save_gt(g, A('gt.npz'))
    t, xy, yaw = g['stamp_ns'], g['pos'][:, :2], gtm.yaws(g)
    report = {}

    # D-001: vx/vy are body-frame velocities (no heading inside)
    v = state['v_mps']
    report['vy_abs_max_moving'] = float(np.abs(state['vy_mps'][v > 3]).max())

    # border orientation vs. driving direction (sampled), needed for signed offsets
    idx = np.arange(0, len(t), 200)
    report['border_orientation_cos'] = track.check_orientation(xy[idx], yaw[idx])

    # laps (D-002)
    a, b = laps.gate_segment(track, cfg['laps']['gate_point'])
    t0, t1 = bag.bag_time_range(cfg['bag'])
    table = laps.lap_table(t, xy, a, b, int(t[0]), int(t[-1]))
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    gtv = np.interp(t, state['stamp_ns'], state['v_mps'])
    split = {name: role for role in ('train', 'tune', 'val') for name in cfg['split'][role]}
    margins_all = np.array([track.margins(p) for p in xy[::10]], dtype=object)
    in_pit = np.array([m['pit'] > 0 and m['track'] <= 0 for m in margins_all])
    ts = t[::10]
    print(f'\n{"lap":4} {"role":5} {"start s":>8} {"dur s":>6} {"dist m":>7} {"vmax":>5} {"vmean":>5} {"pit s":>6}')
    for lap in table:
        m = (t >= lap['start_ns']) & (t < lap['end_ns'])
        mp = (ts >= lap['start_ns']) & (ts < lap['end_ns'])
        lap.update(role=split.get(lap['name'], 'unused'),
                   start_s=(lap['start_ns'] - t[0]) / 1e9, duration_s=(lap['end_ns'] - lap['start_ns']) / 1e9,
                   distance_m=float(dist[m][-1] - dist[m][0]), vmax=float(gtv[m].max()),
                   vmean=float(gtv[m].mean()), pit_s=float(in_pit[mp].sum() * np.median(np.diff(ts)) / 1e9))
        print(f'{lap["name"]:4} {lap["role"]:5} {lap["start_s"]:8.1f} {lap["duration_s"]:6.1f} '
              f'{lap["distance_m"]:7.0f} {lap["vmax"]:5.1f} {lap["vmean"]:5.1f} {lap["pit_s"]:6.1f}')
    with open(A('laps.json'), 'w') as f:
        json.dump({'gate': [a.tolist(), b.tolist()], 'laps': table}, f, indent=1)

    # per-lap GT files for the official scorer (D-004)
    for lap in table:
        gtm.write_eval_csv(g, A('gt_laps', f'{lap["name"]}.csv'), lap['start_ns'], lap['end_ns'])
    gtm.write_eval_csv(g, A('gt_laps', 'full.csv'))

    # D-008: how far does the GT base_link go outside the drivable area?
    inside = np.array([max(m.values()) for m in margins_all])
    report['gt_inside_margin_m'] = {f'p{p}': float(np.percentile(inside, p)) for p in (0.1, 1, 5, 50)}
    report['gt_outside_share'] = float(np.mean(inside < 0))
    # ground height: GT z minus track-surface z from the borders (for PnP sanity checks)
    dz = np.array([p[2] - track.ground_z(p[:2]) for p in g['pos'][::50]])
    report['gt_z_minus_ground_m'] = {'median': float(np.median(dz)), 'p1': float(np.percentile(dz, 1)),
                                     'p99': float(np.percentile(dz, 99))}
    with open(A('prepare_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('\n' + json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
