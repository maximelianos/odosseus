#!/usr/bin/env python3
"""Errors of one lap of a continuous run over time, shaded by map coverage of the driven line
(D-021, D-022): where the lap's GT path is more than 1 m from every map lap's GT path, the map has
no keyframes on that line.

    python scripts/plot_coverage.py --config configs/cv_f2_swapped.yaml --run results/cv/f2_swapped/fullrun/submission.csv --lap L2
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import config, metrics  # noqa: E402

OFF_LINE_M = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--lap', required=True)
    args = ap.parse_args()
    cfg = config.load(args.config)
    laps = {lap['name']: lap for lap in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}
    gt = metrics.load_csv(config.artifact(cfg, 'gt_laps', 'full.csv'))

    def gt_path(name):
        w = laps[name]
        m = (gt['timestamp_ns'] >= w['start_ns']) & (gt['timestamp_ns'] < w['end_ns'])
        return np.c_[gt['x'][m], gt['y'][m]]

    w = laps[args.lap]
    sub = metrics.load_csv(args.run)
    diag = dict(np.load(os.path.splitext(args.run)[0] + '_diag.npz'))
    m = (sub['timestamp_ns'] >= w['start_ns']) & (sub['timestamp_ns'] < w['end_ns'])
    md = (diag['stamp_ns'] >= w['start_ns']) & (diag['stamp_ns'] < w['end_ns'])
    e = metrics.errors(gt, sub['timestamp_ns'][m], sub['x'][m], sub['y'][m], sub['yaw'][m])
    t = (sub['timestamp_ns'][m] - w['start_ns']) / 1e9
    dist, _ = cKDTree(np.vstack([gt_path(n) for n in cfg['split']['train']])).query(np.c_[e['gt_x'], e['gt_y']])
    off = dist > OFF_LINE_M
    fix = diag['n_accepted'][md] > 0

    fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True)
    series = [(e['pos'], 'position [m]', 'k'), (e['lat'], 'lateral [m]', 'tab:red'),
              (e['lon'], 'longitudinal [m]', 'tab:blue'), (e['dyaw_deg'], 'heading [deg]', 'tab:green')]
    for a, (v, lab, c) in zip(axes, series):
        a.plot(t, v, lw=0.8, color=c)
        a.axhline(0, color='k', lw=0.4)
        a.set_ylabel(lab)
    axes[0].axhline(1.0, color='tab:red', lw=0.6, ls='--', label='1 m')
    axes[4].plot(t, dist, lw=0.8, color='tab:brown')
    axes[4].axhline(OFF_LINE_M, color='k', lw=0.5, ls='--')
    axes[4].set_ylabel('distance to\nnearest map lap [m]')
    win = np.ones(10)
    rate = np.convolve(fix.astype(float), win, 'same') / np.convolve(np.ones(len(fix)), win, 'same') * 100
    axes[5].plot(t, rate, lw=0.8, color='tab:purple', label='image sets with an accepted fix [%] (1 s window)')
    axes[5].plot(t, diag['sigma_xy'][md] * 100, lw=0.8, color='tab:orange', label='filter σ_xy [cm]')
    axes[5].set_ylim(0, 105)
    axes[5].set_xlabel(f'time in {args.lap} [s]')
    axes[5].legend(fontsize=8, loc='lower left')
    edges = np.flatnonzero(np.diff(np.r_[0, off.astype(int), 0]))
    for a in axes:
        for i, j in zip(edges[::2], edges[1::2]):
            a.axvspan(t[i], t[j - 1], color='tab:orange', alpha=0.12, lw=0)
        a.grid(alpha=0.3)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
    fig.suptitle(f'{args.lap}, {os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.run))))}: '
                 f'map {" ".join(cfg["split"]["train"])}.  Shaded = line not in the map (> {OFF_LINE_M:.0f} m, '
                 f'{100 * off.mean():.0f} % of the lap).\n'
                 f'pos RMSE {rms(e["pos"]):.3f} m (on line {rms(e["pos"][~off]):.3f}, off line {rms(e["pos"][off]):.3f}); '
                 f'lateral {rms(e["lat"]):.3f} m; heading {rms(e["dyaw_deg"]):.3f}°; '
                 f'fixes {100 * fix[~off].mean():.0f} % on / {100 * fix[off].mean():.0f} % off line', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(os.path.dirname(args.run), f'{args.lap}_coverage.png')
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
