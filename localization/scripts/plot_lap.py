#!/usr/bin/env python3
"""Plot a lap result against GT: trajectory on the track map (coloured by position error) and the
lateral / longitudinal / yaw errors over time with the camera-fix rate.

    python scripts/plot_lap.py --run results/iter1/fullrun/submission.csv --laps L4 L7   # laps of a continuous run
    python scripts/plot_lap.py --tag tune_final --laps L5                                # a lap-restart run (D-004)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import config, metrics  # noqa: E402


def load_run_lap(cfg, run_csv, lap):
    """Slice a continuous run (scripts/run.py output + its _diag.npz) to one lap."""
    import json
    w = {x['name']: x for x in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}[lap]
    sub = metrics.load_csv(run_csv)
    diag = dict(np.load(os.path.splitext(run_csv)[0] + '_diag.npz'))
    m = (sub['timestamp_ns'] >= w['start_ns']) & (sub['timestamp_ns'] < w['end_ns'])
    md = (diag['stamp_ns'] >= w['start_ns']) & (diag['stamp_ns'] < w['end_ns'])
    return {k: v[m] for k, v in sub.items()}, {k: (v[md] if np.ndim(v) else v) for k, v in diag.items()}


def plot(cfg, tag, lap, compare=None, run_csv=None):
    if run_csv:
        d = os.path.dirname(os.path.abspath(run_csv))
        tag = f'continuous run, {os.path.basename(d)}'
        sub, diag = load_run_lap(cfg, run_csv, lap)
    else:
        d = os.path.join(cfg['results'], tag)
        sub = metrics.load_csv(os.path.join(d, f'{lap}.csv'))
        diag = dict(np.load(os.path.join(d, f'{lap}_diag.npz')))
    gt = metrics.load_csv(config.artifact(cfg, 'gt_laps', f'{lap}.csv'))
    bounds = dict(np.load(config.artifact(cfg, 'track_bounds.npz')))
    e = metrics.errors(gt, sub['timestamp_ns'], sub['x'], sub['y'], sub['yaw'])
    m = metrics.score(gt, sub['timestamp_ns'], sub['x'], sub['y'], sub['yaw'])
    t = (sub['timestamp_ns'] - sub['timestamp_ns'][0]) / 1e9

    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_axes([0.03, 0.06, 0.42, 0.88])
    for name, pts in bounds.items():
        ax.plot(pts[:, 0], pts[:, 1], color='0.6', lw=0.6, ls='--' if 'pit' in name else '-')
    ax.plot(gt['x'], gt['y'], color='k', lw=0.8, label='ground truth')
    sc = ax.scatter(sub['x'], sub['y'], c=np.clip(e['pos'], 0, 2), s=3, cmap='viridis', vmin=0, vmax=2,
                    label='estimate (colour = error)')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='position error [m] (clipped at 2)')
    ax.set_aspect('equal')
    ax.set_title(f'{lap} ({tag}): pos RMSE {m["pos_rmse_m"]:.2f} m, lateral {m["lat_rmse_m"]:.2f} m, '
                 f'yaw {m["yaw_rmse_deg"]:.2f}°, <1 m {m["within_1.0_pct"]:.0f} %', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    axes = [fig.add_axes([0.53, 0.70 - 0.21 * i, 0.44, 0.18]) for i in range(4)]
    axes[0].plot(t, e['lat'], lw=0.8, color='tab:red')
    axes[0].set_ylabel('lateral [m]')
    axes[1].plot(t, e['lon'], lw=0.8, color='tab:blue')
    axes[1].set_ylabel('longitudinal [m]')
    axes[2].plot(t, e['dyaw_deg'], lw=0.8, color='tab:green')
    axes[2].set_ylabel('yaw [deg]')
    if compare:
        csub = metrics.load_csv(os.path.join(cfg['results'], compare, f'{lap}.csv'))
        ce = metrics.errors(gt, csub['timestamp_ns'], csub['x'], csub['y'], csub['yaw'])
        ct = (csub['timestamp_ns'] - sub['timestamp_ns'][0]) / 1e9
        for a, k in zip(axes[:3], ('lat', 'lon', 'dyaw_deg')):
            a.plot(ct, ce[k], lw=0.6, color='0.6', label=compare, zorder=0)
        axes[0].legend(fontsize=8, loc='upper right')
    for a in axes[:3]:
        a.axhline(0, color='k', lw=0.5)
        a.grid(alpha=0.3)
    win = np.ones(10)
    hit = (diag['n_accepted'] > 0).astype(float)
    rate = np.convolve(hit, win, mode='same') / np.convolve(np.ones_like(hit), win, mode='same') * 100
    axes[3].plot(t, rate, lw=0.8, color='tab:purple', label='image sets with an accepted fix [%]')
    axes[3].plot(t, diag['sigma_xy'] * 100, lw=0.8, color='tab:orange', label='filter sigma_xy [cm]')
    axes[3].set_ylim(0, 105)
    axes[3].set_xlabel('time in lap [s]')
    axes[3].legend(fontsize=8, loc='lower left')
    axes[3].grid(alpha=0.3)
    out = os.path.join(d, f'{lap}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tag', help='a lap-restart result directory under results/<iteration>/')
    src.add_argument('--run', help='a continuous-run submission CSV (with its _diag.npz)')
    ap.add_argument('--laps', nargs='+', required=True)
    ap.add_argument('--compare', help='another tag to draw in grey')
    args = ap.parse_args()
    cfg = config.load(args.config)
    for lap in args.laps:
        print(plot(cfg, args.tag, lap, args.compare, args.run))


if __name__ == '__main__':
    main()
