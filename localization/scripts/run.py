#!/usr/bin/env python3
"""Single entry point: raw mcap -> submission CSV (timestamp_ns,x,y,yaw), causal (D-005).

    python scripts/run.py --bag <run.mcap> --out submission.csv [--config configs/final.yaml]

Read from <run.mcap>, through camloc.allowed only: camera images, track borders, the car-state
motion fields and the first map -> base_link pose of the run. The relocalization map, masks and
camera calibration come from artifacts/ (built from the training bag, see run.sh). Ground truth is
never imported (tests/test_rules.py).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import bag, config, pipeline  # noqa: E402
from camloc.reloc.query import Relocalizer  # noqa: E402
from camloc.trackmap import TrackMap  # noqa: E402


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == 'Darwin':
            cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        elif os.path.exists('/proc/cpuinfo'):
            cpu = next(line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name'))
    except Exception:
        pass
    return {'cpu': cpu, 'cores': os.cpu_count(), 'system': platform.platform(), 'python': platform.python_version(),
            'gpu_used': False}


def check_calibration(cfg, bag_path):
    """The map was built with the training calibration; warn if the run's calibration differs."""
    ref = json.load(open(config.artifact(cfg, 'calibration.json')))
    cur = bag.read_calibration(bag_path)
    worst = 0.0
    for c in cfg['reloc']['cameras']:
        worst = max(worst, float(np.abs(np.array(ref[c]['K']) - cur[c].K).max()),
                    100 * float(np.abs(np.array(ref[c]['T_base_cam']) - cur[c].T_base_cam).max()))
    if worst > 1.0:
        print(f'WARNING: camera calibration of this run differs from the map\'s (max diff {worst:.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config')
    ap.add_argument('--first-pose', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
                    help='use instead of the first /tf pose of the bag')
    ap.add_argument('--no-cameras', action='store_true', help='ablation: car-state odometry (+ track constraint) only')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = config.apply_overrides(config.load(args.config), args.set)

    t = time.time()
    state = bag.read_state(args.bag)
    track = TrackMap(bag.read_track_bounds(args.bag))
    if args.first_pose:
        init = (int(state['stamp_ns'][0]), *args.first_pose)
    else:
        init = bag.read_first_pose(args.bag)
    reloc = None
    if not args.no_cameras:
        check_calibration(cfg, args.bag)
        reloc = Relocalizer(cfg, track=track)
    print(f'first pose ({init[1]:.2f}, {init[2]:.2f}, {np.degrees(init[3]):.1f} deg), '
          f'cameras {reloc.cameras if reloc else "none"}, map "{cfg["reloc"]["map_name"] if reloc else "-"}"')
    res = pipeline.run(cfg, args.bag, state, track, init, reloc=reloc, log_every=1000)
    pipeline.write_csv(res, args.out)
    np.savez(os.path.splitext(args.out)[0] + '_diag.npz', **res)    # per-image-set filter diagnostics

    proc = res['proc_s'] * 1e3
    runtime = {'image_sets': int(len(proc)), 'ms_per_set_mean': float(proc.mean()), 'ms_per_set_median': float(np.median(proc)),
               'ms_per_set_p95': float(np.percentile(proc, 95)), 'wall_s_total': time.time() - t,
               'fix_rate_pct': 100.0 * float(np.mean(res['n_accepted'] > 0)), 'hardware': hardware(),
               'cameras': reloc.cameras if reloc else [], 'per_camera': reloc.stats if reloc else {}}
    with open(os.path.splitext(args.out)[0] + '_runtime.json', 'w') as f:
        json.dump(runtime, f, indent=1)
    print(f'{len(proc)} poses -> {args.out}; {runtime["ms_per_set_mean"]:.1f} ms per image set (mean), '
          f'{runtime["ms_per_set_p95"]:.1f} ms p95 on {runtime["hardware"]["cpu"]}')


if __name__ == '__main__':
    main()
