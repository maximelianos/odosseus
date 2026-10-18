#!/usr/bin/env python3
"""Format checks on a submission CSV before handing it in (D-005, D-016): exact header, one row per
image set, strictly increasing integer stamps that are real image stamps of the bag, finite values,
yaw in [-pi, pi], and coverage of the 1 s bins of the bag's image time span (>= 90 % to be ranked).

    python scripts/check_submission.py <submission.csv> --bag <run.mcap>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import allowed, bag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--bag', required=True)
    args = ap.parse_args()
    with open(args.submission) as f:
        header = f.readline().strip()
    d = np.genfromtxt(args.submission, delimiter=',', skip_header=1, dtype=None, encoding=None,
                      names=('t', 'x', 'y', 'yaw'))
    t = np.asarray(d['t'], dtype=np.int64)
    xyz = np.c_[d['x'], d['y'], d['yaw']].astype(float)
    image_stamps = set()
    for s in bag.iter_image_sets(args.bag, allowed.CAMERAS):
        image_stamps.update(s.stamps.values())
    t_img = np.array(sorted(image_stamps), dtype=np.int64)
    bins = np.arange(t_img[0], t_img[-1], 1_000_000_000)
    covered = len(np.unique(np.searchsorted(bins, t, side='right') - 1)) / len(bins)
    checks = {
        'header is timestamp_ns,x,y,yaw': header == 'timestamp_ns,x,y,yaw',
        'stamps strictly increasing': bool(np.all(np.diff(t) > 0)),
        'every stamp is an image header stamp': bool(np.isin(t, t_img).all()),
        'all values finite': bool(np.isfinite(xyz).all()),
        'yaw within [-pi, pi]': bool(np.all(np.abs(xyz[:, 2]) <= np.pi + 1e-9)),
        f'coverage of 1 s bins >= 90 % ({100 * covered:.1f} %)': covered >= 0.9,
    }
    print(f'{len(t)} poses, {len(t_img)} image stamps in the bag')
    for k, ok in checks.items():
        print(f'  [{"ok" if ok else "FAIL"}] {k}')
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
