#!/usr/bin/env python3
"""Static-feature masks per camera (D-011): image regions where the SAME feature (descriptor match)
appears at the SAME pixel in frames taken at different places on the track. Such features sit on the
car body or the lens, move with the car, waste the feature budget and add outliers. Built from images
only (no GT).

Rejected alternatives (see D-011): pixel intensity variance (at night the body brightness changes
with the floodlights as much as the background), and keypoint-location frequency (keypoints are
always dense along the horizon, which is the most useful band).

    python scripts/build_masks.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import allowed, bag, config  # noqa: E402
from camloc.reloc.features import Extractor, decode_half_gray  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--max-shift-px', type=float, default=3.0, help='full-res pixels')
    ap.add_argument('--freq-thr', type=float, default=0.05, help='share of frame pairs with a static match')
    args = ap.parse_args()
    cfg = config.load(args.config)
    state = dict(np.load(config.artifact(cfg, 'state.npz')))
    cams = {c: bag.Camera.from_dict(d) for c, d in json.load(open(config.artifact(cfg, 'calibration.json'))).items()}
    fcfg = cfg['reloc']['features']

    # frames spread over the run (seconds apart) where the car is fast: nothing in the scene can
    # stay at the same pixel, only things attached to the car
    fast = state['stamp_ns'][state['v_mps'] > 20]
    picks = set(fast[np.linspace(0, len(fast) - 1, args.frames).astype(int)] // 10**8)
    ext = {c: Extractor(cams[c], fcfg) for c in allowed.CAMERAS}
    feats = {c: [] for c in allowed.CAMERAS}
    first = {}
    for s in bag.iter_image_sets(cfg['bag'], allowed.CAMERAS):
        if s.stamp_ns // 10**8 not in picks:
            continue
        for c, data in s.images.items():
            g = decode_half_gray(data)
            _, d, px = ext[c](g)
            feats[c].append((px, d.astype(np.float32)))
            first.setdefault(c, g)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    previews, report = [], {}
    for c in allowed.CAMERAS:
        h, w = first[c].shape
        hits = np.zeros((h, w), np.float32)
        pairs = 0
        for (pa, da), (pb, db) in zip(feats[c][:-1], feats[c][1:]):
            if len(da) < 2 or len(db) < 2:
                continue
            pairs += 1
            hit = np.zeros((h, w), np.uint8)
            for m in bf.knnMatch(da, db, k=2):
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
                    a, b = pa[m[0].queryIdx], pb[m[0].trainIdx]
                    if np.linalg.norm(a - b) < args.max_shift_px:
                        u, v = ((a - 0.5) / 2.0).astype(int)
                        cv2.circle(hit, (int(u), int(v)), 4, 1, -1)
            hits += hit
        f = hits / max(pairs, 1)
        static = cv2.dilate((f > args.freq_thr).astype(np.uint8), np.ones((11, 11), np.uint8))
        mask = np.where(static > 0, 0, 255).astype(np.uint8)
        cv2.imwrite(config.artifact(cfg, 'masks', f'{c}.png'), mask)
        report[c] = {'masked_pct': float(100 * (mask == 0).mean()), 'pairs': pairs, 'max_freq': float(f.max())}
        print(f'{c}: masked {report[c]["masked_pct"]:5.1f} % ({pairs} frame pairs, max static-match frequency {f.max():.2f})')
        vis = cv2.cvtColor(first[c], cv2.COLOR_GRAY2BGR)
        vis[mask == 0] = (0.4 * vis[mask == 0] + [0, 0, 150]).astype(np.uint8)
        cv2.putText(vis, c, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        previews.append(vis)
    with open(config.artifact(cfg, 'masks', 'report.json'), 'w') as fo:
        json.dump(report, fo, indent=1)
    blank = np.zeros_like(previews[0])
    grid = np.vstack([np.hstack(previews[:4]), np.hstack(previews[4:] + [blank])])
    cv2.imwrite(os.path.join(cfg['results'], 'masks_preview.jpg'), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()
