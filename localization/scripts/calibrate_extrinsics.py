#!/usr/bin/env python3
"""Refine each camera's mounting rotation from image motion (D-012; uses GT of the training laps).

The README says the extrinsic yaw is reliable only for r and rl (+-1 deg for fl/fr, older calibration
for cl/cr) and roll is not calibrated. On the narrow fl camera 1 deg is ~45 px, which breaks the
epipolar check used for triangulation. For consecutive frames, the camera motion predicted from the
GT base motion and the extrinsic must agree with the image matches; we find the small rotation
correction dR (3 parameters per camera) that minimizes the Sampson epipolar error of 5-point-RANSAC
inlier matches, R_base_cam' = R_base_cam * Exp(dR). The translation (mount values) is kept.

    python scripts/calibrate_extrinsics.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import allowed, bag, config, gt as gtm  # noqa: E402
from camloc.geometry import inv_T  # noqa: E402
from camloc.reloc.features import Extractor, decode_half_gray  # noqa: E402
from camloc.reloc.mapbuild import load_mask, skew  # noqa: E402


def sampson_px(params, pairs, T_bc, fx):
    dR = Rotation.from_rotvec(params).as_matrix()
    T = T_bc.copy()
    T[:3, :3] = T_bc[:3, :3] @ dR
    out = []
    for Bk, Bj, xk, xj in pairs:
        Tjk = inv_T(Bj @ T) @ (Bk @ T)
        E = skew(Tjk[:3, 3]) @ Tjk[:3, :3]
        Ex, Etx = xk @ E.T, xj @ E
        num = np.sum(xj * Ex, axis=1)
        den = np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2) + 1e-18
        out.append(num / den * fx)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--laps', nargs='+', default=['L1', 'L2', 'L3', 'L6'])
    ap.add_argument('--pairs', type=int, default=120)
    args = ap.parse_args()
    cfg = config.load(args.config)
    assert not set(args.laps) & set(cfg['split']['val'] + cfg['split']['tune'])
    calib = {c: bag.Camera.from_dict(d) for c, d in json.load(open(config.artifact(cfg, 'calibration.json'))).items()}
    gti = gtm.interpolator(gtm.load_gt(config.artifact(cfg, 'gt.npz')))
    state = dict(np.load(config.artifact(cfg, 'state.npz')))
    laps = {x['name']: x for x in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}
    ext = {c: Extractor(calib[c], cfg['reloc']['features'], load_mask(cfg, c)) for c in allowed.CAMERAS}
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # sample times spread over the laps where the car moves > 10 m/s
    per_lap = int(np.ceil(args.pairs / len(args.laps)))
    samples = []
    for n in args.laps:
        a, b = laps[n]['start_ns'], laps[n]['end_ns']
        m = (state['stamp_ns'] > a) & (state['stamp_ns'] < b) & (state['v_mps'] > 10)
        ts = state['stamp_ns'][m]
        samples += list(ts[np.linspace(0, len(ts) - 1, per_lap + 2).astype(int)[1:-1]])
    samples = sorted(samples)

    pairs = {c: [] for c in allowed.CAMERAS}
    prev = {}
    si = 0
    for n in args.laps:
        for s in bag.iter_image_sets(cfg['bag'], allowed.CAMERAS, laps[n]['start_ns'], laps[n]['end_ns']):
            while si < len(samples) and samples[si] < s.stamp_ns - 200_000_000:
                si += 1
            if si >= len(samples):
                break
            take = samples[si] <= s.stamp_ns
            for c, data in s.images.items():
                if c in prev:
                    tk, fk = prev.pop(c)
                    tj = s.stamps[c]
                    uj, dj, _ = ext[c](decode_half_gray(data))
                    uk, dk = fk
                    mt = [p[0] for p in bf.knnMatch(dk.astype(np.float32), dj.astype(np.float32), k=2)
                          if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
                    if len(mt) < 30:
                        continue
                    xk = uk[[m.queryIdx for m in mt]].astype(np.float64)
                    xj = uj[[m.trainIdx for m in mt]].astype(np.float64)
                    E, inl = cv2.findEssentialMat(xk, xj, np.eye(3), cv2.RANSAC, 0.999, 1.0 / calib[c].K[0, 0])
                    if inl is None or inl.sum() < 20:
                        continue
                    inl = inl.ravel() > 0
                    pairs[c].append((gti(tk), gti(tj), np.c_[xk[inl], np.ones(inl.sum())], np.c_[xj[inl], np.ones(inl.sum())]))
                elif take:
                    u, d, _ = ext[c](decode_half_gray(data))
                    prev[c] = (s.stamps[c], (u, d))
            if take:
                si += 1

    report = {}
    refined = json.load(open(config.artifact(cfg, 'calibration.json')))
    for c in allowed.CAMERAS:
        P, fx, T_bc = pairs[c], calib[c].K[0, 0], calib[c].T_base_cam
        if len(P) < 10:
            print(f'{c}: only {len(P)} pairs, keeping the bag extrinsics')
            continue
        r0 = np.abs(sampson_px(np.zeros(3), P, T_bc, fx))
        sol = least_squares(sampson_px, np.zeros(3), args=(P, T_bc, fx), loss='soft_l1', f_scale=1.0)
        r1 = np.abs(sampson_px(sol.x, P, T_bc, fx))
        # express the correction as rotation about the camera axes: x right (tilt/pitch), y down (pan/yaw), z forward (roll)
        d = np.degrees(sol.x)
        report[c] = {'pairs': len(P), 'matches': int(len(r0)), 'sampson_px_median_before': float(np.median(r0)),
                     'sampson_px_median_after': float(np.median(r1)), 'within_2px_before_pct': float(100 * np.mean(r0 < 2)),
                     'within_2px_after_pct': float(100 * np.mean(r1 < 2)),
                     'correction_deg_about_cam_x_y_z': d.tolist(), 'correction_deg_total': float(np.linalg.norm(d))}
        T = T_bc.copy()
        T[:3, :3] = T_bc[:3, :3] @ Rotation.from_rotvec(sol.x).as_matrix()
        refined[c]['T_base_cam'] = T.tolist()
        print(f'{c}: {len(P):3d} pairs {len(r0):6d} matches  Sampson median {np.median(r0):6.2f} -> {np.median(r1):5.2f} px  '
              f'<2px {100 * np.mean(r0 < 2):5.1f} -> {100 * np.mean(r1 < 2):5.1f} %  correction (cam x,y,z) '
              f'{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f} deg')
    with open(config.artifact(cfg, 'calibration_refined.json'), 'w') as f:
        json.dump(refined, f, indent=1)
    with open(os.path.join(cfg['results'], 'extrinsics_refinement.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
