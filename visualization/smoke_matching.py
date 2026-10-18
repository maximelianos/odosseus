#!/usr/bin/env python3
"""Smoke test: do daylight frames find enough matches in night frames of the same place?

SuperPoint + LightGlue (the alternate approach on origin/main, odo/) next to SIFT (the current approach,
localization/camloc), on the same image pairs, with night-vs-night pairs as the reference.

Places: every 2.5 s of the first 30 s of the daytime lap, located by distance along the track from the
start line (car-state speed integrated; the lap starts 6 m along L7, daybag.stand_in_first_pose). At each
place and camera:
    query: the day frame, and as the reference the night frame of L7 at the same distance
    map:   night frames of L3 and L6 at that distance (L7 is in neither)
Matching uses no pose guess, then an essential-matrix RANSAC on undistorted coordinates (2 px at full
resolution) keeps only geometrically consistent matches:
    sp_lg  SuperPoint (2048 keypoints, long side 1024 px, car body cropped as odo/prep.py CROP_BOTTOM) +
           LightGlue (no early stopping), as the alternate approach's map building and relocalisation
    sp_nn  SuperPoint mutual nearest neighbours with cosine similarity > 0.75 (odo/online.py sim1): its
           tracking matcher without the projection window around the predicted pose
    sift   camloc's Extractor (CLAHE, half resolution, car-body masks) + ratio test 0.8 (D-010, D-013), its
           keypoints cut to the same image region as SuperPoint's (CROP_BOTTOM)
A fix needs >= 30 inliers in the alternate approach (min_inl) and >= 15 in camloc (checks.min_inliers);
the counts here are two-view inliers, a proxy for how many correct correspondences exist.

    conda run -n odosseus-sp python visualization/smoke_matching.py      # env with torch + LightGlue
Writes visualization/outputs/smoke_matching/: pairs.csv, summary.json, match pictures.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402
import daybag  # noqa: E402
from camloc import bag, config  # noqa: E402
from camloc.reloc.features import Extractor as SiftExtractor, decode_half_gray  # noqa: E402
from camloc.reloc.mapbuild import load_cameras, load_mask  # noqa: E402

CAMS = ('fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr')
CROP_BOTTOM = {'fl': 1.0, 'fr': 0.55, 'cl': 0.6, 'cr': 0.6, 'r': 1.0, 'rl': 1.0, 'rr': 0.6}   # origin/main odo/prep.py
METHODS = ('sp_lg', 'sp_nn', 'sift')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'smoke_matching')


class SuperPointLG:
    """SuperPoint + LightGlue with the alternate approach's settings (odo/feature_vo.py Extractor)."""

    def __init__(self, dev):
        from lightglue import LightGlue, SuperPoint
        self.dev = dev
        self.sp = SuperPoint(max_num_keypoints=2048).eval().to(dev)
        self.lg = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(dev)

    @torch.no_grad()
    def features(self, gray, cam):
        f = self.sp.extract(torch.from_numpy(gray).float().div(255)[None, None].to(self.dev))
        keep = f['keypoints'][0, :, 1] < CROP_BOTTOM[cam] * gray.shape[0]          # car body (odo/map_build.crop_feats)
        return {k: v[:, keep] if v.dim() > 1 and v.shape[1] == keep.shape[0] else v for k, v in f.items()}

    @torch.no_grad()
    def lightglue(self, f0, f1):
        return self.lg({'image0': f0, 'image1': f1})['matches'][0].cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def mutual_nn(f0, f1, sim=0.75):
        s = f0['descriptors'][0] @ f1['descriptors'][0].T                      # unit-length descriptors
        b01, b10 = s.argmax(1), s.argmax(0)
        i = torch.arange(len(b01), device=s.device)
        ok = (b10[b01] == i) & (s[i, b01] > sim)
        return torch.stack([i[ok], b01[ok]], 1).cpu().numpy()


def crop_sift(f, cam, H=728):
    """The same image region as SuperPoint: camloc's car-body masks leave parts of the bodywork, and points that
    move with the camera pass any two-view check."""
    uv, desc, px = f
    keep = px[:, 1] < CROP_BOTTOM[cam] * H
    return uv[keep], desc[keep], px[keep]


def sift_match(d0, d1, ratio=0.8):
    if len(d0) < 2 or len(d1) < 2:
        return np.zeros((0, 2), int)
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0.astype(np.float32), d1.astype(np.float32), k=2)
    return np.array([(a.queryIdx, a.trainIdx) for a, b in (x for x in m if len(x) == 2)
                     if a.distance < ratio * b.distance], int).reshape(-1, 2)


def verify(n0, n1, f_px, thr_px):
    """Essential-matrix RANSAC on normalized coordinates -> inlier mask."""
    if len(n0) < 8:
        return np.zeros(len(n0), bool)
    E, inl = cv2.findEssentialMat(n0.astype(np.float64), n1.astype(np.float64), np.eye(3), method=cv2.USAC_MAGSAC,
                                  prob=0.99999, threshold=thr_px / f_px, maxIters=10000)
    return np.zeros(len(n0), bool) if inl is None else inl.ravel().astype(bool)


def lap_distance(lap):
    """GT stamps and distance travelled since the lap start (the start/finish line) of a training lap."""
    L = ref.laps()[lap]
    g = ref.cut(ref.ground_truth(), L['start_ns'], L['end_ns'])
    return g['timestamp_ns'], np.r_[0, np.cumsum(np.hypot(np.diff(g['x']), np.diff(g['y'])))]


def image_set_at(bag_path, stamp, cams):
    sets = list(bag.iter_image_sets(bag_path, cams, int(stamp) - 60_000_000, int(stamp) + 60_000_000))
    return min(sets, key=lambda s: abs(s.stamp_ns - stamp))


def draw(q_img, m_img, kq, km, inl, title):
    """Query | map side by side (half size) with the verified matches."""
    a, b = cv2.cvtColor(q_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(np.hstack([a, b]), None, fx=0.5, fy=0.5)
    w = a.shape[1] * 0.5
    rng = np.random.default_rng(0)
    for i in rng.permutation(np.flatnonzero(inl))[:150]:
        p, q = tuple(np.int32(kq[i] * 0.5)), tuple(np.int32(km[i] * 0.5 + [w, 0]))
        cv2.line(canvas, p, q, (26, 175, 27), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=float, default=2.5, help='s between places')
    ap.add_argument('--t1', type=float, default=30.0, help='end of the day chunk, s')
    ap.add_argument('--map-laps', nargs='+', default=['L3', 'L6'])
    ap.add_argument('--thr-px', type=float, default=2.0, help='RANSAC threshold, full-resolution pixels')
    ap.add_argument('--pictures', nargs='*', default=['5:fr', '17.5:cl', '25:rr'], help='place:camera pairs to draw')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cfg = config.load(os.path.join(ref.LOC, 'configs', 'final.yaml'))
    calib = load_cameras(cfg)                         # intrinsics are identical in the day bag
    sift = {c: SiftExtractor(calib[c], cfg['reloc']['features'], load_mask(cfg, c)) for c in CAMS}
    splg = SuperPointLG(dev)
    state = daybag.read_state(ref.DAY_BAG)
    first = daybag.first_set_stamp(ref.DAY_BAG, CAMS)
    start_m = daybag.stand_in_first_pose(state, first)[1]['start_along_ref_m']
    ts = (state['stamp_ns'] - state['stamp_ns'][0]) / 1e9
    sd = np.cumsum(state['vx_mps'] * np.diff(ts, prepend=ts[0]))
    laps = {lap: lap_distance(lap) for lap in ['L7'] + list(args.map_laps)}
    pictures = {(float(p.split(':')[0]), p.split(':')[1]) for p in args.pictures}
    rows, timing = [], {'sp_extract_ms': [], 'lg_match_ms': [], 'sift_extract_ms': []}
    t_run = time.time()
    for tm in np.arange(0.0, args.t1 - 1e-6, args.step):
        dist = float(np.interp((first - state['stamp_ns'][0]) / 1e9 + tm, ts, sd) + start_m)
        sets = {'day': image_set_at(ref.DAY_BAG, first + int(tm * 1e9), CAMS)}
        for lap, (gs, gd) in laps.items():
            sets[lap] = image_set_at(ref.TRAIN_BAG, gs[min(np.searchsorted(gd, dist), len(gs) - 1)], CAMS)
        for cam in CAMS:
            if not all(cam in s.images for s in sets.values()):
                continue
            gray, spf, sf = {}, {}, {}
            for k, s in sets.items():
                gray[k] = cv2.imdecode(np.frombuffer(s.images[cam], np.uint8), cv2.IMREAD_GRAYSCALE)
                t0 = time.perf_counter()
                spf[k] = splg.features(gray[k], cam)
                timing['sp_extract_ms'].append(1e3 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                sf[k] = crop_sift(sift[cam](decode_half_gray(s.images[cam])), cam, gray[k].shape[0])
                timing['sift_extract_ms'].append(1e3 * (time.perf_counter() - t0))
            f_px = 0.5 * (calib[cam].K[0, 0] + calib[cam].K[1, 1])
            kp = {k: f['keypoints'][0].cpu().numpy() for k, f in spf.items()}
            nrm = {k: cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), calib[cam].K, calib[cam].D).reshape(-1, 2)
                   for k, p in kp.items()}
            for q in ('day', 'L7'):
                for mlap in args.map_laps:
                    for method in METHODS:
                        if method == 'sift':
                            m = sift_match(sf[q][1], sf[mlap][1])
                            n0, n1 = sf[q][0][m[:, 0]], sf[mlap][0][m[:, 1]]
                            p0, p1 = sf[q][2][m[:, 0]], sf[mlap][2][m[:, 1]]
                            nq, nm = len(sf[q][1]), len(sf[mlap][1])
                        else:
                            t0 = time.perf_counter()
                            m = splg.lightglue(spf[q], spf[mlap]) if method == 'sp_lg' else splg.mutual_nn(spf[q], spf[mlap])
                            if method == 'sp_lg':
                                timing['lg_match_ms'].append(1e3 * (time.perf_counter() - t0))
                            n0, n1 = nrm[q][m[:, 0]], nrm[mlap][m[:, 1]]
                            p0, p1 = kp[q][m[:, 0]], kp[mlap][m[:, 1]]
                            nq, nm = len(kp[q]), len(kp[mlap])
                        inl = verify(n0, n1, f_px, args.thr_px)
                        rows.append({'t_day_s': round(float(tm), 2), 'dist_m': round(dist, 1), 'cam': cam,
                                     'query': 'day' if q == 'day' else 'night_L7', 'map_lap': mlap, 'method': method,
                                     'kp_query': nq, 'kp_map': nm, 'matches': len(m), 'inliers': int(inl.sum())})
                        if (round(float(tm), 2), cam) in pictures and mlap == args.map_laps[-1] and method in ('sp_lg', 'sift'):
                            name = f'pair_t{tm:g}_{cam}_{rows[-1]["query"]}_vs_{mlap}_{method}.jpg'
                            title = f'{rows[-1]["query"]} vs night {mlap}, {cam}, {method}: {int(inl.sum())} verified of {len(m)}'
                            cv2.imwrite(os.path.join(OUT, name), draw(gray[q], gray[mlap], p0, p1, inl, title),
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f'  place t={tm:4.1f} s, {dist:6.0f} m from the start line done ({time.time() - t_run:.0f} s)', flush=True)

    with open(os.path.join(OUT, 'pairs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {'pairs_per_group': None, 'thr_px': args.thr_px, 'map_laps': args.map_laps, 'device': dev,
               'timing_ms_median': {k: float(np.median(v)) for k, v in timing.items()}, 'groups': {}, 'per_camera': {}}
    for q in ('night_L7', 'day'):
        for method in METHODS:
            v = np.array([r['inliers'] for r in rows if r['query'] == q and r['method'] == method])
            mt = np.array([r['matches'] for r in rows if r['query'] == q and r['method'] == method])
            summary['pairs_per_group'] = int(len(v))
            summary['groups'][f'{q}/{method}'] = {
                'median_inliers': float(np.median(v)), 'p10_inliers': float(np.percentile(v, 10)),
                'median_matches': float(np.median(mt)), 'share_ge30_pct': 100.0 * float(np.mean(v >= 30)),
                'share_ge15_pct': 100.0 * float(np.mean(v >= 15))}
            summary['per_camera'][f'{q}/{method}'] = {
                c: float(np.median([r['inliers'] for r in rows if r['query'] == q and r['method'] == method and r['cam'] == c]))
                for c in CAMS}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\n{summary["pairs_per_group"]} pairs per group (places x cameras x map laps), RANSAC {args.thr_px:g} px, {dev}')
    print(f'{"query / method":20s} {"median inl":>10s} {"p10":>6s} {"median matches":>15s} {">=30 inl":>9s} {">=15 inl":>9s}')
    for k, g in summary['groups'].items():
        print(f'{k:20s} {g["median_inliers"]:10.0f} {g["p10_inliers"]:6.0f} {g["median_matches"]:15.0f} '
              f'{g["share_ge30_pct"]:8.0f}% {g["share_ge15_pct"]:8.0f}%')
    print('\nmedian verified inliers per camera:')
    print(f'{"query / method":20s} ' + ' '.join(f'{c:>5s}' for c in CAMS))
    for k, pc in summary['per_camera'].items():
        print(f'{k:20s} ' + ' '.join(f'{pc[c]:5.0f}' for c in CAMS))
    print('timing (median ms):', summary['timing_ms_median'])


if __name__ == '__main__':
    main()
