#!/usr/bin/env python3
"""Match images from an unknown camera (e.g. screenshots of onboard footage) against the
relocalization map, with no pose prior and no intrinsics (D-024). A sanity check of how far the
night map transfers to other cameras and other light; not part of the method.

Per image:
1. SIFT exactly as in the pipeline (CLAHE, D-010), on the image scaled to the width of our
   half-resolution frames (the scale the map's features come from).
2. Nearest map points over all mapped cameras. Every place is in the map once per lap, so the ratio
   test compares the best point with the best point of a *different place* (> 3 m away).
3. Vote for a place: the densest cluster of matched keyframe positions (two candidates).
4. PnP RANSAC over a grid of horizontal fields of view and radial distortion (principal point at the
   image centre, square pixels); the hypothesis with the most inliers wins.
5. Checks: enough inliers, camera height above the track surface, roll and pitch, inside the track.
Writes summary.json / summary.txt and, per image, the inlier matches next to the matched night
keyframe (decoded from the training bag).

    python scripts/match_external.py --images external_frames/ --out results/external/<run>
    python scripts/match_external.py --extract L6 fr 6 --images <dir> --config configs/cv_f1.yaml   # control frames
    python scripts/match_external.py --images <dir> --config configs/cv_f1.yaml --gt-cam fr --out ...  # control with GT
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import bag, config  # noqa: E402
from camloc.geometry import inv_T, make_T  # noqa: E402
from camloc.reloc.mapbuild import load_cameras  # noqa: E402
from camloc.trackmap import TrackMap  # noqa: E402

WIDTH = 753                  # our half-res frame width (1506 / 2)
K_NN = 8                     # nearest map points per mapped camera
OTHER_PLACE_M = 3.0          # the ratio test's second neighbour must be at least this far away
RATIO = 0.8
CLUSTER_M = 30.0
FOV_DEG = np.arange(24, 142, 3)
K1 = (0.0, -0.1, -0.2, -0.3, -0.4)
RANSAC_PX = 3.0              # at the scaled image size
MIN_INLIERS = 25
MAX_HEIGHT_M, MAX_ROLL_DEG, PITCH_DEG, MIN_INSIDE_M = 3.0, 10.0, (-20.0, 15.0), -5.0


def features(gray, fcfg, ignore_top=0.0, ignore_bottom=0.0):
    s = WIDTH / gray.shape[1]
    small = cv2.resize(gray, (WIDTH, int(round(gray.shape[0] * s))), interpolation=cv2.INTER_AREA)
    mask = np.full(small.shape, 255, np.uint8)
    mask[:int(ignore_top * small.shape[0])] = 0
    mask[small.shape[0] - int(ignore_bottom * small.shape[0]):] = 0
    clahe = cv2.createCLAHE(clipLimit=fcfg['clahe_clip'], tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=fcfg['n_features'], contrastThreshold=fcfg['contrast_threshold'])
    kps, desc = sift.detectAndCompute(clahe.apply(small), mask)
    if desc is None:
        return small, np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    px = np.array([k.pt for k in kps], np.float32)
    return small, px, np.clip(np.rint(desc), 0, 255).astype(np.float32)


def load_maps(cfg):
    maps = {}
    for c in cfg['reloc']['cameras']:
        d = np.load(config.artifact(cfg, cfg['reloc']['map_name'], f'{c}.npz'))
        maps[c] = {k: d[k] for k in ('kf_stamp', 'kf_T_map_cam', 'kf_base', 'pt_offset', 'pt_xyz', 'pt_desc')}
    return maps


def nearest(maps, desc):
    """K_NN nearest map points per mapped camera for every query descriptor, with exact L2 distances.
    -> cam index, point index, distance; each (n, K_NN * cameras)."""
    cam, pt, dist = [], [], []
    for ci, m in enumerate(maps.values()):
        data = m['pt_desc'].astype(np.float32)
        index = cv2.flann_Index(data, dict(algorithm=1, trees=4))
        idx, _ = index.knnSearch(desc, K_NN, params=dict(checks=128))
        idx = idx.astype(np.int64)
        cam.append(np.full(idx.shape, ci))
        pt.append(idx)
        dist.append(np.linalg.norm(data[idx] - desc[:, None, :], axis=2))
        del index, data
    return np.hstack(cam), np.hstack(pt), np.hstack(dist)


def ratio_matches(maps, cam, pt, dist):
    """Best map point per query feature, kept if it beats the best point of a different place."""
    xyz = [m['pt_xyz'] for m in maps.values()]
    keep = []
    for q in range(len(dist)):
        o = np.argsort(dist[q])
        X0 = xyz[cam[q, o[0]]][pt[q, o[0]]]
        second = dist[q, o[-1]]
        for j in o[1:]:
            if np.linalg.norm(xyz[cam[q, j]][pt[q, j]] - X0) > OTHER_PLACE_M:
                second = dist[q, j]
                break
        if dist[q, o[0]] < RATIO * second:
            keep.append((q, cam[q, o[0]], pt[q, o[0]]))
    return keep


def places(maps, matches, n=2):
    """Up to n candidate places: the densest clusters of the matched keyframes' positions."""
    cams = list(maps.values())
    kf = [int(np.searchsorted(cams[c]['pt_offset'], p, side='right') - 1) for _, c, p in matches]
    pos = np.array([cams[c]['kf_base'][k, :2] for (_, c, _), k in zip(matches, kf)]).reshape(-1, 2)
    out, free = [], np.ones(len(pos), bool)
    for _ in range(n):
        if free.sum() < 6:
            break
        d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        count = ((d < CLUSTER_M) & free[None]).sum(1) * free
        centre = int(np.argmax(count))
        member = (d[centre] < CLUSTER_M) & free
        out.append({'centre_xy': pos[centre], 'members': np.where(member)[0], 'kf': kf})
        free &= d[centre] > 3 * CLUSTER_M
    return out


def grid_hypotheses(size):
    """Unknown camera: principal point at the centre, square pixels, a grid of fields of view and k1."""
    W, H = size
    out = []
    for fov in FOV_DEG:
        f = (W / 2) / np.tan(np.radians(fov) / 2)
        for k1 in K1:
            out.append({'fov': float(fov), 'k1': k1, 'f': f, 'K': np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]),
                        'D': np.array([k1, 0, 0, 0, 0.0])})
    return out


def known_hypothesis(cam, size):
    """Control only: the true calibration of one of our cameras, scaled to the image size used."""
    s = size[0] / cam.width
    K = cam.K.copy()
    K[:2] *= s
    return [{'fov': float(2 * np.degrees(np.arctan(cam.width / 2 / cam.K[0, 0]))), 'k1': float(cam.D[0]),
             'f': float(K[0, 0]), 'K': K, 'D': cam.D}]


def pnp_grid(px, X, hyps):
    """PnP RANSAC for every camera hypothesis; the one with the most inliers wins."""
    c0 = X.mean(0)
    Xc = (X - c0).astype(np.float64)
    best = {'inliers': 0}
    for h in hyps:
        uv = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), h['K'], h['D']).reshape(-1, 2)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(Xc, uv, np.eye(3), None, iterationsCount=2000,
                                                 reprojectionError=RANSAC_PX / h['f'], confidence=0.999,
                                                 flags=cv2.SOLVEPNP_AP3P)
        if ok and inl is not None and len(inl) > best['inliers']:
            best = {'inliers': int(len(inl)), 'fov': h['fov'], 'f': h['f'], 'k1': h['k1'], 'rvec': rvec, 'tvec': tvec,
                    'inl': inl[:, 0], 'uv': uv}
    if best['inliers'] < 6:
        return best
    i = best['inl']
    rvec, tvec = cv2.solvePnPRefineLM(Xc[i], best['uv'][i], np.eye(3), None, best['rvec'], best['tvec'])
    R = cv2.Rodrigues(rvec)[0]
    proj = Xc[i] @ R.T + tvec.ravel()
    best['rms_px'] = float(np.sqrt(np.mean(np.sum((proj[:, :2] / proj[:, 2:3] - best['uv'][i]) ** 2, 1))) * best['f'])
    best['T_map_cam'] = inv_T(make_T(R, tvec.ravel() - R @ c0))
    return best


def check(best, track, n_matches):
    T = best['T_map_cam']
    C = T[:3, 3]
    h = float(C[2] - track.ground_z(C[:2]))
    roll = float(np.degrees(np.arcsin(np.clip(T[2, 0], -1, 1))))          # camera x axis vs horizontal
    pitch = float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1))))        # + = looking down
    inside = float(track.inside_margin(C[:2]))
    fails = []
    if best['inliers'] < MIN_INLIERS:
        fails.append('inliers')
    if not 0.0 < h < MAX_HEIGHT_M:
        fails.append('height')
    if abs(roll) > MAX_ROLL_DEG:
        fails.append('roll')
    if not PITCH_DEG[0] < pitch < PITCH_DEG[1]:
        fails.append('pitch')
    if inside < MIN_INSIDE_M:
        fails.append('outside_track')
    return {'x': float(C[0]), 'y': float(C[1]), 'height_m': h, 'roll_deg': roll, 'pitch_down_deg': pitch,
            'inside_track_m': inside, 'heading_deg': float(np.degrees(np.arctan2(T[1, 2], T[0, 2]))),
            'inlier_ratio': best['inliers'] / max(n_matches, 1), 'fails': fails, 'matched': not fails}


def where(cfg, cam, stamp):
    for lap in json.load(open(config.artifact(cfg, 'laps.json')))['laps']:
        if lap['start_ns'] <= stamp < lap['end_ns']:
            return lap['name'], (stamp - lap['start_ns']) / 1e9
    return '-', float('nan')


def keyframe_image(cfg, cam, stamp):
    for s in bag.iter_image_sets(cfg['bag'], [cam], stamp - 1_000_000, stamp + 1_000_000):
        if cam in s.images:
            return cv2.imdecode(np.frombuffer(s.images[cam], np.uint8), cv2.IMREAD_GRAYSCALE)
    return None


def draw(cfg, calib, maps, small, px, res, out_png):
    """Query image (left) next to the night keyframe that holds the most inliers (right)."""
    cams = list(maps)
    inl = res['inlier_matches']                                  # (query idx, cam idx, point idx, kf)
    kfs = {}
    for q, c, p, k in inl:
        kfs.setdefault((c, k), []).append((q, p))
    (c, k), pairs = max(kfs.items(), key=lambda kv: len(kv[1]))
    cam, m = cams[c], maps[cams[c]]
    img = keyframe_image(cfg, cam, int(m['kf_stamp'][k]))
    left = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for q, _, _, _ in inl:
        cv2.circle(left, tuple(int(v) for v in px[q]), 3, (0, 200, 255), 1)
    if img is None:
        cv2.imwrite(out_png, left)
        return
    right = cv2.cvtColor(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)), cv2.COLOR_GRAY2BGR)
    T_cam_map = inv_T(m['kf_T_map_cam'][k])
    X = np.array([m['pt_xyz'][p] for _, p in pairs], np.float64)
    uv, _ = cv2.projectPoints(X, cv2.Rodrigues(T_cam_map[:3, :3])[0], T_cam_map[:3, 3], calib[cam].K, calib[cam].D)
    uv = uv.reshape(-1, 2) / 2.0
    H = max(left.shape[0], right.shape[0])
    canvas = np.zeros((H + 40, left.shape[1] + right.shape[1], 3), np.uint8)
    canvas[40:40 + left.shape[0], :left.shape[1]] = left
    canvas[40:40 + right.shape[0], left.shape[1]:] = right
    for (q, _), u in zip(pairs, uv):
        a = (int(px[q][0]), int(px[q][1]) + 40)
        b = (int(u[0]) + left.shape[1], int(u[1]) + 40)
        cv2.line(canvas, a, b, (0, 255, 0), 1, cv2.LINE_AA)
    lap, t = where(cfg, cam, int(m['kf_stamp'][k]))
    txt = (f"{os.path.basename(res['image'])}: {'MATCHED' if res.get('matched') else 'no match'}  inliers "
           f"{res['inliers']}  HFOV {res.get('fov', 0):.0f} deg  |  night keyframe: camera {cam}, {lap} "
           f"+{t:.1f} s ({len(pairs)} of the inliers)")
    cv2.putText(canvas, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, canvas)


def extract(cfg, lap_name, cam, n, out_dir):
    """Control frames from the training bag: n full-res frames of one camera spread over a lap, saved
    as <stamp>.png, plus a web-like copy (JPEG quality 35) and a mirrored copy (must not match)."""
    lap = {x['name']: x for x in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}[lap_name]
    for sub in ('raw', 'jpeg35', 'mirrored'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    step = (lap['end_ns'] - lap['start_ns']) // (n + 1)
    for i in range(1, n + 1):
        t = lap['start_ns'] + i * step
        for s in bag.iter_image_sets(cfg['bag'], [cam], t, t + 200_000_000):
            img = cv2.imdecode(np.frombuffer(s.images[cam], np.uint8), cv2.IMREAD_GRAYSCALE)
            st = s.stamps[cam]
            cv2.imwrite(os.path.join(out_dir, 'raw', f'{st}.png'), img)
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 35])
            cv2.imwrite(os.path.join(out_dir, 'jpeg35', f'{st}.png'), cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE))
            cv2.imwrite(os.path.join(out_dir, 'mirrored', f'{st}.png'), cv2.flip(img, 1))
            break
    print(f'{n} frames of {cam} in {lap_name} -> {out_dir}/{{raw,jpeg35,mirrored}}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(config.ROOT, 'configs', 'final.yaml'))
    ap.add_argument('--images', required=True, help='folder of images (png/jpg)')
    ap.add_argument('--out', default='results/external/run')
    ap.add_argument('--ignore-top', type=float, default=0.0, help='fraction of the image height to ignore (sky, graphics)')
    ap.add_argument('--ignore-bottom', type=float, default=0.0, help='fraction to ignore (cockpit, car body)')
    ap.add_argument('--gt-cam', help='control only: images are <stamp>.png of this camera of the training bag')
    ap.add_argument('--known-intrinsics', action='store_true', help='control only: use the true calibration of --gt-cam')
    ap.add_argument('--extract', nargs=3, metavar=('LAP', 'CAM', 'N'), help='write control frames to --images and stop')
    args = ap.parse_args()
    cfg = config.load(args.config)
    if args.extract:
        extract(cfg, args.extract[0], args.extract[1], int(args.extract[2]), args.images)
        return
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(args.images, '*')) if p.lower().endswith(('.png', '.jpg', '.jpeg')))
    calib = load_cameras(cfg)
    track = TrackMap(dict(np.load(config.artifact(cfg, 'track_bounds.npz'))))
    maps = load_maps(cfg)
    cams = list(maps)
    feats = [features(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cfg['reloc']['features'], args.ignore_top, args.ignore_bottom)
             for p in paths]
    n_q = [len(f[2]) for f in feats]
    cam, pt, dist = nearest(maps, np.vstack([f[2] for f in feats]))
    gti = None
    if args.gt_cam:
        from camloc import gt as gtm                              # control only: never on the inference path
        gti = gtm.interpolator(gtm.load_gt(config.artifact(cfg, 'gt.npz')))
    results, a = [], 0
    for path, (small, px, desc), n in zip(paths, feats, n_q):
        sl = slice(a, a + n)
        a += n
        res = {'image': path, 'features': n, 'inliers': 0, 'matched': False}
        matches = ratio_matches(maps, cam[sl], pt[sl], dist[sl])
        res['ratio_matches'] = len(matches)
        best = None
        for pl in places(maps, matches):
            sel = [matches[i] for i in pl['members']]
            if len(sel) < 6:
                continue
            X = np.array([maps[cams[c]]['pt_xyz'][p] for _, c, p in sel], np.float64)
            size = (small.shape[1], small.shape[0])
            hyps = known_hypothesis(calib[args.gt_cam], size) if args.known_intrinsics else grid_hypotheses(size)
            b = pnp_grid(px[[q for q, _, _ in sel]], X, hyps)
            if b['inliers'] >= 6 and (best is None or b['inliers'] > best[0]['inliers']):
                best = (b, sel, [pl['kf'][i] for i in pl['members']])
        if best is not None:
            b, sel, kf = best
            res.update({k: b[k] for k in ('inliers', 'fov', 'k1', 'rms_px')})
            res['place_matches'] = len(sel)
            res.update(check(b, track, len(sel)))
            res['inlier_matches'] = [(int(sel[i][0]), int(sel[i][1]), int(sel[i][2]), int(kf[i])) for i in b['inl']]
            kfc = max(set((c, k) for _, c, _, k in res['inlier_matches']),
                      key=lambda ck: sum(1 for _, c, _, k in res['inlier_matches'] if (c, k) == ck))
            res['night_keyframe'] = {'camera': cams[kfc[0]], 'stamp_ns': int(maps[cams[kfc[0]]]['kf_stamp'][kfc[1]])}
            res['night_keyframe']['lap'], res['night_keyframe']['t_in_lap_s'] = where(cfg, *res['night_keyframe'].values())
            if gti is not None:
                st = int(os.path.splitext(os.path.basename(path))[0])
                C_gt = (gti(st) @ calib[args.gt_cam].T_base_cam)[:3, 3]
                res['gt_error_m'] = float(np.linalg.norm(C_gt[:2] - np.array([res['x'], res['y']])))
                c_kf = cams.index(res['night_keyframe']['camera'])
                k_kf = int(np.searchsorted(maps[cams[c_kf]]['kf_stamp'], res['night_keyframe']['stamp_ns']))
                res['place_error_m'] = float(np.linalg.norm(C_gt[:2] - maps[cams[c_kf]]['kf_T_map_cam'][k_kf][:2, 3]))
                res['true_hfov_deg'] = float(2 * np.degrees(np.arctan(calib[args.gt_cam].width / 2 / calib[args.gt_cam].K[0, 0])))
            draw(cfg, calib, maps, small, px, res, os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + '_match.png'))
        results.append(res)
    lines = [f'{"image":28} {"feat":>5} {"ratio":>5} {"place":>5} {"inl":>4} {"HFOV":>5} {"k1":>5} {"h m":>5} '
             f'{"roll":>5} {"pitch":>5} {"result":8} {"night keyframe":22}' + ('  pose err  place err' if gti else '')]
    for r in results:
        nk = r.get('night_keyframe', {})
        lines.append(f'{os.path.basename(r["image"])[:28]:28} {r["features"]:5d} {r["ratio_matches"]:5d} '
                     f'{r.get("place_matches", 0):5d} {r["inliers"]:4d} {r.get("fov", float("nan")):5.0f} '
                     f'{r.get("k1", float("nan")):5.1f} {r.get("height_m", float("nan")):5.2f} '
                     f'{r.get("roll_deg", float("nan")):5.1f} {r.get("pitch_down_deg", float("nan")):5.1f} '
                     f'{"MATCHED" if r["matched"] else ",".join(r.get("fails", ["none"])):8} '
                     f'{nk.get("camera", "-")} {nk.get("lap", "-")} +{nk.get("t_in_lap_s", float("nan")):.1f} s'
                     + (f'  {r["gt_error_m"]:7.2f} m  {r["place_error_m"]:7.2f} m' if 'gt_error_m' in r else ''))
    print('\n'.join(lines))
    open(os.path.join(args.out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    for r in results:
        r.pop('inlier_matches', None)
    json.dump(results, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
