"""Building the relocalization map from training laps (D-009, D-012). Uses ground truth.

Per camera: keyframes every `spacing_m` of travel inside the training laps. Each keyframe stores its
SIFT features that could be triangulated against 1-3 later keyframes of the same pass, with the
camera poses known from GT (T_map_cam = T_map_base(GT at image stamp) * T_base_cam). Points are
stored in the map frame, so a query match gives a 2D-3D correspondence directly.
"""
import json
import os
from collections import deque
from multiprocessing import Pool

import cv2
import numpy as np

from .. import bag, config
from ..geometry import inv_T, yaw_of
from .features import Extractor, decode_half_gray

_W = {}   # per-worker state


def load_cameras(cfg):
    return {c: bag.Camera.from_dict(d) for c, d in json.load(open(config.artifact(cfg, 'calibration.json'))).items()}


def load_mask(cfg, cam):
    p = config.artifact(cfg, 'masks', f'{cam}.png')
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None


def _init_worker(cfg):
    cams = load_cameras(cfg)
    _W['ext'] = {c: Extractor(cams[c], cfg['reloc']['features'], load_mask(cfg, c)) for c in cams}
    _W['cfg'] = cfg
    _W['fx'] = {c: cams[c].K[0, 0] for c in cams}


def _extract(job):
    cam, stamp, data = job
    uv, desc, _ = _W['ext'][cam](decode_half_gray(data))
    return cam, stamp, uv, desc


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def triangulate_pair(fk, fj, Tk, Tj, mcfg, fx, bf):
    """Triangulate matches between keyframe k and a later keyframe j of the same camera.
    fk/fj: (uv_norm, desc float32); Tk/Tj: map <- cam. Returns (indices into k, X_map, parallax_deg)."""
    uk, dk = fk
    uj, dj = fj
    if len(dk) < 8 or len(dj) < 8:
        return None
    T = inv_T(Tj) @ Tk                         # cam_k -> cam_j
    R, t = T[:3, :3], T[:3, 3]
    m = [p[0] for p in bf.knnMatch(dk, dj, k=2) if len(p) == 2 and p[0].distance < mcfg['ratio'] * p[1].distance]
    if len(m) < 8:
        return None
    qi = np.array([x.queryIdx for x in m])
    ti = np.array([x.trainIdx for x in m])
    xk = np.c_[uk[qi], np.ones(len(qi))]
    xj = np.c_[uj[ti], np.ones(len(ti))]
    E = skew(t) @ R
    Ex, Etx = xk @ E.T, xj @ E
    num = np.sum(xj * Ex, axis=1)
    samp = num ** 2 / (Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2 + 1e-18)
    thr = mcfg['max_err_px'] / fx
    keep = samp < thr ** 2
    if keep.sum() < 4:
        return None
    qi, xk, xj = qi[keep], xk[keep], xj[keep]
    Pk = np.hstack([np.eye(3), np.zeros((3, 1))])
    Pj = np.hstack([R, t[:, None]])
    Xh = cv2.triangulatePoints(Pk, Pj, xk[:, :2].T, xj[:, :2].T)
    Xk = (Xh[:3] / Xh[3]).T
    Xj = Xk @ R.T + t
    zk, zj = Xk[:, 2], Xj[:, 2]
    ok = (zk > mcfg['min_depth_m']) & (zk < mcfg['max_depth_m']) & (zj > mcfg['min_depth_m'])
    with np.errstate(divide='ignore', invalid='ignore'):
        ek = np.linalg.norm(Xk[:, :2] / zk[:, None] - xk[:, :2], axis=1)
        ej = np.linalg.norm(Xj[:, :2] / zj[:, None] - xj[:, :2], axis=1)
    ok &= (ek < thr) & (ej < thr)
    cj = -R.T @ t                               # centre of camera j in frame k
    rk = Xk / np.linalg.norm(Xk, axis=1, keepdims=True)
    rj = (Xk - cj) / np.linalg.norm(Xk - cj, axis=1, keepdims=True)
    par = np.degrees(np.arccos(np.clip(np.sum(rk * rj, axis=1), -1, 1)))
    ok &= par >= mcfg['min_parallax_deg']
    if not ok.any():
        return None
    Xm = Xk[ok] @ Tk[:3, :3].T + Tk[:3, 3]
    return qi[ok], Xm, par[ok]


def _triangulate(job):
    cam, k_feat, Tk, neighbours = job
    cfg = _W['cfg']
    mcfg = cfg['reloc']['map']
    bf = cv2.BFMatcher(cv2.NORM_L2)
    uk, dk = k_feat
    best_par = np.zeros(len(dk))
    best_X = np.zeros((len(dk), 3))
    for fj, Tj in neighbours:
        r = triangulate_pair((uk, dk.astype(np.float32)), (fj[0], fj[1].astype(np.float32)), Tk, Tj, mcfg, _W['fx'][cam], bf)
        if r is None:
            continue
        qi, Xm, par = r
        better = par > best_par[qi]
        best_par[qi[better]] = par[better]
        best_X[qi[better]] = Xm[better]
    sel = np.where(best_par > 0)[0]
    if len(sel) > mcfg['max_points_per_kf']:
        sel = sel[np.argsort(-best_par[sel])[:mcfg['max_points_per_kf']]]
    return best_X[sel].astype(np.float32), dk[sel], uk[sel].astype(np.float16), best_par[sel].astype(np.float16)


def select_keyframes(cfg, gti, cams, windows, spacing):
    """Stream image sets in the training windows and yield (cam, stamp, jpeg) for keyframes:
    every `spacing` metres of GT travel per camera, restarting at each window."""
    for (a, b) in windows:
        last = {c: None for c in cams}
        for s in bag.iter_image_sets(cfg['bag'], cams, a, b):
            for c, data in s.images.items():
                t = s.stamps[c]
                if not gti.covers(t):
                    continue
                p = gti(t)[:2, 3]
                if last[c] is None or np.linalg.norm(p - last[c]) >= spacing:
                    last[c] = p
                    yield c, t, data


def _chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def build(cfg, cams_sel, windows, spacing, out_path, workers=8, chunk=256, log=print):
    """Build and save the map for cameras `cams_sel` from GT-time windows [(start_ns, end_ns)].

    Streaming, so memory stays bounded: the bag is read in chunks of `chunk` keyframes, features are
    extracted in a process pool, and each keyframe is triangulated as soon as its next `neighbours`
    keyframes of the same camera and pass are known. Only the triangulated points are kept."""
    from .. import gt as gtm
    gti = gtm.interpolator(gtm.load_gt(config.artifact(cfg, 'gt.npz')))
    cams = load_cameras(cfg)
    mc = cfg['reloc']['map']
    win_of = lambda t: next(i for i, (a, b) in enumerate(windows) if a <= t <= b)  # noqa: E731
    queue = {c: deque() for c in cams_sel}        # recent keyframes: (stamp, window, T_map_cam, uv, desc)
    out = {c: [] for c in cams_sel}               # finished keyframes: (stamp, window, T, points...)

    def ready_jobs(c, final):
        """Triangulation jobs for the keyframes of camera c whose neighbours are all known."""
        q, jobs = queue[c], []
        while q and (final or len(q) > mc['neighbours'] or q[-1][1] != q[0][1]):
            t, w, T, uv, desc = q.popleft()
            nb = []
            for (tj, wj, Tj, uvj, dj) in list(q)[:mc['neighbours']]:
                base = np.linalg.norm(Tj[:3, 3] - T[:3, 3])
                if wj != w or base > mc['max_baseline_m']:
                    break
                if base >= mc['min_baseline_m']:
                    nb.append(((uvj, dj), Tj))
            jobs.append(((c, t, w, T), (c, (uv, desc), T, nb)))
        return jobs

    def run_jobs(pool, jobs):
        for (key, _), r in zip(jobs, pool.map(_triangulate, [j for _, j in jobs], chunksize=4)):
            out[key[0]].append(key[1:] + r)

    with Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        n = 0
        for batch in _chunks(select_keyframes(cfg, gti, cams_sel, windows, spacing), chunk):
            jobs = []
            for c, t, uv, desc in pool.map(_extract, batch, chunksize=8):
                w = win_of(t)
                if queue[c] and queue[c][-1][1] != w:
                    jobs += ready_jobs(c, final=True)      # a new pass starts: finish the old one
                queue[c].append((t, w, gti(t) @ cams[c].T_base_cam, uv, desc))
                jobs += ready_jobs(c, final=False)
            run_jobs(pool, jobs)
            n += len(batch)
            log(f'  {n} keyframes processed', flush=True)
        run_jobs(pool, [j for c in cams_sel for j in ready_jobs(c, final=True)])

    maps = {}
    for c in cams_sel:
        kfs = sorted(out[c], key=lambda k: k[0])
        npts = np.array([len(k[3]) for k in kfs])
        cat = lambda i, shape, dt: np.concatenate([k[i] for k in kfs]) if npts.sum() else np.zeros(shape, dt)  # noqa: E731
        maps[c] = {
            'kf_stamp': np.array([k[0] for k in kfs], np.int64),
            'kf_window': np.array([k[1] for k in kfs], np.int16),
            'kf_T_map_cam': np.array([k[2] for k in kfs]),
            'kf_base': np.array([[*(k[2] @ inv_T(cams[c].T_base_cam))[:2, 3], yaw_of((k[2] @ inv_T(cams[c].T_base_cam))[:3, :3])]
                                 for k in kfs]),
            'pt_offset': np.concatenate([[0], np.cumsum(npts)]).astype(np.int64),
            'pt_xyz': cat(3, (0, 3), np.float32), 'pt_desc': cat(4, (0, 128), np.uint8),
            'pt_uv': cat(5, (0, 2), np.float16), 'pt_parallax': cat(6, (0,), np.float16),
        }
        np.savez(out_path.format(cam=c), **maps[c])
        log(f'  {c}: {len(kfs)} keyframes, {int(npts.sum())} points (median {int(np.median(npts))}/keyframe, '
            f'{100 * np.mean(npts < 30):.1f} % keyframes < 30 points)', flush=True)
    return maps
