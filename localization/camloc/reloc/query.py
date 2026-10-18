"""Relocalization of one image against the map (D-009, D-013).

Candidate keyframes of the same camera near the predicted pose -> SIFT ratio-test matches to their
3D points -> guided filter: keep matches whose 3D point is seen within the angular uncertainty of the
predicted pose -> PnP RANSAC + LM refinement -> base_link pose in map -> sanity checks (inliers,
height above the track surface, roll and pitch). Uses no ground truth.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .. import config
from ..geometry import inv_T, make_T, roll_pitch_of, wrap, yaw_of
from ..trackmap import TrackMap
from .features import Extractor, decode_half_gray
from .mapbuild import load_cameras, load_mask


class CameraMap:
    def __init__(self, path, min_parallax_deg=0.0):
        d = np.load(path)
        self.kf_base = d['kf_base']
        self.kf_window = d['kf_window']
        self.off = d['pt_offset']
        self.xyz = d['pt_xyz']
        self.desc = d['pt_desc']
        if min_parallax_deg > 0:                              # E-2a: only points with well-determined depth
            keep = d['pt_parallax'].astype(np.float32) >= min_parallax_deg
            kf_of_pt = np.repeat(np.arange(len(self.off) - 1), np.diff(self.off))
            self.off = np.r_[0, np.cumsum(np.bincount(kf_of_pt[keep], minlength=len(self.off) - 1))]
            self.xyz, self.desc = self.xyz[keep], self.desc[keep]
        self.n_pts = np.diff(self.off)
        self.tree = cKDTree(self.kf_base[:, :2])

    def candidates(self, x, y, yaw, radius, max_yaw_diff, k, min_points):
        """Up to k keyframes within `radius` of (x, y) and heading within `max_yaw_diff`, nearest first."""
        idx = np.array(self.tree.query_ball_point([x, y], radius), dtype=int)
        if not len(idx):
            return []
        idx = idx[(np.abs(wrap(self.kf_base[idx, 2] - yaw)) < max_yaw_diff) & (self.n_pts[idx] >= min_points)]
        d = np.hypot(self.kf_base[idx, 0] - x, self.kf_base[idx, 1] - y)
        return list(idx[np.argsort(d)][:k])

    def points(self, i):
        s = slice(self.off[i], self.off[i + 1])
        return self.xyz[s], self.desc[s]


class Relocalizer:
    def __init__(self, cfg, map_name=None, cameras=None, track=None):
        self.cfg = cfg
        self.rc = cfg['reloc']
        self.cameras = list(cameras or self.rc['cameras'])
        calib = load_cameras(cfg)
        self.calib = calib
        self.ext = {c: Extractor(calib[c], self.rc['features'], load_mask(cfg, c)) for c in self.cameras}
        name = map_name or self.rc['map_name']
        self.maps = {c: CameraMap(config.artifact(cfg, name, f'{c}.npz'), self.rc['query'].get('min_point_parallax_deg', 0.0))
                     for c in self.cameras}
        self.T_cam_base = {c: inv_T(calib[c].T_base_cam) for c in self.cameras}
        self.track = track or TrackMap(dict(np.load(config.artifact(cfg, 'track_bounds.npz'))))
        self.stats = {c: {'tried': 0, 'fix': 0, 'accepted': 0} for c in self.cameras}
        # one thread per camera: SIFT, matching and PnP release the GIL
        self.pool = ThreadPoolExecutor(max_workers=self.rc.get('threads') or len(self.cameras))
        # robustness experiments only (D-017): synthetic darkening/brightening, blur, camera blackout
        st = self.rc.get('stress') or {}
        g = st.get('gamma', 1.0)
        self.lut = None if g == 1.0 else np.clip(255 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)
        self.blur = int(st.get('blur_px', 0))
        self.blackout = st.get('blackout_s') or []
        # E-4: keep only a fraction of the fixes (random image sets, or periodic blocks of time)
        self.keep = float(st.get('keep_fraction', 1.0))
        self.keep_mode = st.get('keep_mode', 'random')
        self.block_period = float(st.get('block_period_s', 10.0))
        self.rng = np.random.default_rng(int(st.get('seed', 0)))
        self.dropped = 0
        self.t_first = None
        # E-3: the features of the visual-odometry camera are kept for every frame
        vo = cfg.get('vo') or {}
        self.feature_cam = vo['camera'] if vo.get('enabled') else None
        self.features = {}

    @staticmethod
    def decode(cam, data):
        return decode_half_gray(data)

    def search_radius(self, sigma_xy):
        q = self.rc['query']
        return float(np.clip(q['radius_m'] + 3.0 * sigma_xy, q['radius_m'], q['max_radius_m']))

    def localize(self, image_set, ekf):
        """Fixes for every configured camera present in the set, with the filter estimate as prior.
        Images may be JPEG bytes (decoded here, in the camera's thread) or decoded half-res gray."""
        x, y, yaw = ekf.pose
        sxy, syaw = np.sqrt(0.5 * (ekf.P[0, 0] + ekf.P[1, 1])), np.sqrt(ekf.P[2, 2])
        self.t_first = self.t_first or image_set.stamp_ns
        self.features = {}
        if self.blackout and self.blackout[0] <= (image_set.stamp_ns - self.t_first) / 1e9 < self.blackout[1]:
            return []

        def one(c):
            img = image_set.images[c]
            if isinstance(img, (bytes, bytearray)):
                img = decode_half_gray(img)
            if self.lut is not None:
                img = cv2.LUT(img, self.lut)
            if self.blur > 1:
                img = cv2.blur(img, (self.blur, 1))
            return self.localize_camera(c, img, x, y, yaw, sxy, syaw)

        cams = [c for c in self.cameras if c in image_set.images]
        out = []
        for c, f in zip(cams, self.pool.map(one, cams)):
            self.stats[c]['tried'] += 1
            if f.get('ok'):
                self.stats[c]['fix'] += 1
                out.append(f)
        if self.keep < 1.0 and out:
            t = (image_set.stamp_ns - self.t_first) / 1e9
            if self.keep_mode == 'blocks':
                kept = (t % self.block_period) < self.keep * self.block_period
            else:
                kept = self.rng.random() < self.keep
            if not kept:
                self.dropped += len(out)
                return []
        return out

    def feedback(self, image_set, fixes, ekf):
        for f in fixes:
            self.stats[f['cam']]['accepted'] += int(f.get('accepted', False))

    def prior_camera(self, cam, x, y, yaw):
        """Camera pose (map <- cam) predicted from a base pose, with the typical roll, zero pitch and the
        track-surface height (these three are not estimated by the filter)."""
        chk = self.rc['checks']
        R = Rotation.from_euler('ZYX', [yaw, 0.0, np.radians(chk['roll0_deg'])]).as_matrix()
        z = self.track.ground_z((x, y)) + chk['z_offset_m']
        return make_T(R, [x, y, z]) @ self.calib[cam].T_base_cam

    def guided(self, cam, uv, X, x, y, yaw, sigma_xy, sigma_yaw):
        """Mask of matches (bearing uv <-> 3D point X) consistent with the prior pose: the angle between
        the observed and the predicted bearing must be below n sigma of what the position (scaled by
        1/depth), heading and attitude uncertainty allow."""
        g = self.rc['guided']
        T = self.prior_camera(cam, x, y, yaw)
        v = (X - T[:3, 3]) @ T[:3, :3]                        # point in the predicted camera frame
        depth = np.linalg.norm(v, axis=1)
        b = np.c_[uv, np.ones(len(uv))]
        cosang = np.sum(v * b, axis=1) / (depth * np.linalg.norm(b, axis=1))
        ang = np.arccos(np.clip(cosang, -1, 1))
        s_pos = np.hypot(sigma_xy, g['sigma_z_m']) / np.maximum(depth, 1.0)
        s_rot = np.hypot(sigma_yaw, np.radians(g['sigma_att_deg']))
        lim = g['n_sigma'] * np.hypot(s_pos, s_rot) + g['pixel_px'] / self.calib[cam].K[0, 0]
        return (ang < np.minimum(lim, np.radians(g['max_deg']))) & (v[:, 2] > 0)

    def localize_camera(self, cam, gray, x, y, yaw, sigma_xy=1.0, sigma_yaw=0.01):
        """-> dict with 'ok' and, if a pose was found, x, y, yaw, z, roll, pitch, inliers, ... and the
        reason it was rejected ('fail') otherwise."""
        q, chk = self.rc['query'], self.rc['checks']
        cmap = self.maps[cam]
        radius = self.search_radius(sigma_xy)
        feats = None
        if cam == self.feature_cam:                           # E-3: needed every frame, keyframes or not
            feats = self.ext[cam](gray)
            self.features[cam] = feats[:2]
        cands = cmap.candidates(x, y, yaw, radius, np.radians(q['max_yaw_diff_deg']), q['keyframes'], q['min_kf_points'])
        if not cands:
            return {'cam': cam, 'ok': False, 'fail': 'no_keyframe'}
        uv, desc, _ = feats if feats is not None else self.ext[cam](gray)
        if len(desc) < q['min_matches']:
            return {'cam': cam, 'ok': False, 'fail': 'few_features'}
        dq = desc.astype(np.float32)
        best_d = np.full(len(dq), np.inf)
        best_X = np.zeros((len(dq), 3))
        bf = cv2.BFMatcher(cv2.NORM_L2)                       # per call: matchers are not thread-safe
        for i in cands:
            xyz, dk = cmap.points(i)
            if len(dk) < 2:
                continue
            for m in bf.knnMatch(dq, dk.astype(np.float32), k=2):
                if len(m) == 2 and m[0].distance < q['ratio'] * m[1].distance and m[0].distance < best_d[m[0].queryIdx]:
                    best_d[m[0].queryIdx] = m[0].distance
                    best_X[m[0].queryIdx] = xyz[m[0].trainIdx]
        sel = np.where(np.isfinite(best_d))[0]
        n_tentative = int(len(sel))
        if self.rc['guided']['enabled'] and len(sel):
            sel = sel[self.guided(cam, uv[sel], best_X[sel], x, y, yaw, sigma_xy, sigma_yaw)]
        if len(sel) < q['min_matches']:
            return {'cam': cam, 'ok': False, 'fail': 'few_matches', 'matches': int(len(sel)), 'tentative': n_tentative}
        img_pts = uv[sel].astype(np.float64)
        obj = best_X[sel]
        c0 = obj.mean(0)
        objc = obj - c0                                       # centred for conditioning
        thr = q['ransac_px'] / self.calib[cam].K[0, 0]
        ok, rvec, tvec, inl = cv2.solvePnPRansac(objc, img_pts, np.eye(3), None, iterationsCount=q['ransac_iters'],
                                                 reprojectionError=thr, confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < chk['min_inliers']:
            return {'cam': cam, 'ok': False, 'fail': 'few_inliers', 'matches': int(len(sel)),
                    'inliers': 0 if inl is None else int(len(inl))}
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(objc[inl], img_pts[inl], np.eye(3), None, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        proj = objc[inl] @ R.T + tvec.ravel()
        res = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - img_pts[inl], axis=1) * self.calib[cam].K[0, 0]
        T_map_cam = inv_T(make_T(R, tvec.ravel() - R @ c0))
        T_map_base = T_map_cam @ self.T_cam_base[cam]
        xb, yb, zb = T_map_base[:3, 3]
        roll, pitch = roll_pitch_of(T_map_base[:3, :3])
        f = {'cam': cam, 'x': float(xb), 'y': float(yb), 'yaw': yaw_of(T_map_base[:3, :3]), 'z': float(zb),
             'roll': float(np.degrees(roll)), 'pitch': float(np.degrees(pitch)), 'matches': int(len(sel)),
             'tentative': n_tentative, 'inliers': int(len(inl)), 'rms_px': float(np.sqrt(np.mean(res ** 2))),
             'n_kf': len(cands)}
        f['dz'] = float(zb - self.track.ground_z((xb, yb)) - chk['z_offset_m'])
        # E-6: lateral distance of the fix from the nearest candidate keyframe (~ from the mapped line)
        kb = cmap.kf_base[cands]
        near = int(np.argmin(np.hypot(kb[:, 0] - xb, kb[:, 1] - yb)))
        f['kf_lateral_m'] = float(abs(-np.sin(kb[near, 2]) * (xb - kb[near, 0]) + np.cos(kb[near, 2]) * (yb - kb[near, 1])))
        if abs(f['dz']) > chk['max_dz_m']:
            f.update(ok=False, fail='height')
        elif abs(f['roll'] - chk['roll0_deg']) > chk['max_droll_deg'] or abs(f['pitch']) > chk['max_pitch_deg']:
            f.update(ok=False, fail='attitude')
        elif f['inliers'] / len(sel) < chk['min_inlier_ratio']:
            f.update(ok=False, fail='inlier_ratio')
        else:
            s = self.rc['sigma']
            k = np.sqrt(s['ref_inliers'] / f['inliers'])
            sxy = s['xy_m'] * k
            if s.get('offline_gain', 0.0) > 0:                 # E-6: fixes get worse off the mapped line
                slat = float(np.hypot(sxy, s['offline_gain'] * f['kf_lateral_m']))
                if s.get('offline_mode', 'isotropic') == 'lateral':
                    # E-6b: only across the mapped line (left normal of the nearest keyframe's heading)
                    nv = np.array([-np.sin(kb[near, 2]), np.cos(kb[near, 2])])
                    slon_c = float(np.clip(sxy, s['xy_min_m'], s['xy_max_m']))
                    slat_c = float(np.clip(slat, s['xy_min_m'], s['xy_max_m']))
                    f['R_xy'] = slon_c ** 2 * (np.eye(2) - np.outer(nv, nv)) + slat_c ** 2 * np.outer(nv, nv)
                else:
                    sxy = slat
            f.update(ok=True, sigma_xy=float(np.clip(sxy, s['xy_min_m'], s['xy_max_m'])),
                     sigma_yaw=float(np.radians(np.clip(s['yaw_deg'] * k, s['yaw_min_deg'], s['yaw_max_deg']))))
        return f
