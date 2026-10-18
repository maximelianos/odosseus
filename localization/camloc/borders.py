"""E-1: lateral position and heading from the track borders seen in the images (README, Experiments).

The 3D track borders (track_bounds, an allowed input) are projected into a full-resolution camera
image at the predicted pose. Along the image normal of each projected border point, oriented outwards
(away from the track), the strongest dark-to-bright edge within a search band is taken as the observed
border: the asphalt-to-white-line (or asphalt-to-wall) transition the borders lie on (E-1 step 1).
A robust least-squares fit to these image offsets estimates the base pose's lateral offset (along the
left normal of the heading) and heading, together with pitch, roll and height as nuisance parameters
with priors (the filter does not estimate them, and at a camera height of ~0.35 m they move the
borders near the horizon by tens of pixels). One measurement (lateral, heading) per camera, fused as
one filter update. Uses no ground truth; a constant per-camera detector offset (`bias_m`, from
training laps) can be removed.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import inv_T, make_T

OUTWARD = {'track_left': +1, 'pitlane_left': +1, 'track_right': -1, 'pitlane_right': -1}   # left normal sign


def resample(P, step):
    """Polyline -> points every `step` metres with their unit 3D tangent."""
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    q = np.arange(0, s[-1], step)
    pts = np.c_[[np.interp(q, s, P[:, k]) for k in range(3)]].T
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    return pts, tan


def rot(axis, a):
    return Rotation.from_euler(axis, a).as_matrix()


class BorderMeasurement:
    def __init__(self, bcfg, calib, bounds, masks, checks, track, visual=None):
        """bcfg: config block `borders`; calib: bag.Camera per camera; bounds: track_bounds arrays;
        masks: half-res car-body masks (0 = car); checks: reloc.checks (roll0, z offset); track: TrackMap;
        visual: optional learned visible edge (scripts/exp_e1_learn_borders.py: pts, valid), same
        resampling as here."""
        self.c = bcfg
        self.calib = calib
        self.masks = masks
        self.track = track
        self.roll0 = np.radians(checks['roll0_deg'])
        self.zoff = checks['z_offset_m']
        pts, tan, out = [], [], []
        for name, P in bounds.items():
            p, t = resample(np.asarray(P, float), bcfg['step_m'])
            sgn = OUTWARD[name]
            pts.append(p)
            tan.append(t)
            out.append(np.c_[-t[:, 1] * sgn, t[:, 0] * sgn, np.zeros(len(t))])
        self.pts, self.tan, self.out = np.vstack(pts), np.vstack(tan), np.vstack(out)
        self.out /= np.maximum(np.linalg.norm(self.out, axis=1, keepdims=True), 1e-9)
        if visual is not None:
            assert len(visual['pts']) == len(self.pts), 'visual borders were learned with another resampling'
            keep = np.asarray(visual['valid'], bool)
            self.pts, self.tan, self.out = np.asarray(visual['pts'])[keep], self.tan[keep], self.out[keep]
        self.tree = cKDTree(self.pts[:, :2])
        self.stats = {c: {'tried': 0, 'ok': 0} for c in bcfg['cameras']}
        self.debug = False
        self.last_debug = None

    def road_pitch(self, x, y, yaw):
        """Pitch of the track surface along the heading, from the 3D borders (positive = nose down)."""
        f = 4.0 * np.array([np.cos(yaw), np.sin(yaw)])
        p = np.array([x, y])
        return -np.arctan2(self.track.ground_z(p + f) - self.track.ground_z(p - f), 8.0)

    def T_map_base(self, x, y, yaw, ay=None):
        """Predicted base pose: the filter's x, y, yaw; pitch from the road slope and roll from the lateral
        acceleration (gains from training laps, E-1a), or zero pitch and the typical roll; the track surface."""
        c = self.c
        pitch = c['pitch_slope_gain'] * self.road_pitch(x, y, yaw) if c.get('pitch_from_slope') else 0.0
        roll = np.radians(c['roll_ay_gain_deg'] * ay + c['roll_bias_deg']) if (ay is not None and c.get('roll_from_ay')) else self.roll0
        return make_T(rot('ZYX', [yaw, pitch, roll]), [x, y, self.track.ground_z((x, y)) + self.zoff])

    def project(self, cam, T_map_base, P):
        """-> pixels (distorted, full res) and a mask of points in front and inside the field of view
        (checked before distortion: with strong distortion, points far outside can fold back in)."""
        cal = self.calib[cam]
        T_cam_map = inv_T(T_map_base @ cal.T_base_cam)
        Pc = P @ T_cam_map[:3, :3].T + T_cam_map[:3, 3]
        z = Pc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            xn, yn = Pc[:, 0] / z, Pc[:, 1] / z
        fov = (z > 1.0) & (np.abs(xn) < 1.1 * cal.width / 2 / cal.K[0, 0]) & (np.abs(yn) < 1.1 * cal.height / 2 / cal.K[1, 1])
        uv, _ = cv2.projectPoints(P, cv2.Rodrigues(T_cam_map[:3, :3])[0], T_cam_map[:3, 3], cal.K, cal.D)
        uv = uv.reshape(-1, 2)
        uv[~fov] = -1e4
        return uv, fov

    def perturbed(self, T0, nrm):
        """Pose perturbations for the Jacobian columns: lateral (world, along nrm), yaw (world z),
        pitch (body y), roll (body x), height (world z)."""
        e = self.EPS
        Tn = T0.copy()
        Tn[:2, 3] += e[0] * nrm
        Ty = T0.copy()
        Ty[:3, :3] = rot('z', e[1]) @ T0[:3, :3]
        Tp = T0 @ make_T(rot('y', e[2]), np.zeros(3))
        Tr = T0 @ make_T(rot('x', e[3]), np.zeros(3))
        Tz = T0.copy()
        Tz[2, 3] += e[4]
        return (Tn, Ty, Tp, Tr, Tz)

    EPS = (0.05, np.radians(0.05), np.radians(0.05), np.radians(0.05), 0.01)

    def measure(self, cam, gray, x, y, yaw, sigma_n=0.3, sigma_yaw=np.radians(0.3), T_base=None, ay=None):
        """gray: full-resolution grayscale image of `cam`. T_base: optional full prior pose (diagnostics:
        GT attitude), otherwise built from x, y, yaw (and the lateral acceleration ay for the roll).
        -> dict with ok, dn (lateral correction along the left normal of the heading), dyaw, cov (2x2
        for dn, dyaw), normal, and diagnostics."""
        c = self.c
        self.stats[cam]['tried'] += 1
        cal = self.calib[cam]
        T0 = self.T_map_base(x, y, yaw, ay) if T_base is None else T_base
        cpos = (T0 @ cal.T_base_cam)[:3, 3]
        idx = np.array(self.tree.query_ball_point(cpos[:2], c['max_dist_m']), dtype=int)
        if len(idx) < c['min_points']:
            return {'cam': cam, 'ok': False, 'fail': 'no_border'}
        idx = idx[np.linalg.norm(self.pts[idx, :2] - cpos[:2], axis=1) > c['min_dist_m']]
        P = self.pts[idx]
        nrm = np.array([-np.sin(yaw), np.cos(yaw)])
        uv0, fov = self.project(cam, T0, P)
        H, W = gray.shape
        m = fov & (uv0[:, 0] > 5) & (uv0[:, 0] < W - 5) & (uv0[:, 1] > 5) & (uv0[:, 1] < H - 5)
        mask = self.masks.get(cam)
        if mask is not None and m.any():
            ui = np.clip((uv0[m, 0] / 2).astype(int), 0, mask.shape[1] - 1)
            vi = np.clip((uv0[m, 1] / 2).astype(int), 0, mask.shape[0] - 1)
            m[np.where(m)[0][mask[vi, ui] == 0]] = False
        if m.sum() < c['min_points']:
            return {'cam': cam, 'ok': False, 'fail': 'not_visible'}
        P, idx, uv0 = P[m], idx[m], uv0[m]
        uvt, _ = self.project(cam, T0, P + 0.25 * self.tan[idx])
        uvo, _ = self.project(cam, T0, P + 0.25 * self.out[idx])
        t_img = uvt - uv0
        t_img /= np.maximum(np.linalg.norm(t_img, axis=1, keepdims=True), 1e-9)
        n_img = np.c_[-t_img[:, 1], t_img[:, 0]]
        n_img[np.sum(n_img * (uvo - uv0), axis=1) < 0] *= -1      # oriented outwards (away from the track)
        J = np.c_[[np.sum(n_img * (self.project(cam, Tk, P)[0] - uv0), axis=1) / e
                   for Tk, e in zip(self.perturbed(T0, nrm), self.EPS)]].T
        prior = np.array([sigma_n, sigma_yaw, np.radians(c['prior_pitch_deg']), np.radians(c['prior_roll_deg']), c['prior_z_m']])
        if T_base is not None:                                    # attitude and height known (diagnostics)
            prior[2:] = [1e-6, 1e-6, 1e-6]
        # search band per point: 3 sigma of what the prior uncertainty moves it, plus a margin
        R = np.clip(3 * np.sqrt(np.sum((J * prior) ** 2, axis=1)) + c['band_margin_px'], c['band_min_px'], c['band_max_px'])
        K = int(2 * c['band_max_px'] / c['sample_px']) + 1
        s = np.linspace(-c['band_max_px'], c['band_max_px'], K)
        mx = (uv0[:, 0:1] + n_img[:, 0:1] * s[None]).astype(np.float32)
        my = (uv0[:, 1:2] + n_img[:, 1:2] * s[None]).astype(np.float32)
        prof = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=c['smooth_px'] / c['sample_px'], sigmaY=1e-3)
        grad = np.gradient(prof, axis=1) / c['sample_px']        # outward dark -> bright is positive
        grad[np.abs(s)[None, :] > R[:, None]] = -np.inf
        # candidate edges: up to n_candidates local maxima per point above the threshold (sub-pixel)
        pk = (grad[:, 1:-1] > grad[:, :-2]) & (grad[:, 1:-1] >= grad[:, 2:]) & (grad[:, 1:-1] >= c['min_gradient'])
        cand_pt, cand_s, cand_g = [], [], []
        for i in range(len(P)):
            kk = np.where(pk[i])[0] + 1
            if not len(kk):
                continue
            kk = kk[np.argsort(-grad[i, kk])[:c['n_candidates']]]
            y0, y1, y2 = grad[i, kk - 1], grad[i, kk], grad[i, kk + 1]
            den = y0 - 2 * y1 + y2
            sub = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)
            cand_pt += [i] * len(kk)
            cand_s += list(s[kk] + np.clip(sub, -1, 1) * c['sample_px'])
            cand_g += list(y1)
        if len(set(cand_pt)) < c['min_points']:
            return {'cam': cam, 'ok': False, 'fail': 'few_edges', 'points': int(len(P))}
        cand_pt, cand_s = np.array(cand_pt), np.array(cand_s)
        # RANSAC over the candidates for (dn, dyaw), pitch / roll / height at their priors: two points
        # with one candidate each give a hypothesis; the one that explains the most points wins
        rng = np.random.default_rng(len(cand_pt))
        best, best_n = None, 0
        Jc = J[cand_pt][:, :2]
        for _ in range(c['ransac_iters']):
            a, b = rng.choice(len(cand_pt), 2, replace=False)
            if cand_pt[a] == cand_pt[b]:
                continue
            A = Jc[[a, b]]
            if abs(np.linalg.det(A)) < 1e-3:
                continue
            d = np.linalg.solve(A, cand_s[[a, b]])
            if abs(d[0]) > 3 * sigma_n + 0.2 or abs(d[1]) > 3 * sigma_yaw + np.radians(0.2):
                continue
            res = np.abs(cand_s - Jc @ d)
            ok = res < c['ransac_px']
            n_pts = len(set(cand_pt[ok]))
            if n_pts > best_n:
                best, best_n = d, n_pts
        if best is None or best_n < c['min_points']:
            return {'cam': cam, 'ok': False, 'fail': 'few_inliers', 'edges': int(len(cand_pt))}
        # per point, the candidate closest to the consensus hypothesis
        res = np.abs(cand_s - Jc @ best)
        gi, obs = [], []
        for i in np.unique(cand_pt):
            j = np.where(cand_pt == i)[0]
            j = j[np.argmin(res[j])]
            if res[j] < 2 * c['ransac_px']:
                gi.append(i)
                obs.append(cand_s[j])
        gi, obs = np.array(gi), np.array(obs)
        Jg = J[gi]
        # robust (Huber) least squares with priors on all five parameters; observation = J @ delta
        Lam = np.diag(1.0 / prior ** 2)
        w = np.ones(len(gi))
        scale = c['min_sigma_px']
        for _ in range(10):
            A = Jg * (w / scale ** 2)[:, None]
            info = A.T @ Jg + Lam
            delta = np.linalg.solve(info, A.T @ obs)
            r = obs - Jg @ delta
            scale = max(1.4826 * np.median(np.abs(r)), c['min_sigma_px'])
            w = np.minimum(1.0, c['huber_k'] * scale / np.maximum(np.abs(r), 1e-9))
        inl = np.abs(r) < 3 * scale
        if inl.sum() < c['min_points']:
            return {'cam': cam, 'ok': False, 'fail': 'few_inliers', 'edges': int(len(gi))}
        info = (Jg[inl] * (w[inl] / scale ** 2)[:, None]).T @ Jg[inl] + Lam
        cov5 = np.linalg.inv(info)
        cov = cov5[:2, :2] * c['cov_inflation']
        dn = float(delta[0] - c['bias_m'].get(cam, 0.0))
        if self.debug:
            self.last_debug = {'obs': obs, 'J': Jg, 'uv': uv0[gi], 'inl': inl}
        f = {'cam': cam, 'dn': dn, 'dyaw': float(delta[1]), 'dpitch_deg': float(np.degrees(delta[2])),
             'droll_deg': float(np.degrees(delta[3])), 'dz': float(delta[4]), 'cov': cov, 'normal': nrm,
             'points': int(inl.sum()), 'edges': int(len(gi)), 'rms_px': float(np.sqrt(np.mean(r[inl] ** 2))),
             'sigma_n': float(np.sqrt(cov[0, 0])),
             'lateral_info_share': float(1.0 - cov5[0, 0] / sigma_n ** 2)}   # how much the image constrained dn
        if abs(dn) > c['max_correction_m']:
            f.update(ok=False, fail='large')
        elif f['sigma_n'] > c['max_sigma_m']:
            f.update(ok=False, fail='weak')
        else:
            f['ok'] = True
            self.stats[cam]['ok'] += 1
        return f
