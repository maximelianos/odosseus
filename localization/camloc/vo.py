"""E-3: visual-odometry rotation between consecutive frames of one camera (README, Experiments).

The relocalizer extracts SIFT for every camera and frame (undistorted normalized coordinates, car-body
masked); the features of two consecutive frames of one camera are matched (ratio test), an essential
matrix is fitted with RANSAC, and the rotation between the frames is recovered. Its yaw component in
the base frame is the heading change between the two image stamps, which the filter fuses as a
relative measurement (PoseEKF.clone_yaw / update_relative_yaw). Uses no ground truth.
"""
import cv2
import numpy as np

from .geometry import yaw_of


class VisualOdometry:
    def __init__(self, vcfg, cam):
        """vcfg: config block `vo`; cam: bag.Camera of the camera used."""
        self.c = vcfg
        self.R_bc = cam.T_base_cam[:3, :3]
        self.thr = vcfg['ransac_px'] / cam.K[0, 0]
        self.prev = None
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
        self.stats = {'pairs': 0, 'ok': 0}

    def step(self, stamp_ns, uv, desc):
        """Features of the next frame -> (dyaw [rad], inliers, dt [s]) since the previous frame, or None."""
        prev, self.prev = self.prev, (stamp_ns, uv, desc)
        if prev is None or len(desc) < self.c['min_inliers'] or len(prev[2]) < self.c['min_inliers']:
            return None
        dt = (stamp_ns - prev[0]) / 1e9
        if not 0 < dt <= self.c['max_dt_s']:
            return None
        self.stats['pairs'] += 1
        m = [p[0] for p in self.matcher.knnMatch(prev[2].astype(np.float32), desc.astype(np.float32), k=2)
             if len(p) == 2 and p[0].distance < self.c['ratio'] * p[1].distance]
        if len(m) < self.c['min_inliers']:
            return None
        p1 = prev[1][[x.queryIdx for x in m]].astype(np.float64)
        p2 = uv[[x.trainIdx for x in m]].astype(np.float64)
        E, mask = cv2.findEssentialMat(p1, p2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=self.thr)
        if E is None or E.shape != (3, 3):
            return None
        n, R, _, _ = cv2.recoverPose(E, p1, p2, np.eye(3), mask=mask)
        if n < self.c['min_inliers']:
            return None
        R_c1c2 = R.T                                  # recoverPose: X2 = R X1 + t (frame 1 -> frame 2)
        dyaw = yaw_of(self.R_bc @ R_c1c2 @ self.R_bc.T)
        self.stats['ok'] += 1
        return dyaw, int(n), dt
