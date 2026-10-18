"""Local features for relocalization: SIFT on CLAHE-equalized half-resolution grayscale (D-010).

Keypoints are returned as undistorted normalized image coordinates (x/z, y/z in the optical frame),
so the rest of the pipeline works without intrinsics.
"""
import cv2
import numpy as np


def decode_half_gray(data):
    """JPEG -> half-resolution grayscale (libjpeg DCT scaling: about 3x faster than full decode)."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)


class Extractor:
    def __init__(self, cam, fcfg, mask=None):
        """cam: bag.Camera; fcfg: config reloc.features; mask: half-res uint8, 0 = ignore (car body)."""
        self.cam = cam
        self.sift = cv2.SIFT_create(nfeatures=fcfg['n_features'], contrastThreshold=fcfg['contrast_threshold'])
        self.clahe = cv2.createCLAHE(clipLimit=fcfg['clahe_clip'], tileGridSize=(8, 8))
        self.mask = mask

    def __call__(self, gray_half):
        """-> (uv_norm (N,2) float32, desc (N,128) uint8, uv_px (N,2) full-res distorted pixels)."""
        img = self.clahe.apply(gray_half)
        kps, desc = self.sift.detectAndCompute(img, self.mask)
        if desc is None or not kps:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), np.zeros((0, 2), np.float32)
        # half-res pixel i covers full-res pixels 2i and 2i+1
        px = np.array([k.pt for k in kps], np.float32) * 2.0 + 0.5
        uv = cv2.undistortPoints(px.reshape(-1, 1, 2), self.cam.K, self.cam.D).reshape(-1, 2)
        return uv.astype(np.float32), np.clip(np.rint(desc), 0, 255).astype(np.uint8), px
