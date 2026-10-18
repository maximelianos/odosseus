"""Track map from /constructor0/track_bounds: corridors (track, pit lane) between a left and a right
border, signed distances to the borders, the inside-track test and the ground height."""
import numpy as np
from scipy.spatial import cKDTree

CORRIDORS = {'track': ('track_left', 'track_right'), 'pit': ('pitlane_left', 'pitlane_right')}


class Polyline:
    """2D polyline (with z per vertex), densified so the nearest vertex finds the nearest segment."""

    def __init__(self, pts, closed, step=0.5):
        pts = np.asarray(pts, float)
        if closed and np.linalg.norm(pts[0, :2] - pts[-1, :2]) > 1e-6:
            pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-6])
        pts = pts[keep]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
        n = max(int(np.ceil(s[-1] / step)) + 1, 2)
        si = np.linspace(0, s[-1], n)
        self.p = np.stack([np.interp(si, s, pts[:, k]) for k in range(3)], axis=1)
        self.s = si
        self.closed = closed
        self.tree = cKDTree(self.p[:, :2])

    def project(self, xy):
        """Signed distance (left of the polyline direction = positive), arc length s, z, tangent."""
        xy = np.asarray(xy, float)
        _, i = self.tree.query(xy)
        best = None
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.p):
                continue
            a, b = self.p[j, :2], self.p[j + 1, :2]
            ab = b - a
            u = np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            q = a + u * ab
            d = np.linalg.norm(xy - q)
            if best is None or d < best[0]:
                cross = ab[0] * (xy[1] - a[1]) - ab[1] * (xy[0] - a[0])
                z = self.p[j, 2] + u * (self.p[j + 1, 2] - self.p[j, 2])
                best = (d, np.sign(cross) * d, self.s[j] + u * (self.s[j + 1] - self.s[j]), z, ab / np.linalg.norm(ab))
        _, sd, s, z, tan = best
        return sd, s, z, tan

    def beyond_ends(self, s):
        """True if the projection fell on an end vertex of an open polyline (point is past its ends)."""
        return not self.closed and (s <= 1e-6 or s >= self.s[-1] - 1e-6)


class TrackMap:
    def __init__(self, bounds):
        self.raw = bounds
        self.lines = {
            'track_left': Polyline(bounds['track_left'], closed=True),
            'track_right': Polyline(bounds['track_right'], closed=True),
            'pitlane_left': Polyline(bounds['pitlane_left'], closed=False),
            'pitlane_right': Polyline(bounds['pitlane_right'], closed=False),
        }

    def margins(self, xy):
        """{corridor: metres inside (>0) or outside (<0) the corridor at xy}. Positive means right of
        the left border and left of the right border (borders run in the driving direction). Past the
        ends of the open pit-lane borders the pit corridor does not exist (-inf)."""
        out = {}
        for name, (l, r) in CORRIDORS.items():
            sl, s_l, _, _ = self.lines[l].project(xy)
            sr, s_r, _, _ = self.lines[r].project(xy)
            if self.lines[l].beyond_ends(s_l) or self.lines[r].beyond_ends(s_r):
                out[name] = -np.inf
            else:
                out[name] = min(-sl, sr)
        return out

    def corridor_of(self, xy):
        """(corridor name, margin) of the corridor that contains xy most deeply."""
        m = self.margins(xy)
        name = max(m, key=m.get)
        return name, m[name]

    def inside_margin(self, xy):
        """Metres inside the drivable area (track or pit lane); negative = outside."""
        return self.corridor_of(xy)[1]

    def nearest_border_point(self, xy, corridor):
        """Closest point on either border of a corridor, and the inward unit normal there."""
        best = None
        for side, name in zip(('left', 'right'), CORRIDORS[corridor]):
            sd, s, z, tan = self.lines[name].project(xy)
            if best is None or abs(sd) < abs(best[0]):
                left_normal = np.array([-tan[1], tan[0]])
                q = np.asarray(xy) - sd * left_normal
                inward = -left_normal if side == 'left' else left_normal
                best = (sd, q, inward)
        return best[1], best[2]

    def ground_z(self, xy):
        """Surface height at xy: mean z of the two borders (at their nearest points) of the corridor
        the point is in. The pit exit runs under the track, so track borders alone are wrong there."""
        l, r = CORRIDORS[self.corridor_of(xy)[0]]
        return 0.5 * (self.lines[l].project(xy)[2] + self.lines[r].project(xy)[2])

    def check_orientation(self, xy, heading):
        """Mean cos between each border's tangent and the driving heading at samples (should be > 0)."""
        out = {}
        for name, line in self.lines.items():
            c = []
            for p, h in zip(xy, heading):
                sd, _, _, tan = line.project(p)
                if abs(sd) < 30:
                    c.append(tan[0] * np.cos(h) + tan[1] * np.sin(h))
            out[name] = float(np.mean(c)) if c else float('nan')
        return out
