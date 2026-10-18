"""Laps: crossings of a start/finish gate by the GT trajectory (D-002)."""
import numpy as np


def gate_segment(track, point):
    """Gate across the track through `point`: from its projection on the left border to the one on the
    right border."""
    ends = []
    for name in ('track_left', 'track_right'):
        line = track.lines[name]
        sd, _, _, tan = line.project(point)
        left_normal = np.array([-tan[1], tan[0]])
        ends.append(np.asarray(point, float) - sd * left_normal)
    return ends[0], ends[1]


def crossings(stamp_ns, xy, a, b):
    """Stamps where the trajectory crosses segment a-b, keeping only the dominant (driving) direction."""
    d = b - a
    side = d[0] * (xy[:, 1] - a[1]) - d[1] * (xy[:, 0] - a[0])
    out = []
    for i in np.where(np.sign(side[:-1]) != np.sign(side[1:]))[0]:
        p, q = xy[i], xy[i + 1]
        # p + w r = a + u d, solved with 2D cross products
        r = q - p
        den = r[0] * d[1] - r[1] * d[0]
        if abs(den) < 1e-12:
            continue
        u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
        w = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den
        if 0 <= u <= 1 and 0 <= w <= 1:
            out.append((int(stamp_ns[i] + w * (stamp_ns[i + 1] - stamp_ns[i])), float(side[i + 1] - side[i])))
    if not out:
        return []
    # keep crossings in the dominant direction (the driving direction through the gate)
    sgn = np.sign(np.median([o[1] for o in out]))
    return [t for t, s in out if np.sign(s) == sgn]


def lap_table(stamp_ns, xy, a, b, t_start, t_end):
    """[{'name': 'L0', 'start_ns', 'end_ns'}...]: L0 from the start of the run to the first crossing,
    the last lap from the last crossing to the end of the run."""
    edges = [t_start] + crossings(stamp_ns, xy, a, b) + [t_end]
    return [{'name': f'L{i}', 'start_ns': int(edges[i]), 'end_ns': int(edges[i + 1])}
            for i in range(len(edges) - 1)]
