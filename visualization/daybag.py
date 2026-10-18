"""The daytime sample (hackFiles/DAYTIME_TEST_...). Its layout differs from the training bag, so the
visualizer reads it itself and hands camloc's pipeline what pipeline.run already takes as arguments;
nothing in localization/ is changed.

- Car state: /constructor0/validation/twist + /accel (frame base_link, 106 Hz), the motion fields of the
  car's state estimation ("SE"): twist.linear.x / y = vx / vy, angular.z = wz, accel.linear.x / y =
  ax / ay. v_mps is not included, so v = |(vx, vy)|.
- Track borders: the training bag's (same map frame), from localization/artifacts.
- First pose: none in this bag (the final test gives it). For the visualizer only, a stand-in: the
  whole lap's motion data placed on the track by matching its curvature over distance against a
  training lap's GT path. It uses the whole lap (not causal); it only replaces the given first pose.
"""
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

import reference as ref
from camloc import bag
from camloc.geometry import inv_T, wrap

TWIST, ACCEL = '/constructor0/validation/twist', '/constructor0/validation/accel'


def read_state(path):
    """Car state in the format of camloc.bag.read_state (allowed motion fields only), sorted by stamp."""
    tw, ac = [], {}
    with open(path, 'rb') as f:
        for _, ch, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[TWIST, ACCEL]):
            t = bag.stamp_ns(m.header)
            if ch.topic == TWIST:
                tw.append((t, m.twist.linear.x, m.twist.linear.y, m.twist.angular.z))
            else:
                ac[t] = (m.accel.linear.x, m.accel.linear.y)
    tw.sort()
    t = np.array([r[0] for r in tw], dtype=np.int64)
    a = np.array([r[1:] for r in tw], float)
    ka = np.array(sorted(ac), dtype=np.int64)
    va = np.array([ac[k] for k in ka], float)
    return {'stamp_ns': t, 'vx_mps': a[:, 0], 'vy_mps': a[:, 1], 'v_mps': np.hypot(a[:, 0], a[:, 1]),
            'wz_radps': a[:, 2], 'ax_mps2': np.interp(t, ka, va[:, 0]), 'ay_mps2': np.interp(t, ka, va[:, 1])}


def first_set_stamp(path, cams, after_ns=None):
    """Earliest image stamp of the first image set at or after after_ns: a run started there gets every
    camera of that set (camloc.bag.iter_image_sets selects images by their own stamps)."""
    s = next(bag.iter_image_sets(path, cams, after_ns, None))
    return min(s.stamps.values())


def _curvature(yaw, ds, window_m=10.0):
    k = np.gradient(yaw, ds)
    w = max(1, int(window_m / ds))
    return np.convolve(k, np.ones(w) / w, mode='same')


def stand_in_first_pose(state, t_ns, lap='L7', ds=0.5):
    """(x, y, yaw) at t_ns and a dict of how it was found. Curvature over distance of the whole lap
    (wz / speed integrated over the car-state distance) against the training lap's GT, best shift by
    correlation; the heading offset is the median over the matched distance; then dead reckoning of
    the whole lap from the matched start, read at t_ns."""
    gx, gy, gyaw = ref.lap_line(lap)
    sr = np.r_[0, np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))]
    g = np.arange(0, sr[-1], ds)
    rx, ry, ryaw = np.interp(g, sr, gx), np.interp(g, sr, gy), np.interp(g, sr, gyaw)
    t = (state['stamp_ns'] - state['stamp_ns'][0]) / 1e9
    dt = np.diff(t, prepend=t[0])
    vx, vy, wz = state['vx_mps'], state['vy_mps'], state['wz_radps']
    sd, yd = np.cumsum(vx * dt), np.cumsum(wz * dt)
    yds = np.interp(np.arange(0, sd[-1], ds), sd, yd)
    kr, kd = _curvature(ryaw, ds), _curvature(yds, ds)
    n = min(len(kd), len(kr))
    kd, yds = kd[:n] - kd[:n].mean(), yds[:n]
    kr2 = np.r_[kr, kr]                                  # the lap is closed: the reference wraps around
    score = np.array([np.dot(kr2[i:i + n] - kr2[i:i + n].mean(), kd) for i in range(len(kr))])
    sh = int(np.argmax(score))
    ryaw2 = np.r_[ryaw, ryaw + (ryaw[-1] - ryaw[0])]
    off = float(np.median(ryaw2[sh:sh + n] - yds))
    psi = off + yd
    x = rx[sh] + np.cumsum((vx * np.cos(psi) - vy * np.sin(psi)) * dt)
    y = ry[sh] + np.cumsum((vx * np.sin(psi) + vy * np.cos(psi)) * dt)
    tq = (t_ns - state['stamp_ns'][0]) / 1e9
    if tq <= 0:                                          # before the first car-state sample: extrapolate back
        x0 = x[0] + (vx[0] * np.cos(psi[0]) - vy[0] * np.sin(psi[0])) * tq
        y0 = y[0] + (vx[0] * np.sin(psi[0]) + vy[0] * np.cos(psi[0])) * tq
        a0 = psi[0] + wz[0] * tq
    else:
        x0, y0, a0 = np.interp(tq, t, x), np.interp(tq, t, y), np.interp(tq, t, psi)
    return (float(x0), float(y0), float(wrap(a0))), {
        'method': f'curvature over distance matched against {lap} GT (whole lap, not causal)',
        'start_along_ref_m': sh * ds, 'heading_offset_deg': float(np.degrees(off)), 'ref_lap': lap}


def use_bag_calibration(reloc, path, cams):
    """Query with this bag's own extrinsics for `cams` (the map keeps the training calibration;
    intrinsics are identical in both bags)."""
    cal = bag.read_calibration(path)
    for c in cams:
        reloc.calib[c] = cal[c]
        reloc.T_cam_base[c] = inv_T(cal[c].T_base_cam)
