"""Reading the ROS 2 mcap bag, restricted to the topics and fields in camloc.allowed.

Latched topics (camera_info, tf_static, track_bounds) are repeated in the bag; the first message of
each is used. All stamps are header stamps in ns.
"""
from dataclasses import dataclass, field

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from . import allowed
from .geometry import make_T, quat_to_rot


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def _reader(f):
    return make_reader(f, decoder_factories=[DecoderFactory()])


def bag_time_range(path):
    """(first, last) log time of the bag in ns."""
    with open(path, 'rb') as f:
        s = _reader(f).get_summary().statistics
        return s.message_start_time, s.message_end_time


@dataclass
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray                 # 3x3, full-resolution pixels
    D: np.ndarray                 # plumb_bob k1 k2 p1 p2 k3
    T_base_cam: np.ndarray        # 4x4, optical frame -> base_link

    def to_dict(self):
        return {'name': self.name, 'width': self.width, 'height': self.height, 'K': self.K.tolist(),
                'D': self.D.tolist(), 'T_base_cam': self.T_base_cam.tolist()}

    @staticmethod
    def from_dict(d):
        return Camera(d['name'], d['width'], d['height'], np.array(d['K']), np.array(d['D']),
                      np.array(d['T_base_cam']))


def read_calibration(path, cams=allowed.CAMERAS):
    """Intrinsics (first camera_info per camera) and extrinsics (first /tf_static)."""
    info, extr = {}, {}
    topics = [allowed.INFO_TOPIC.format(c) for c in cams] + [allowed.TF_STATIC_TOPIC]
    with open(path, 'rb') as f:
        for _, ch, _, msg in _reader(f).iter_decoded_messages(topics=topics):
            if ch.topic == allowed.TF_STATIC_TOPIC:
                if extr:
                    continue
                for t in msg.transforms:
                    q, p = t.transform.rotation, t.transform.translation
                    extr[t.child_frame_id] = (t.header.frame_id,
                                              make_T(quat_to_rot([q.x, q.y, q.z, q.w]), [p.x, p.y, p.z]))
            else:
                c = ch.topic.split('camera_')[1].split('/')[0]
                info.setdefault(c, msg)
            if extr and len(info) == len(cams):
                break
    out = {}
    for c in cams:
        parent, T = extr[f'constructor0/camera_{c}']
        assert parent == 'constructor0/base_link', parent
        m = info[c]
        out[c] = Camera(c, m.width, m.height, np.array(m.k, float).reshape(3, 3), np.array(m.d, float), T)
    return out


def read_track_bounds(path):
    """{marker namespace: (N, 3) points in map}, from the first /constructor0/track_bounds message."""
    with open(path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[allowed.TRACK_TOPIC]):
            return {m.ns: np.array([(p.x, p.y, p.z) for p in m.points]) for m in msg.markers}
    raise ValueError(f'no {allowed.TRACK_TOPIC} in {path}')


def read_state(path):
    """Car state as {'stamp_ns': int64 array, field: float array} for the allowed fields only, sorted."""
    cols = {k: [] for k in ('stamp_ns',) + allowed.ALLOWED_STATE_FIELDS}
    with open(path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[allowed.STATE_TOPIC]):
            cols['stamp_ns'].append(stamp_ns(msg.header))
            for k in allowed.ALLOWED_STATE_FIELDS:
                cols[k].append(getattr(msg, k))
    out = {k: np.asarray(v, dtype=np.int64 if k == 'stamp_ns' else float) for k, v in cols.items()}
    o = np.argsort(out['stamp_ns'], kind='stable')
    return {k: v[o] for k, v in out.items()}


def read_first_pose(path):
    """(stamp_ns, x, y, yaw) of the first map -> constructor0/base_link transform: the one GT pose the
    rules allow as the initial pose. Nothing after it is read."""
    from .geometry import yaw_of
    with open(path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[allowed.FIRST_POSE_TOPIC]):
            for t in msg.transforms:
                if t.header.frame_id == 'map' and t.child_frame_id == 'constructor0/base_link':
                    q, p = t.transform.rotation, t.transform.translation
                    return stamp_ns(t.header), p.x, p.y, yaw_of(quat_to_rot([q.x, q.y, q.z, q.w]))
    raise ValueError(f'no map -> base_link transform in {path}')


@dataclass
class ImageSet:
    stamp_ns: int                                 # max header stamp over the set (D-005)
    stamps: dict = field(default_factory=dict)    # camera -> header stamp
    images: dict = field(default_factory=dict)    # camera -> decoded image (or jpeg bytes)


def iter_image_sets(path, cams, start_ns=None, end_ns=None, decode=None, tol_ns=2_000_000):
    """Yield ImageSets in stamp order. Images of different cameras whose header stamps lie within
    tol_ns form one set (the cameras are hardware-triggered: measured spread 0.04 ms). A set is
    yielded with whatever cameras arrived; decode(cam, jpeg_bytes) -> image (None: keep bytes).

    start_ns/end_ns select by header stamp; the log-time window is padded because messages are
    logged ~40-60 ms after exposure.
    """
    topics = {allowed.IMAGE_TOPIC.format(c): c for c in cams}
    pad = 1_000_000_000
    kw = {}
    if start_ns is not None:
        kw['start_time'] = max(0, start_ns - pad)
    if end_ns is not None:
        kw['end_time'] = end_ns + pad
    pending = {}                                    # stamp key -> ImageSet

    def flush(upto):
        for k in sorted(k for k in pending if k < upto):
            s = pending.pop(k)
            s.stamp_ns = max(s.stamps.values())
            yield s

    with open(path, 'rb') as f:
        for _, ch, _, msg in _reader(f).iter_decoded_messages(topics=list(topics), **kw):
            t = stamp_ns(msg.header)
            if (start_ns is not None and t < start_ns) or (end_ns is not None and t > end_ns):
                continue
            c = topics[ch.topic]
            key = next((k for k in pending if abs(k - t) <= tol_ns), t)
            s = pending.setdefault(key, ImageSet(key))
            data = bytes(msg.data)
            s.images[c] = decode(c, data) if decode else data
            s.stamps[c] = t
            # sets older than 0.5 s cannot receive more images (logging delay is < 0.1 s)
            yield from flush(t - 500_000_000)
    yield from flush(np.iinfo(np.int64).max)
