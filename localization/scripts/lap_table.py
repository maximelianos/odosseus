#!/usr/bin/env python3
"""Lap table as CSV: timestamps, kind, split role and speed metrics of every lap (D-002).

    python scripts/lap_table.py            # -> results/laps.csv

Timestamps are header stamps in ns (the clock of the images and of the submission CSV), also as UTC
and as Abu Dhabi local time (UTC+4, the time zone of the bag name); `start_s` / `end_s` count from
the first GT pose of the run. Lap boundaries are the start/finish gate crossings
of D-002; L0 starts at the first and the last lap ends at the last GT pose.

Kind (rule-based, so it is reproducible):
  out-lap   first lap: from the run start in the pit lane to the first gate crossing
  in-lap    last lap: from the last gate crossing into the pit lane
  flying    complete lap, >= 80 % of its distance at >= 85 % of the fastest lap's speed at the same point
  partial   complete lap, 40-80 %: part of the lap at pace, part slow (see `slow_sections`)
  warm-up   complete lap, < 40 %, before the first flying lap
  slow      complete lap, < 40 %, after the first flying lap
"Same point" = same distance travelled since the gate. Speeds are the car-state speed `v_mps` (an
allowed input), time-weighted; distances and the pit-lane time come from the GT path.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import allowed, bag, config, gt as gtm  # noqa: E402
from camloc.trackmap import TrackMap  # noqa: E402

PUSH_SPEED = 0.85        # fraction of the reference lap's speed that counts as "at pace"
SLOW_SPEED = 0.60        # below this fraction a stretch counts as slow
FLYING, PARTIAL = 80.0, 40.0


GST = timezone(timedelta(hours=4))   # Abu Dhabi local time (no DST); the bag is named in it


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def local(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=GST).isoformat(timespec='milliseconds')


def sections(t_s, slow, min_s=10.0, gap_s=5.0):
    """'a-b s' ranges (time in lap) where `slow` holds for >= min_s, gaps < gap_s merged."""
    out, start, last = [], None, None
    for t, s in zip(t_s, slow):
        if s:
            if start is None or t - last > gap_s:
                if start is not None and last - start >= min_s:
                    out.append((start, last))
                start = t
            last = t
    if start is not None and last - start >= min_s:
        out.append((start, last))
    return '; '.join(f'{a:.0f}-{b:.0f} s' for a, b in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--out')
    args = ap.parse_args()
    cfg = config.load(args.config)
    A = lambda *p: config.artifact(cfg, *p)  # noqa: E731
    laps = json.load(open(A('laps.json')))['laps']
    st = dict(np.load(A('state.npz')))
    g = gtm.load_gt(A('gt.npz'))
    track = TrackMap(dict(np.load(A('track_bounds.npz'))))
    role = {n: r for r in ('train', 'tune', 'val') for n in cfg['split'][r]}

    # car-state speed, time-weighted (each sample holds until the next one; gaps capped at 0.2 s)
    ts, v = st['stamp_ns'], st['v_mps']
    w = np.clip(np.diff(ts, append=ts[-1]) / 1e9, 0, 0.2)
    # GT path: distance travelled and pit-lane membership
    tg, xy = g['stamp_ns'], g['pos'][:, :2]
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    sub = np.arange(0, len(tg), 5)
    in_pit = np.array([m['pit'] > 0 and m['track'] <= 0 for m in (track.margins(p) for p in xy[sub])])
    wg = np.clip(np.diff(tg[sub], append=tg[sub][-1]) / 1e9, 0, 1.0)
    vg = np.interp(tg, ts, v)
    # image sets (any camera), as the pipeline forms them
    img = np.array([s.stamp_ns for s in bag.iter_image_sets(cfg['bag'], allowed.CAMERAS)], np.int64)

    # reference = fastest complete lap; speed profile over distance since the gate
    complete = [lap for lap in laps[1:-1]]
    ref = min(complete, key=lambda lap: lap['duration_s'])
    grid = np.arange(0.0, 2970.0, 1.0)

    def profile(lap):
        m = (tg >= lap['start_ns']) & (tg < lap['end_ns'])
        return m, dist[m] - dist[m][0]

    mr, dr = profile(ref)
    v_ref = np.interp(grid, dr, vg[mr])
    first_flying = None
    rows = []
    for i, lap in enumerate(laps):
        a, b = lap['start_ns'], lap['end_ns']
        m = (ts >= a) & (ts < b)
        vv, ww = v[m], w[m]
        mg, dg = profile(lap)
        mp = (tg[sub] >= a) & (tg[sub] < b)
        mi = (img >= a) & (img < b)
        is_complete = 0 < i < len(laps) - 1
        push = slow = float('nan')
        slow_txt = ''
        if is_complete:
            vl = np.interp(grid, dg, vg[mg])
            push = 100.0 * float(np.mean(vl >= PUSH_SPEED * v_ref))
            slow_lap = vg[mg] < SLOW_SPEED * np.interp(dg, grid, v_ref)
            slow = 100.0 * float(np.mean(vl < SLOW_SPEED * v_ref))
            slow_txt = sections((tg[mg] - a) / 1e9, slow_lap)
            if push >= FLYING:
                kind = 'flying'
                first_flying = i if first_flying is None else first_flying
            elif push >= PARTIAL:
                kind = 'partial'
            else:
                kind = 'warm-up' if first_flying is None else 'slow'
        else:
            kind = 'out-lap' if i == 0 else 'in-lap'
        rows.append({
            'lap': lap['name'], 'kind': kind, 'complete_lap': int(is_complete), 'split_role': role.get(lap['name'], 'unused'),
            'start_ns': a, 'end_ns': b, 'start_utc': utc(a), 'end_utc': utc(b),
            'start_local_uae': local(a), 'end_local_uae': local(b),
            'start_s': round((a - tg[0]) / 1e9, 3), 'end_s': round((b - tg[0]) / 1e9, 3),
            'duration_s': round((b - a) / 1e9, 3),
            'first_image_ns': int(img[mi][0]), 'last_image_ns': int(img[mi][-1]), 'image_sets': int(mi.sum()),
            'distance_m': round(float(dg[-1]), 1),
            'v_max_mps': round(float(vv.max()), 2), 'v_max_kmh': round(3.6 * float(vv.max()), 1),
            'v_mean_mps': round(float(np.sum(vv * ww) / np.sum(ww)), 2),
            'v_mean_kmh': round(3.6 * float(np.sum(vv * ww) / np.sum(ww)), 1),
            'v_median_mps': round(float(np.median(vv)), 2), 'v_p95_mps': round(float(np.percentile(vv, 95)), 2),
            'v_min_mps': round(float(vv.min()), 2),
            'time_above_30mps_s': round(float(ww[vv > 30].sum()), 1), 'time_below_10mps_s': round(float(ww[vv < 10].sum()), 1),
            'standstill_s': round(float(ww[vv < 0.05].sum()), 1), 'pit_lane_s': round(float(wg[mp][in_pit[mp]].sum()), 1),
            'push_share_pct': '' if np.isnan(push) else round(push, 1),
            'slow_share_pct': '' if np.isnan(slow) else round(slow, 1),
            'slow_sections': slow_txt,
        })
    for r in rows:   # warm-up only applies before the first flying lap
        if r['kind'] == 'warm-up' and first_flying is not None and int(r['lap'][1:]) > first_flying:
            r['kind'] = 'slow'

    out = args.out or os.path.join(os.path.dirname(cfg['results']), 'laps.csv')
    with open(out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'reference (fastest complete lap): {ref["name"]} {ref["duration_s"]:.1f} s -> {out}')
    print(f'{"lap":4} {"kind":8} {"role":6} {"start_utc":24} {"dur s":>6} {"dist m":>7} {"vmax":>5} {"vmean":>5} '
          f'{"push%":>5}  slow sections')
    for r in rows:
        print(f'{r["lap"]:4} {r["kind"]:8} {r["split_role"]:6} {r["start_utc"]:24} {r["duration_s"]:6.1f} {r["distance_m"]:7.0f} '
              f'{r["v_max_mps"]:5.1f} {r["v_mean_mps"]:5.1f} {str(r["push_share_pct"]):>5}  {r["slow_sections"]}')


if __name__ == '__main__':
    main()
