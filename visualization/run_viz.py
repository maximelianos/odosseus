#!/usr/bin/env python3
"""Real-time visualization of the camera localization: overhead track view of ground truth and
estimate, live lateral and heading errors, camera fixes and running metrics, as a window paced at the
10 Hz image-set rate and/or an mp4 at 10 fps.

    python visualization/run_viz.py mapped                  # L7, first 30 s from the gate, fold-2 run -> mp4
    python visualization/run_viz.py mapped --live           # also play it in a window at 10 Hz
    python visualization/run_viz.py mapped --lap L2 --t0 40 --t1 70
    python visualization/run_viz.py test --live             # daytime lap, first 30 s: the pipeline live -> window + mp4
    python visualization/run_viz.py test --calib bag        # query with the day bag's own rr/rl extrinsics

Ground truth is used only for the errors, the metrics and the drawn true path, never as an input of
the pipeline. The daytime bag has none: its view shows health signals. Outputs go to
visualization/outputs/<name>/ (gitignored).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402
import replay  # noqa: E402
from live_metrics import RunningMetrics  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
F2_RUN = os.path.join(ref.LOC, 'results', 'cv', 'f2', 'fullrun', 'submission.csv')


def figure(interactive):
    """16:9 figure: 1920x1080 for the mp4, a smaller window on screen (same layout, other pixel density)."""
    if interactive:
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(16, 9), dpi=80)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, 9), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def with_ground_truth(d):
    rm = RunningMetrics(ref.ground_truth(), d['stamp_ns'], d['x'], d['y'], d['yaw'], d['proc_ms'], d['n_acc'])
    d.update(t=(d['stamp_ns'] - d['stamp_ns'][0]) / 1e9, gt_x=rm.e['gt_x'], gt_y=rm.e['gt_y'], gt_yaw=rm.gt_yaw,
             lat=rm.e['lat'], lon=rm.e['lon'], pos=rm.e['pos'], dyaw_deg=rm.e['dyaw_deg'], ok=rm.ok,
             running=[rm.at(k) for k in range(len(d['stamp_ns']))])
    return rm


def write_poses(d, path):
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for t, x, y, a in zip(d['stamp_ns'], d['x'], d['y'], d['yaw']):
            f.write(f'{int(t)},{x:.4f},{y:.4f},{a:.6f}\n')


def mapped(args):
    from dashboard import TruthDashboard as Dashboard
    import player
    t = time.time()
    d = replay.load(args.run, args.lap, args.t0, args.t1, thumbs=not args.no_thumbs)
    rm = with_ground_truth(d)
    d['bounds'], d['ref_xy'] = ref.track_bounds(), (rm.gt['x'], rm.gt['y'])
    chunk = rm.final()                                   # = camloc.metrics.score; asserts the running values agree
    name = args.name or f'mapped_{args.lap}_{args.t0:g}-{args.t1:g}s'
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)

    # cross-checks: the official tools/evaluate.py on the chunk, and the whole lap against the README
    write_poses(d, os.path.join(out, 'chunk_poses.csv'))
    ref.write_gt_csv(rm.gt, os.path.join(out, 'chunk_gt.csv'))
    official = ref.metrics.run_official(ref.OFFICIAL_EVAL, os.path.join(out, 'chunk_gt.csv'),
                                        os.path.join(out, 'chunk_poses.csv'))
    lap = ref.laps()[args.lap]
    sub = ref.metrics.load_csv(args.run)
    ml = (sub['timestamp_ns'] >= lap['start_ns']) & (sub['timestamp_ns'] < lap['end_ns'])
    lap_gt = ref.metrics.load_csv(os.path.join(ref.ART, 'gt_laps', f'{args.lap}.csv'))
    whole = ref.metrics.score(lap_gt, sub['timestamp_ns'][ml], sub['x'][ml], sub['y'][ml], sub['yaw'][ml])
    gx, gy = rm.gt['x'], rm.gt['y']
    summary = {'run': os.path.relpath(args.run, ref.ROOT), 'lap': args.lap, 'chunk_s': [args.t0, args.t1],
               'frames': len(d['t']), 'distance_m': float(np.hypot(np.diff(gx), np.diff(gy)).sum()),
               'speed_kmh': [float(d['speed_kmh'].min()), float(d['speed_kmh'].max())],
               'chunk': chunk, 'runtime_ms': {'mean': float(d['proc_ms'].mean()), 'p95': float(np.percentile(d['proc_ms'], 95))},
               'fix_rate_pct': float(100 * np.mean(d['n_acc'] > 0)), 'whole_lap': whole}
    with open(os.path.join(out, 'official_evaluate_chunk.txt'), 'w') as f:
        f.write(official)
    print(official)
    print(f'chunk {args.lap} {args.t0:g}-{args.t1:g} s: {len(d["t"])} frames, {summary["distance_m"]:.0f} m, '
          f'pos {chunk["pos_rmse_m"]:.3f} m, lateral {chunk["lat_rmse_m"]:.3f} m, heading {chunk["yaw_rmse_deg"]:.3f} deg, '
          f'<1 m {chunk["within_1.0_pct"]:.1f} %, coverage {chunk["coverage_pct"]:.0f} %')
    print(f'whole lap {args.lap}: pos {whole["pos_rmse_m"]:.3f} m, lateral {whole["lat_rmse_m"]:.3f} m, '
          f'heading {whole["yaw_rmse_deg"]:.3f} deg (data prepared in {time.time() - t:.0f} s)')

    title = (f'Mapped mode · validation lap {args.lap}: {args.t1 - args.t0:g} s from '
             + ('the start/finish line' if args.t0 == 0 else f'{args.t0:g} s into the lap'))
    subtitle = None                                    # no second header line in mapped mode (title only)
    if not args.no_mp4:
        dash = Dashboard(d, figure(False), title, subtitle)
        summary['mp4'] = player.mp4(dash, os.path.join(out, f'{name}.mp4'))
        dash.save_images(out)
        print(f'mp4: {summary["mp4"]["frames"]} frames in {summary["mp4"]["render_s"]:.0f} s -> {out}/{name}.mp4')
    if args.live:
        dash = Dashboard(d, figure(True), title, subtitle)
        summary['live'] = player.live(dash, d['stamp_ns'], args.speed, hold=not args.no_hold)
        print('live window:', json.dumps(summary['live']))
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=1)


def test(args):
    """The daytime lap: camloc's pipeline.run (unchanged, configs/final.yaml) in a worker process at the
    10 Hz image-set rate, drawn as frames arrive; afterwards the mp4 of every frame."""
    import multiprocessing as mp
    import queue
    import daybag
    import live_day
    from camloc import config
    from dashboard import HealthDashboard
    if args.from_log:
        return test_render(args, **_load(args.from_log))
    t = time.time()
    cfg_path = os.path.join(ref.LOC, 'configs', 'final.yaml')
    cams = config.load(cfg_path)['reloc']['cameras']
    state = daybag.read_state(ref.DAY_BAG)
    first = daybag.first_set_stamp(ref.DAY_BAG, cams)
    t0 = first if args.t0 <= 0 else daybag.first_set_stamp(ref.DAY_BAG, cams, first + int(args.t0 * 1e9) - 5_000_000)
    end_ns = t0 + int((args.t1 - args.t0) * 1e9) - 50_000_000          # last frame at chunk end - 0.1 s
    pose, start = daybag.stand_in_first_pose(state, t0)
    start.update(pose=pose, sigma0_xy_m=args.sigma0_xy, sigma0_yaw_deg=args.sigma0_yaw)
    print(f'stand-in first pose ({pose[0]:.1f}, {pose[1]:.1f}, {np.degrees(pose[2]):.1f} deg): {start["method"]}, '
          f'{start["start_along_ref_m"]:.0f} m along {start["ref_lap"]} (prepared in {time.time() - t:.0f} s)')
    chunk = args.t1 - args.t0
    log = live_day.Log(int(round(chunk * 10)), cams, state, ref.track_bounds(), ref.racing_line(),
                       f'stand-in, σ {args.sigma0_xy:g} m / {args.sigma0_yaw:g}°', chunk)
    params = {'config': cfg_path, 'bag': ref.DAY_BAG, 'state': state, 'init': (int(t0), *pose), 'end_ns': int(end_ns),
              'overrides': [f'filter.sigma0_xy={args.sigma0_xy}', f'filter.sigma0_yaw_deg={args.sigma0_yaw}'],
              'calib': args.calib, 'pace': args.live, 'speed': args.speed}
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = ctx.Process(target=live_day.worker, args=(q, params), daemon=True)
    proc.start()
    title = test_title(args.t0, args.t1)
    subtitle = ('No ground truth in this bag: health signals only  ·  map of all 9 night laps, iteration 1, 5 cameras  ·  '
                f'{args.calib} calibration  ·  pipeline at the 10 Hz image-set rate')
    dash = None
    if args.live:
        import matplotlib.pyplot as plt
        dash = HealthDashboard(log.d, figure(True), title, subtitle)
        plt.show(block=False)
        dash.fig.canvas.draw()
        dash.fig.canvas.start_event_loop(0.3)
    done, drawn, draw_ms, skipped = None, -1, [], 0
    while done is None:
        try:
            items = [q.get(timeout=0.05)]
        except queue.Empty:
            if not proc.is_alive():
                raise RuntimeError('the pipeline worker exited without a result')
            if dash is not None:
                dash.fig.canvas.start_event_loop(0.01)
            continue
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
        for kind, payload in items:
            if kind == 'frame':
                log.add(payload)
            elif kind == 'done':
                done = payload
            elif kind == 'error':
                raise RuntimeError('pipeline worker failed:\n' + payload)
            elif kind == 'ready':
                print(f'pipeline ready: cameras {payload["cameras"]}, map {payload["map"]}')
        if dash is not None and log.k > drawn:
            late = log.d['late_ms'][log.k]
            skipped += max(0, log.k - drawn - 1)
            td = time.perf_counter()
            dash.render(log.k, lag=max(0, log.k - drawn - 1), show=True,
                        note='live' if late < 50 else f'processing {late / 1e3:.1f} s behind the 10 Hz camera clock (no pose dropped)')
            dash.fig.canvas.flush_events()
            draw_ms.append(1e3 * (time.perf_counter() - td))
            drawn = log.k
    proc.join(timeout=10)

    d = log.trimmed(done['proc_ms'])                    # runtime as measured by pipeline.run itself
    h = d['health'][-1]
    name = args.name or f'test_{args.t0:g}-{args.t1:g}s' + ('_bagcalib' if args.calib == 'bag' else '')
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    write_poses(d, os.path.join(out, 'poses.csv'))
    summary = {'bag': os.path.relpath(ref.DAY_BAG, ref.ROOT), 'chunk_s': [args.t0, args.t1], 'frames': len(d['t']),
               'config': os.path.relpath(cfg_path, ref.ROOT), 'calibration': args.calib, 'start_pose': start,
               'reference': "organizers' racing line (update map.mcap), offsets = track-frame n",
               'racing_line_s_m': [float(d['ref_s'][0]), float(d['ref_s'][-1])],
               'guard': done['guard'], 'health': h, 'per_camera': done['per_camera'], 'recoveries': done['recoveries'],
               'speed_kmh': [float(d['speed_kmh'].min()), float(d['speed_kmh'].max())],
               'pipeline_late_ms_max': float(d['late_ms'].max())}
    if dash is not None:
        summary['live_window'] = {'drawn': len(draw_ms), 'skipped': skipped, 'draw_ms_mean': float(np.mean(draw_ms)),
                                  'draw_ms_p95': float(np.percentile(draw_ms, 95))}
        print('live window:', json.dumps(summary['live_window']))
    print(f'{len(d["t"])} frames; guard: {done["guard"]}; frames with an accepted fix {h["fix_frames_pct"]:.0f} %, '
          f'fixes {h["accepted"]} accepted / {h["produced"]} produced / {h["tried"]} tried, median matches '
          f'{h["median_matches"]:.0f}; filter sigma {h["sigma_now"]:.2f} m (max {h["sigma_max"]:.2f}); track constraint '
          f'{h["pushed_pct"]:.0f} % of frames; runtime {h["ms_mean"]:.0f} ms mean / {h["ms_p95"]:.0f} ms p95; '
          f'pipeline late by at most {summary["pipeline_late_ms_max"]:.0f} ms')
    print('per camera:', json.dumps(done['per_camera']))
    import pickle
    with open(os.path.join(out, 'log.pkl'), 'wb') as f:      # re-render without the pipeline: test --from-log
        pickle.dump({'d': d, 'title': title, 'subtitle': subtitle, 'summary': summary, 'name': name}, f)
    if dash is not None:
        dash.render(len(d['t']) - 1, final=True, show=True)
    test_render(args, d, title, subtitle, summary, name, out=out, live_dash=dash)


def test_title(t0, t1):
    return f'Test mode · daytime lap: {t1 - t0:g} s from ' + ('the start/finish line' if t0 == 0 else f'{t0:g} s into the lap')


def _load(path):
    import pickle
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    saved['out'] = os.path.dirname(os.path.abspath(path))
    saved['title'] = test_title(*saved['summary']['chunk_s'])        # current naming, also for older recordings
    return saved


def offset_after(d, out, t_from):
    """Offset from the racing line from t_from s on (after the stand-in start has locked on): RMS, mean, max,
    printed, and plotted as plots/racing_line_offset_after_<t_from>s.png."""
    from dashboard import EST, INK, INK2, MUTED, AXIS, GRID, PAGE, SURFACE
    import matplotlib.pyplot as plt
    m = d['t'] >= t_from
    t, off, sig = d['t'][m], d['ref_off'][m], d['sigma_xy'][m] / np.sqrt(2)
    r = {'from_s': t_from, 'frames': int(m.sum()), 'rms_m': float(np.sqrt(np.mean(off ** 2))),
         'mean_m': float(off.mean()), 'abs_max_m': float(np.abs(off).max()),
         'abs_p95_m': float(np.percentile(np.abs(off), 95)), 'within_0.5_m_pct': 100.0 * float(np.mean(np.abs(off) < 0.5))}
    print(f'offset from the racing line from {t_from:g} s ({r["frames"]} frames): RMS {r["rms_m"]:.3f} m, '
          f'mean {r["mean_m"]:+.3f} m, |p95| {r["abs_p95_m"]:.3f} m, |max| {r["abs_max_m"]:.3f} m, '
          f'within 0.5 m {r["within_0.5_m_pct"]:.1f} %')
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=120, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.fill_between(t, off - sig, off + sig, color=EST, alpha=0.10, lw=0, label='filter ±1σ (per axis)')
    ax.plot(t, off, color=EST, lw=2.0, label='estimate')
    ax.set_xlim(t_from, t[-1] + 0.1)
    lim = max(0.5, float(np.ceil(1.1 * np.abs(np.r_[off - sig, off + sig]).max() * 10) / 10))
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('time in chunk [s]', color=INK2)
    ax.set_ylabel('offset [m], + = left', color=INK2)
    ax.set_title(f'Offset from the racing line after {t_from:g} s: RMS {r["rms_m"]:.3f} m, mean {r["mean_m"]:+.3f} m, '
                 f'|max| {r["abs_max_m"]:.2f} m', loc='left', color=INK, fontsize=11)
    ax.text(1.0, 1.01, "context, not an error: n of the organizers' track frame", transform=ax.transAxes,
            ha='right', va='bottom', color=MUTED, fontsize=8.5)
    ax.legend(loc='upper right', frameon=False, fontsize=8.5)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', f'racing_line_offset_after_{t_from:g}s.png'), facecolor=PAGE)
    plt.close(fig)
    return r


def test_render(args, d, title, subtitle, summary, name, out, live_dash=None):
    """mp4 + final frame + metrics.json of a recorded day run; with --from-log --live, replay it at 10 Hz."""
    import player
    from dashboard import HealthDashboard
    if not args.no_mp4:
        dash = HealthDashboard(d, figure(False), title, subtitle)
        summary['mp4'] = player.mp4(dash, os.path.join(out, f'{name}.mp4'))
        dash.save_images(out)
        print(f'mp4: {summary["mp4"]["frames"]} frames in {summary["mp4"]["render_s"]:.0f} s -> {out}/{name}.mp4')
    summary['offset_after'] = offset_after(d, out, args.offset_from)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    if args.live and live_dash is None:                       # replay of a recorded run
        print('replay:', json.dumps(player.live(HealthDashboard(d, figure(True), title, subtitle), d['stamp_ns'],
                                                args.speed, hold=not args.no_hold)))
    elif live_dash is not None and not args.no_hold:
        import matplotlib.pyplot as plt
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='source', required=True)
    n = sp.add_parser('mapped', help='replay a continuous run of the mapped (night training) bag, with ground truth')
    n.add_argument('--run', default=F2_RUN, help='scripts/run.py output CSV (its _diag.npz next to it)')
    n.add_argument('--lap', default='L7')
    n.add_argument('--t0', type=float, default=0.0, help='chunk start, s after the lap start')
    n.add_argument('--t1', type=float, default=30.0, help='chunk end, s after the lap start')
    n.add_argument('--no-thumbs', action='store_true', help='no camera thumbnail (skips reading images)')
    dy = sp.add_parser('test', help='run the pipeline on the test-format daytime sample (no ground truth)')
    dy.add_argument('--t0', type=float, default=0.0, help='chunk start, s after the first image')
    dy.add_argument('--t1', type=float, default=30.0, help='chunk end, s after the first image')
    dy.add_argument('--calib', choices=('training', 'bag'), default='training',
                    help='rr/rl extrinsics used at query time: the training calibration (what the submission does) or the bag\'s own')
    dy.add_argument('--sigma0-xy', type=float, default=3.0, help='initial position sigma of the stand-in first pose, m')
    dy.add_argument('--sigma0-yaw', type=float, default=1.0, help='initial heading sigma of the stand-in first pose, deg')
    dy.add_argument('--offset-from', type=float, default=10.0,
                    help='s from which the racing-line offset RMS is computed and plotted (after lock-on)')
    dy.add_argument('--from-log', help="a recorded run's log.pkl: render (and with --live replay) it without the pipeline")
    for p in (n, dy):
        p.add_argument('--live', action='store_true', help='show a window at the image-set rate')
        p.add_argument('--speed', type=float, default=1.0, help='live speed (1 = real time)')
        p.add_argument('--no-hold', action='store_true', help='close the window at the end')
        p.add_argument('--no-mp4', action='store_true')
        p.add_argument('--name', help='output folder name under visualization/outputs/')
    args = ap.parse_args()
    {'mapped': mapped, 'test': test}[args.source](args)


if __name__ == '__main__':
    main()
