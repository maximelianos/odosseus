"""Showing a dashboard: a live window paced at the image-set rate (10 Hz), or an mp4 at 10 fps.

Live: frame k is due at (stamp_k - stamp_0) / speed after the start. If drawing falls behind, the window
jumps to the newest due frame (frames are never computed differently, only not drawn) and says how far
behind it is. The mp4 draws every frame and holds the final one for a few seconds.
"""
import subprocess
import time

import numpy as np


def mp4(dash, path, fps=10, hold_s=3.0):
    canvas = dash.fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
           '-preset', 'medium', '-movflags', '+faststart', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t = time.perf_counter()
    for k in range(dash.n):
        dash.render(k)
        p.stdin.write(bytes(canvas.buffer_rgba()))
    dash.render(dash.n - 1, final=True)
    last = bytes(canvas.buffer_rgba())
    for _ in range(int(hold_s * fps)):
        p.stdin.write(last)
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    return {'frames': dash.n, 'render_s': time.perf_counter() - t, 'size': [w, h]}


def live(dash, stamps, speed=1.0, hold=True):
    import matplotlib.pyplot as plt
    due = (np.asarray(stamps) - stamps[0]) / 1e9 / speed
    canvas = dash.fig.canvas
    plt.show(block=False)
    canvas.draw()
    # waits run the GUI event loop directly: plt.pause() would redraw the stale figure without the
    # blitted (animated) artists and make them flicker
    canvas.start_event_loop(0.3)                     # let the window appear before the clock starts
    t0 = time.perf_counter()
    k, skipped, draw_ms = -1, 0, []
    while k < len(due) - 1:
        newest = int(np.searchsorted(due, time.perf_counter() - t0, side='right')) - 1
        k_next = max(k + 1, newest)
        skipped += k_next - k - 1
        k = k_next
        td = time.perf_counter()
        dash.render(k, lag=max(newest - k, 0), show=True)
        canvas.flush_events()
        draw_ms.append(1e3 * (time.perf_counter() - td))
        if k < len(due) - 1:
            wait = due[k + 1] - (time.perf_counter() - t0)
            if wait > 0:
                canvas.start_event_loop(wait)
    dash.render(k, final=True, show=True)
    stats = {'frames': len(due), 'drawn': len(draw_ms), 'skipped': skipped,
             'draw_ms_mean': float(np.mean(draw_ms)), 'draw_ms_p95': float(np.percentile(draw_ms, 95)),
             'wall_s': time.perf_counter() - t0, 'chunk_s': float(due[-1] + 0.1 / speed)}
    if hold:
        plt.show()
    else:
        plt.close(dash.fig)
    return stats
