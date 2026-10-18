"""The dashboard: one figure for the live window and the mp4 (same drawing code, 16:9).

    overview map | camera thumbnail      | metrics
                 | chase view            | time series 1
                 | error scope /         | time series 2
                 |   camera matching     | camera fixes per frame

TruthDashboard (night, with ground truth): errors against the truth, the official metrics.
HealthDashboard (day, no ground truth): L7's driven line as context, the filter's uncertainty, what
each camera matched; the metrics that can be measured without ground truth.

Colors follow the reference palette of the dataviz guidance (light mode, validated): ink for the
ground truth, categorical slot 1 (blue) for the estimate, a one-hue blue ramp for the size of its
error (night) or its uncertainty (day), slots 3 / 2 (aqua / orange) for accepted / rejected fixes.

`render(k)` draws frame k from per-frame arrays only, so frames can be drawn in any order (the live
window skips frames when drawing falls behind; the mp4 draws every frame). The static parts are drawn
once and kept as a background; per frame only the moving artists are drawn on top (blitting), unless
an axis range changed.
"""
import numpy as np
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

from reference import GATE

SURFACE, PAGE = '#fcfcfb', '#f9f9f7'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
EST, ACC, REJ = '#2a78d6', '#1baf7a', '#eb6834'
RAMP = LinearSegmentedColormap.from_list('blue_seq', ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
# schematic car footprint in base_link (x forward, y left), metres
CAR = np.array([[-1.0, -0.95], [3.2, -0.95], [4.0, 0.0], [3.2, 0.95], [-1.0, 0.95]])
CHASE_X, CHASE_Y = (-22, 22), (-14, 30)     # chase view window, m (x right, y ahead)
OFFSET_LIM_M = 5.0              # test mode: offset from the racing line always shown at +-5 m

mpl.rcParams.update({
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.labelsize': 9,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'legend.frameon': False,
})


def _style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    if grid:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def _title(ax, text, sub=None, legend=None, below=False):
    """Panel title on the left; a muted note on the right (or under the title with below=True, for narrow
    panels), or, with a legend (handles, labels), the note under the title and the legend on the right
    of the title row (so it never covers data)."""
    if legend is None and below:
        ax.set_title(text, loc='left', color=INK, fontsize=10.5, pad=17)
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, ha='left', va='bottom', color=MUTED, fontsize=8.5)
        return
    if legend is None:
        ax.set_title(text, loc='left', color=INK, fontsize=10.5, pad=5)
        if sub:
            ax.text(1.0, 1.0, sub, transform=ax.transAxes, ha='right', va='bottom', color=MUTED, fontsize=8.5)
        return
    ax.set_title(text, loc='left', color=INK, fontsize=10.5, pad=17 if sub else 5)
    if sub:
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, ha='left', va='bottom', color=MUTED, fontsize=8.5)
    ax.legend(*legend, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=len(legend[0]), borderaxespad=0.3,
              handlelength=1.6, columnspacing=1.2)


def _to_screen(px, py, cx, cy, cyaw):
    """Map points -> chase-view screen of a car at (cx, cy, cyaw): x right, y forward (heading up)."""
    dx, dy = px - cx, py - cy
    fwd = np.cos(cyaw) * dx + np.sin(cyaw) * dy
    left = -np.sin(cyaw) * dx + np.cos(cyaw) * dy
    return -left, fwd


def _in_view(sx, sy, margin=15.0):
    """Only the points near the chase window, with NaN breaks between separate runs (fast to draw)."""
    keep = (sx > CHASE_X[0] - margin) & (sx < CHASE_X[1] + margin) & (sy > CHASE_Y[0] - margin) & (sy < CHASE_Y[1] + margin)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return [], []
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.insert(sx[idx], brk, np.nan), np.insert(sy[idx], brk, np.nan)


def _footprint(lat=0.0, lon=0.0, dyaw=0.0):
    """Screen polygon of a car offset by (lat, lon) and rotated by dyaw relative to the view's car."""
    c, s = np.cos(dyaw), np.sin(dyaw)
    sx, sy = -CAR[:, 1], CAR[:, 0]
    return np.c_[c * sx - s * sy - lat, s * sx + c * sy + lon]


def _step_limit(v, steps):
    return next((s for s in steps if v <= s), steps[-1])


def _steps(t, v):
    """x, y of a step line: each value holds until the next frame."""
    te = np.r_[t, t[-1] + 0.1]
    return np.repeat(te, 2)[1:-1], np.repeat(v, 2)


class Dashboard:
    """Shared layout, header, overview, camera thumbnail, chase view, fixes panel and blitting.
    Subclasses fill in what depends on having ground truth."""
    RAMP_MAX, RAMP_TICKS, RAMP_LABEL = 0.5, ['0', '0.25', '≥ 0.5'], '|lateral error| [m]'
    METRIC_ROWS = []
    METRICS_TITLE = ('Running metrics', '')

    def __init__(self, d, fig, title, subtitle):
        self.d, self.fig = d, fig
        self.n = len(d['t'])
        self.dt_end = float(d['dt_end']) if 'dt_end' in d else float(np.round(d['t'][-1] + 0.1))
        self.dynamic, self.bg, self.limits = [], None, {}
        fig.set_facecolor(PAGE)
        fig.text(0.01, 0.968, title, fontsize=15, color=INK, va='center')
        if subtitle:                                   # mapped mode passes none (only the title on the left)
            fig.text(0.01, 0.937, subtitle, fontsize=10, color=INK2, va='center')
        self.clock = self._dyn(fig.text(0.99, 0.968, '', fontsize=13, color=INK, ha='right', va='center'))
        self.clock2 = self._dyn(fig.text(0.99, 0.937, '', fontsize=10, color=INK2, ha='right', va='center'))
        self.panels = {'track': fig.add_axes([0.01, 0.04, 0.31, 0.86]), 'camera': fig.add_axes([0.335, 0.73, 0.205, 0.176]),
                       'chase': fig.add_axes([0.35, 0.43, 0.175, 0.27]), 'middle': fig.add_axes([0.35, 0.075, 0.175, 0.28]),
                       'metrics': fig.add_axes([0.575, 0.68, 0.41, 0.23])}
        self._overview(self.panels['track'])
        self._thumb(self.panels['camera'])
        self._chase(self.panels['chase'])
        self._middle(self.panels['middle'])
        self._metrics(self.panels['metrics'])
        self.ax_s1 = fig.add_axes([0.575, 0.45, 0.41, 0.175])
        self.ax_s2 = fig.add_axes([0.575, 0.255, 0.41, 0.135], sharex=self.ax_s1)
        self.ax_fix = fig.add_axes([0.575, 0.06, 0.41, 0.13], sharex=self.ax_s1)
        for ax in (self.ax_s1, self.ax_s2, self.ax_fix):
            _style(ax)
        self.ax_s1.set_xlim(0, self.dt_end)
        for ax in (self.ax_s1, self.ax_s2):
            ax.tick_params(labelbottom=False)
        self.ax_fix.set_xlabel('time in chunk [s]')
        self._series1(self.ax_s1)
        self._series2(self.ax_s2)
        self._fixes(self.ax_fix)
        self.panels.update(series1=self.ax_s1, series2=self.ax_s2, fixes=self.ax_fix)
        canvas = fig.canvas
        self.blit = bool(getattr(canvas, 'supports_blit', False))
        if self.blit:
            for a in self.dynamic:
                a.set_animated(True)
            canvas.mpl_connect('draw_event', lambda ev: setattr(self, 'bg', canvas.copy_from_bbox(fig.bbox)))

    def _dyn(self, artist):
        self.dynamic.append(artist)
        return artist

    # ---------------------------------------------------------------- shared panels
    def _overview(self, ax):
        d = self.d
        _style(ax, grid=False)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        allp = np.vstack([p[:, :2] for p in d['bounds'].values()])
        for name, p in d['bounds'].items():
            ax.plot(p[:, 0], p[:, 1], color=GRID if 'pit' in name else AXIS, lw=0.9, zorder=1)
        ax.plot(*d['ref_xy'], color=MUTED, lw=0.9, alpha=0.6, zorder=2)
        ax.scatter([GATE[0]], [GATE[1]], marker='s', s=14, color=INK2, zorder=3)
        ax.annotate('start / finish', GATE, xytext=(8, 0), textcoords='offset points', color=INK2,
                    fontsize=8.5, va='center')
        self.ov_est = self._dyn(LineCollection([], cmap=RAMP, norm=Normalize(0, self.RAMP_MAX), lw=3.5, zorder=4, capstyle='round'))
        ax.add_collection(self.ov_est)
        self.ov_dot = self._dyn(ax.scatter([], [], s=55, color=EST, edgecolor=SURFACE, linewidth=2, zorder=6))
        m = 25.0
        ax.set_xlim(allp[:, 0].min() - m, allp[:, 0].max() + m)
        ax.set_ylim(allp[:, 1].min() - m, allp[:, 1].max() + m)
        cax = ax.inset_axes([0.53, 0.045, 0.42, 0.016])        # bottom right: empty on this track
        cb = self.fig.colorbar(ScalarMappable(Normalize(0, self.RAMP_MAX), RAMP), cax=cax, orientation='horizontal')
        cb.set_ticks([0, self.RAMP_MAX / 2, self.RAMP_MAX])
        cb.set_ticklabels(self.RAMP_TICKS)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, labelsize=8)
        cb.set_label(self.RAMP_LABEL, fontsize=8.5, color=INK2)
        self._overview_extra(ax)

    def _thumb(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(AXIS)
        th = self.d.get('thumbs')
        first = next((im for im in th if im is not None), None) if th is not None else None
        self.th = ax.imshow(first if first is not None else np.full((182, 377, 3), 235, np.uint8), aspect='auto')
        _title(ax, 'Front camera fr', 'as recorded')
        if th is None:
            ax.set_visible(False)
        else:
            self._dyn(self.th)

    def _chase(self, ax):
        _style(ax)
        ax.set_aspect('equal')
        ax.set_xlim(*CHASE_X)
        ax.set_ylim(*CHASE_Y)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_yticks([-10, 0, 10, 20, 30])
        self.ch_bounds = [self._dyn(ax.plot([], [], color=GRID if 'pit' in name else AXIS, lw=1.0)[0])
                          for name in self.d['bounds']]
        self.ch_ref = self._dyn(ax.plot([], [], color=MUTED, lw=1.0, alpha=0.6)[0])
        self.ch_est = self._dyn(ax.plot([], [], color=EST, lw=3.5, solid_capstyle='round')[0])
        self._chase_extra(ax)
        _title(ax, 'Chase view', 'heading up, m')

    def _metrics(self, ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=SURFACE, ec=GRID, lw=0.8))
        _title(ax, *self.METRICS_TITLE)
        self.mtext = []
        y = 0.91
        for label, note in self.METRIC_ROWS:
            if label is None:
                ax.plot([0.03, 0.97], [y + 0.04, y + 0.04], color=GRID, lw=0.8)
                y -= 0.035
                continue
            ax.text(0.03, y, label, color=INK2, fontsize=10, va='center')
            ax.text(0.42, y, note, color=MUTED, fontsize=8.5, va='center')
            self.mtext.append(self._dyn(ax.text(0.97, y, '', color=INK, fontsize=11, va='center', ha='right')))
            y -= 0.092

    def _line_panel(self, ax, band=True):
        """A growing time series: one line, its end dot and value, optionally a shaded band."""
        p = {'ax': ax}
        if band:
            p['band'] = self._dyn(ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=to_rgba(EST, 0.10), ec='none')))
        p['line'] = self._dyn(ax.plot([], [], color=EST, lw=2.0, solid_capstyle='round', solid_joinstyle='round')[0])
        p['dot'] = self._dyn(ax.scatter([], [], s=40, color=EST, edgecolor=SURFACE, linewidth=2, zorder=5))
        p['lbl'] = self._dyn(ax.text(0, 0, '', color=INK, fontsize=9, va='center'))
        return p

    def _fixes(self, ax):
        self.fix_wash = self._dyn(ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=to_rgba(ACC, 0.12), ec='none')))
        self.fix_acc = self._dyn(ax.plot([], [], color=ACC, lw=2.0)[0])
        self.fix_rej = self._dyn(ax.plot([], [], color=REJ, lw=2.0)[0])
        ax.set_ylim(0, 5.6)
        ax.set_yticks(range(6))
        _title(ax, 'Camera fixes per frame', 'of 5 cameras',
               ([Line2D([], [], color=ACC, lw=2), Line2D([], [], color=REJ, lw=2)],
                ['accepted by the filter', 'rejected at the gate']))

    # ---------------------------------------------------------------- per frame
    def _set_limit(self, key, setter, value):
        """Change an axis range; returns True if it changed (the background must then be redrawn)."""
        if self.limits.get(key) == value:
            return False
        self.limits[key] = value
        setter(value)
        return True

    def _end_label(self, txt, tk, v, text):
        right = tk < 0.85 * self.dt_end
        txt.set_position((tk + 0.35, v) if right else (tk - 0.35, v))
        txt.set_ha('left' if right else 'right')
        txt.set_text(text)

    def _update_line(self, key, p, t, v, lo, hi, text, steps, symmetric=True, window=None, fixed=None):
        """window: frames the y range is fitted to (None = all), so that detail after a large early value
        stays visible; older values may then run off the panel."""
        p['line'].set_data(t, v)
        if 'band' in p:
            p['band'].set_xy(np.c_[np.r_[t, t[::-1]], np.r_[hi, lo[::-1]]])
        p['dot'].set_offsets([[t[-1], v[-1]]])
        self._end_label(p['lbl'], t[-1], v[-1], text)
        if fixed is not None:                             # a fixed range: +-fixed (symmetric) or 0..fixed
            return self._set_limit(key, lambda x: p['ax'].set_ylim(-x if symmetric else 0, x), fixed)
        w = slice(-window, None) if window else slice(None)
        if symmetric:
            lim = _step_limit(1.1 * max(np.abs(v[w]).max(), np.abs(hi[w]).max(), np.abs(lo[w]).max()), steps)
            return self._set_limit(key, lambda x: p['ax'].set_ylim(-x, x), lim)
        lim = _step_limit(1.1 * max(v[w].max(), hi[w].max()), steps)
        return self._set_limit(key, lambda x: p['ax'].set_ylim(0, x), lim)

    def update(self, k, lag=0, final=False, note=''):
        """Set every moving artist to frame k. Returns True if an axis range changed."""
        d, s = self.d, slice(0, k + 1)
        t, tk = d['t'][s], d['t'][k]
        self.final = final
        if not note and 'late_ms' in d and d['late_ms'][k] >= 50:          # recorded live run: was it real time?
            note = f'processing {d["late_ms"][k] / 1e3:.1f} s behind the 10 Hz camera clock (no pose dropped)'
        state = 'final' if final else (f'display {lag} frames behind' if lag > 0 else note)
        self.clock.set_text(f't = {tk:4.1f} s of {self.dt_end:.0f} s')
        self.clock2.set_text(f'{d["speed_kmh"][k]:3.0f} km/h  ·  frame {k + 1} / {self.n}' + (f'  ·  {state}' if state else ''))
        pts = np.c_[d['x'][s], d['y'][s]]
        self.ov_est.set_segments(np.stack([pts[:-1], pts[1:]], axis=1) if k > 0 else [])
        self.ov_est.set_array(self._trail_values(s)[1:])
        self.ov_dot.set_offsets(pts[-1:])
        if d.get('thumbs') is not None and d['thumbs'][k] is not None:
            self.th.set_data(d['thumbs'][k])
        cx, cy, cyaw = self._center(k)
        for line, p in zip(self.ch_bounds, d['bounds'].values()):
            line.set_data(*_in_view(*_to_screen(p[:, 0], p[:, 1], cx, cy, cyaw)))
        self.ch_ref.set_data(*_in_view(*_to_screen(np.asarray(d['ref_xy'][0]), np.asarray(d['ref_xy'][1]), cx, cy, cyaw)))
        self.ch_est.set_data(*_in_view(*_to_screen(d['x'][s], d['y'][s], cx, cy, cyaw)))   # whole path so far
        full = self._update_extra(k, s, t, cx, cy, cyaw)
        xa, ya = _steps(t, d['n_acc'][s])
        self.fix_acc.set_data(xa, ya)
        self.fix_wash.set_xy(np.c_[np.r_[xa, xa[::-1]], np.r_[ya, np.zeros_like(ya)]])
        self.fix_rej.set_data(*_steps(t, d['n_fix'][s] - d['n_acc'][s]))
        for txt, v in zip(self.mtext, self._metric_values(k)):
            txt.set_text(v)
        return full

    def save_images(self, out, k=None):
        """The final frame as final_frame.png, and each panel of it as plots/<panel>.png."""
        import os
        k = self.n - 1 if k is None else k
        self.update(k, final=True)
        for a in self.dynamic:                       # blitted artists are skipped by a normal draw
            a.set_animated(False)
        names = {'middle': self.MIDDLE_NAME, 'series1': self.SERIES_NAMES[0], 'series2': self.SERIES_NAMES[1]}
        try:
            self.fig.savefig(os.path.join(out, 'final_frame.png'), facecolor=self.fig.get_facecolor())
            os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
            r = self.fig.canvas.get_renderer()
            for key, ax in self.panels.items():
                if not ax.get_visible():
                    continue
                box = ax.get_tightbbox(r).transformed(self.fig.dpi_scale_trans.inverted()).padded(0.08)
                self.fig.savefig(os.path.join(out, 'plots', f'{names.get(key, key)}.png'), bbox_inches=box,
                                 facecolor=self.fig.get_facecolor())
        finally:
            for a in self.dynamic:
                a.set_animated(self.blit)
            self.bg = None

    def render(self, k, lag=0, final=False, show=False, note=''):
        """Draw frame k: background + moving artists. show=True pushes it to a window (blit)."""
        canvas = self.fig.canvas
        full = self.update(k, lag, final, note)
        if not self.blit:
            canvas.draw()
            return
        if full or self.bg is None:
            canvas.draw()                    # the draw_event stores the new background (moving artists excluded)
        else:
            canvas.restore_region(self.bg)
        for a in self.dynamic:
            (a.axes or self.fig).draw_artist(a)
        if show:
            canvas.blit(self.fig.bbox)


class TruthDashboard(Dashboard):
    """Mapped mode (night training bag, with ground truth): errors across / along the true heading and the
    official metrics."""
    MIDDLE_NAME, SERIES_NAMES = 'position_error', ('lateral_error', 'heading_error')

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # layout: right column lateral, position, heading error; camera fixes under the chase view
        mid, s2, fix = self.pm['ax'], self.ax_s2, self.ax_fix
        rect_mid, rect_s2, rect_fix = mid.get_position(), s2.get_position(), fix.get_position()
        mid.set_position(rect_s2)
        mid.set_xlabel('')
        mid.tick_params(labelbottom=False)
        s2.set_position(rect_fix)
        s2.tick_params(labelbottom=True)
        s2.set_xlabel('time in chunk [s]')
        fix.set_position(rect_mid)
        fix.get_legend().remove()                        # narrow slot: legend on its own line under the title
        for txt in fix.texts:
            txt.remove()
        fix.set_title('Camera fixes per frame (of 5)', loc='left', color=INK, fontsize=10.5, pad=17)
        fix.legend([Line2D([], [], color=ACC, lw=2), Line2D([], [], color=REJ, lw=2)], ['accepted', 'rejected at the gate'],
                   loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3, handlelength=1.6, columnspacing=1.0)
    METRIC_ROWS = [('Position RMSE', '25 %'), ('Lateral RMSE', '20 %'), ('Heading RMSE', '10 %'),
                   ('Within 1 m', '10 %'), ('Runtime per frame', '5 %'), ('Coverage', 'ranked ≥ 90 %'),
                   (None, None), ('Longitudinal RMSE', ''), ('Median · p95 · max', ''), ('Frames with an accepted fix', '')]
    METRICS_TITLE = ('Running metrics', 'official scoring math, frames so far')

    def _overview_extra(self, ax):
        self.ov_gt = self._dyn(ax.plot([], [], color=INK, lw=0.9, zorder=5, solid_capstyle='round')[0])
        ax.legend([Line2D([], [], color=INK, lw=0.9), Line2D([], [], color=EST, lw=3.5)],
                  ['ground truth', 'estimate (darker = larger lateral error)'], loc='upper right')
        _title(ax, 'Track, map frame', 'grey: whole chunk path')

    def _chase_extra(self, ax):
        self.ch_gt = self._dyn(ax.plot([], [], color=INK, lw=1.0)[0])      # ground truth, whole chunk, over the estimate
        ax.add_patch(Polygon(_footprint(), closed=True, fc='none', ec=INK, lw=1.3, zorder=4))
        self._dyn(ax.legend([Line2D([], [], color=INK, lw=1), Line2D([], [], color=EST, lw=3)],
                  ['ground truth', 'predicted path'], loc='upper right', fontsize=7.5,
                  frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.9))   # drawn after the moving lines (blitting), so they never cross it
        self.ch_car = self._dyn(ax.add_patch(Polygon(_footprint(), closed=True, fc=to_rgba(EST, 0.12), ec=EST, lw=1.3, zorder=5)))

    def _middle(self, ax):
        _style(ax)
        ax.set_xlim(0, self.dt_end)
        self.pm = self._line_panel(ax, band=False)
        ax.set_xlabel('time in chunk [s]')
        _title(ax, 'Position error [m]', 'distance from the truth')

    def _series1(self, ax):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        self.p1 = self._line_panel(ax)
        _title(ax, 'Lateral error [m]', 'across the true heading, + = estimate left of truth',
               ([Line2D([], [], color=EST, lw=2), Rectangle((0, 0), 1, 1, fc=to_rgba(EST, 0.10))],
                ['lateral error', 'filter ±1σ (per axis)']))

    def _series2(self, ax):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        self.p2 = self._line_panel(ax)
        _title(ax, 'Heading error [deg]', 'shaded: filter ±1σ')

    def _trail_values(self, s):
        return np.abs(self.d['lat'][s])

    def _center(self, k):
        return self.d['gt_x'][k], self.d['gt_y'][k], self.d['gt_yaw'][k]

    def _update_extra(self, k, s, t, cx, cy, cyaw):
        d = self.d
        self.ov_gt.set_data(d['gt_x'][s], d['gt_y'][s])
        self.ch_ref.set_data([], [])                                           # replaced by ch_gt, drawn on top
        self.ch_gt.set_data(*_in_view(*_to_screen(np.asarray(d['ref_xy'][0]), np.asarray(d['ref_xy'][1]), cx, cy, cyaw)))
        self.ch_car.set_xy(_footprint(d['lat'][k], d['lon'][k], np.radians(d['dyaw_deg'][k])))
        pos = d['pos'][s]
        full = self._update_line('mid', self.pm, t, pos, pos, pos, f'{100 * pos[-1]:.1f} cm',
                                 (0.3, 0.6, 1.2, 2.5, 5.0, 10.0), symmetric=False)
        slat = d['sigma_xy'][s] / np.sqrt(2)
        full |= self._update_line('s1', self.p1, t, d['lat'][s], -slat, slat, f'{100 * d["lat"][k]:+.1f} cm',
                                  (0.3, 0.6, 1.2, 2.5, 5.0, 10.0, 25.0))
        syaw = np.degrees(d['sigma_yaw'][s])
        full |= self._update_line('s2', self.p2, t, d['dyaw_deg'][s], -syaw, syaw, f'{d["dyaw_deg"][k]:+.2f}°',
                                  (0.3, 0.6, 1.2, 2.5, 5.0, 10.0))
        return full

    def _metric_values(self, k):
        m = self.d['running'][k]
        return [f'{m["pos_rmse_m"]:.3f} m', f'{m["lat_rmse_m"]:.3f} m', f'{m["yaw_rmse_deg"]:.3f}°',
                f'{m["within_1.0_pct"]:.1f} %', f'{m["ms_mean"]:.0f} ms mean · {m["ms_p95"]:.0f} ms p95',
                f'{m["coverage_pct"]:.0f} %', f'{m["lon_rmse_m"]:.3f} m',
                f'{m["pos_median_m"]:.3f} · {m["pos_p95_m"]:.3f} · {m["pos_max_m"]:.3f} m', f'{m["fix_rate_pct"]:.0f} %']


class HealthDashboard(Dashboard):
    MIDDLE_NAME, SERIES_NAMES = 'camera_matching', ('racing_line_offset', 'filter_sigma')
    """Test mode (daytime sample, no ground truth): where the estimate is against the organizers' racing line (context, not an
    error), how sure the filter is, what each camera matched, and the metrics that need no truth."""
    RAMP_MAX, RAMP_TICKS, RAMP_LABEL = 3.0, ['0', '1.5', '≥ 3'], 'filter position σ [m]'
    METRIC_ROWS = [('Frames with an accepted fix', ''), ('Camera fixes', 'accepted · produced · tried'),
                   ('Matches after the pose filter', 'median, fix needs ≥ 20'), ('Filter position σ', 'now · max'),
                   ('Track constraint active', 'share of frames'), ('Runtime per frame', '5 %'),
                   ('Coverage', 'ranked ≥ 90 %'), (None, None), ('Offset from the racing line', 'RMS, context only'),
                   ('Ground truth', ''), ('Start pose', '')]
    METRICS_TITLE = ('Health metrics', 'no ground truth in this bag, frames so far')
    CAM_COLOR = {0: MUTED, 1: REJ, 2: ACC}

    def _overview_extra(self, ax):
        ax.legend([Line2D([], [], color=MUTED, lw=0.9), Line2D([], [], color=EST, lw=3.5)],
                  ["racing line (organizers' reference, not this lap's truth)", 'estimate (darker = less certain)'],
                  loc='upper right')
        _title(ax, 'Track, map frame', 'no ground truth')

    def _chase_extra(self, ax):
        ax.add_patch(Polygon(_footprint(), closed=True, fc=to_rgba(EST, 0.12), ec=EST, lw=1.3, zorder=5))
        self._dyn(ax.legend([Line2D([], [], color=AXIS, lw=1), Line2D([], [], color=MUTED, lw=1), Line2D([], [], color=EST, lw=3)],
                  ['borders', 'racing line', 'estimate, last 5 s'], loc='lower right', fontsize=7.5,
                  frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.9))   # drawn after the moving lines (blitting), so they never cross it

    def _middle(self, ax):
        _style(ax, grid=False)
        ax.grid(True, axis='x', color=GRID, lw=0.6)
        cams = self.d['cams']
        y = np.arange(len(cams))[::-1].astype(float)
        ax.set_yticks(y, cams)
        ax.set_ylim(-0.7, len(cams) - 0.3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 50, 100])
        ax.axvline(20, color=INK2, lw=1.0, zorder=3)
        self.cam_bars = [self._dyn(ax.add_patch(Rectangle((0, yi - 0.28), 0, 0.56, fc=MUTED, ec='none', zorder=2))) for yi in y]
        self.cam_lbls = [self._dyn(ax.text(2, yi, '', color=INK2, fontsize=8.5, va='center', zorder=4)) for yi in y]
        ax.set_xlabel('matches after the pose filter (PnP needs ≥ 20)')
        _title(ax, 'Camera matching, this frame', 'bars: grey no fix · aqua accepted · orange rejected', below=True)

    def _series1(self, ax):
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        self.p1 = self._line_panel(ax)
        _title(ax, 'Offset from the racing line [m]', "context, not an error: n of the organizers' track frame, + = left of the line",
               ([Line2D([], [], color=EST, lw=2), Rectangle((0, 0), 1, 1, fc=to_rgba(EST, 0.10))],
                ['estimate', 'filter ±1σ (per axis)']))

    def _series2(self, ax):
        self.p2 = self._line_panel(ax, band=False)
        _title(ax, 'Filter position σ [m]', 'grows while no camera fix is accepted')

    def _trail_values(self, s):
        return self.d['sigma_xy'][s]

    def _center(self, k):
        return self.d['x'][k], self.d['y'][k], self.d['yaw'][k]

    def _update_extra(self, k, s, t, cx, cy, cyaw):
        d = self.d
        for bar, lbl, m, st, txt in zip(self.cam_bars, self.cam_lbls, d['cam_matches'][k], d['cam_state'][k], d['cam_text'][k]):
            bar.set_width(min(m, 100))
            bar.set_facecolor(self.CAM_COLOR[int(st)])
            lbl.set_position((min(m, 100) + 2 if m < 62 else 2, lbl.get_position()[1]))
            lbl.set_text(txt)
        slat = d['sigma_xy'][s] / np.sqrt(2)
        off = d['ref_off'][s]
        window = None if self.final else 60                   # y range: the last 6 s (whole chunk at the end)
        full = self._update_line('s1', self.p1, t, off, off - slat, off + slat, f'{off[-1]:+.2f} m',
                                 (5.0,), fixed=OFFSET_LIM_M)
        sig = d['sigma_xy'][s]
        full |= self._update_line('s2', self.p2, t, sig, sig, sig, f'{sig[-1]:.2f} m', (0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                                  symmetric=False, window=window)
        return full

    def _metric_values(self, k):
        m = self.d['health'][k]
        return [f'{m["fix_frames_pct"]:.0f} %', f'{m["accepted"]} · {m["produced"]} · {m["tried"]}',
                f'{m["median_matches"]:.0f}', f'{m["sigma_now"]:.2f} m · {m["sigma_max"]:.2f} m',
                f'{m["pushed_pct"]:.0f} %', f'{m["ms_mean"]:.0f} ms mean · {m["ms_p95"]:.0f} ms p95',
                f'{m["coverage_pct"]:.0f} %', f'{m["off_rms"]:.2f} m', 'none in this bag', self.d['start_note']]
