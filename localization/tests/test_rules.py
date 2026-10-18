"""Checks for the rules that disqualify a submission (D-001, D-005).

    python -m unittest discover -s tests -v
"""
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camloc import allowed  # noqa: E402
from camloc.ekf import PoseEKF  # noqa: E402
from camloc.odometry import OdometryStream  # noqa: E402

NOISE = {'v_long': 0.05, 'v_lat': 0.05, 'v_scale': 0.002, 'wz': 0.0015, 'bias_rw': 1e-5}


def synthetic_state(n=500, rate_hz=105.0):
    t = (np.arange(n) * 1e9 / rate_hz).astype(np.int64) + 10**18
    return {'stamp_ns': t, 'vx_mps': np.full(n, 20.0), 'vy_mps': np.zeros(n), 'v_mps': np.full(n, 20.0),
            'wz_radps': np.full(n, 0.1), 'ax_mps2': np.zeros(n), 'ay_mps2': np.zeros(n)}


class TestAllowedFields(unittest.TestCase):
    def test_disjoint(self):
        self.assertFalse(set(allowed.ALLOWED_STATE_FIELDS) & set(allowed.FORBIDDEN_STATE_FIELDS))

    def test_known_leaks_forbidden(self):
        # yaw_vel_rad equals yaw + beta (absolute course): it must never be read
        for f in ('yaw_vel_rad', 'yaw_rad', 'x_m', 'y_m', 'sn_map_state'):
            self.assertIn(f, allowed.FORBIDDEN_STATE_FIELDS)

    def test_read_state_uses_whitelist_only(self):
        import inspect
        from camloc import bag
        src = inspect.getsource(bag.read_state)
        self.assertIn('allowed.ALLOWED_STATE_FIELDS', src)
        for f in allowed.FORBIDDEN_STATE_FIELDS:
            self.assertNotIn(f"'{f}'", src)


class TestCausality(unittest.TestCase):
    def test_odometry_never_reads_future(self):
        st = synthetic_state()
        odo = OdometryStream(st)
        t0 = int(st['stamp_ns'][10])
        ekf = PoseEKF(t0, 0, 0, 0, 0, (0.1, 0.01, 1e-4), NOISE)
        odo.seek(t0)
        for t in np.linspace(t0, st['stamp_ns'][-1] + 10**7, 37).astype(np.int64):
            odo.advance(ekf, int(t))
            self.assertLessEqual(odo.max_read, t)
            self.assertEqual(ekf.t, t)

    def test_time_cannot_run_backwards(self):
        st = synthetic_state()
        odo = OdometryStream(st)
        t0 = int(st['stamp_ns'][10])
        ekf = PoseEKF(t0, 0, 0, 0, 0, (0.1, 0.01, 1e-4), NOISE)
        odo.seek(t0)
        odo.advance(ekf, t0 + 10**8)
        with self.assertRaises(AssertionError):
            odo.advance(ekf, t0)

    def test_constant_turn_integration(self):
        # 20 m/s, 0.1 rad/s for ~4.7 s: yaw = 0.1 * T exactly, position on the circle of radius 200 m
        st = synthetic_state()
        odo = OdometryStream(st)
        t0 = int(st['stamp_ns'][0])
        ekf = PoseEKF(t0, 0, 0, 0, 0, (0.1, 0.01, 1e-4), NOISE)
        odo.seek(t0)
        t1 = int(st['stamp_ns'][-1])
        odo.advance(ekf, t1)
        T = (t1 - t0) / 1e9
        self.assertAlmostEqual(ekf.x[2], 0.1 * T, places=6)
        self.assertAlmostEqual(ekf.x[0], 200 * np.sin(0.1 * T), places=3)
        self.assertAlmostEqual(ekf.x[1], 200 * (1 - np.cos(0.1 * T)), places=3)


class TestInferencePathHasNoGroundTruth(unittest.TestCase):
    def test_pipeline_modules_do_not_load_gt(self):
        # import everything the inference entry point imports, in a fresh interpreter
        code = textwrap.dedent('''
            import sys
            sys.path.insert(0, %r)
            import camloc.pipeline, camloc.reloc.query, camloc.bag, camloc.trackmap
            import importlib.util
            spec = importlib.util.spec_from_file_location('run_mod', %r)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            print('camloc.gt' in sys.modules)
        ''') % (ROOT, os.path.join(ROOT, 'scripts', 'run.py'))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
