"""Tests for the 3D Clohessy-Wiltshire dynamics and state-transition matrix."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import cw
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestCW(unittest.TestCase):
    def setUp(self):
        self.n = cw.cw_mean_motion(DEFAULT_RADIUS, MU_EARTH)
        self.period = 2.0 * np.pi / self.n

    def test_system_matrix_shapes(self):
        A, B = cw.cw_system_matrices(self.n)
        self.assertEqual(A.shape, (6, 6))
        self.assertEqual(B.shape, (6, 3))

    def test_stm_at_zero_is_identity(self):
        self.assertTrue(np.allclose(cw.cw_state_transition(self.n, 0.0), np.eye(6)))

    def test_numeric_matches_stm(self):
        x0 = np.array([120.0, -80.0, 40.0, 0.3, 0.1, 0.2])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propagate_cw(x0, t, self.n)
        analytic = cw.propagate_cw_analytic(x0, t, self.n)
        self.assertLess(np.max(np.abs(states - analytic)), 1e-6)

    def test_alongtrack_offset_is_equilibrium(self):
        x0 = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 2001)
        states, _ = cw.propagate_cw(x0, t, self.n)
        self.assertTrue(np.allclose(states[-1], x0, atol=1e-6))

    def test_cross_track_is_harmonic_at_orbital_period(self):
        # z'' = -n^2 z is an oscillator of period 2*pi/n; after one period the
        # cross-track state returns to its start and stays decoupled from x, y.
        x0 = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propagate_cw(x0, t, self.n)
        self.assertAlmostEqual(states[-1, 2], 50.0, places=2)
        self.assertTrue(np.allclose(states[:, 0], 0.0, atol=1e-6))  # no in-plane coupling
        self.assertTrue(np.allclose(states[:, 1], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
