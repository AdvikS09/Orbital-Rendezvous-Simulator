"""Tests for the PD rendezvous controller."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import controller, cw
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestController(unittest.TestCase):
    def setUp(self):
        self.n = cw.cw_mean_motion(DEFAULT_RADIUS, MU_EARTH)
        self.period = 2.0 * np.pi / self.n

    def test_critically_damped_gains(self):
        kp, kd = controller.critically_damped_gains(0.05)
        self.assertAlmostEqual(kp, 0.05**2)
        self.assertAlmostEqual(kd, 2.0 * 0.05)

    def test_gain_matrix_coercion(self):
        c_scalar = controller.PDController(1.0, 2.0)
        c_vector = controller.PDController([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(c_scalar.Kp, np.eye(3)))
        self.assertTrue(np.allclose(c_vector.Kd, 2.0 * np.eye(3)))

    def test_pd_drives_state_to_target(self):
        pd = controller.PDController.critically_damped(6.0 * self.n)
        x0 = np.array([150.0, -200.0, 60.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propagate_cw(x0, t, self.n, pd)
        self.assertLess(np.linalg.norm(states[-1, :3]), 0.1)
        self.assertLess(np.linalg.norm(states[-1, 3:]), 1e-3)

    def test_saturation(self):
        pd = controller.PDController.critically_damped(6.0 * self.n, max_accel=0.01)
        u = pd.control(np.array([1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0]))
        self.assertLessEqual(np.linalg.norm(u), 0.01 + 1e-12)

    def test_cross_track_convergence(self):
        # Pure cross-track offset should also be driven to zero by the PD law.
        pd = controller.PDController.critically_damped(6.0 * self.n)
        x0 = np.array([0.0, 0.0, 120.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propagate_cw(x0, t, self.n, pd)
        self.assertLess(abs(states[-1, 2]), 0.1)


if __name__ == "__main__":
    unittest.main()
