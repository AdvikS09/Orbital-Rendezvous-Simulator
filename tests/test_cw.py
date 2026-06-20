"""Tests for Clohessy-Wiltshire dynamics, the STM and LVLH frame transforms."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import cw, physics, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


class TestClohessyWiltshire(unittest.TestCase):
    def setUp(self):
        self.n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
        self.period = 2.0 * np.pi / self.n

    def test_numeric_cw_matches_analytic_stm(self):
        x0 = np.array([120.0, -80.0, 0.4, 0.2])
        res = simulation.simulate_cw(
            x0, (0.0, self.period), self.period / 4000, self.n, method="rk4"
        )
        analytic = cw.propagate_cw_analytic(x0, res.t, self.n)
        max_err = np.max(np.abs(res.states - analytic))
        self.assertLess(max_err, 1e-6)

    def test_stm_at_zero_is_identity(self):
        phi = cw.cw_state_transition(self.n, 0.0)
        self.assertTrue(np.allclose(phi, np.eye(4)))

    def test_alongtrack_offset_is_equilibrium(self):
        # A pure along-track offset is a CW equilibrium: it should not move.
        x0 = np.array([0.0, 100.0, 0.0, 0.0])
        res = simulation.simulate_cw(
            x0, (0.0, self.period), self.period / 2000, self.n, method="rk4"
        )
        self.assertTrue(np.allclose(res.states[-1], x0, atol=1e-6))

    def test_lvlh_roundtrip(self):
        target = physics.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH)
        rel = np.array([300.0, -150.0, 0.2, 0.5])
        deputy = cw.lvlh_to_inertial(target, rel)
        rel_back = cw.inertial_to_lvlh(target, deputy)
        self.assertTrue(np.allclose(rel, rel_back, atol=1e-9))

    def test_cw_matches_nonlinear_for_small_separation(self):
        # For small separations over one orbit, linear CW should track the full
        # non-linear two-body relative motion to within a fraction of a metre.
        target = physics.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH)
        rel0 = np.array([20.0, 40.0, 0.0, 0.0])
        t_eval = np.linspace(0.0, self.period, 300)
        nl = simulation.propagate_nonlinear_relative(target, rel0, t_eval, MU_EARTH)
        lin = cw.propagate_cw_analytic(rel0, t_eval, self.n)
        max_err = np.max(np.linalg.norm(nl.states[:, :2] - lin[:, :2], axis=1))
        self.assertLess(max_err, 1.0)


if __name__ == "__main__":
    unittest.main()
