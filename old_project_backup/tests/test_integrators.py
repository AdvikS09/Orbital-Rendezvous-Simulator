"""Tests for the numerical integrators (accuracy and symplectic behaviour)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import analysis, integrators, physics, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


class TestIntegrators(unittest.TestCase):
    def setUp(self):
        self.period = physics.orbital_period(DEFAULT_RADIUS, MU_EARTH)
        self.state0 = physics.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH)
        self.dt = self.period / 2000

    def _energy_drift(self, method):
        res = simulation.propagate_two_body(
            self.state0, (0.0, 2 * self.period), self.dt, MU_EARTH, method
        )
        return analysis.max_abs_relative_drift(analysis.energy_series(res.states, MU_EARTH))

    def test_rk4_far_more_accurate_than_euler(self):
        self.assertLess(self._energy_drift("rk4"), 1e-6)
        self.assertGreater(self._energy_drift("euler"), self._energy_drift("rk4"))

    def test_verlet_energy_drift_is_bounded(self):
        # Symplectic integrator: energy error stays small (no large secular growth).
        self.assertLess(self._energy_drift("verlet"), 1e-3)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            integrators.integrate(
                lambda s, t: s, self.state0, 0.0, 1.0, 0.1, method="bogus"
            )

    def test_rk4_step_matches_analytic_linear_ode(self):
        # Scalar exponential decay y' = -y over one step; RK4's single-step
        # truncation error is ~dt^5/120, so a dt=0.1 step matches e^-dt to ~1e-7.
        deriv = lambda s, t: -s
        y1 = integrators.rk4_step(deriv, np.array([1.0]), 0.0, 0.1)
        self.assertAlmostEqual(y1[0], np.exp(-0.1), places=6)


if __name__ == "__main__":
    unittest.main()
