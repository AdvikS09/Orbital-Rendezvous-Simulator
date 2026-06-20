"""Tests for two-body dynamics, conservation and orbital elements."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import analysis, orbital_elements, physics, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


class TestTwoBody(unittest.TestCase):
    def setUp(self):
        self.radius = DEFAULT_RADIUS
        self.period = physics.orbital_period(self.radius, MU_EARTH)
        self.state0 = physics.circular_orbit_state(self.radius, MU_EARTH)

    def test_circular_speed_matches_vis_viva(self):
        v = physics.circular_speed(self.radius, MU_EARTH)
        self.assertAlmostEqual(v, np.sqrt(MU_EARTH / self.radius), places=6)

    def test_400km_period_is_about_92_minutes(self):
        # A 400 km circular orbit has a period near 92.6 minutes.
        self.assertTrue(90 * 60 < self.period < 95 * 60)

    def test_rk4_conserves_energy_and_momentum(self):
        res = simulation.propagate_two_body(
            self.state0, (0.0, self.period), self.period / 2000, MU_EARTH, "rk4"
        )
        energy = analysis.energy_series(res.states, MU_EARTH)
        ang_mom = analysis.angular_momentum_series(res.states)
        self.assertLess(analysis.max_abs_relative_drift(energy), 1e-9)
        self.assertLess(analysis.max_abs_relative_drift(ang_mom), 1e-9)

    def test_measured_period_matches_analytic(self):
        res = simulation.propagate_two_body(
            self.state0, (0.0, 1.2 * self.period), self.period / 4000, MU_EARTH, "rk4"
        )
        measured = analysis.measure_period(res.t, res.states)
        self.assertAlmostEqual(measured / self.period, 1.0, places=4)


class TestOrbitalElements(unittest.TestCase):
    def test_circular_orbit_has_zero_eccentricity(self):
        state = physics.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH)
        elements = orbital_elements.state_to_elements(state, MU_EARTH)
        self.assertAlmostEqual(elements["a"], DEFAULT_RADIUS, places=0)
        self.assertLess(elements["e"], 1e-9)

    def test_state_to_elements_roundtrip(self):
        # An eccentric orbit should survive a state -> elements -> state round trip.
        state = orbital_elements.elements_to_state(
            a=7.0e6, e=0.1, nu=0.7, omega=0.3, mu=MU_EARTH
        )
        elements = orbital_elements.state_to_elements(state, MU_EARTH)
        state_back = orbital_elements.elements_to_state(
            elements["a"], elements["e"], elements["nu"], elements["omega"], MU_EARTH
        )
        self.assertTrue(np.allclose(state, state_back, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
