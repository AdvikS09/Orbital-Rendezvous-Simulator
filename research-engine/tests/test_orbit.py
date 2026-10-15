"""Tests for core orbital mechanics: two-body, RK4, elements, energy, period."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import orbit, perturbations, validation
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestOrbit(unittest.TestCase):
    def setUp(self):
        self.radius = DEFAULT_RADIUS
        self.period = orbit.orbital_period(self.radius, MU_EARTH)
        self.state0 = orbit.circular_orbit_state(self.radius, MU_EARTH, np.radians(45.0))

    def test_circular_speed(self):
        v = orbit.circular_speed(self.radius, MU_EARTH)
        self.assertAlmostEqual(v, np.sqrt(MU_EARTH / self.radius), places=6)

    def test_400km_period_about_92_minutes(self):
        self.assertTrue(90 * 60 < self.period < 95 * 60)

    def test_rk4_conserves_energy_and_momentum(self):
        t = np.linspace(0.0, self.period, 2001)
        states = orbit.propagate(perturbations.ForceModel.two_body().derivatives, self.state0, t)
        energy = validation.energy_series(states, MU_EARTH)
        ang_mom = validation.angular_momentum_series(states)
        self.assertLess(np.max(np.abs(validation.relative_drift(energy))), 1e-9)
        self.assertLess(np.max(np.abs(validation.relative_drift(ang_mom))), 1e-9)

    def test_measure_period_matches_analytic(self):
        t = np.linspace(0.0, 1.3 * self.period, 4001)
        states = orbit.propagate(perturbations.ForceModel.two_body().derivatives, self.state0, t)
        measured = orbit.measure_period(t, states)
        self.assertAlmostEqual(measured / self.period, 1.0, places=4)

    def test_circular_orbit_zero_eccentricity(self):
        el = orbit.rv_to_elements(self.state0, MU_EARTH)
        self.assertLess(el["e"], 1e-9)
        self.assertAlmostEqual(el["a"], self.radius, places=0)
        self.assertAlmostEqual(el["i"], np.radians(45.0), places=6)

    def test_elements_roundtrip(self):
        state = orbit.elements_to_rv(7.0e6, 0.08, np.radians(35), np.radians(50),
                                     np.radians(70), np.radians(110), MU_EARTH)
        el = orbit.rv_to_elements(state, MU_EARTH)
        state_back = orbit.elements_to_rv(el["a"], el["e"], el["i"], el["raan"],
                                          el["argp"], el["nu"], MU_EARTH)
        self.assertTrue(np.allclose(state, state_back, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
