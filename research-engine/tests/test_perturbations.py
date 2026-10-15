"""Tests for J2 and drag perturbations and the force-model toggles."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import orbit, perturbations, validation
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestPerturbations(unittest.TestCase):
    def setUp(self):
        self.radius = DEFAULT_RADIUS
        self.period = orbit.orbital_period(self.radius, MU_EARTH)
        self.state0 = orbit.circular_orbit_state(self.radius, MU_EARTH, np.radians(51.6))
        self.t = np.linspace(0.0, 2.0 * self.period, 4001)

    def test_j2_zero_az_in_equatorial_plane(self):
        # On the equatorial plane (z = 0) the J2 out-of-plane component vanishes.
        acc = perturbations.j2_acceleration(np.array([self.radius, 0.0, 0.0]))
        self.assertAlmostEqual(acc[2], 0.0, places=15)
        self.assertLess(acc[0], 0.0)  # points inward on +x

    def test_toggle_labels(self):
        self.assertEqual(perturbations.ForceModel.two_body().label, "two-body")
        self.assertEqual(perturbations.ForceModel.j2_only().label, "two-body + J2")
        self.assertEqual(perturbations.ForceModel.drag_only().label, "two-body + drag")
        self.assertEqual(perturbations.ForceModel.j2_drag().label, "two-body + J2 + drag")

    def test_two_body_force_model_matches_orbit(self):
        acc = perturbations.ForceModel.two_body().acceleration(self.state0)
        self.assertTrue(np.allclose(acc, orbit.two_body_acceleration(self.state0[:3])))

    def test_j2_conserves_hz_and_bounds_energy(self):
        states = orbit.propagate(perturbations.ForceModel.j2_only().derivatives, self.state0, self.t)
        hz = np.array([np.cross(s[:3], s[3:])[2] for s in states])
        energy = validation.energy_series(states, MU_EARTH)
        self.assertLess(abs((hz[-1] - hz[0]) / hz[0]), 1e-8)        # axisymmetric -> h_z conserved
        # Two-body energy oscillates at the J2-potential scale (~order J2) but is
        # bounded (no secular growth), unlike drag which decays it monotonically.
        self.assertLess(np.max(np.abs(validation.relative_drift(energy))), 5e-3)

    def test_drag_decays_energy(self):
        fm = perturbations.ForceModel.drag_only(area=1.0, mass=100.0)
        states = orbit.propagate(fm.derivatives, self.state0, self.t)
        energy = validation.energy_series(states, MU_EARTH)
        # Drag removes energy: specific energy becomes more negative.
        self.assertLess(energy[-1], energy[0])

    def test_drag_zero_without_atmosphere(self):
        # Far above the atmosphere the exponential model gives ~0 density.
        acc = perturbations.drag_acceleration(
            np.array([self.radius + 2.0e6, 0.0, 0.0]), np.array([0.0, 7000.0, 0.0])
        )
        self.assertTrue(np.allclose(acc, 0.0, atol=1e-15))


if __name__ == "__main__":
    unittest.main()
