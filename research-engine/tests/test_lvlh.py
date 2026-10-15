"""Tests for the LVLH frame transforms."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import lvlh, orbit, perturbations
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestLVLH(unittest.TestCase):
    def setUp(self):
        self.target = orbit.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH, np.radians(51.6))

    def test_rotation_matrix_orthonormal(self):
        rot = lvlh.lvlh_rotation_matrix(self.target)
        self.assertTrue(np.allclose(rot @ rot.T, np.eye(3), atol=1e-12))
        self.assertAlmostEqual(np.linalg.det(rot), 1.0, places=10)

    def test_roundtrip(self):
        rel = np.array([100.0, -200.0, 50.0, 0.1, 0.2, -0.05])
        deputy = lvlh.lvlh_to_inertial(self.target, rel)
        rel_back = lvlh.inertial_to_lvlh(self.target, deputy)
        self.assertTrue(np.allclose(rel, rel_back, atol=1e-9))

    def test_relative_helpers_consistent(self):
        deputy = lvlh.lvlh_to_inertial(self.target, np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]))
        full = lvlh.inertial_to_lvlh(self.target, deputy)
        self.assertTrue(np.allclose(lvlh.relative_position(self.target, deputy), full[:3]))
        self.assertTrue(np.allclose(lvlh.relative_velocity(self.target, deputy), full[3:]))

    def test_alongtrack_offset_stays_coorbital(self):
        # A pure along-track offset on the same circular orbit is an equilibrium:
        # in LVLH it should stay nearly constant over one orbit.
        period = orbit.orbital_period(DEFAULT_RADIUS, MU_EARTH)
        t = np.linspace(0.0, period, 2001)
        rel0 = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        deputy0 = lvlh.lvlh_to_inertial(self.target, rel0)
        deriv = perturbations.ForceModel.two_body().derivatives
        tgt = orbit.propagate(deriv, self.target, t)
        dep = orbit.propagate(deriv, deputy0, t)
        rel = np.array([lvlh.inertial_to_lvlh(tgt[k], dep[k]) for k in range(t.size)])
        drift = np.max(np.linalg.norm(rel[:, :3] - rel0[:3], axis=1))
        self.assertLess(drift, 1.0)  # < 1 m over a full orbit


if __name__ == "__main__":
    unittest.main()
