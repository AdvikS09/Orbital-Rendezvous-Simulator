"""Tests for the collision / keep-out-zone detection extension."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import collision
from src import scenarios as sc


class TestCollision(unittest.TestCase):
    def test_minimum_distance(self):
        positions = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 0.0]])
        dist, dmin, idx = collision.minimum_distance(positions, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(dmin, 0.0)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(dist[0], 5.0)
        self.assertAlmostEqual(dist[2], 10.0)

    def test_keep_out_violation_flagged(self):
        states = np.zeros((11, 6))
        states[:, 1] = np.linspace(100.0, -100.0, 11)  # along-track sweep through the origin
        a = collision.analyze(np.arange(11), states, keep_out_radius=10.0)
        self.assertTrue(a["keep_out_violation"])
        self.assertLess(a["min_dist_to_target_m"], 10.0)

    def test_keep_out_respected(self):
        states = np.zeros((10, 6))
        states[:, 0] = 50.0  # holds a 50 m radial standoff
        a = collision.analyze(np.arange(10), states, keep_out_radius=10.0)
        self.assertFalse(a["keep_out_violation"])
        self.assertAlmostEqual(a["min_dist_to_target_m"], 50.0)

    def test_debris_collision_flagged(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = np.linspace(0.0, 40.0, 5)  # passes through x = 20
        debris = np.zeros((5, 6))
        debris[:, 0] = 20.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=1.0,
                              debris_states=debris, collision_radius=5.0)
        self.assertTrue(a["collision_risk"])
        self.assertAlmostEqual(a["min_dist_to_debris_m"], 0.0)

    def test_debris_clear(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = np.linspace(0.0, 40.0, 5)
        debris = np.zeros((5, 6))
        debris[:, 0] = 200.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=1.0,
                              debris_states=debris, collision_radius=5.0)
        self.assertFalse(a["collision_risk"])
        self.assertGreater(a["min_dist_to_debris_m"], 5.0)

    def test_summary_columns(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = 50.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=10.0)
        self.assertIn("keep_out_violation", a["summary"].columns)
        self.assertNotIn("collision_risk", a["summary"].columns)  # no debris given

    def test_run_collision_scenario_integration(self):
        res = sc.run_collision_scenario([0.0, 300.0, 40.0], [0.0, 0.0, 0.0], dt_s=20.0,
                                        keep_out_radius=25.0, collision_radius=5.0,
                                        debris_rel_pos=[10.0, 150.0, 20.0])
        self.assertIn("analysis", res)
        self.assertIn("collision_risk", res["analysis"])
        # A PD rendezvous drives the chaser to the target (origin), so it must
        # enter any keep-out zone centred there -- the detector should flag it.
        self.assertTrue(res["analysis"]["keep_out_violation"])


if __name__ == "__main__":
    unittest.main()
