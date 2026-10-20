import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import collision
from src import scenarios as sc


class TestColl(unittest.TestCase):
    def testmindist(self):
        pos = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 0.0]])
        dist, dmin, idx = collision.mindist(pos, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(dmin, 0.0)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(dist[0], 5.0)
        self.assertAlmostEqual(dist[2], 10.0)

    def testkeepoutviol(self):
        states = np.zeros((11, 6))
        states[:, 1] = np.linspace(100.0, -100.0, 11)
        a = collision.analyze(np.arange(11), states, keep_out_radius=10.0)
        self.assertTrue(a["keep_out_violation"])
        self.assertLess(a["min_dist_to_target_m"], 10.0)

    def testkeepoutresp(self):
        states = np.zeros((10, 6))
        states[:, 0] = 50.0
        a = collision.analyze(np.arange(10), states, keep_out_radius=10.0)
        self.assertFalse(a["keep_out_violation"])
        self.assertAlmostEqual(a["min_dist_to_target_m"], 50.0)

    def testdebriscollflagg(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = np.linspace(0.0, 40.0, 5)
        debris = np.zeros((5, 6))
        debris[:, 0] = 20.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=1.0,
                              debris_states=debris, collision_radius=5.0)
        self.assertTrue(a["collision_risk"])
        self.assertAlmostEqual(a["min_dist_to_debris_m"], 0.0)

    def testdebrisclear(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = np.linspace(0.0, 40.0, 5)
        debris = np.zeros((5, 6))
        debris[:, 0] = 200.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=1.0,
                              debris_states=debris, collision_radius=5.0)
        self.assertFalse(a["collision_risk"])
        self.assertGreater(a["min_dist_to_debris_m"], 5.0)

    def testsummcol(self):
        chaser = np.zeros((5, 6))
        chaser[:, 0] = 50.0
        a = collision.analyze(np.arange(5), chaser, keep_out_radius=10.0)
        self.assertIn("keep_out_violation", a["summary"].columns)
        self.assertNotIn("collision_risk", a["summary"].columns)

    def testruncollscen(self):
        res = sc.runcollscen([0.0, 300.0, 40.0], [0.0, 0.0, 0.0], dt_s=20.0,
                             keep_out_radius=25.0, collision_radius=5.0,
                             debrisrelpos=[10.0, 150.0, 20.0])
        self.assertIn("analysis", res)
        self.assertIn("collision_risk", res["analysis"])
        self.assertTrue(res["analysis"]["keep_out_violation"])


if __name__ == "__main__":
    unittest.main()
