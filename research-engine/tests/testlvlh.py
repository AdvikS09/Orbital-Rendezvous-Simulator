import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import lvlh, orbit, perturbations
from src.orbit import DEFRAD, MUEARTH


class TestLVLH(unittest.TestCase):
    def setUp(self):
        self.target = orbit.circorbitstate(DEFRAD, MUEARTH, np.radians(51.6))

    def testrotmatorth(self):
        rot = lvlh.lvlhrotmat(self.target)
        self.assertTrue(np.allclose(rot @ rot.T, np.eye(3), atol=1e-12))
        self.assertAlmostEqual(np.linalg.det(rot), 1.0, places=10)

    def testroundtrip(self):
        rel = np.array([100.0, -200.0, 50.0, 0.1, 0.2, -0.05])
        deputy = lvlh.lvlhinert(self.target, rel)
        relback = lvlh.inertlvlh(self.target, deputy)
        self.assertTrue(np.allclose(rel, relback, atol=1e-9))

    def testrelhelpconsist(self):
        deputy = lvlh.lvlhinert(self.target, np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]))
        full = lvlh.inertlvlh(self.target, deputy)
        self.assertTrue(np.allclose(lvlh.relatpos(self.target, deputy), full[:3]))
        self.assertTrue(np.allclose(lvlh.relatvel(self.target, deputy), full[3:]))

    def testalongoffstays(self):
        period = orbit.orbperiod(DEFRAD, MUEARTH)
        t = np.linspace(0.0, period, 2001)
        rel0 = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        deputy0 = lvlh.lvlhinert(self.target, rel0)
        deriv = perturbations.ForceMod.twobody().derivatives
        tgt = orbit.propagate(deriv, self.target, t)
        dep = orbit.propagate(deriv, deputy0, t)
        rel = np.array([lvlh.inertlvlh(tgt[k], dep[k]) for k in range(t.size)])
        d = np.max(np.linalg.norm(rel[:, :3] - rel0[:3], axis=1))
        self.assertLess(d, 1.0)


if __name__ == "__main__":
    unittest.main()
