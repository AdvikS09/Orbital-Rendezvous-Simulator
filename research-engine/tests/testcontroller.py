import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import controller, cw
from src.orbit import DEFRAD, MUEARTH


class TestCtrl(unittest.TestCase):
    def setUp(self):
        self.n = cw.cwmeanmotion(DEFRAD, MUEARTH)
        self.period = 2.0 * np.pi / self.n

    def testcritdampedgains(self):
        kp, kd = controller.critdampedgains(0.05)
        self.assertAlmostEqual(kp, 0.05**2)
        self.assertAlmostEqual(kd, 2.0 * 0.05)

    def testgainmatcoerc(self):
        cscalar = controller.PDCtrl(1.0, 2.0)
        cvector = controller.PDCtrl([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(cscalar.Kp, np.eye(3)))
        self.assertTrue(np.allclose(cvector.Kd, 2.0 * np.eye(3)))

    def testpddrivesstate(self):
        pd = controller.PDCtrl.critdamped(6.0 * self.n)
        x0 = np.array([150.0, -200.0, 60.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propcw(x0, t, self.n, pd)
        self.assertLess(np.linalg.norm(states[-1, :3]), 0.1)
        self.assertLess(np.linalg.norm(states[-1, 3:]), 1e-3)

    def testsat(self):
        pd = controller.PDCtrl.critdamped(6.0 * self.n, max_accel=0.01)
        u = pd.control(np.array([1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0]))
        self.assertLessEqual(np.linalg.norm(u), 0.01 + 1e-12)

    def testcrosstrackconv(self):
        pd = controller.PDCtrl.critdamped(6.0 * self.n)
        x0 = np.array([0.0, 0.0, 120.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propcw(x0, t, self.n, pd)
        self.assertLess(abs(states[-1, 2]), 0.1)


if __name__ == "__main__":
    unittest.main()
