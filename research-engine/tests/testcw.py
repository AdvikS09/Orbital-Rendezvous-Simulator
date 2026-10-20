import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import cw
from src.orbit import DEFRAD, MUEARTH


class TestCW(unittest.TestCase):
    def setUp(self):
        self.n = cw.cwmeanmotion(DEFRAD, MUEARTH)
        self.period = 2.0 * np.pi / self.n

    def testsystemmatshapes(self):
        A, B = cw.cwsystemmats(self.n)
        self.assertEqual(A.shape, (6, 6))
        self.assertEqual(B.shape, (6, 3))

    def teststmzeroident(self):
        self.assertTrue(np.allclose(cw.cwstatetrans(self.n, 0.0), np.eye(6)))

    def testnummatchstm(self):
        x0 = np.array([120.0, -80.0, 40.0, 0.3, 0.1, 0.2])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propcw(x0, t, self.n)
        analytic = cw.propcwanal(x0, t, self.n)
        self.assertLess(np.max(np.abs(states - analytic)), 1e-6)

    def testalongoffequil(self):
        x0 = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 2001)
        states, _ = cw.propcw(x0, t, self.n)
        self.assertTrue(np.allclose(states[-1], x0, atol=1e-6))

    def testcrosstrackharm(self):
        x0 = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 4001)
        states, _ = cw.propcw(x0, t, self.n)
        self.assertAlmostEqual(states[-1, 2], 50.0, places=2)
        self.assertTrue(np.allclose(states[:, 0], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(states[:, 1], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
