import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import controller, cw, lvlh, orbit, perturbations


class TestCtrlImpl(unittest.TestCase):
    def setUp(self):
        self.radius = orbit.DEFRAD
        self.mu = orbit.MUEARTH
        self.n = cw.cwmeanmotion(self.radius, self.mu)
        self.target0 = orbit.circorbitstate(self.radius, self.mu, np.radians(51.6))
        self.rot = lvlh.lvlhrotmat(self.target0)

    def testctrlvecroundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ulvlh = rng.normal(size=3)
            ueci = self.rot.T @ ulvlh
            uback = self.rot @ ueci
            self.assertLess(np.linalg.norm(uback - ulvlh), 1e-13)

    def testrotpresmag(self):
        self.assertTrue(np.allclose(self.rot @ self.rot.T, np.eye(3), atol=1e-13))
        rng = np.random.default_rng(1)
        for _ in range(20):
            v = rng.normal(size=3)
            self.assertAlmostEqual(np.linalg.norm(self.rot @ v), np.linalg.norm(v), places=12)
            self.assertAlmostEqual(np.linalg.norm(self.rot.T @ v), np.linalg.norm(v), places=12)

    def testsatdircap(self):
        cap = 0.05
        big = np.array([0.3, -0.4, 0.5])
        sat = controller.saturate(big, cap)
        self.assertAlmostEqual(np.linalg.norm(sat), cap, places=15)
        self.assertLess(np.linalg.norm(np.cross(sat, big)), 1e-15)
        self.assertGreater(np.dot(sat, big), 0.0)
        small = np.array([0.01, 0.0, -0.02])
        self.assertTrue(np.allclose(controller.saturate(small, cap), small))

    def testzerostatezero(self):
        pd = controller.PDCtrl.critdamped(6.0 * self.n, max_accel=0.05)
        self.assertTrue(np.allclose(pd.control(np.zeros(6)), np.zeros(3)))
        tgt = np.array([10.0, -5.0, 3.0, 0.1, 0.2, -0.3])
        pdt = controller.PDCtrl.critdamped(6.0 * self.n, target=tgt)
        self.assertTrue(np.allclose(pdt.control(tgt), np.zeros(3)))

    def testpropderivterms(self):
        kp, kd = controller.critdampedgains(6.0 * self.n)
        pd = controller.PDCtrl.critdamped(6.0 * self.n)
        pos = np.array([1.0, -2.0, 0.5])
        self.assertTrue(np.allclose(pd.control(np.concatenate((pos, np.zeros(3)))),
                                    -kp * pos))
        vel = np.array([0.02, 0.01, -0.03])
        self.assertTrue(np.allclose(pd.control(np.concatenate((np.zeros(3), vel))),
                                    -kd * vel))
        both = np.concatenate((pos, vel))
        self.assertTrue(np.allclose(pd.control(both), -kp * pos - kd * vel))

    def testfinitediffcontr(self):
        fm = perturbations.ForceMod.twobody()
        pd = controller.PDCtrl.critdamped(6.0 * self.n, max_accel=0.05)
        rel0 = np.array([3000.0, 1500.0, -800.0, 0.5, -0.3, 0.2])
        deputy0 = lvlh.lvlhinert(self.target0, rel0)

        def depaccel(tgtstate, depstate):
            a = fm.acc3(depstate)
            rel = lvlh.inertlvlh(tgtstate, depstate)
            u = pd.control(rel)
            return a + lvlh.lvlhrotmat(tgtstate).T @ u

        aimpl = depaccel(self.target0, deputy0)

        def combderiv(z, t):
            ts, ds = z[:6], z[6:]
            at = fm.acc3(ts)
            ad = depaccel(ts, ds)
            return np.concatenate((ts[3:], at, ds[3:], ad))

        z0 = np.concatenate((self.target0, deputy0))
        dt = 1e-2
        zf = orbit.rk4step(combderiv, z0, 0.0, dt)
        zb = orbit.rk4step(combderiv, z0, 0.0, -dt)
        afd = (zf[9:12] - zb[9:12]) / (2.0 * dt)
        posfd = (zf[6:9] - 2.0 * z0[6:9] + zb[6:9]) / (dt * dt)

        self.assertLess(np.linalg.norm(afd - aimpl) / np.linalg.norm(aimpl), 1e-6)
        self.assertLess(np.linalg.norm(posfd - aimpl) / np.linalg.norm(aimpl), 1e-6)


if __name__ == "__main__":
    unittest.main()
