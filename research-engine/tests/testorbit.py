import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import orbit, perturbations, validation
from src.orbit import DEFRAD, MUEARTH


class TestOrb(unittest.TestCase):
    def setUp(self):
        self.radius = DEFRAD
        self.period = orbit.orbperiod(self.radius, MUEARTH)
        self.state0 = orbit.circorbitstate(self.radius, MUEARTH, np.radians(45.0))

    def testcircspeed(self):
        v = orbit.circspeed(self.radius, MUEARTH)
        self.assertAlmostEqual(v, np.sqrt(MUEARTH / self.radius), places=6)

    def test400kmperiod(self):
        self.assertTrue(90 * 60 < self.period < 95 * 60)

    def testrk4consenergy(self):
        t = np.linspace(0.0, self.period, 2001)
        states = orbit.propagate(perturbations.ForceMod.twobody().derivatives, self.state0, t)
        energy = validation.energyseries(states, MUEARTH)
        angmom = validation.angmomseries(states)
        self.assertLess(np.max(np.abs(validation.reldrift(energy))), 1e-9)
        self.assertLess(np.max(np.abs(validation.reldrift(angmom))), 1e-9)

    def testmeasperiodmatch(self):
        t = np.linspace(0.0, 1.3 * self.period, 4001)
        states = orbit.propagate(perturbations.ForceMod.twobody().derivatives, self.state0, t)
        measured = orbit.measperiod(t, states)
        self.assertAlmostEqual(measured / self.period, 1.0, places=4)

    def testcircorbitzero(self):
        el = orbit.rvelem(self.state0, MUEARTH)
        self.assertLess(el["e"], 1e-9)
        self.assertAlmostEqual(el["a"], self.radius, places=0)
        self.assertAlmostEqual(el["i"], np.radians(45.0), places=6)

    def testelemroundtrip(self):
        state = orbit.elemrv(7.0e6, 0.08, np.radians(35), np.radians(50),
                             np.radians(70), np.radians(110), MUEARTH)
        el = orbit.rvelem(state, MUEARTH)
        stateback = orbit.elemrv(el["a"], el["e"], el["i"], el["raan"],
                                 el["argp"], el["nu"], MUEARTH)
        self.assertTrue(np.allclose(state, stateback, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
