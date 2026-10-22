import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import orbit, perturbations, validation
from src.orbit import DEFRAD, MUEARTH


class TestPerts(unittest.TestCase):
    def setUp(self):
        self.radius = DEFRAD
        self.period = orbit.orbperiod(self.radius, MUEARTH)
        self.state0 = orbit.circorbitstate(self.radius, MUEARTH, np.radians(51.6))
        self.t = np.linspace(0.0, 2.0 * self.period, 4001)

    def testj2zeroaz(self):
        acc = perturbations.j2acc(np.array([self.radius, 0.0, 0.0]))
        self.assertAlmostEqual(acc[2], 0.0, places=15)
        self.assertLess(acc[0], 0.0)

    def testtogglelabels(self):
        self.assertEqual(perturbations.ForceMod.twobody().label, "two-body")
        self.assertEqual(perturbations.ForceMod.j2only().label, "two-body + J2")
        self.assertEqual(perturbations.ForceMod.dragonly().label, "two-body + drag")
        self.assertEqual(perturbations.ForceMod.j2_drag().label, "two-body + J2 + drag")

    def testtwobodyforce(self):
        acc = perturbations.ForceMod.twobody().acc3(self.state0)
        self.assertTrue(np.allclose(acc, orbit.twobodyacc(self.state0[:3])))

    def testj2conshz(self):
        states = orbit.propagate(perturbations.ForceMod.j2only().derivatives, self.state0, self.t)
        hz = np.array([np.cross(s[:3], s[3:])[2] for s in states])
        energy = validation.energyseries(states, MUEARTH)
        self.assertLess(abs((hz[-1] - hz[0]) / hz[0]), 1e-8)
        self.assertLess(np.max(np.abs(validation.reldrift(energy))), 5e-3)

    def testdragdecaysenergy(self):
        fm = perturbations.ForceMod.dragonly(area=1.0, mass=100.0)
        states = orbit.propagate(fm.derivatives, self.state0, self.t)
        energy = validation.energyseries(states, MUEARTH)
        self.assertLess(energy[-1], energy[0])

    def testdragzerowith(self):
        acc = perturbations.dragacc(
            np.array([self.radius + 2.0e6, 0.0, 0.0]), np.array([0.0, 7000.0, 0.0])
        )
        self.assertTrue(np.allclose(acc, 0.0, atol=1e-15))


if __name__ == "__main__":
    unittest.main()
