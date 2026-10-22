import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import cw, orbit, perturbations, validation
from src.orbit import DEFRAD, MUEARTH


class TestVal(unittest.TestCase):
    def setUp(self):
        self.radius = DEFRAD
        self.period = orbit.orbperiod(self.radius, MUEARTH)
        self.n = cw.cwmeanmotion(self.radius, MUEARTH)
        self.target0 = orbit.circorbitstate(self.radius, MUEARTH, np.radians(51.6))

    def testcheckorbittwo(self):
        t = np.linspace(0.0, self.period, 2001)
        states = orbit.propagate(perturbations.ForceMod.twobody().derivatives, self.target0, t)
        row = validation.checkorbit(t, states, MUEARTH, self.period).iloc[0]
        self.assertLess(row["max_energy_drift_rel"], 1e-9)
        self.assertLess(row["period_rel_err"], 1e-3)

    def testdeltavzero(self):
        t = np.linspace(0.0, 100.0, 11)
        controls = np.zeros((11, 3))
        self.assertEqual(validation.cumdeltav(t, controls), 0.0)

    def testconvtimesem(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        err = np.array([10.0, 5.0, 0.5, 0.2, 0.1])
        self.assertEqual(validation.convtime(t, err, 1.0), 2.0)
        self.assertEqual(validation.convtime(t, np.full(5, 0.1), 1.0), 0.0)
        self.assertTrue(np.isnan(validation.convtime(t, np.full(5, 10.0), 1.0)))

    def testcwmatchhigh(self):
        rel0 = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        fm = perturbations.ForceMod.twobody()
        cmp = validation.cmpfid(self.target0, rel0, t, fm, self.n)
        self.assertLess(cmp["summary"].iloc[0]["max_pos_error_m"], 1.0)

    def testpertsincrdiv(self):
        rel0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        tb = validation.cmpfid(
            self.target0, rel0, t, perturbations.ForceMod.twobody(), self.n
        )["summary"].iloc[0]["max_pos_error_m"]
        j2 = validation.cmpfid(
            self.target0, rel0, t, perturbations.ForceMod.j2only(), self.n
        )["summary"].iloc[0]["max_pos_error_m"]
        self.assertGreater(j2, tb)

    def testgainsweepshape(self):
        rel0 = np.array([100.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        df = validation.gainsweep2(rel0, t, self.n, [4.6e-5, 1e-4], [1.36e-2, 3e-2])
        self.assertEqual(len(df), 4)
        for col in ("kp", "kd", "final_pos_error_m", "convergence_time_s",
                    "max_control_accel_mps2", "delta_v_mps"):
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
