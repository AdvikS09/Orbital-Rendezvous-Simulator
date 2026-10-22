import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src import extval as ext
from src import scenarios as sc


def makeeph(n=50):
    t = np.linspace(0.0, 1000.0, n)
    return pd.DataFrame({
        "t_s": t,
        "x_m": 7.0e6 + t, "y_m": 2.0 * t, "z_m": -1.5 * t,
        "vx_mps": np.full(n, 1.0), "vy_mps": np.full(n, 2.0), "vz_mps": np.full(n, -1.5),
    })


class TestExtVal(unittest.TestCase):
    def setUp(self):
        self.eph = makeeph()

    def testcsvroundtripzero(self):
        csv = ext.tocsvtext(self.eph, {"frame": "ECI", "initial_state_m_mps": [7e6, 0, 0, 0, 1, 0]})
        loaded = ext.loadext(io.StringIO(csv))
        cmp = ext.cmpeph(self.eph, loaded)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def testjsonvalidfull(self):
        doc = json.loads(ext.jsontext(self.eph, {"a": 1, "b": "x"}))
        self.assertIn("metadata", doc)
        self.assertIn("ephemeris", doc)
        self.assertEqual(len(doc["ephemeris"]["t_s"]), len(self.eph))

    def testknownshiftmetrics(self):
        shifted = self.eph.copy()
        shifted["x_m"] = shifted["x_m"] + 100.0
        s = ext.cmpeph(self.eph, shifted)["summary"].iloc[0]
        self.assertAlmostEqual(s["max_position_error_m"], 100.0, places=6)
        self.assertAlmostEqual(s["rms_position_error_m"], 100.0, places=6)
        self.assertAlmostEqual(s["final_position_error_m"], 100.0, places=6)

    def testkmunitload(self):
        km = self.eph.copy()
        for c in ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]:
            km[c] = km[c] / 1000.0
        km.columns = ["t_s", "x", "y", "z", "vx", "vy", "vz"]
        buf = io.StringIO()
        km.to_csv(buf, index=False)
        buf.seek(0)
        loaded = ext.loadext(buf, timecol="t_s", poscols=["x", "y", "z"],
                             velcols=["vx", "vy", "vz"], lenunit="km")
        cmp = ext.cmpeph(self.eph, loaded)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def testautstdcol(self):
        t, p, v = ext.autcol(["t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"])
        self.assertEqual(t, "t_s")
        self.assertEqual(p, ["x_m", "y_m", "z_m"])
        self.assertEqual(v, ["vx_mps", "vy_mps", "vz_mps"])

    def testinterpacrossdiff(self):
        cmp = ext.cmpeph(self.eph, makeeph(123))
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def testnooverlraises(self):
        other = self.eph.copy()
        other["t_s"] = other["t_s"] + 1.0e6
        with self.assertRaises(ValueError):
            ext.cmpeph(self.eph, other)

    def testbuildscenself(self):
        scen = sc.buildvalscen(altitude_km=400, duration_s=2000, dt_s=50,
                               use_j2=True, use_drag=False)
        self.assertEqual(scen["metadata"]["n_samples"], len(scen["ephemeris"]))
        self.assertIn("initial_state_m_mps", scen["metadata"])
        extdf = sc.loadexteph(io.StringIO(scen["csv_text"]))
        cmp = sc.cmpexteph(scen["ephemeris"], extdf)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)


if __name__ == "__main__":
    unittest.main()
