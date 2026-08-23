"""Tests for the external-tool validation workflow"""

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src import external_validation as ext
from src import scenarios as sc


def make_ephemeris(n=50):
    """Synthetic ephemeris that is linear in time (so interpolation is exact)."""
    t = np.linspace(0.0, 1000.0, n)
    return pd.DataFrame({
        "t_s": t,
        "x_m": 7.0e6 + t, "y_m": 2.0 * t, "z_m": -1.5 * t,
        "vx_mps": np.full(n, 1.0), "vy_mps": np.full(n, 2.0), "vz_mps": np.full(n, -1.5),
    })


class TestExternalValidation(unittest.TestCase):
    def setUp(self):
        self.eph = make_ephemeris()

    def test_csv_roundtrip_zero_error(self):
        csv = ext.to_csv_text(self.eph, {"frame": "ECI", "initial_state_m_mps": [7e6, 0, 0, 0, 1, 0]})
        loaded = ext.load_external(io.StringIO(csv))  # autodetect, SI units
        cmp = ext.compare_ephemerides(self.eph, loaded)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def test_json_is_valid_and_complete(self):
        payload = json.loads(ext.to_json_text(self.eph, {"a": 1, "b": "x"}))
        self.assertIn("metadata", payload)
        self.assertIn("ephemeris", payload)
        self.assertEqual(len(payload["ephemeris"]["t_s"]), len(self.eph))

    def test_known_shift_metrics(self):
        shifted = self.eph.copy()
        shifted["x_m"] = shifted["x_m"] + 100.0
        s = ext.compare_ephemerides(self.eph, shifted)["summary"].iloc[0]
        self.assertAlmostEqual(s["max_position_error_m"], 100.0, places=6)
        self.assertAlmostEqual(s["rms_position_error_m"], 100.0, places=6)
        self.assertAlmostEqual(s["final_position_error_m"], 100.0, places=6)

    def test_km_unit_loading(self):
        km = self.eph.copy()
        for c in ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]:
            km[c] = km[c] / 1000.0
        km.columns = ["t_s", "x", "y", "z", "vx", "vy", "vz"]
        buf = io.StringIO()
        km.to_csv(buf, index=False)
        buf.seek(0)
        loaded = ext.load_external(buf, time_col="t_s", pos_cols=["x", "y", "z"],
                                   vel_cols=["vx", "vy", "vz"], length_unit="km")
        cmp = ext.compare_ephemerides(self.eph, loaded)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def test_autodetect_standard_columns(self):
        t, p, v = ext.autodetect_columns(["t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"])
        self.assertEqual(t, "t_s")
        self.assertEqual(p, ["x_m", "y_m", "z_m"])
        self.assertEqual(v, ["vx_mps", "vy_mps", "vz_mps"])

    def test_interpolation_across_different_grids(self):
        cmp = ext.compare_ephemerides(self.eph, make_ephemeris(123))
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)

    def test_no_overlap_raises(self):
        other = self.eph.copy()
        other["t_s"] = other["t_s"] + 1.0e6
        with self.assertRaises(ValueError):
            ext.compare_ephemerides(self.eph, other)

    def test_build_scenario_self_consistency(self):
        scen = sc.build_validation_scenario(altitude_km=400, duration_s=2000, dt_s=50,
                                            use_j2=True, use_drag=False)
        self.assertEqual(scen["metadata"]["n_samples"], len(scen["ephemeris"]))
        self.assertIn("initial_state_m_mps", scen["metadata"])
        ext_df = sc.load_external_ephemeris(io.StringIO(scen["csv_text"]))
        cmp = sc.compare_external_ephemeris(scen["ephemeris"], ext_df)
        self.assertLess(cmp["summary"].iloc[0]["max_position_error_m"], 1e-6)


if __name__ == "__main__":
    unittest.main()
