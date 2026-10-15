"""Tests for the validation utilities: metrics, gain sweep, CW-vs-high-fidelity."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import cw, orbit, perturbations, validation
from src.orbit import DEFAULT_RADIUS, MU_EARTH


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.radius = DEFAULT_RADIUS
        self.period = orbit.orbital_period(self.radius, MU_EARTH)
        self.n = cw.cw_mean_motion(self.radius, MU_EARTH)
        self.target0 = orbit.circular_orbit_state(self.radius, MU_EARTH, np.radians(51.6))

    def test_verify_orbit_two_body(self):
        t = np.linspace(0.0, self.period, 2001)
        states = orbit.propagate(perturbations.ForceModel.two_body().derivatives, self.target0, t)
        row = validation.verify_orbit(t, states, MU_EARTH, self.period).iloc[0]
        self.assertLess(row["max_energy_drift_rel"], 1e-9)
        self.assertLess(row["period_rel_err"], 1e-3)

    def test_delta_v_zero_for_no_control(self):
        t = np.linspace(0.0, 100.0, 11)
        controls = np.zeros((11, 3))
        self.assertEqual(validation.cumulative_delta_v(t, controls), 0.0)

    def test_convergence_time_semantics(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        err = np.array([10.0, 5.0, 0.5, 0.2, 0.1])  # drops below tol=1 after index 1
        self.assertEqual(validation.convergence_time(t, err, 1.0), 2.0)
        self.assertEqual(validation.convergence_time(t, np.full(5, 0.1), 1.0), 0.0)
        self.assertTrue(np.isnan(validation.convergence_time(t, np.full(5, 10.0), 1.0)))

    def test_cw_matches_high_fidelity_two_body_small_sep(self):
        rel0 = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        fm = perturbations.ForceModel.two_body()
        cmp = validation.compare_cw_high_fidelity(self.target0, rel0, t, fm, self.n)
        self.assertLess(cmp["summary"].iloc[0]["max_pos_error_m"], 1.0)

    def test_perturbations_increase_divergence(self):
        rel0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        tb = validation.compare_cw_high_fidelity(
            self.target0, rel0, t, perturbations.ForceModel.two_body(), self.n
        )["summary"].iloc[0]["max_pos_error_m"]
        j2 = validation.compare_cw_high_fidelity(
            self.target0, rel0, t, perturbations.ForceModel.j2_only(), self.n
        )["summary"].iloc[0]["max_pos_error_m"]
        # Adding J2 makes the CW linearisation diverge more from the truth.
        self.assertGreater(j2, tb)

    def test_gain_sweep_shape(self):
        rel0 = np.array([100.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        t = np.linspace(0.0, self.period, 1001)
        df = validation.gain_sweep(rel0, t, self.n, [4.6e-5, 1e-4], [1.36e-2, 3e-2])
        self.assertEqual(len(df), 4)
        for col in ("kp", "kd", "final_pos_error_m", "convergence_time_s",
                    "max_control_accel_mps2", "delta_v_mps"):
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
