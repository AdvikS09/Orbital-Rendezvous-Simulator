import json
import os
import unittest

import numpy as np

from src import controller, cw, orbit, perturbations, validation
from src import scenarios as sc

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_engine_outputs.json")
with open(BASELINE_PATH, encoding="utf-8") as fh:
    BASE = json.load(fh)

RTOL = 1e-6
ATOL = 1e-11

RADIUS = orbit.DEFAULT_RADIUS
MU = orbit.MU_EARTH
PERIOD = orbit.orbital_period(RADIUS, MU)
INC = np.radians(51.6)
TARGET0 = orbit.circular_orbit_state(RADIUS, MU, INC)
N = cw.cw_mean_motion(RADIUS, MU)


class _BaselineAssertions(unittest.TestCase):
    def assertClose(self, actual, expected, msg=""):
        self.assertTrue(
            np.isclose(actual, expected, rtol=RTOL, atol=ATOL),
            f"{msg}: got {actual!r}, baseline {expected!r} "
            f"(rtol={RTOL}, atol={ATOL})",
        )


class TestEngineRegression(_BaselineAssertions):
    def test_two_body_conservation_and_period(self):
        base = BASE["two_body"]
        times = np.linspace(0.0, 2.0 * PERIOD, 4001)
        states = orbit.propagate(orbit.two_body_derivatives, TARGET0, times)
        row = validation.verify_orbit(times, states, MU, analytic_period=PERIOD).iloc[0]
        self.assertClose(PERIOD, base["analytic_period_s"], "analytic period")
        self.assertClose(row["max_energy_drift_rel"], base["max_energy_drift_rel"],
                         "energy drift")
        self.assertClose(row["max_ang_mom_drift_rel"], base["max_ang_mom_drift_rel"],
                         "angular-momentum drift")
        self.assertClose(row["period_rel_err"], base["period_rel_err"], "period error")
        self.assertLess(row["max_energy_drift_rel"], 1e-11)
        self.assertLess(row["period_rel_err"], 1e-9)

    def test_mean_motion(self):
        self.assertClose(N, BASE["mean_motion_rad_s"], "mean motion")

    def test_cw_rk4_matches_state_transition_matrix(self):
        times = np.linspace(0.0, PERIOD, 2001)
        state0 = np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0])
        rk4, _ = cw.propagate_cw(state0, times, N, None)
        exact = cw.propagate_cw_analytic(state0, times, N)
        err = float(np.max(np.linalg.norm(rk4[:, :3] - exact[:, :3], axis=1)))
        self.assertClose(err, BASE["cw_rk4_vs_stm_max_err_m"], "CW RK4 vs STM")
        self.assertLess(err, 1e-6)

    def test_pd_rendezvous_metrics(self):
        base = BASE["pd_rendezvous"]
        times = np.linspace(0.0, PERIOD, 1112)
        ctrl = controller.PDController.critically_damped(6.0 * N, max_accel=0.05)
        states, controls = cw.propagate_cw(
            np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0]), times, N, ctrl)
        row = validation.rendezvous_metrics(times, states, controls, pos_tol=1.0).iloc[0]
        for key, expected in base.items():
            self.assertClose(float(row[key]), expected, f"rendezvous {key}")

    def test_cw_vs_high_fidelity_toggles(self):
        times = np.linspace(0.0, PERIOD, 1001)
        rel0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
        models = {
            "two_body": perturbations.ForceModel.two_body(),
            "j2": perturbations.ForceModel.j2_only(),
            "drag": perturbations.ForceModel.drag_only(),
            "j2_drag": perturbations.ForceModel.j2_drag(),
        }
        for name, force_model in models.items():
            expected = BASE["cw_vs_hf_toggles"][name]
            summary = validation.compare_cw_high_fidelity(
                TARGET0, rel0, times, force_model, N, controller=None)["summary"].iloc[0]
            for key, value in expected.items():
                self.assertClose(float(summary[key]), value, f"{name} {key}")

    def test_j2_roughly_doubles_cw_error(self):
        two_body = BASE["cw_vs_hf_toggles"]["two_body"]["max_pos_error_m"]
        with_j2 = BASE["cw_vs_hf_toggles"]["j2"]["max_pos_error_m"]
        self.assertGreater(with_j2 / two_body, 1.8)
        self.assertLess(with_j2 / two_body, 2.5)

    def test_drag_is_nearly_common_mode(self):
        two_body = BASE["cw_vs_hf_toggles"]["two_body"]["max_pos_error_m"]
        with_drag = BASE["cw_vs_hf_toggles"]["drag"]["max_pos_error_m"]
        self.assertLess(abs(with_drag - two_body), 0.01)

    def test_gain_sweep_critically_damped_point(self):
        times = np.linspace(0.0, PERIOD, 556)
        table = validation.gain_sweep(
            np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0]), times, N,
            [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4], [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2],
            max_accel=None, pos_tol=1.0)
        self.assertEqual(len(table), BASE["gain_sweep_rows"])
        row = table[(table.kp == 4.6e-5) & (table.kd == 1.36e-2)].iloc[0]
        for key, expected in BASE["gain_sweep_critdamp"].items():
            self.assertClose(float(row[key]), expected, f"gain sweep {key}")


class TestScenarioApiRegression(_BaselineAssertions):
    def test_run_cw_rendezvous_defaults(self):
        result = sc.run_cw_rendezvous([200, -300, 80], [0, 0, 0])
        row = result["metrics"].iloc[0]
        for key, expected in BASE["api_run_cw_rendezvous"].items():
            self.assertClose(float(row[key]), expected, f"api rendezvous {key}")

    def test_run_orbit_defaults(self):
        result = sc.run_orbit(400.0, 51.6, 2.0)
        base = BASE["api_run_orbit"]
        self.assertClose(result["period_s"], base["period_s"], "api period")
        self.assertClose(float(result["validation"].iloc[0]["max_energy_drift_rel"]),
                         base["max_energy_drift_rel"], "api energy drift")

    def test_run_relative_scenario_comparison(self):
        result = sc.run_relative_scenario(
            [0, 1000, 0], [0, 0, 0], model="comparison", controlled=False,
            use_j2=True, use_drag=False, dt_s=10.0)
        row = result["comparison"]["summary"].iloc[0]
        for key, expected in BASE["api_cw_vs_hf_j2"].items():
            self.assertClose(float(row[key]), expected, f"api comparison {key}")


if __name__ == "__main__":
    unittest.main()
