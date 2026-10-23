import unittest

import numpy as np

from src import controller, cw, orbit, perturbations, validation


RADIUS = orbit.DEFAULT_RADIUS
MU = orbit.MU_EARTH
PERIOD = orbit.orbital_period(RADIUS, MU)
N = cw.cw_mean_motion(RADIUS, MU)
TARGET0 = orbit.circular_orbit_state(RADIUS, MU, np.radians(51.6))


def _cw_integration_error(n_steps, state0):
    times = np.linspace(0.0, PERIOD, n_steps + 1)
    numeric, _ = cw.propagate_cw(state0, times, N, None)
    exact = cw.propagate_cw_analytic(state0, times, N)
    return float(np.max(np.linalg.norm(numeric[:, :3] - exact[:, :3], axis=1)))


class TestRK4Order(unittest.TestCase):
    def test_cw_propagation_is_fourth_order(self):
        state0 = np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0])
        steps = [125, 250, 500, 1000]
        errors = [_cw_integration_error(s, state0) for s in steps]

        for coarse, fine in zip(errors, errors[1:]):
            self.assertLess(fine, coarse)

        orders = [np.log2(c / f) for c, f in zip(errors, errors[1:])]
        for order in orders:
            self.assertGreater(order, 3.6, f"observed order {order:.2f} is below 4th")
            self.assertLess(order, 4.4, f"observed order {order:.2f} is above 4th")

    def test_two_body_energy_drift_falls_with_step(self):
        drifts = []
        for n_steps in (500, 1000, 2000):
            times = np.linspace(0.0, PERIOD, n_steps + 1)
            states = orbit.propagate(orbit.two_body_derivatives, TARGET0, times)
            drift = float(np.max(np.abs(
                validation.relative_drift(validation.energy_series(states, MU)))))
            drifts.append(drift)
        for coarse, fine in zip(drifts, drifts[1:]):
            self.assertLess(fine, coarse)


class TestKnownSolutions(unittest.TestCase):
    def test_circular_orbit_stays_circular(self):
        times = np.linspace(0.0, PERIOD, 2001)
        states = orbit.propagate(orbit.two_body_derivatives, TARGET0, times)
        radii = np.linalg.norm(states[:, :3], axis=1)
        self.assertLess(float(np.ptp(radii)) / RADIUS, 1e-10)

    def test_circular_orbit_returns_to_start_after_one_period(self):
        times = np.linspace(0.0, PERIOD, 4001)
        states = orbit.propagate(orbit.two_body_derivatives, TARGET0, times)
        closing_error = float(np.linalg.norm(states[-1, :3] - states[0, :3]))
        self.assertLess(closing_error, 1e-3)

    def test_cross_track_is_harmonic_at_the_orbit_rate(self):
        times = np.linspace(0.0, PERIOD, 2001)
        z0 = 250.0
        states, _ = cw.propagate_cw(
            np.array([0.0, 0.0, z0, 0.0, 0.0, 0.0]), times, N, None)
        analytic = z0 * np.cos(N * times)
        self.assertLess(float(np.max(np.abs(states[:, 2] - analytic))), 1e-6)

    def test_along_track_offset_is_a_cw_equilibrium(self):
        times = np.linspace(0.0, PERIOD, 2001)
        states, _ = cw.propagate_cw(
            np.array([0.0, 500.0, 0.0, 0.0, 0.0, 0.0]), times, N, None)
        self.assertLess(float(np.max(np.abs(states[:, 1] - 500.0))), 1e-6)
        self.assertLess(float(np.max(np.abs(states[:, 0]))), 1e-6)

    def test_radial_offset_drifts_minus_twelve_pi_x0_per_orbit(self):
        x0 = 100.0
        times = np.linspace(0.0, PERIOD, 4001)
        states, _ = cw.propagate_cw(
            np.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0]), times, N, None)
        expected = -12.0 * np.pi * x0
        self.assertAlmostEqual(states[-1, 1] / expected, 1.0, places=6)


class TestReportedMetricsAreStepIndependent(unittest.TestCase):
    def test_cw_vs_reference_divergence_is_converged(self):
        rel0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
        force_model = perturbations.ForceModel.two_body()
        values = []
        for n_steps in (500, 1000, 2000):
            times = np.linspace(0.0, PERIOD, n_steps + 1)
            summary = validation.compare_cw_high_fidelity(
                TARGET0, rel0, times, force_model, N, controller=None)["summary"].iloc[0]
            values.append(float(summary["max_pos_error_m"]))
        finest = values[-1]
        for value in values:
            self.assertLess(abs(value - finest) / finest, 1e-3,
                            f"divergence metric moves with step size: {values}")

    def test_delta_v_is_converged(self):
        rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
        ctrl = controller.PDController.critically_damped(6.0 * N, max_accel=0.05)
        values = []
        for n_steps in (500, 1000, 2000):
            times = np.linspace(0.0, PERIOD, n_steps + 1)
            states, controls = cw.propagate_cw(rel0, times, N, ctrl)
            values.append(validation.cumulative_delta_v(times, controls))
        finest = values[-1]
        for value in values:
            self.assertLess(abs(value - finest) / finest, 1e-3,
                            f"delta-v moves with step size: {values}")


if __name__ == "__main__":
    unittest.main()
