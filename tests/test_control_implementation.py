import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import controller, cw, lvlh, orbit, perturbations


class TestControlImplementation(unittest.TestCase):
    def setUp(self):
        self.radius = orbit.DEFAULT_RADIUS
        self.mu = orbit.MU_EARTH
        self.n = cw.cw_mean_motion(self.radius, self.mu)
        self.target0 = orbit.circular_orbit_state(self.radius, self.mu, np.radians(51.6))
        self.rot = lvlh.lvlh_rotation_matrix(self.target0)

    def test_control_vector_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            u_lvlh = rng.normal(size=3)
            u_eci = self.rot.T @ u_lvlh
            u_back = self.rot @ u_eci
            self.assertLess(np.linalg.norm(u_back - u_lvlh), 1e-13)

    def test_rotation_preserves_magnitude(self):
        self.assertTrue(np.allclose(self.rot @ self.rot.T, np.eye(3), atol=1e-13))
        rng = np.random.default_rng(1)
        for _ in range(20):
            v = rng.normal(size=3)
            self.assertAlmostEqual(np.linalg.norm(self.rot @ v), np.linalg.norm(v), places=12)
            self.assertAlmostEqual(np.linalg.norm(self.rot.T @ v), np.linalg.norm(v), places=12)

    def test_saturation_direction_and_cap(self):
        cap = 0.05
        big = np.array([0.3, -0.4, 0.5])
        sat = controller.saturate(big, cap)
        self.assertAlmostEqual(np.linalg.norm(sat), cap, places=15)
        self.assertLess(np.linalg.norm(np.cross(sat, big)), 1e-15)
        self.assertGreater(np.dot(sat, big), 0.0)
        small = np.array([0.01, 0.0, -0.02])
        self.assertTrue(np.allclose(controller.saturate(small, cap), small))

    def test_zero_state_zero_command(self):
        pd = controller.PDController.critically_damped(6.0 * self.n, max_accel=0.05)
        self.assertTrue(np.allclose(pd.control(np.zeros(6)), np.zeros(3)))
        tgt = np.array([10.0, -5.0, 3.0, 0.1, 0.2, -0.3])
        pd_t = controller.PDController.critically_damped(6.0 * self.n, target=tgt)
        self.assertTrue(np.allclose(pd_t.control(tgt), np.zeros(3)))

    def test_proportional_and_derivative_terms(self):
        kp, kd = controller.critically_damped_gains(6.0 * self.n)
        pd = controller.PDController.critically_damped(6.0 * self.n)
        pos = np.array([1.0, -2.0, 0.5])
        self.assertTrue(np.allclose(pd.control(np.concatenate((pos, np.zeros(3)))),
                                    -kp * pos))
        vel = np.array([0.02, 0.01, -0.03])
        self.assertTrue(np.allclose(pd.control(np.concatenate((np.zeros(3), vel))),
                                    -kd * vel))
        both = np.concatenate((pos, vel))
        self.assertTrue(np.allclose(pd.control(both), -kp * pos - kd * vel))

    def test_finite_difference_controlled_acceleration(self):
        fm = perturbations.ForceModel.two_body()
        pd = controller.PDController.critically_damped(6.0 * self.n, max_accel=0.05)
        rel0 = np.array([3000.0, 1500.0, -800.0, 0.5, -0.3, 0.2])
        deputy0 = lvlh.lvlh_to_inertial(self.target0, rel0)

        def deputy_accel(target_state, deputy_state):
            a = fm.acceleration(deputy_state)
            rel = lvlh.inertial_to_lvlh(target_state, deputy_state)
            u = pd.control(rel)
            return a + lvlh.lvlh_rotation_matrix(target_state).T @ u

        a_impl = deputy_accel(self.target0, deputy0)

        def combined_deriv(z, t):
            ts, ds = z[:6], z[6:]
            a_t = fm.acceleration(ts)
            a_d = deputy_accel(ts, ds)
            return np.concatenate((ts[3:], a_t, ds[3:], a_d))

        z0 = np.concatenate((self.target0, deputy0))
        dt = 1e-2
        zf = orbit.rk4_step(combined_deriv, z0, 0.0, dt)
        zb = orbit.rk4_step(combined_deriv, z0, 0.0, -dt)
        a_fd = (zf[9:12] - zb[9:12]) / (2.0 * dt)
        pos_fd = (zf[6:9] - 2.0 * z0[6:9] + zb[6:9]) / (dt * dt)

        self.assertLess(np.linalg.norm(a_fd - a_impl) / np.linalg.norm(a_impl), 1e-6)
        self.assertLess(np.linalg.norm(pos_fd - a_impl) / np.linalg.norm(a_impl), 1e-6)


if __name__ == "__main__":
    unittest.main()
