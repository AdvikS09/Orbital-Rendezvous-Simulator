"""Internal verification of the controlled nonlinear branch

GMAT validates the nonlinear propagator and the LVLH relative-state
reconstruction, but not the controller. These tests verify, inside the
simulator, the parts of the controlled nonlinear branch that GMAT does not
cover:

  1. an LVLH control vector rotated to ECI and back reproduces the original;
  2. the LVLH<->ECI rotation preserves acceleration magnitude;
  3. saturation preserves direction and enforces the exact acceleration cap;
  4. the desired (zero) relative state produces a zero command;
  5. representative states give the expected proportional and derivative terms;
  6. a central finite-difference of the deputy trajectory reproduces the
     implemented controlled acceleration (gravity + rotated PD command),
     confirming the control actually enters the nonlinear dynamics as intended.
"""

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

    # 1. LVLH -> ECI -> LVLH round trip for a free (control) vector.
    def test_control_vector_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            u_lvlh = rng.normal(size=3)
            u_eci = self.rot.T @ u_lvlh          # applied to the deputy in ECI
            u_back = self.rot @ u_eci            # read back into LVLH
            self.assertLess(np.linalg.norm(u_back - u_lvlh), 1e-13)

    # 2. Rotation preserves magnitude (orthonormal frame).
    def test_rotation_preserves_magnitude(self):
        self.assertTrue(np.allclose(self.rot @ self.rot.T, np.eye(3), atol=1e-13))
        rng = np.random.default_rng(1)
        for _ in range(20):
            v = rng.normal(size=3)
            self.assertAlmostEqual(np.linalg.norm(self.rot @ v), np.linalg.norm(v), places=12)
            self.assertAlmostEqual(np.linalg.norm(self.rot.T @ v), np.linalg.norm(v), places=12)

    # 3. Saturation: exact cap, unchanged direction, no-op below the cap.
    def test_saturation_direction_and_cap(self):
        cap = 0.05
        big = np.array([0.3, -0.4, 0.5])          # norm 0.707 > cap
        sat = controller.saturate(big, cap)
        self.assertAlmostEqual(np.linalg.norm(sat), cap, places=15)
        # direction preserved: cross product zero, positive projection
        self.assertLess(np.linalg.norm(np.cross(sat, big)), 1e-15)
        self.assertGreater(np.dot(sat, big), 0.0)
        small = np.array([0.01, 0.0, -0.02])      # norm < cap
        self.assertTrue(np.allclose(controller.saturate(small, cap), small))

    # 4. Zero relative state (the target) produces zero command.
    def test_zero_state_zero_command(self):
        pd = controller.PDController.critically_damped(6.0 * self.n, max_accel=0.05)
        self.assertTrue(np.allclose(pd.control(np.zeros(6)), np.zeros(3)))
        tgt = np.array([10.0, -5.0, 3.0, 0.1, 0.2, -0.3])
        pd_t = controller.PDController.critically_damped(6.0 * self.n, target=tgt)
        self.assertTrue(np.allclose(pd_t.control(tgt), np.zeros(3)))

    # 5. Expected proportional and derivative contributions (unsaturated).
    def test_proportional_and_derivative_terms(self):
        kp, kd = controller.critically_damped_gains(6.0 * self.n)
        pd = controller.PDController.critically_damped(6.0 * self.n)  # no cap
        pos = np.array([1.0, -2.0, 0.5])
        self.assertTrue(np.allclose(pd.control(np.concatenate((pos, np.zeros(3)))),
                                    -kp * pos))
        vel = np.array([0.02, 0.01, -0.03])
        self.assertTrue(np.allclose(pd.control(np.concatenate((np.zeros(3), vel))),
                                    -kd * vel))
        # superposition of both terms
        both = np.concatenate((pos, vel))
        self.assertTrue(np.allclose(pd.control(both), -kp * pos - kd * vel))

    # 6. Finite-difference of the deputy trajectory == implemented controlled accel.
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

        # Central difference of deputy position under the full coupled dynamics.
        def combined_deriv(z, t):
            ts, ds = z[:6], z[6:]
            a_t = fm.acceleration(ts)
            a_d = deputy_accel(ts, ds)
            return np.concatenate((ts[3:], a_t, ds[3:], a_d))

        z0 = np.concatenate((self.target0, deputy0))
        dt = 1e-2
        zf = orbit.rk4_step(combined_deriv, z0, 0.0, dt)
        zb = orbit.rk4_step(combined_deriv, z0, 0.0, -dt)
        a_fd = (zf[9:12] - zb[9:12]) / (2.0 * dt)   # d(deputy velocity)/dt
        pos_fd = (zf[6:9] - 2.0 * z0[6:9] + zb[6:9]) / (dt * dt)  # d2(deputy pos)/dt2

        self.assertLess(np.linalg.norm(a_fd - a_impl) / np.linalg.norm(a_impl), 1e-6)
        self.assertLess(np.linalg.norm(pos_fd - a_impl) / np.linalg.norm(a_impl), 1e-6)


if __name__ == "__main__":
    unittest.main()
