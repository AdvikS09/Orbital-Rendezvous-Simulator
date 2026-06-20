"""Tests for the PD rendezvous controller and APF obstacle avoidance."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import controllers, cw, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


class TestPDController(unittest.TestCase):
    def setUp(self):
        self.n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
        self.period = 2.0 * np.pi / self.n

    def test_pd_drives_state_to_target(self):
        controller = controllers.PDController.critically_damped(6.0 * self.n)
        x0 = np.array([100.0, -200.0, 0.0, 0.0])
        res = simulation.simulate_cw(
            x0, (0.0, self.period), self.period / 4000, self.n, controller=controller
        )
        final_pos = np.linalg.norm(res.states[-1, :2])
        final_vel = np.linalg.norm(res.states[-1, 2:])
        self.assertLess(final_pos, 0.1)
        self.assertLess(final_vel, 1e-3)

    def test_saturation_limits_acceleration(self):
        controller = controllers.PDController.critically_damped(6.0 * self.n, max_accel=0.01)
        u = controller.control(np.array([1.0e4, 1.0e4, 0.0, 0.0]))
        self.assertLessEqual(np.linalg.norm(u), 0.01 + 1e-12)

    def test_critically_damped_gains(self):
        kp, kd = controllers.critically_damped_gains(0.05)
        self.assertAlmostEqual(kp, 0.05**2)
        self.assertAlmostEqual(kd, 2.0 * 0.05)


class TestObstacleAvoidance(unittest.TestCase):
    def setUp(self):
        self.n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
        self.period = 2.0 * np.pi / self.n

    def test_apf_pushes_away_from_obstacle(self):
        obstacle = controllers.Obstacle(center=[0.0, 0.0], radius=10.0, influence=50.0)
        # A point just outside the keep-out radius should be pushed radially out.
        accel = controllers.apf_repulsion(np.array([12.0, 0.0]), [obstacle], eta=1.0)
        self.assertGreater(accel[0], 0.0)
        self.assertAlmostEqual(accel[1], 0.0)

    def test_avoidance_keeps_clearance(self):
        # Zone placed on the chaser's natural PD path (which bows to -radial).
        obstacle = controllers.Obstacle(center=[-15.0, 140.0], radius=30.0, influence=90.0)
        x0 = np.array([0.0, 300.0, 0.0, 0.0])

        # Plain PD flies straight through the zone (negative clearance) -- this
        # makes the avoidance test non-trivial.
        plain = controllers.PDController.critically_damped(6.0 * self.n, max_accel=0.05)
        res_plain = simulation.simulate_cw(
            x0, (0.0, self.period), self.period / 4000, self.n, controller=plain
        )
        plain_clear = min(obstacle.clearance(s[:2]) for s in res_plain.states)
        self.assertLess(plain_clear, 0.0)

        # With APF avoidance the chaser stays outside the keep-out radius and
        # still reaches the target.
        pd = controllers.PDController.critically_damped(6.0 * self.n, max_accel=0.05)
        avoider = controllers.ObstacleAvoidanceController(
            pd, [obstacle], eta=50.0, max_accel=0.05
        )
        res = simulation.simulate_cw(
            x0, (0.0, self.period), self.period / 4000, self.n, controller=avoider
        )
        min_clear = min(obstacle.clearance(s[:2]) for s in res.states)
        self.assertGreater(min_clear, 0.0)
        self.assertLess(np.linalg.norm(res.states[-1, :2]), 1.0)


if __name__ == "__main__":
    unittest.main()
