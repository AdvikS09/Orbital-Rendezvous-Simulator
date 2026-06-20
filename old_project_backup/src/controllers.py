"""Feedback controllers for relative-motion rendezvous.

Controllers return a commanded acceleration ``[ax, ay]`` (force per unit mass)
expressed in the LVLH frame. They expose both a ``control(state, t)`` method and
a ``__call__`` alias so they can be passed directly as the ``control`` argument
of :func:`src.cw.cw_derivatives`.
"""

import numpy as np


def _as_gain_matrix(gain):
    """Coerce a scalar, length-2 vector, or 2x2 matrix into a 2x2 gain matrix."""
    gain = np.asarray(gain, dtype=float)
    if gain.ndim == 0:
        return gain * np.eye(2)
    if gain.shape == (2,):
        return np.diag(gain)
    if gain.shape == (2, 2):
        return gain
    raise ValueError("gain must be a scalar, length-2 vector, or 2x2 matrix")


def critically_damped_gains(omega_n):
    """Return (kp, kd) for a critically damped 2nd-order response (unit mass).

    For ``x'' + kd x' + kp x = 0`` critical damping is ``kd = 2*sqrt(kp)``; with
    ``kp = omega_n^2`` this gives ``kd = 2*omega_n``.
    """
    return omega_n**2, 2.0 * omega_n


def saturate_acceleration(u, max_accel):
    """Clip the magnitude of acceleration ``u`` to ``max_accel`` (if not None)."""
    if max_accel is None:
        return u
    mag = np.linalg.norm(u)
    if mag > max_accel and mag > 0.0:
        return u * (max_accel / mag)
    return u


class PDController:
    """Proportional-Derivative controller driving the state to a target.

    u = -Kp @ (pos - pos_target) - Kd @ (vel - vel_target)

    Gains may be scalars, length-2 vectors (per-axis), or 2x2 matrices.
    """

    def __init__(self, kp, kd, target=None, max_accel=None):
        self.Kp = _as_gain_matrix(kp)
        self.Kd = _as_gain_matrix(kd)
        self.target = np.zeros(4) if target is None else np.asarray(target, dtype=float)
        self.max_accel = max_accel

    @classmethod
    def critically_damped(cls, omega_n, target=None, max_accel=None):
        """Build a critically damped PD controller from a natural frequency."""
        kp, kd = critically_damped_gains(omega_n)
        return cls(kp, kd, target=target, max_accel=max_accel)

    def control(self, state, t=0.0):
        state = np.asarray(state, dtype=float)
        pos_err = state[:2] - self.target[:2]
        vel_err = state[2:] - self.target[2:]
        u = -self.Kp @ pos_err - self.Kd @ vel_err
        return saturate_acceleration(u, self.max_accel)

    __call__ = control


class Obstacle:
    """Circular keep-out zone in the LVLH frame.

    ``center``    : 2D centre [m]
    ``radius``    : hard keep-out radius [m]
    ``influence`` : range beyond the radius over which repulsion acts [m]
    """

    def __init__(self, center, radius, influence):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.influence = float(influence)

    def clearance(self, position):
        """Signed distance from ``position`` to the keep-out boundary [m]."""
        return np.linalg.norm(np.asarray(position, float) - self.center) - self.radius


def apf_repulsion(position, obstacles, eta=1.0, inside_gain=10.0):
    """Artificial-potential-field repulsive acceleration from keep-out zones.

    Uses the classic FIRAS potential gradient outside the boundary, plus a
    capped outward push if the position has penetrated a zone.
    """
    position = np.asarray(position, dtype=float)
    accel = np.zeros(2)
    for ob in obstacles:
        offset = position - ob.center
        dist = np.linalg.norm(offset)
        if dist == 0.0:
            continue
        direction = offset / dist
        edge = dist - ob.radius  # distance to the keep-out boundary
        if edge <= 0.0:
            # Inside the zone: strong, bounded outward push.
            accel += inside_gain * eta * direction
        elif edge < ob.influence:
            # FIRAS repulsive gradient, vanishing at the influence radius.
            mag = eta * (1.0 / edge - 1.0 / ob.influence) / (edge * edge)
            accel += mag * direction
    return accel


class ObstacleAvoidanceController:
    """PD attraction to the target plus APF repulsion from keep-out zones."""

    def __init__(self, pd_controller, obstacles, eta=1.0, max_accel=None):
        self.pd = pd_controller
        self.obstacles = list(obstacles)
        self.eta = eta
        # If no explicit cap is given, inherit the PD controller's cap.
        self.max_accel = max_accel if max_accel is not None else pd_controller.max_accel

    def control(self, state, t=0.0):
        state = np.asarray(state, dtype=float)
        u_attract = self.pd.control(state, t)
        u_repel = apf_repulsion(state[:2], self.obstacles, eta=self.eta)
        return saturate_acceleration(u_attract + u_repel, self.max_accel)

    __call__ = control


__all__ = [
    "critically_damped_gains",
    "saturate_acceleration",
    "PDController",
    "Obstacle",
    "apf_repulsion",
    "ObstacleAvoidanceController",
]
