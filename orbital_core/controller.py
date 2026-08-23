"""PD rendezvous controller for relative motion (3D, LVLH frame).

The controller commands an acceleration ``u = [ax, ay, az]`` (force per unit
mass) to drive the relative state to a target (default: the LVLH origin with
zero relative velocity):

    u = -Kp (r - r_target) - Kd (v - v_target)

Gains ``Kp`` and ``Kd`` may be scalars, length-3 vectors (per-axis) or 3x3
matrices. An optional acceleration magnitude limit models thrust saturation.
The controller is callable so it can be passed directly to
:func:`src.cw.propagate_cw`.
"""

import numpy as np


def _as_gain_matrix(gain):
    """Coerce a scalar / length-3 vector / 3x3 matrix into a 3x3 gain matrix."""
    gain = np.asarray(gain, dtype=float)
    if gain.ndim == 0:
        return gain * np.eye(3)
    if gain.shape == (3,):
        return np.diag(gain)
    if gain.shape == (3, 3):
        return gain
    raise ValueError("gain must be a scalar, length-3 vector, or 3x3 matrix")


def critically_damped_gains(omega_n):
    """Return (kp, kd) for a critically damped 2nd-order response (unit mass).

    For ``x'' + kd x' + kp x = 0`` critical damping is ``kd = 2*sqrt(kp)``; with
    ``kp = omega_n^2`` this gives ``kd = 2*omega_n``.
    """
    return omega_n**2, 2.0 * omega_n


def saturate(u, max_accel):
    """Clip the magnitude of ``u`` to ``max_accel`` (no-op if max_accel is None)."""
    if max_accel is None:
        return u
    mag = np.linalg.norm(u)
    if mag > max_accel and mag > 0.0:
        return u * (max_accel / mag)
    return u


class PDController:
    """Proportional-Derivative controller for 3D relative-motion rendezvous."""

    def __init__(self, kp, kd, target=None, max_accel=None):
        self.Kp = _as_gain_matrix(kp)
        self.Kd = _as_gain_matrix(kd)
        self.target = np.zeros(6) if target is None else np.asarray(target, dtype=float)
        self.max_accel = max_accel

    @classmethod
    def critically_damped(cls, omega_n, target=None, max_accel=None):
        """Build a critically damped PD controller from a natural frequency."""
        kp, kd = critically_damped_gains(omega_n)
        return cls(kp, kd, target=target, max_accel=max_accel)

    def control(self, state, t=0.0):
        """Commanded control acceleration ``[ax, ay, az]`` for the given state."""
        state = np.asarray(state, dtype=float)
        pos_err = state[:3] - self.target[:3]
        vel_err = state[3:] - self.target[3:]
        u = -self.Kp @ pos_err - self.Kd @ vel_err
        return saturate(u, self.max_accel)

    __call__ = control


__all__ = [
    "critically_damped_gains",
    "saturate",
    "PDController",
]
