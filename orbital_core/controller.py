import numpy as np


def _as_gain_matrix(gain):
    gain = np.asarray(gain, dtype=float)
    if gain.ndim == 0:
        return gain * np.eye(3)
    if gain.shape == (3,):
        return np.diag(gain)
    if gain.shape == (3, 3):
        return gain
    raise ValueError("gain must be a scalar, length-3 vector, or 3x3 matrix")


def critically_damped_gains(omega_n):
    return omega_n**2, 2.0 * omega_n


def saturate(u, max_accel):
    if max_accel is None:
        return u
    mag = np.linalg.norm(u)
    if mag > max_accel and mag > 0.0:
        return u * (max_accel / mag)
    return u


class PDController:
    def __init__(self, kp, kd, target=None, max_accel=None):
        self.Kp = _as_gain_matrix(kp)
        self.Kd = _as_gain_matrix(kd)
        self.target = np.zeros(6) if target is None else np.asarray(target, dtype=float)
        self.max_accel = max_accel

    @classmethod
    def critically_damped(cls, omega_n, target=None, max_accel=None):
        kp, kd = critically_damped_gains(omega_n)
        return cls(kp, kd, target=target, max_accel=max_accel)

    def control(self, state, t=0.0):
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
