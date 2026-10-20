import numpy as np


def gainmat(gain):
    gain = np.asarray(gain, dtype=float)
    if gain.ndim == 0:
        return gain * np.eye(3)
    if gain.shape == (3,):
        return np.diag(gain)
    if gain.shape == (3, 3):
        return gain
    raise ValueError("gain must be a scalar, length-3 vector, or 3x3 matrix")


def critdampedgains(omegan):
    return omegan**2, 2.0 * omegan


def saturate(u, max_accel):
    if max_accel is None:
        return u
    mag = np.linalg.norm(u)
    if mag > max_accel and mag > 0.0:
        return u * (max_accel / mag)
    return u


class PDCtrl:

    def __init__(self, kp, kd, target=None, max_accel=None):
        self.Kp = gainmat(kp)
        self.Kd = gainmat(kd)
        self.target = np.zeros(6) if target is None else np.asarray(target, dtype=float)
        self.max_accel = max_accel

    @classmethod
    def critdamped(cls, omegan, target=None, max_accel=None):
        kp, kd = critdampedgains(omegan)
        return cls(kp, kd, target=target, max_accel=max_accel)

    def control(self, state, t=0.0):
        state = np.asarray(state, dtype=float)
        poserr = state[:3] - self.target[:3]
        velerr = state[3:] - self.target[3:]
        u = -self.Kp @ poserr - self.Kd @ velerr
        return saturate(u, self.max_accel)

    __call__ = control
