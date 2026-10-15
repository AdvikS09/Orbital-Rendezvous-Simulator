"""PD controller with direction-preserving saturation (SPEC section 4)."""

import math

import numpy as np


class PDController:
    """a_raw = -Kp rho - Kd rhodot, Kp = wn^2, Kd = 2 wn, wn = wn_ratio * n.

    If a_max is not None and |a_raw| > a_max the command is scaled to length
    a_max keeping its direction.
    """

    def __init__(self, n, wn_ratio, a_max):
        self.wn = float(wn_ratio) * float(n)
        self.kp = self.wn ** 2
        self.kd = 2.0 * self.wn
        self.a_max = None if a_max is None else float(a_max)

    # ---- fast scalar versions (used inside right-hand sides and events) ----
    def raw_scalar(self, p0, p1, p2, q0, q1, q2):
        kp, kd = self.kp, self.kd
        return (-kp * p0 - kd * q0, -kp * p1 - kd * q1, -kp * p2 - kd * q2)

    def raw_norm_scalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.raw_scalar(p0, p1, p2, q0, q1, q2)
        return math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)

    def applied_scalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.raw_scalar(p0, p1, p2, q0, q1, q2)
        am = self.a_max
        if am is not None:
            nr = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if nr > am:
                s = am / nr
                return a0 * s, a1 * s, a2 * s
        return a0, a1, a2

    # ---- vectorised versions (component-first arrays, shape (3, ...)) ----
    def raw(self, rho, rhodot):
        return -self.kp * np.asarray(rho, dtype=float) - self.kd * np.asarray(rhodot, dtype=float)

    def saturate(self, a_raw):
        a_raw = np.asarray(a_raw, dtype=float)
        if self.a_max is None:
            return a_raw.copy()
        nr = np.sqrt(np.sum(a_raw * a_raw, axis=0))
        over = nr > self.a_max
        safe = np.where(over, nr, 1.0)
        scale = np.where(over, self.a_max / safe, 1.0)
        return a_raw * scale

    def applied(self, rho, rhodot):
        return self.saturate(self.raw(rho, rhodot))

    def frozen(self, saturated):
        """Copy of this controller with the saturation mode frozen.

        Used only to locate a saturation switch (integrators.py): on each side
        of the switching surface the law is smooth, and continuing the current
        side's law a little past the surface gives a smooth vector field whose
        dense output can be root-searched without a kink inside the step.
        Before the switch it is identical to the true law.
        """
        return FrozenModeController(self, saturated)


class FrozenModeController(PDController):
    """PD law with the saturation decision fixed: ``saturated=True`` always
    scales a_raw to length a_max, ``saturated=False`` never scales."""

    def __init__(self, base, saturated):
        self.wn, self.kp, self.kd, self.a_max = base.wn, base.kp, base.kd, base.a_max
        self.saturated = bool(saturated)

    def applied_scalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.raw_scalar(p0, p1, p2, q0, q1, q2)
        if self.saturated and self.a_max is not None:
            nr = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if nr > 0.0:
                s = self.a_max / nr
                return a0 * s, a1 * s, a2 * s
        return a0, a1, a2

    def saturate(self, a_raw):
        a_raw = np.asarray(a_raw, dtype=float)
        if not self.saturated or self.a_max is None:
            return a_raw.copy()
        nr = np.sqrt(np.sum(a_raw * a_raw, axis=0))
        safe = np.where(nr > 0.0, nr, 1.0)
        return a_raw * np.where(nr > 0.0, self.a_max / safe, 1.0)

    def frozen(self, saturated):
        return FrozenModeController(self, saturated)
