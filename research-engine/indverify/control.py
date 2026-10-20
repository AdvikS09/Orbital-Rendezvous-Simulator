import math

import numpy as np


class PDCtrl:

    def __init__(self, n, wn_ratio, a_max):
        self.wn = float(wn_ratio) * float(n)
        self.kp = self.wn ** 2
        self.kd = 2.0 * self.wn
        self.a_max = None if a_max is None else float(a_max)

    def rawscalar(self, p0, p1, p2, q0, q1, q2):
        kp, kd = self.kp, self.kd
        return (-kp * p0 - kd * q0, -kp * p1 - kd * q1, -kp * p2 - kd * q2)

    def rawnormscalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.rawscalar(p0, p1, p2, q0, q1, q2)
        return math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)

    def applscalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.rawscalar(p0, p1, p2, q0, q1, q2)
        am = self.a_max
        if am is not None:
            nr = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if nr > am:
                s = am / nr
                return a0 * s, a1 * s, a2 * s
        return a0, a1, a2

    def raw(self, rho, rhodot):
        return -self.kp * np.asarray(rho, dtype=float) - self.kd * np.asarray(rhodot, dtype=float)

    def saturate(self, araw):
        araw = np.asarray(araw, dtype=float)
        if self.a_max is None:
            return araw.copy()
        nr = np.sqrt(np.sum(araw * araw, axis=0))
        over = nr > self.a_max
        safe = np.where(over, nr, 1.0)
        scale = np.where(over, self.a_max / safe, 1.0)
        return araw * scale

    def applied(self, rho, rhodot):
        return self.saturate(self.raw(rho, rhodot))

    def frozen(self, saturated):
        return FrozenModeCtrl(self, saturated)


class FrozenModeCtrl(PDCtrl):

    def __init__(self, base, saturated):
        self.wn, self.kp, self.kd, self.a_max = base.wn, base.kp, base.kd, base.a_max
        self.saturated = bool(saturated)

    def applscalar(self, p0, p1, p2, q0, q1, q2):
        a0, a1, a2 = self.rawscalar(p0, p1, p2, q0, q1, q2)
        if self.saturated and self.a_max is not None:
            nr = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if nr > 0.0:
                s = self.a_max / nr
                return a0 * s, a1 * s, a2 * s
        return a0, a1, a2

    def saturate(self, araw):
        araw = np.asarray(araw, dtype=float)
        if not self.saturated or self.a_max is None:
            return araw.copy()
        nr = np.sqrt(np.sum(araw * araw, axis=0))
        safe = np.where(nr > 0.0, nr, 1.0)
        return araw * np.where(nr > 0.0, self.a_max / safe, 1.0)

    def frozen(self, saturated):
        return FrozenModeCtrl(self, saturated)
