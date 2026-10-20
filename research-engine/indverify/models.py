import copy
import math

import numpy as np

from .constants import (ALTFLOOR, DIVLIMIT, MU,
                        POSINERT, POSREL, R_E,
                        VELINERT, VELREL)
from .frames import inertrel, relinert

BRANCHES = ("ref", "cw", "so")


class RelModel:

    name = "?"
    nstate = 6
    hasalt = False

    def __init__(self, orbit, ctrl, rel0):
        self.orbit = orbit
        self.ctrl = ctrl
        self.n = orbit.n
        self.R = orbit.R
        self.y0 = np.array(rel0, dtype=float).reshape(6)
        n = self.n
        self._3n2 = 3.0 * n * n
        self._2n = 2.0 * n
        self._n2 = n * n

    @property
    def hassat(self):
        return self.ctrl.a_max is not None

    def atol(self, rtol):
        p, v = rtol * POSREL, rtol * VELREL
        return np.array([p, p, p, v, v, v])

    def relscalar(self, y):
        return tuple(np.asarray(y, dtype=float).tolist())

    def relmany(self, Y):
        Y = np.asarray(Y, dtype=float)
        return Y[0:3], Y[3:6]

    def gsat(self, t, y):
        return self.ctrl.rawnormscalar(*self.relscalar(y)) - self.ctrl.a_max

    def alt(self, t, y):
        raise NotImplementedError

    def frozen(self, saturated):
        m = copy.copy(self)
        m.ctrl = self.ctrl.frozen(saturated)
        return m

    def badstate(self, y):
        m = np.abs(np.asarray(y, dtype=float)).max()
        if m <= DIVLIMIT:
            return None
        if not np.isfinite(m):
            return "non-finite state"
        return "state magnitude exceeded %.0e (divergence guard)" % DIVLIMIT


class CWModel(RelModel):

    name = "cw"

    def rhs(self, t, y):
        x, yy, z, vx, vy, vz = y.tolist()
        a0, a1, a2 = self.ctrl.applscalar(x, yy, z, vx, vy, vz)
        return np.array([vx, vy, vz,
                         self._3n2 * x + self._2n * vy + a0,
                         -self._2n * vx + a1,
                         -self._n2 * z + a2])


class SOModel(RelModel):

    name = "so"

    def __init__(self, orbit, ctrl, rel0):
        super().__init__(orbit, ctrl, rel0)
        self.chalf = self._3n2 / (2.0 * self.R)
        self.cfull = self._3n2 / self.R

    def rhs(self, t, y):
        x, yy, z, vx, vy, vz = y.tolist()
        a0, a1, a2 = self.ctrl.applscalar(x, yy, z, vx, vy, vz)
        ch, cf = self.chalf, self.cfull
        return np.array([vx, vy, vz,
                         self._3n2 * x + self._2n * vy
                         - ch * (2.0 * x * x - yy * yy - z * z) + a0,
                         -self._2n * vx + cf * x * yy + a1,
                         -self._n2 * z + cf * x * z + a2])


class RefModel:

    name = "ref"
    nstate = 12
    hasalt = True

    def __init__(self, orbit, ctrl, rel0):
        self.orbit = orbit
        self.ctrl = ctrl
        rel0 = np.array(rel0, dtype=float).reshape(6)
        rt, vt = orbit.initstate()
        rc, vc = relinert(rt, vt, rel0[:3], rel0[3:])
        self.y0 = np.concatenate([rt, vt, rc, vc])

    @property
    def hassat(self):
        return self.ctrl.a_max is not None

    def atol(self, rtol):
        p, v = rtol * POSINERT, rtol * VELINERT
        return np.array([p, p, p, v, v, v, p, p, p, v, v, v])

    def rhs(self, t, y):
        (rtx, rty, rtz, vtx, vty, vtz,
         rcx, rcy, rcz, vcx, vcy, vcz) = y.tolist()
        rt2 = rtx * rtx + rty * rty + rtz * rtz
        rt = math.sqrt(rt2)
        kt = -MU / (rt2 * rt)
        xx, xy, xz = rtx / rt, rty / rt, rtz / rt
        hx = rty * vtz - rtz * vty
        hy = rtz * vtx - rtx * vtz
        hz = rtx * vty - rty * vtx
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        zx, zy, zz = hx / hn, hy / hn, hz / hn
        yx = zy * xz - zz * xy
        yy_ = zz * xx - zx * xz
        yz = zx * xy - zy * xx
        w = hn / rt2
        dx, dy, dz = rcx - rtx, rcy - rty, rcz - rtz
        dvx, dvy, dvz = vcx - vtx, vcy - vty, vcz - vtz
        p0 = xx * dx + xy * dy + xz * dz
        p1 = yx * dx + yy_ * dy + yz * dz
        p2 = zx * dx + zy * dy + zz * dz
        q0 = xx * dvx + xy * dvy + xz * dvz + w * p1
        q1 = yx * dvx + yy_ * dvy + yz * dvz - w * p0
        q2 = zx * dvx + zy * dvy + zz * dvz
        a0, a1, a2 = self.ctrl.applscalar(p0, p1, p2, q0, q1, q2)
        rc2 = rcx * rcx + rcy * rcy + rcz * rcz
        rc = math.sqrt(rc2)
        kc = -MU / (rc2 * rc)
        return np.array([vtx, vty, vtz,
                         kt * rtx, kt * rty, kt * rtz,
                         vcx, vcy, vcz,
                         kc * rcx + a0 * xx + a1 * yx + a2 * zx,
                         kc * rcy + a0 * xy + a1 * yy_ + a2 * zy,
                         kc * rcz + a0 * xz + a1 * yz + a2 * zz])

    def relscalar(self, y):
        (rtx, rty, rtz, vtx, vty, vtz,
         rcx, rcy, rcz, vcx, vcy, vcz) = np.asarray(y, dtype=float).tolist()
        rt2 = rtx * rtx + rty * rty + rtz * rtz
        rt = math.sqrt(rt2)
        xx, xy, xz = rtx / rt, rty / rt, rtz / rt
        hx = rty * vtz - rtz * vty
        hy = rtz * vtx - rtx * vtz
        hz = rtx * vty - rty * vtx
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        zx, zy, zz = hx / hn, hy / hn, hz / hn
        yx = zy * xz - zz * xy
        yy_ = zz * xx - zx * xz
        yz = zx * xy - zy * xx
        w = hn / rt2
        dx, dy, dz = rcx - rtx, rcy - rty, rcz - rtz
        dvx, dvy, dvz = vcx - vtx, vcy - vty, vcz - vtz
        p0 = xx * dx + xy * dy + xz * dz
        p1 = yx * dx + yy_ * dy + yz * dz
        p2 = zx * dx + zy * dy + zz * dz
        q0 = xx * dvx + xy * dvy + xz * dvz + w * p1
        q1 = yx * dvx + yy_ * dvy + yz * dvz - w * p0
        q2 = zx * dvx + zy * dvy + zz * dvz
        return p0, p1, p2, q0, q1, q2

    def relmany(self, Y):
        Y = np.asarray(Y, dtype=float)
        return inertrel(Y[0:3], Y[3:6], Y[6:9], Y[9:12])

    def gsat(self, t, y):
        return self.ctrl.rawnormscalar(*self.relscalar(y)) - self.ctrl.a_max

    def alt(self, t, y):
        rcx, rcy, rcz = float(y[6]), float(y[7]), float(y[8])
        return math.sqrt(rcx * rcx + rcy * rcy + rcz * rcz) - R_E - ALTFLOOR

    def frozen(self, saturated):
        m = copy.copy(self)
        m.ctrl = self.ctrl.frozen(saturated)
        return m

    def badstate(self, y):
        if np.isfinite(np.asarray(y, dtype=float)).all():
            return None
        return "non-finite state"


MODELCLASS = {"ref": RefModel, "cw": CWModel, "so": SOModel}


def makemodel(branch, orbit, ctrl, rel0):
    try:
        cls = MODELCLASS[branch]
    except KeyError:
        raise ValueError("unknown branch %r (expected one of %s)" % (branch, BRANCHES))
    return cls(orbit, ctrl, rel0)
