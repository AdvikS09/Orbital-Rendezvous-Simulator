"""Branch dynamics (SPEC sections 4 and 5).

Each model provides
  * ``y0``            initial state
  * ``rhs(t, y)``     right-hand side (pure-Python arithmetic for speed)
  * ``rel_scalar(y)`` (rho, rhodot) of one state as six floats
  * ``rel_many(Y)``   (rho, rhodot) of many states, Y has shape (n_state, m)
  * ``gsat(t, y)``    saturation switch function |a_raw| - a_max
  * ``alt(t, y)``     (reference only) chaser altitude above the 100 km floor
  * ``atol(rtol)``    per-component absolute tolerances (SPEC section 7)
  * ``bad_state(y)``  None, or a reason string if the state is unusable
  * ``frozen(sat)``   copy whose controller has its saturation mode frozen
                      (used only to locate saturation switches, integrators.py)
"""

import copy
import math

import numpy as np

from .constants import (ALTITUDE_FLOOR, DIVERGENCE_LIMIT, MU,
                        POS_ATOL_SCALE_INERTIAL, POS_ATOL_SCALE_REL, R_E,
                        VEL_ATOL_SCALE_INERTIAL, VEL_ATOL_SCALE_REL)
from .frames import inertial_to_rel, rel_to_inertial

BRANCHES = ("ref", "cw", "so")


class _RelativeModel:
    """Common part of the CW and quadratic branches: state (x, y, z, xd, yd, zd)."""

    name = "?"
    n_state = 6
    has_alt = False

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
    def has_sat(self):
        return self.ctrl.a_max is not None

    def atol(self, rtol):
        p, v = rtol * POS_ATOL_SCALE_REL, rtol * VEL_ATOL_SCALE_REL
        return np.array([p, p, p, v, v, v])

    def rel_scalar(self, y):
        return tuple(np.asarray(y, dtype=float).tolist())

    def rel_many(self, Y):
        Y = np.asarray(Y, dtype=float)
        return Y[0:3], Y[3:6]

    def gsat(self, t, y):
        return self.ctrl.raw_norm_scalar(*self.rel_scalar(y)) - self.ctrl.a_max

    def alt(self, t, y):  # pragma: no cover - not used for relative branches
        raise NotImplementedError

    def frozen(self, saturated):
        m = copy.copy(self)
        m.ctrl = self.ctrl.frozen(saturated)
        return m

    def bad_state(self, y):
        m = np.abs(np.asarray(y, dtype=float)).max()
        if m <= DIVERGENCE_LIMIT:          # False for NaN as well
            return None
        if not np.isfinite(m):
            return "non-finite state"
        return "state magnitude exceeded %.0e (divergence guard)" % DIVERGENCE_LIMIT


class CWModel(_RelativeModel):
    """Clohessy-Wiltshire branch."""

    name = "cw"

    def rhs(self, t, y):
        x, yy, z, vx, vy, vz = y.tolist()
        a0, a1, a2 = self.ctrl.applied_scalar(x, yy, z, vx, vy, vz)
        return np.array([vx, vy, vz,
                         self._3n2 * x + self._2n * vy + a0,
                         -self._2n * vx + a1,
                         -self._n2 * z + a2])


class SOModel(_RelativeModel):
    """Quadratic (second-order) relative-motion branch."""

    name = "so"

    def __init__(self, orbit, ctrl, rel0):
        super().__init__(orbit, ctrl, rel0)
        self._c_half = self._3n2 / (2.0 * self.R)   # 3 n^2 / (2 R)
        self._c_full = self._3n2 / self.R           # 3 n^2 / R

    def rhs(self, t, y):
        x, yy, z, vx, vy, vz = y.tolist()
        a0, a1, a2 = self.ctrl.applied_scalar(x, yy, z, vx, vy, vz)
        ch, cf = self._c_half, self._c_full
        return np.array([vx, vy, vz,
                         self._3n2 * x + self._2n * vy
                         - ch * (2.0 * x * x - yy * yy - z * z) + a0,
                         -self._2n * vx + cf * x * yy + a1,
                         -self._n2 * z + cf * x * z + a2])


class RefModel:
    """Inertial two-spacecraft reference: state (r_t, v_t, r_c, v_c)."""

    name = "ref"
    n_state = 12
    has_alt = True

    def __init__(self, orbit, ctrl, rel0):
        self.orbit = orbit
        self.ctrl = ctrl
        rel0 = np.array(rel0, dtype=float).reshape(6)
        r_t, v_t = orbit.initial_state()
        r_c, v_c = rel_to_inertial(r_t, v_t, rel0[:3], rel0[3:])
        self.y0 = np.concatenate([r_t, v_t, r_c, v_c])

    @property
    def has_sat(self):
        return self.ctrl.a_max is not None

    def atol(self, rtol):
        p, v = rtol * POS_ATOL_SCALE_INERTIAL, rtol * VEL_ATOL_SCALE_INERTIAL
        return np.array([p, p, p, v, v, v, p, p, p, v, v, v])

    def rhs(self, t, y):
        (rtx, rty, rtz, vtx, vty, vtz,
         rcx, rcy, rcz, vcx, vcy, vcz) = y.tolist()
        rt2 = rtx * rtx + rty * rty + rtz * rtz
        rt = math.sqrt(rt2)
        kt = -MU / (rt2 * rt)
        # LVLH axes of the current target state
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
        # relative state (section 3)
        dx, dy, dz = rcx - rtx, rcy - rty, rcz - rtz
        dvx, dvy, dvz = vcx - vtx, vcy - vty, vcz - vtz
        p0 = xx * dx + xy * dy + xz * dz
        p1 = yx * dx + yy_ * dy + yz * dz
        p2 = zx * dx + zy * dy + zz * dz
        q0 = xx * dvx + xy * dvy + xz * dvz + w * p1
        q1 = yx * dvx + yy_ * dvy + yz * dvz - w * p0
        q2 = zx * dvx + zy * dvy + zz * dvz
        a0, a1, a2 = self.ctrl.applied_scalar(p0, p1, p2, q0, q1, q2)
        rc2 = rcx * rcx + rcy * rcy + rcz * rcz
        rc = math.sqrt(rc2)
        kc = -MU / (rc2 * rc)
        # chaser acceleration = gravity + C^T a
        return np.array([vtx, vty, vtz,
                         kt * rtx, kt * rty, kt * rtz,
                         vcx, vcy, vcz,
                         kc * rcx + a0 * xx + a1 * yx + a2 * zx,
                         kc * rcy + a0 * xy + a1 * yy_ + a2 * zy,
                         kc * rcz + a0 * xz + a1 * yz + a2 * zz])

    def rel_scalar(self, y):
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

    def rel_many(self, Y):
        Y = np.asarray(Y, dtype=float)
        return inertial_to_rel(Y[0:3], Y[3:6], Y[6:9], Y[9:12])

    def gsat(self, t, y):
        return self.ctrl.raw_norm_scalar(*self.rel_scalar(y)) - self.ctrl.a_max

    def alt(self, t, y):
        """|r_c| - R_E - 100 km (negative = below the physical-domain floor)."""
        rcx, rcy, rcz = float(y[6]), float(y[7]), float(y[8])
        return math.sqrt(rcx * rcx + rcy * rcy + rcz * rcz) - R_E - ALTITUDE_FLOOR

    def frozen(self, saturated):
        m = copy.copy(self)
        m.ctrl = self.ctrl.frozen(saturated)
        return m

    def bad_state(self, y):
        if np.isfinite(np.asarray(y, dtype=float)).all():
            return None
        return "non-finite state"


_MODEL_CLASSES = {"ref": RefModel, "cw": CWModel, "so": SOModel}


def make_model(branch, orbit, ctrl, rel0):
    try:
        cls = _MODEL_CLASSES[branch]
    except KeyError:
        raise ValueError("unknown branch %r (expected one of %s)" % (branch, BRANCHES))
    return cls(orbit, ctrl, rel0)
