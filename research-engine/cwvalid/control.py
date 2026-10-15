"""PD law with direction-preserving saturation, and command sources.

The law is the manuscript's (Section 2.5)::

    a_raw = -Kp rho - Kd rho_dot          (desired state = LVLH origin at rest)
    a     = a_raw                         if |a_raw| <= a_max
          = a_raw * a_max / |a_raw|       otherwise (direction preserved)

with Kp = omega_n^2, Kd = 2 omega_n (scalar, isotropic). ``a_max = None``
gives the *unrestricted* diagnostic law (not a physically available actuator).

Command sources (how the command reaches the dynamics):

* ``continuous`` - the law is evaluated from the branch's own current state at
  every right-hand-side evaluation (every Runge-Kutta stage).
* ``sampled``    - the law is evaluated from the state at update instants
  t_k = k*Ts and held constant (zero-order hold) until t_{k+1}; integration is
  restarted at every update instant (never tied to adaptive solver steps).
* ``prescribed`` - an externally supplied applied-acceleration history u(t)
  (e.g. the CW closed-loop command) replayed open-loop ("common-command
  propagation"; a diagnostic, not feedback).
* ``none``       - free drift.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PDLaw:
    kp: float
    kd: float
    a_max: float | None = None  # None -> unrestricted (diagnostic)

    @classmethod
    def from_bandwidth(cls, omega_n, a_max=None):
        """Critical-damping rule Kp = wn^2, Kd = 2 wn (manuscript eq.)."""
        return cls(omega_n * omega_n, 2.0 * omega_n, a_max)

    def raw(self, x, y, z, vx, vy, vz):
        kp, kd = self.kp, self.kd
        return (-kp * x - kd * vx, -kp * y - kd * vy, -kp * z - kd * vz)

    def command(self, x, y, z, vx, vy, vz):
        """Applied (saturated) command, scalar inputs -> 3-tuple."""
        kp, kd = self.kp, self.kd
        ax = -kp * x - kd * vx
        ay = -kp * y - kd * vy
        az = -kp * z - kd * vz
        am = self.a_max
        if am is not None:
            m = math.sqrt(ax * ax + ay * ay + az * az)
            if m > am:
                s = am / m
                return ax * s, ay * s, az * s
        return ax, ay, az

    # vectorised helpers for post-processing -----------------------------
    def raw_vec(self, states):
        s = np.asarray(states, dtype=float)
        return -self.kp * s[:, :3] - self.kd * s[:, 3:6]

    def applied_vec(self, states):
        raw = self.raw_vec(states)
        if self.a_max is None:
            return raw
        m = np.linalg.norm(raw, axis=1)
        scale = np.where(m > self.a_max, self.a_max / np.where(m > 0, m, 1.0), 1.0)
        return raw * scale[:, None]

    @property
    def saturation_distance(self):
        """Zero-velocity position error at which the command first saturates: a_max/Kp."""
        return None if self.a_max is None else self.a_max / self.kp

    def as_dict(self):
        return {"kp": self.kp, "kd": self.kd, "a_max": self.a_max}


def saturate_scalar(ax, ay, az, a_max):
    if a_max is None:
        return ax, ay, az
    m = math.sqrt(ax * ax + ay * ay + az * az)
    if m > a_max:
        s = a_max / m
        return ax * s, ay * s, az * s
    return ax, ay, az


@dataclass
class CommandSpec:
    """How the command is generated for a branch.

    mode: 'continuous' | 'sampled' | 'prescribed' | 'none'
    law:  PDLaw (continuous / sampled)
    sample_period: Ts [s] for 'sampled'
    prescribed: callable t -> (ax, ay, az) for 'prescribed' (scalar t), and
                ``prescribed_vec`` callable (N,) -> (N,3) for post-processing.
    """

    mode: str = "continuous"
    law: PDLaw | None = None
    sample_period: float | None = None
    prescribed: object = None
    prescribed_vec: object = None
    label: str = field(default="")
    breakpoints: tuple = ()   # extra integration restart times (e.g. kinks of a replayed command)

    def describe(self):
        d = {"mode": self.mode}
        if self.law is not None:
            d.update(self.law.as_dict())
        if self.mode == "sampled":
            d["sample_period_s"] = self.sample_period
        if self.label:
            d["label"] = self.label
        return d


__all__ = ["PDLaw", "saturate_scalar", "CommandSpec"]
