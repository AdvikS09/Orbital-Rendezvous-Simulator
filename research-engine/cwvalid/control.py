import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PDLaw:
    kp: float
    kd: float
    a_max: float | None = None

    @classmethod
    def bandw(cls, omegan, a_max=None):
        return cls(omegan * omegan, 2.0 * omegan, a_max)

    def raw(self, x, y, z, vx, vy, vz):
        kp, kd = self.kp, self.kd
        return (-kp * x - kd * vx, -kp * y - kd * vy, -kp * z - kd * vz)

    def command(self, x, y, z, vx, vy, vz):
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

    def rawvec(self, states):
        s = np.asarray(states, dtype=float)
        return -self.kp * s[:, :3] - self.kd * s[:, 3:6]

    def applvec(self, states):
        raw = self.rawvec(states)
        if self.a_max is None:
            return raw
        m = np.linalg.norm(raw, axis=1)
        scale = np.where(m > self.a_max, self.a_max / np.where(m > 0, m, 1.0), 1.0)
        return raw * scale[:, None]

    @property
    def saturdist(self):
        return None if self.a_max is None else self.a_max / self.kp

    def asdict2(self):
        return {"kp": self.kp, "kd": self.kd, "a_max": self.a_max}


def satscalar(ax, ay, az, a_max):
    if a_max is None:
        return ax, ay, az
    m = math.sqrt(ax * ax + ay * ay + az * az)
    if m > a_max:
        s = a_max / m
        return ax * s, ay * s, az * s
    return ax, ay, az


@dataclass
class CmdSpec:

    mode: str = "continuous"
    law: PDLaw | None = None
    sample_period: float | None = None
    prescribed: object = None
    prescribed_vec: object = None
    label: str = field(default="")
    breakpoints: tuple = ()

    def describe(self):
        d = {"mode": self.mode}
        if self.law is not None:
            d.update(self.law.asdict2())
        if self.mode == "sampled":
            d["sample_period_s"] = self.sample_period
        if self.label:
            d["label"] = self.label
        return d
