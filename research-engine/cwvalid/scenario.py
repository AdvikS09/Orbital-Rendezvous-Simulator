import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import DEFALT, MUEARTH, REARTH

AXES = {
    "+R": (1.0, 0.0, 0.0), "-R": (-1.0, 0.0, 0.0),
    "+V": (0.0, 1.0, 0.0), "-V": (0.0, -1.0, 0.0),
    "+H": (0.0, 0.0, 1.0), "-H": (0.0, 0.0, -1.0),
}
AXISNAMES = {"+R": "+radial", "-R": "-radial", "+V": "+along-track", "-V": "-along-track",
             "+H": "+cross-track", "-H": "-cross-track"}


def velmatch(x0, n):
    return -2.0 * n * x0


def velradial(x0, R, mu=MUEARTH):
    rc = R + x0
    if not rc > 0:
        raise ValueError("exact energy matching undefined for this x0 (need -R < x0 < R)")
    arg = mu * (2.0 / rc - 1.0 / R)
    if not arg > 0:
        raise ValueError("exact energy matching undefined for this x0 (need -R < x0 < R)")
    n = math.sqrt(mu / R**3)
    return math.sqrt(arg) - n * rc


@dataclass
class Scenario:
    family: str
    rel0: tuple
    wn_ratio: float
    a_max: float | None
    horizon_orbits: float = 1.0
    altitude: float = DEFALT
    meta: dict = field(default_factory=dict)
    sid: str = ""

    def __post_init__(self):
        self.rel0 = tuple(float(v) for v in self.rel0)
        if not self.sid:
            self.sid = makeid(self.family, self.physkey())

    def physkey(self):
        return {
            "rel0": [float(f"{v:.12g}") for v in self.rel0],
            "wn_ratio": float(f"{self.wn_ratio:.12g}"),
            "a_max": None if self.a_max is None else float(f"{self.a_max:.12g}"),
            "horizon_orbits": float(self.horizon_orbits),
            "altitude": float(self.altitude),
            "variant": self.meta.get("variant", ""),
        }

    @property
    def R(self):
        return REARTH + self.altitude

    @property
    def n(self):
        return math.sqrt(MUEARTH / self.R**3)

    @property
    def horizon(self):
        return self.horizon_orbits * 2.0 * math.pi / self.n

    @property
    def separation(self):
        return float(np.linalg.norm(self.rel0[:3]))

    def to_dict(self):
        return {"sid": self.sid, "family": self.family, "rel0": list(self.rel0),
                "wn_ratio": self.wn_ratio, "a_max": self.a_max,
                "horizon_orbits": self.horizon_orbits, "altitude": self.altitude,
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        return cls(family=d["family"], rel0=tuple(d["rel0"]), wn_ratio=d["wn_ratio"],
                   a_max=d["a_max"], horizon_orbits=d.get("horizon_orbits", 1.0),
                   altitude=d.get("altitude", DEFALT), meta=d.get("meta", {}),
                   sid=d.get("sid", ""))


def makeid(family, key):
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{family}-{h}"


def axisscen(family, axis, sep, wn_ratio, a_max, horizon_orbits=1.0, velocity="zero",
             altitude=DEFALT, extra_meta=None):
    u = np.array(AXES[axis])
    pos = u * sep
    R = REARTH + altitude
    n = math.sqrt(MUEARTH / R**3)
    vel = np.zeros(3)
    if velocity == "zero":
        pass
    elif velocity == "em1":
        vel[1] = velmatch(pos[0], n)
    elif velocity == "em_exact":
        if axis not in ("+R", "-R"):
            raise ValueError("exact energy matching implemented for pure radial offsets only")
        vel[1] = velradial(pos[0], R)
    else:
        raise ValueError(velocity)
    meta = {"axis": axis, "sep_m": float(sep), "velocity": velocity}
    if velocity != "zero":
        meta["variant"] = velocity
    if extra_meta:
        meta.update(extra_meta)
    return Scenario(family, tuple(np.concatenate((pos, vel))), wn_ratio, a_max,
                    horizon_orbits, altitude, meta)
