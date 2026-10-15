"""Scenario specification, deterministic IDs and initial-state constructors."""

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_ALTITUDE, MU_EARTH, R_EARTH

AXES = {
    "+R": (1.0, 0.0, 0.0), "-R": (-1.0, 0.0, 0.0),
    "+V": (0.0, 1.0, 0.0), "-V": (0.0, -1.0, 0.0),
    "+H": (0.0, 0.0, 1.0), "-H": (0.0, 0.0, -1.0),
}
AXIS_NAMES = {"+R": "+radial", "-R": "-radial", "+V": "+along-track", "-V": "-along-track",
              "+H": "+cross-track", "-H": "-cross-track"}


def vel_first_order_energy_matched(x0, n):
    """CW bounded-motion (first-order energy/SMA matching) along-track rate: -2 n x0."""
    return -2.0 * n * x0


def vel_exact_energy_matched_radial(x0, R, mu=MU_EARTH):
    """Exact equal-specific-energy along-track rate for a pure radial offset.

    Chaser at radius R + x0 with zero radial and cross-track relative velocity;
    its inertial speed is purely tangential, v = sqrt(mu (2/(R+x0) - 1/R)),
    so its specific energy (and semi-major axis) equals the circular target's.
    In the rotating LVLH frame the transport velocity at that point is
    n (R + x0), hence ydot0 = v - n (R + x0) (prograde branch).
    Domain: -R < x0 < R. First-order expansion: -2 n x0 + n x0^2/(2R) + O(x0^3/R^2).
    """
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
    altitude: float = DEFAULT_ALTITUDE
    meta: dict = field(default_factory=dict)
    sid: str = ""

    def __post_init__(self):
        self.rel0 = tuple(float(v) for v in self.rel0)
        if not self.sid:
            self.sid = make_id(self.family, self.physics_key())

    def physics_key(self):
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
        return R_EARTH + self.altitude

    @property
    def n(self):
        return math.sqrt(MU_EARTH / self.R**3)

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
                   altitude=d.get("altitude", DEFAULT_ALTITUDE), meta=d.get("meta", {}),
                   sid=d.get("sid", ""))


def make_id(family, key):
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{family}-{h}"


def axis_scenario(family, axis, sep, wn_ratio, a_max, horizon_orbits=1.0, velocity="zero",
                  altitude=DEFAULT_ALTITUDE, extra_meta=None):
    """Pure-axis offset with a named initial-velocity rule."""
    u = np.array(AXES[axis])
    pos = u * sep
    R = R_EARTH + altitude
    n = math.sqrt(MU_EARTH / R**3)
    vel = np.zeros(3)
    if velocity == "zero":
        pass
    elif velocity == "em1":
        vel[1] = vel_first_order_energy_matched(pos[0], n)
    elif velocity == "em_exact":
        if axis not in ("+R", "-R"):
            raise ValueError("exact energy matching implemented for pure radial offsets only")
        vel[1] = vel_exact_energy_matched_radial(pos[0], R)
    else:
        raise ValueError(velocity)
    meta = {"axis": axis, "sep_m": float(sep), "velocity": velocity}
    if velocity != "zero":
        meta["variant"] = velocity
    if extra_meta:
        meta.update(extra_meta)
    return Scenario(family, tuple(np.concatenate((pos, vel))), wn_ratio, a_max,
                    horizon_orbits, altitude, meta)


__all__ = ["AXES", "AXIS_NAMES", "Scenario", "make_id", "axis_scenario",
           "vel_first_order_energy_matched", "vel_exact_energy_matched_radial"]
