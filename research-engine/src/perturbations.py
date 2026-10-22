from dataclasses import dataclass

import numpy as np

from .orbit import J2EARTH, MUEARTH, OMEGAEARTH, REARTH, twobodyacc


def j2acc(position, mu=MUEARTH, r_earth=REARTH, j2=J2EARTH):
    x, y, z = np.asarray(position, dtype=float)
    r = np.sqrt(x * x + y * y + z * z)
    factor = -1.5 * j2 * mu * r_earth**2 / r**5
    zr2 = 5.0 * z * z / (r * r)
    return factor * np.array([x * (1.0 - zr2), y * (1.0 - zr2), z * (3.0 - zr2)])


def expdens(altitude, rho0=3.9e-12, h_ref=400_000.0, scale_height=60_000.0):
    if altitude < -h_ref:
        return 0.0
    return rho0 * np.exp(-(altitude - h_ref) / scale_height)


def dragacc(
    position, velocity, cd=2.2, area=1.0, mass=100.0,
    r_earth=REARTH, omega_earth=OMEGAEARTH, densfunc=expdens,
):
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    altitude = np.linalg.norm(position) - r_earth
    rho = densfunc(altitude)
    if rho <= 0.0:
        return np.zeros(3)
    omegavec = np.array([0.0, 0.0, omega_earth])
    vrel = velocity - np.cross(omegavec, position)
    speed = np.linalg.norm(vrel)
    ball = cd * area / mass
    return -0.5 * rho * ball * speed * vrel


@dataclass
class ForceMod:

    use_j2: bool = False
    use_drag: bool = False
    mu: float = MUEARTH
    r_earth: float = REARTH
    j2: float = J2EARTH
    cd: float = 2.2
    area: float = 1.0
    mass: float = 100.0
    omega_earth: float = OMEGAEARTH

    @classmethod
    def twobody(cls, **kw):
        return cls(use_j2=False, use_drag=False, **kw)

    @classmethod
    def j2only(cls, **kw):
        return cls(use_j2=True, use_drag=False, **kw)

    @classmethod
    def dragonly(cls, **kw):
        return cls(use_j2=False, use_drag=True, **kw)

    @classmethod
    def j2_drag(cls, **kw):
        return cls(use_j2=True, use_drag=True, **kw)

    @property
    def label(self):
        if self.use_j2 and self.use_drag:
            return "two-body + J2 + drag"
        if self.use_j2:
            return "two-body + J2"
        if self.use_drag:
            return "two-body + drag"
        return "two-body"

    def acc3(self, state, t=0.0):
        state = np.asarray(state, dtype=float)
        pos, vel = state[:3], state[3:]
        acc = twobodyacc(pos, self.mu)
        if self.use_j2:
            acc = acc + j2acc(pos, self.mu, self.r_earth, self.j2)
        if self.use_drag:
            acc = acc + dragacc(
                pos, vel, self.cd, self.area, self.mass, self.r_earth, self.omega_earth
            )
        return acc

    def derivatives(self, state, t=0.0):
        state = np.asarray(state, dtype=float)
        return np.concatenate((state[3:], self.acc3(state, t)))
