import math

import numpy as np

from .constants import J2EARTH, MUEARTH, OMEGAEARTH, REARTH


def acccw(n, R, x, y, z, vx, vy, vz):
    n2 = n * n
    return (3.0 * n2 * x + 2.0 * n * vy, -2.0 * n * vx, -n2 * z)


# Second-order (quadratic) relative acceleration about a circular chief
def accso(n, R, x, y, z, vx, vy, vz):
    n2 = n * n
    c = 3.0 * n2 / R
    return (
        3.0 * n2 * x + 2.0 * n * vy - 0.5 * c * (2.0 * x * x - y * y - z * z),
        -2.0 * n * vx + c * x * y,
        -n2 * z + c * x * z,
    )


# Exact nonlinear relative acceleration, rearranged to avoid cancellation. Algebraically the same as
# 2 n vy + n^2 (R + x) - mu (R + x) / r^3 with mu = n^2 R^3.
def accnerm(n, R, x, y, z, vx, vy, vz):
    n2 = n * n
    q = (2.0 * R * x + x * x + y * y + z * z) / (R * R)
    f = -math.expm1(-1.5 * math.log1p(q))
    return (
        2.0 * n * vy + n2 * (R + x) * f,
        -2.0 * n * vx + n2 * y * f,
        -n2 * z * (1.0 - f),
    )


# Direct form of the same acceleration, kept so the tests can compare against it
def accnermnaive(n, R, x, y, z, vx, vy, vz, mu=MUEARTH):
    rx = R + x
    r = math.sqrt(rx * rx + y * y + z * z)
    k = mu / (r * r * r)
    n2 = n * n
    return (2.0 * n * vy + n2 * rx - k * rx, -2.0 * n * vx + n2 * y - k * y, -k * z)


RELMODELS = {"cw": acccw, "so": accso, "nerm": accnerm}


def relaccvec(model, n, R, states):
    s = np.asarray(states, dtype=float)
    x, y, z, vx, vy, vz = (s[:, i] for i in range(6))
    n2 = n * n
    if model == "cw":
        return np.column_stack((3 * n2 * x + 2 * n * vy, -2 * n * vx, -n2 * z))
    if model == "so":
        c = 3 * n2 / R
        return np.column_stack((
            3 * n2 * x + 2 * n * vy - 0.5 * c * (2 * x * x - y * y - z * z),
            -2 * n * vx + c * x * y,
            -n2 * z + c * x * z,
        ))
    if model == "nerm":
        q = (2 * R * x + x * x + y * y + z * z) / (R * R)
        f = -np.expm1(-1.5 * np.log1p(q))
        return np.column_stack((2 * n * vy + n2 * (R + x) * f, -2 * n * vx + n2 * y * f, -n2 * z * (1 - f)))
    raise ValueError(f"unknown relative model {model!r}")


class InertForce:

    __slots__ = ("mu", "use_j2", "use_drag", "r_earth", "j2", "cd", "area", "mass",
                 "omega_earth", "rho0", "h_ref", "scale_height", "_bc")

    def __init__(self, use_j2=False, use_drag=False, mu=MUEARTH, r_earth=REARTH,
                 j2=J2EARTH, cd=2.2, area=1.0, mass=100.0, omega_earth=OMEGAEARTH,
                 rho0=3.9e-12, h_ref=400_000.0, scale_height=60_000.0):
        self.mu = mu
        self.use_j2 = bool(use_j2)
        self.use_drag = bool(use_drag)
        self.r_earth = r_earth
        self.j2 = j2
        self.cd = cd
        self.area = area
        self.mass = mass
        self.omega_earth = omega_earth
        self.rho0 = rho0
        self.h_ref = h_ref
        self.scale_height = scale_height
        self._bc = cd * area / mass

    @property
    def label(self):
        parts = ["two-body"]
        if self.use_j2:
            parts.append("J2")
        if self.use_drag:
            parts.append(f"drag(Cd*A/m={self._bc:.4g} m^2/kg)")
        return " + ".join(parts)

    def asdict2(self):
        return {k: getattr(self, k) for k in self.__slots__ if not k.startswith("_")}

    def pert2(self, px, py, pz, vx, vy, vz):
        ax = ay = az = 0.0
        r2 = px * px + py * py + pz * pz
        r = math.sqrt(r2)
        if self.use_j2:
            factor = -1.5 * self.j2 * self.mu * self.r_earth ** 2 / (r2 * r2 * r)
            zr2 = 5.0 * pz * pz / r2
            ax += factor * px * (1.0 - zr2)
            ay += factor * py * (1.0 - zr2)
            az += factor * pz * (3.0 - zr2)
        if self.use_drag:
            alt = r - self.r_earth
            if alt >= -self.h_ref:
                rho = self.rho0 * math.exp(-(alt - self.h_ref) / self.scale_height)
                we = self.omega_earth
                wx = vx + we * py
                wy = vy - we * px
                wz = vz
                sp = math.sqrt(wx * wx + wy * wy + wz * wz)
                k = -0.5 * rho * self._bc * sp
                ax += k * wx
                ay += k * wy
                az += k * wz
        return ax, ay, az

    def acc3(self, px, py, pz, vx, vy, vz):
        r2 = px * px + py * py + pz * pz
        r = math.sqrt(r2)
        k = -self.mu / (r2 * r)
        ax, ay, az = k * px, k * py, k * pz
        if self.use_j2 or self.use_drag:
            bx, by, bz = self.pert2(px, py, pz, vx, vy, vz)
            ax += bx
            ay += by
            az += bz
        return ax, ay, az


def lvlhbasis(tx, ty, tz, tvx, tvy, tvz):
    r = math.sqrt(tx * tx + ty * ty + tz * tz)
    xh = (tx / r, ty / r, tz / r)
    hx = ty * tvz - tz * tvy
    hy = tz * tvx - tx * tvz
    hz = tx * tvy - ty * tvx
    h = math.sqrt(hx * hx + hy * hy + hz * hz)
    zh = (hx / h, hy / h, hz / h)
    yh = (zh[1] * xh[2] - zh[2] * xh[1], zh[2] * xh[0] - zh[0] * xh[2], zh[0] * xh[1] - zh[1] * xh[0])
    return xh, yh, zh, r, h


# LVLH angular velocity: wz = h / r^2 and wx = r a_h / h, where a_h is the target's out-of-plane
# perturbing acceleration (wx = 0 for an unperturbed target)
def framerate(r, h, aperph):
    return r * aperph / h, h / (r * r)


def ecilvlhscalar(t6, c6, tgtpert=None):
    tx, ty, tz, tvx, tvy, tvz = t6
    xh, yh, zh, r, h = lvlhbasis(tx, ty, tz, tvx, tvy, tvz)
    dx, dy, dz = c6[0] - tx, c6[1] - ty, c6[2] - tz
    dvx, dvy, dvz = c6[3] - tvx, c6[4] - tvy, c6[5] - tvz
    px = xh[0] * dx + xh[1] * dy + xh[2] * dz
    py = yh[0] * dx + yh[1] * dy + yh[2] * dz
    pz = zh[0] * dx + zh[1] * dy + zh[2] * dz
    if tgtpert is None:
        wx, wz = 0.0, h / (r * r)
    else:
        ah = zh[0] * tgtpert[0] + zh[1] * tgtpert[1] + zh[2] * tgtpert[2]
        wx, wz = framerate(r, h, ah)
    vx = xh[0] * dvx + xh[1] * dvy + xh[2] * dvz + wz * py
    vy = yh[0] * dvx + yh[1] * dvy + yh[2] * dvz - (wz * px - wx * pz)
    vz = zh[0] * dvx + zh[1] * dvy + zh[2] * dvz - wx * py
    return (px, py, pz, vx, vy, vz), (xh, yh, zh)


def lvlhtoeci(target6, rel6, tgtpert=None):
    t = np.asarray(target6, dtype=float)
    rel = np.asarray(rel6, dtype=float)
    xh, yh, zh, r, h = lvlhbasis(*t)
    C = np.array([xh, yh, zh])
    if tgtpert is None:
        wx, wz = 0.0, h / (r * r)
    else:
        ah = float(np.dot(zh, tgtpert))
        wx, wz = framerate(r, h, ah)
    w = np.array([wx, 0.0, wz])
    rho, rhod = rel[:3], rel[3:]
    dr = C.T @ rho
    dv = C.T @ (rhod + np.cross(w, rho))
    return np.concatenate((t[:3] + dr, t[3:] + dv))


def ecilvlhvec(tgtstates, chaserstates, tgtpert=None):
    T = np.asarray(tgtstates, dtype=float)
    Cs = np.asarray(chaserstates, dtype=float)
    rvec, vvec = T[:, :3], T[:, 3:]
    r = np.linalg.norm(rvec, axis=1)
    xh = rvec / r[:, None]
    hvec = np.cross(rvec, vvec)
    h = np.linalg.norm(hvec, axis=1)
    zh = hvec / h[:, None]
    yh = np.cross(zh, xh)
    dr = Cs[:, :3] - rvec
    dv = Cs[:, 3:] - vvec
    px = np.einsum("ij,ij->i", xh, dr)
    py = np.einsum("ij,ij->i", yh, dr)
    pz = np.einsum("ij,ij->i", zh, dr)
    wz = h / r**2
    if tgtpert is None:
        wx = np.zeros_like(wz)
    else:
        ah = np.einsum("ij,ij->i", zh, np.asarray(tgtpert, dtype=float))
        wx = r * ah / h
    vx = np.einsum("ij,ij->i", xh, dv) + wz * py
    vy = np.einsum("ij,ij->i", yh, dv) - (wz * px - wx * pz)
    vz = np.einsum("ij,ij->i", zh, dv) - wx * py
    return np.column_stack((px, py, pz, vx, vy, vz)), (xh, yh, zh)
