import math

import numpy as np

from .constants import DEFALT, INC, MU, R_E


class TgtOrbit:

    def __init__(self, altitude=DEFALT):
        self.altitude = float(altitude)
        self.R = R_E + self.altitude
        self.n = math.sqrt(MU / self.R ** 3)
        self.period = 2.0 * math.pi / self.n

    def initstate(self):
        ci, si = math.cos(INC), math.sin(INC)
        r = np.array([self.R, 0.0, 0.0])
        v = self.n * self.R * np.array([0.0, ci, si])
        return r, v

    def state(self, t):
        th = self.n * np.asarray(t, dtype=float)
        ci, si = math.cos(INC), math.sin(INC)
        c, s = np.cos(th), np.sin(th)
        r = self.R * np.array([c, s * ci, s * si])
        v = self.n * self.R * np.array([-s, c * ci, c * si])
        return r, v


def lvlhaxes(r, v):
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    xh = r / rn
    h = cross(r, v)
    hn = np.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
    zh = h / hn
    yh = cross(zh, xh)
    w = hn / (rn * rn)
    return xh, yh, zh, w


def cross(a, b):
    return np.stack([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def dcmecilvlh(r, v):
    xh, yh, zh, w = lvlhaxes(r, v)
    return np.vstack([xh, yh, zh]), float(w)


def inertrel(rt3, vt2, rc3, vc2):
    xh, yh, zh, w = lvlhaxes(rt3, vt2)
    dr = np.asarray(rc3, dtype=float) - np.asarray(rt3, dtype=float)
    dv = np.asarray(vc2, dtype=float) - np.asarray(vt2, dtype=float)
    rho = np.stack([dot(xh, dr), dot(yh, dr), dot(zh, dr)])
    cdv = np.stack([dot(xh, dv), dot(yh, dv), dot(zh, dv)])
    rhodot = np.stack([cdv[0] + w * rho[1],
                       cdv[1] - w * rho[0],
                       cdv[2]])
    return rho, rhodot


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def relinert(rt3, vt2, rho, rhodot):
    rt3 = np.asarray(rt3, dtype=float)
    vt2 = np.asarray(vt2, dtype=float)
    rho = np.asarray(rho, dtype=float)
    rhodot = np.asarray(rhodot, dtype=float)
    xh, yh, zh, w = lvlhaxes(rt3, vt2)
    vrel = np.stack([rhodot[0] - w * rho[1],
                     rhodot[1] + w * rho[0],
                     rhodot[2]])
    rc = rt3 + xh * rho[0] + yh * rho[1] + zh * rho[2]
    vc = vt2 + xh * vrel[0] + yh * vrel[1] + zh * vrel[2]
    return rc, vc
