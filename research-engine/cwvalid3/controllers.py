import hashlib
import json
import math

import numpy as np
from scipy.linalg import solve_continuous_are

from .sampled import cwmatrix

B6 = np.vstack((np.zeros((3, 3)), np.eye(3)))


class MatLaw:
    def __init__(self, K, a_max=None, design=None):
        K = np.array(K, dtype=float).reshape(3, 6)
        if not np.all(np.isfinite(K)):
            raise ValueError("gain matrix must be finite")
        self.K = K
        self.a_max = None if a_max is None else float(a_max)
        self.design = dict(design or {})
        self._k = tuple(tuple(float(v) for v in row) for row in K)

    def raw(self, x, y, z, vx, vy, vz):
        (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5), (c0, c1, c2, c3, c4, c5) = self._k
        return (-(a0 * x + a1 * y + a2 * z + a3 * vx + a4 * vy + a5 * vz),
                -(b0 * x + b1 * y + b2 * z + b3 * vx + b4 * vy + b5 * vz),
                -(c0 * x + c1 * y + c2 * z + c3 * vx + c4 * vy + c5 * vz))

    def command(self, x, y, z, vx, vy, vz):
        ax, ay, az = self.raw(x, y, z, vx, vy, vz)
        am = self.a_max
        if am is not None:
            m = math.sqrt(ax * ax + ay * ay + az * az)
            if m > am:
                s = am / m
                return ax * s, ay * s, az * s
        return ax, ay, az

    def rawvec(self, states):
        S = np.asarray(states, dtype=float)[:, :6]
        return -(S @ self.K.T)

    def applvec(self, states):
        raw = self.rawvec(states)
        if self.a_max is None:
            return raw
        m = np.linalg.norm(raw, axis=1)
        scale = np.where(m > self.a_max, self.a_max / np.where(m > 0, m, 1.0), 1.0)
        return raw * scale[:, None]

    def jacobian(self, s):
        s = np.asarray(s, dtype=float)[:6]
        a = -self.K @ s
        m = float(np.linalg.norm(a))
        if self.a_max is None or m <= self.a_max:
            return -self.K.copy(), False
        ah = a / m
        return (self.a_max / m) * (np.eye(3) - np.outer(ah, ah)) @ (-self.K), True

    @property
    def saturdist(self):
        return None

    def ispd(self, rtol=0.0):
        K = self.K
        kp, kd = K[0, 0], K[0, 3]
        P = np.hstack((kp * np.eye(3), kd * np.eye(3)))
        return bool(np.all(np.abs(K - P) <= rtol * np.abs(P).max()))

    def asdict2(self):
        return {"K": self.K.tolist(), "a_max": self.a_max, "design": self.design, "K_sha256": self.khash()}

    def khash(self):
        return hashlib.sha256(json.dumps(self.K.tolist()).encode()).hexdigest()[:16]


def pdlaw(wn, a_max=None):
    K = np.hstack((wn * wn * np.eye(3), 2.0 * wn * np.eye(3)))
    return MatLaw(K, a_max, {"rule": "scalar_pd", "omega_n_radps": wn, "kp": wn * wn, "kd": 2.0 * wn})


def cwlqrgain(n, wn, rhos=1.0):
    A = cwmatrix(n)
    vs, s = wn * rhos, wn * wn * rhos
    Q = np.diag([1.0 / rhos ** 2] * 3 + [1.0 / vs ** 2] * 3) * s ** 2
    Rm = np.eye(3) / s ** 2 * s ** 2
    P = solve_continuous_are(A, B6, Q, Rm)
    K = np.linalg.solve(Rm, B6.T @ P)
    return K, {"A": A, "Q": Q, "R": Rm, "P": P}


def lqrlaw(n, wn, a_max=None, rhos=1.0):
    K, d = cwlqrgain(n, wn, rhos)
    ev = np.linalg.eigvals(cwmatrix(n) - B6 @ K)
    design = {"rule": "cw_lqr_bryson", "omega_n_radps": wn, "n_radps": n, "rho_s_m": rhos,
              "v_s_mps": wn * rhos, "a_s_mps2": wn * wn * rhos,
              "Q_diag_normalised": np.diag(d["Q"]).tolist(), "R_normalised": "I3",
              "closed_loop_max_real_eig": float(np.max(ev.real)),
              "closed_loop_eigs": [[float(e.real), float(e.imag)] for e in ev],
              "pos_block_offdiag_ratio": float(np.linalg.norm(K[:, :3] - np.diag(np.diag(K[:, :3]))) /
                                               np.linalg.norm(np.diag(K[:, :3]))),
              "vel_block_offdiag_ratio": float(np.linalg.norm(K[:, 3:] - np.diag(np.diag(K[:, 3:]))) /
                                               np.linalg.norm(np.diag(K[:, 3:]))),
              "kp_diag_over_wn2": (np.diag(K[:, :3]) / wn ** 2).tolist(),
              "kd_diag_over_wn": (np.diag(K[:, 3:]) / wn).tolist()}
    return MatLaw(K, a_max, design)


def unsatstab(law, n):
    ev = np.linalg.eigvals(cwmatrix(n) - B6 @ law.K)
    return float(np.max(ev.real))
