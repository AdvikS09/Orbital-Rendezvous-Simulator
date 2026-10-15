"""General matrix feedback U(s) = normsat(-K s) with K a serialized 3x6 matrix (P07).

The scalar PD law of the paper is the special case K = [kp I3, kd I3]. ``MatrixLaw`` is a
drop-in replacement for ``cwvalid.control.PDLaw`` wherever the production code only calls
``command``, ``raw_vec``, ``applied_vec``, ``a_max`` and ``as_dict`` (cwvalid.simulate right-hand
sides, cwvalid.metrics). It deliberately has NO ``kp``/``kd`` attributes: code that silently
assumes scalar gains (cwvalid.simulate's saturation-switch event, cwvalid.estimate*,
cwvalid.experiment.evaluate's kp/kd/sat_distance fields) must fail loudly or be replaced by the
matrix-aware versions in cwvalid3.continuous, never reuse a reconstructed scalar law.

Row arithmetic is written so that, for a PD matrix, every command and Jacobian product is
bit-identical to the scalar PDLaw expressions (adding exact zeros does not change a sum).

Jacobian of the law at s (|K s| != a_max):
  unsaturated  J_U = -K
  saturated    J_U = (a_max/|a|) (I - a_hat a_hat^T) (-K),  a = -K s, a_hat = a/|a|
At |a| = a_max the law is continuous but J_U jumps; callers that integrate across that switch
must restart there (continuous feedback) or evaluate J_U only at update instants (held commands).

D5 design rule (fixed before any D5 data were generated; see methods/derivations.md section 6):
continuous-time CW LQR with Bryson-type physical scaling
  rho_s = 1 m, v_s = omega_n * rho_s, a_s = omega_n^2 * rho_s,
  cost = integral( |rho|^2/rho_s^2 + |rho_dot|^2/v_s^2 + |u|^2/a_s^2 ) dt,
equivalently Q = diag(omega_n^4 I3, omega_n^2 I3), R = I3 after multiplying by a_s^2.
In the double-integrator limit (n -> 0) this gives kp = omega_n^2 and kd = sqrt(3) omega_n
(damping ratio sqrt(3)/2), and the CW terms add radial/along-track cross-coupling.
"""

import hashlib
import json
import math

import numpy as np
from scipy.linalg import solve_continuous_are

from .sampled import cw_matrix

B6 = np.vstack((np.zeros((3, 3)), np.eye(3)))


class MatrixLaw:
    def __init__(self, K, a_max=None, design=None):
        K = np.array(K, dtype=float).reshape(3, 6)
        if not np.all(np.isfinite(K)):
            raise ValueError("gain matrix must be finite")
        self.K = K
        self.a_max = None if a_max is None else float(a_max)
        self.design = dict(design or {})
        self._k = tuple(tuple(float(v) for v in row) for row in K)

    # scalar interface used inside right-hand sides -------------------------------------
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

    # vectorised post-processing ------------------------------------------------------------
    def raw_vec(self, states):
        S = np.asarray(states, dtype=float)[:, :6]
        return -(S @ self.K.T)

    def applied_vec(self, states):
        raw = self.raw_vec(states)
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
    def saturation_distance(self):
        """Not defined for a general matrix (no single kp); see controller_compatibility report."""
        return None

    def is_pd(self, rtol=0.0):
        K = self.K
        kp, kd = K[0, 0], K[0, 3]
        P = np.hstack((kp * np.eye(3), kd * np.eye(3)))
        return bool(np.all(np.abs(K - P) <= rtol * np.abs(P).max()))

    def as_dict(self):
        return {"K": self.K.tolist(), "a_max": self.a_max, "design": self.design, "K_sha256": self.k_hash()}

    def k_hash(self):
        return hashlib.sha256(json.dumps(self.K.tolist()).encode()).hexdigest()[:16]


def pd_law(wn, a_max=None):
    """Matrix form of the paper's scalar PD (kp = wn^2, kd = 2 wn)."""
    K = np.hstack((wn * wn * np.eye(3), 2.0 * wn * np.eye(3)))
    return MatrixLaw(K, a_max, {"rule": "scalar_pd", "omega_n_radps": wn, "kp": wn * wn, "kd": 2.0 * wn})


def cw_lqr_gain(n, wn, rho_s=1.0):
    """Continuous CW LQR with Bryson-type physical scaling (module docstring)."""
    A = cw_matrix(n)
    v_s, a_s = wn * rho_s, wn * wn * rho_s
    Q = np.diag([1.0 / rho_s ** 2] * 3 + [1.0 / v_s ** 2] * 3) * a_s ** 2
    Rm = np.eye(3) / a_s ** 2 * a_s ** 2
    P = solve_continuous_are(A, B6, Q, Rm)
    K = np.linalg.solve(Rm, B6.T @ P)
    return K, {"A": A, "Q": Q, "R": Rm, "P": P}


def lqr_law(n, wn, a_max=None, rho_s=1.0):
    K, d = cw_lqr_gain(n, wn, rho_s)
    ev = np.linalg.eigvals(cw_matrix(n) - B6 @ K)
    design = {"rule": "cw_lqr_bryson", "omega_n_radps": wn, "n_radps": n, "rho_s_m": rho_s,
              "v_s_mps": wn * rho_s, "a_s_mps2": wn * wn * rho_s,
              "Q_diag_normalised": np.diag(d["Q"]).tolist(), "R_normalised": "I3",
              "closed_loop_max_real_eig": float(np.max(ev.real)),
              "closed_loop_eigs": [[float(e.real), float(e.imag)] for e in ev],
              # unit-consistent structure measures (each block compared with its own diagonal)
              "pos_block_offdiag_ratio": float(np.linalg.norm(K[:, :3] - np.diag(np.diag(K[:, :3]))) /
                                               np.linalg.norm(np.diag(K[:, :3]))),
              "vel_block_offdiag_ratio": float(np.linalg.norm(K[:, 3:] - np.diag(np.diag(K[:, 3:]))) /
                                               np.linalg.norm(np.diag(K[:, 3:]))),
              "kp_diag_over_wn2": (np.diag(K[:, :3]) / wn ** 2).tolist(),
              "kd_diag_over_wn": (np.diag(K[:, 3:]) / wn).tolist()}
    return MatrixLaw(K, a_max, design)


def unsaturated_stability(law, n):
    ev = np.linalg.eigvals(cw_matrix(n) - B6 @ law.K)
    return float(np.max(ev.real))


__all__ = ["MatrixLaw", "pd_law", "cw_lqr_gain", "lqr_law", "unsaturated_stability", "B6"]
