"""CW-only first-order estimate of the CW-to-nonlinear closed-loop discrepancy.

Given only the CW closed-loop run, the error e = x_NL - x_CW is propagated with
the linearised (variational) closed-loop dynamics along the CW trajectory,
driven by the acceleration residual of the CW model evaluated on the CW
trajectory:

    e' = (A_CW + B dU/dx(x_CW(t))) e + B r(t),      e(0) = 0
    r(t) = a_exact(x_CW(t)) - a_CW(x_CW(t))   (analytic gravity residual)

dU/dx is the Jacobian of the saturated PD law: -[Kp I, Kd I] when unsaturated,
and (a_max/|a_raw|)(I - a_hat a_hat^T)(-[Kp I, Kd I]) when saturated. The
integration restarts at the CW loop's saturation switches (Jacobian jumps).

No nonlinear propagation is performed: the inputs are the CW trajectory and
an analytic acceleration function, so this is admissible as a screening input.
It is a first-order estimate; it is not expected to be accurate once the
discrepancy is large or the loops diverge (it is used only to rank risk).

Implementation note: the right-hand side is written in scalar arithmetic and
reads the CW dense output segment by segment (same equations as the earlier
numpy version, `first_order_discrepancy_reference`, kept for testing).
"""

import math
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import acc_cw, acc_nerm, relative_acc_vectorized



def _validate_cw_coverage(cw_sim, t_end):
    """Reject incomplete CW input rather than extrapolating a screening feature."""
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("estimator horizon must be finite and positive")
    tol = 32 * np.finfo(float).eps * max(1.0, abs(t_end))
    if cw_sim.status != "ok" or not np.isfinite(cw_sim.t_final) or cw_sim.t_final < t_end - tol:
        raise ValueError("CW branch did not complete the requested horizon")
    covered = 0.0
    for seg in cw_sim.segments:
        if not np.isfinite(seg.t0) or not np.isfinite(seg.t1) or seg.t0 > covered + tol or seg.t1 < seg.t0:
            raise ValueError("CW dense-output coverage is incomplete")
        covered = max(covered, seg.t1)
    if not cw_sim.segments or covered < t_end - tol:
        raise ValueError("CW dense-output coverage is incomplete")


def _validate_estimator_segment(out, a, b):
    """Check completion before any dense-output sampling beyond the solved arc."""
    tol = 32 * np.finfo(float).eps * max(1.0, abs(b))
    if (not out.success or out.sol is None or len(out.t) == 0
            or not np.all(np.isfinite(out.t)) or out.t[-1] < b - tol
            or not np.all(np.isfinite(out.y))
            or out.sol.t_min > a + tol or out.sol.t_max < b - tol):
        raise ValueError("estimator segment failed or stopped before its endpoint")


def first_order_discrepancy(cw_sim, law, n, R, t_end, dt_out=2.0, rtol=1e-9, atol=1e-9):
    """Return dict with the predicted peak position discrepancy and related values."""
    _validate_cw_coverage(cw_sim, t_end)
    if not np.isfinite(dt_out) or dt_out <= 0:
        raise ValueError("dt_out must be positive and finite")
    segs = cw_sim.segments
    t0s = [s.t0 for s in segs]
    kp, kd, am = law.kp, law.kd, law.a_max
    n2 = n * n
    sqrt = math.sqrt

    def x_at(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = x_at(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = acc_nerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acc_cw(n, R, x0, x1, x2, x3, x4, x5)
        w0 = -kp * e0 - kd * e3
        w1 = -kp * e1 - kd * e4
        w2 = -kp * e2 - kd * e5
        if am is not None:
            rx, ry, rz = -kp * x0 - kd * x3, -kp * x1 - kd * x4, -kp * x2 - kd * x5
            m = sqrt(rx * rx + ry * ry + rz * rz)
            if m > am:
                ax, ay, az = rx / m, ry / m, rz / m
                dot = ax * w0 + ay * w1 + az * w2
                s = am / m
                w0, w1, w2 = s * (w0 - ax * dot), s * (w1 - ay * dot), s * (w2 - az * dot)
        return [e3, e4, e5,
                3.0 * n2 * e0 + 2.0 * n * e4 + w0 + (an[0] - ac[0]),
                -2.0 * n * e3 + w1 + (an[1] - ac[1]),
                -n2 * e2 + w2 + (an[2] - ac[2])]

    _validate_cw_coverage(cw_sim, t_end)
    breaks = [0.0] + sorted(set(t for t in cw_sim.sat_switch_times if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    ts, es = [], []
    nfev = 0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=rtol, atol=atol, max_step=20.0, dense_output=True)
        _validate_estimator_segment(out, a, b)
        nfev += out.nfev
        k = max(2, int(math.ceil((b - a) / dt_out)) + 1)
        tt = np.linspace(a, b, k)
        ts.append(tt)
        es.append(out.sol(tt).T)
        e = out.y[:, -1]
    T = np.concatenate(ts)
    E = np.vstack(es)
    if not np.all(np.isfinite(E)):
        raise ValueError("non-finite estimator samples")
    dp = np.linalg.norm(E[:, :3], axis=1)
    i = int(np.argmax(dp))
    return {"d1_peak_m": float(dp[i]), "d1_t_peak_s": float(T[i]), "d1_final_m": float(dp[-1]),
            "d1_peak_vel_mps": float(np.linalg.norm(E[:, 3:], axis=1).max()), "d1_nfev": nfev}


def _jacobian_u(law, x):
    kp, kd = law.kp, law.kd
    K = np.hstack((kp * np.eye(3), kd * np.eye(3)))
    raw = -K @ x
    m = float(np.linalg.norm(raw))
    if law.a_max is None or m <= law.a_max:
        return -K
    a = raw / m
    return (law.a_max / m) * (np.eye(3) - np.outer(a, a)) @ (-K)


def first_order_discrepancy_reference(cw_sim, law, n, R, t_end, dt_out=2.0):
    """Original (numpy, slow) implementation of the same estimate; used in tests."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4], A[4, 3], A[5, 2] = 3 * n * n, 2 * n, -2 * n, -n * n

    def resid(x):
        s = x[None, :]
        return (relative_acc_vectorized("nerm", n, R, s) - relative_acc_vectorized("cw", n, R, s))[0]

    def rhs(t, e):
        x = cw_sim.states([t])[0]
        M = A.copy()
        M[3:6, :] += _jacobian_u(law, x)
        de = M @ e
        de[3:6] += resid(x)
        return de

    _validate_cw_coverage(cw_sim, t_end)
    breaks = [0.0] + sorted(set(t for t in cw_sim.sat_switch_times if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    best = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=1e-9, atol=1e-9, max_step=20.0, dense_output=True)
        _validate_estimator_segment(out, a, b)
        tt = np.linspace(a, b, max(2, int(math.ceil((b - a) / dt_out)) + 1))
        best = max(best, float(np.linalg.norm(out.sol(tt)[:3].T, axis=1).max()))
        e = out.y[:, -1]
    return {"d1_peak_m": best}


__all__ = ["first_order_discrepancy", "first_order_discrepancy_reference"]
