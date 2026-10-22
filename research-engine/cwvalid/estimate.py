import math
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import acccw, accnerm, relaccvec


def valcwcov(cwsim, t_end):
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("estimator horizon must be finite and positive")
    tol = 32 * np.finfo(float).eps * max(1.0, abs(t_end))
    if cwsim.status != "ok" or not np.isfinite(cwsim.tfinal) or cwsim.tfinal < t_end - tol:
        raise ValueError("CW branch did not complete the requested horizon")
    covered = 0.0
    for seg in cwsim.segments:
        if not np.isfinite(seg.t0) or not np.isfinite(seg.t1) or seg.t0 > covered + tol or seg.t1 < seg.t0:
            raise ValueError("CW dense-output coverage is incomplete")
        covered = max(covered, seg.t1)
    if not cwsim.segments or covered < t_end - tol:
        raise ValueError("CW dense-output coverage is incomplete")


def valestseg(out, a, b):
    tol = 32 * np.finfo(float).eps * max(1.0, abs(b))
    if (not out.success or out.sol is None or len(out.t) == 0
            or not np.all(np.isfinite(out.t)) or out.t[-1] < b - tol
            or not np.all(np.isfinite(out.y))
            or out.sol.t_min > a + tol or out.sol.t_max < b - tol):
        raise ValueError("estimator segment failed or stopped before its endpoint")


# Propagates e = x_NL - x_CW with the closed-loop variational equations along the CW run, driven
# by the acceleration residual, and returns the predicted peak position discrepancy
def firstorderdisc(cwsim, law, n, R, t_end, dt_out=2.0, rtol=1e-9, atol=1e-9):
    valcwcov(cwsim, t_end)
    if not np.isfinite(dt_out) or dt_out <= 0:
        raise ValueError("dt_out must be positive and finite")
    segs = cwsim.segments
    t0s = [s.t0 for s in segs]
    kp, kd, am = law.kp, law.kd, law.a_max
    n2 = n * n
    sqrt = math.sqrt

    def xat(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = xat(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = accnerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acccw(n, R, x0, x1, x2, x3, x4, x5)
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

    valcwcov(cwsim, t_end)
    breaks = [0.0] + sorted(set(t for t in cwsim.satswitchtimes if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    ts, es = [], []
    nfev = 0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=rtol, atol=atol, max_step=20.0, dense_output=True)
        valestseg(out, a, b)
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


def jacobianu(law, x):
    kp, kd = law.kp, law.kd
    K = np.hstack((kp * np.eye(3), kd * np.eye(3)))
    raw = -K @ x
    m = float(np.linalg.norm(raw))
    if law.a_max is None or m <= law.a_max:
        return -K
    a = raw / m
    return (law.a_max / m) * (np.eye(3) - np.outer(a, a)) @ (-K)


# Slow reference version of the same estimate, used in the tests
def firstref(cwsim, law, n, R, t_end, dt_out=2.0):
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4], A[4, 3], A[5, 2] = 3 * n * n, 2 * n, -2 * n, -n * n

    def resid(x):
        s = x[None, :]
        return (relaccvec("nerm", n, R, s) - relaccvec("cw", n, R, s))[0]

    def rhs(t, e):
        x = cwsim.states([t])[0]
        M = A.copy()
        M[3:6, :] += jacobianu(law, x)
        de = M @ e
        de[3:6] += resid(x)
        return de

    valcwcov(cwsim, t_end)
    breaks = [0.0] + sorted(set(t for t in cwsim.satswitchtimes if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    best = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=1e-9, atol=1e-9, max_step=20.0, dense_output=True)
        valestseg(out, a, b)
        tt = np.linspace(a, b, max(2, int(math.ceil((b - a) / dt_out)) + 1))
        best = max(best, float(np.linalg.norm(out.sol(tt)[:3].T, axis=1).max()))
        e = out.y[:, -1]
    return {"d1_peak_m": best}
