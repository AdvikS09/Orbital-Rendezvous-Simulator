"""v2 extension: estimator variants for the CW-to-nonlinear closed-loop discrepancy.

All variants use only the CW closed-loop run (no nonlinear propagation) and the
analytic CW acceleration residual evaluated on the CW trajectory,

    r(t) = a_exact(x_CW(t)) - a_CW(x_CW(t))      [m/s^2]

(a_exact: exact circular-chief relative gravity/kinematics, a_CW: the CW model;
the Coriolis terms cancel, so r depends on position only).

A  ``d1``     the full first-order estimate (cwvalid/estimate.py; same equations,
              with completion, coverage and finiteness guards revised 13 October 2026):
              e' = (A_CW + B dU/dx(x_CW(t))) e + B r(t),  e(0) = 0,
              dU/dx = Jacobian of the SATURATED PD law along the CW trajectory.
B  ``d1u``    identical to A except that the Jacobian is the UNSATURATED -K = -[Kp I, Kd I]
              at all times:  e' = (A_CW - B K) e + B r(t),  e(0) = 0.
              Same CW trajectory, same residual, same integrator settings
              (DOP853, rtol = atol = 1e-9, max step 20 s), same restarts at the CW
              loop's saturation switch times, same 2 s output grid.
C  ``cres``   residual-only diagnostic (no closed-loop response terms):
              e_pos'' = r(t),  e_pos(0) = 0, e_vel(0) = 0.
              Same integrator settings, restarts and output grid.
              This is a diagnostic comparison, not a complete physical error model.

Each variant's scalar feature is the peak of ||e_position(t)|| over the CW branch's
time span [m]. If the integration fails or produces non-finite values the
feature is reported as NaN with status 'failed' (never silently replaced).
"""

import math
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import acc_cw, acc_nerm
from .estimate import first_order_discrepancy, _validate_cw_coverage, _validate_estimator_segment

RTOL = 1e-9
ATOL = 1e-9
DT_OUT = 2.0
MAX_STEP = 20.0


def _segment_eval(cw_sim):
    segs = cw_sim.segments
    t0s = [s.t0 for s in segs]

    def x_at(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)
    return x_at


def _integrate(rhs, cw_sim, t_end):
    _validate_cw_coverage(cw_sim, t_end)
    breaks = [0.0] + sorted(set(t for t in cw_sim.sat_switch_times if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    ts, es = [], []
    nfev = 0
    status = "ok"
    def failed(reason):
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "final_m": float("nan"),
                "peak_vel_mps": float("nan"), "nfev": nfev, "status": "failed",
                "failure_reason": reason}
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True)
        nfev += out.nfev
        try:
            _validate_estimator_segment(out, a, b)
        except ValueError as exc:
            return failed(f"segment [{a}, {b}]: {exc}")
        k = max(2, int(math.ceil((b - a) / DT_OUT)) + 1)
        tt = np.linspace(a, b, k)
        ts.append(tt)
        es.append(out.sol(tt).T)
        e = out.y[:, -1]
    if not ts:
        return failed("no completed estimator segments")
    T = np.concatenate(ts)
    E = np.vstack(es)
    if not np.all(np.isfinite(E)):
        return failed("non-finite estimator dense-output samples")
    dp = np.linalg.norm(E[:, :3], axis=1)
    i = int(np.argmax(dp))
    return {"peak_m": float(dp[i]) if status == "ok" else float("nan"), "t_peak_s": float(T[i]),
            "final_m": float(dp[-1]), "peak_vel_mps": float(np.linalg.norm(E[:, 3:], axis=1).max()),
            "nfev": nfev, "status": status}


def unsaturated_jacobian_estimate(cw_sim, law, n, R, t_end):
    """Variant B: as the full estimate but with the unsaturated Jacobian -K throughout."""
    x_at = _segment_eval(cw_sim)
    kp, kd = law.kp, law.kd
    n2 = n * n

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = x_at(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = acc_nerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acc_cw(n, R, x0, x1, x2, x3, x4, x5)
        w0 = -kp * e0 - kd * e3
        w1 = -kp * e1 - kd * e4
        w2 = -kp * e2 - kd * e5
        return [e3, e4, e5,
                3.0 * n2 * e0 + 2.0 * n * e4 + w0 + (an[0] - ac[0]),
                -2.0 * n * e3 + w1 + (an[1] - ac[1]),
                -n2 * e2 + w2 + (an[2] - ac[2])]
    return _integrate(rhs, cw_sim, t_end)


def residual_only_diagnostic(cw_sim, n, R, t_end):
    """Variant C: e_pos'' = r(t) with zero initial error (no closed-loop terms)."""
    x_at = _segment_eval(cw_sim)

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = x_at(t)[:6].tolist()
        an = acc_nerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acc_cw(n, R, x0, x1, x2, x3, x4, x5)
        return [e[3], e[4], e[5], an[0] - ac[0], an[1] - ac[1], an[2] - ac[2]]
    return _integrate(rhs, cw_sim, t_end)


def estimator_variants(cw_sim, law, n, R, t_end):
    """Return the scalar features of variants A, B and C plus their status fields."""
    out = {}
    try:
        a = first_order_discrepancy(cw_sim, law, n, R, t_end)
        ok = np.isfinite(a["d1_peak_m"])
        out.update({"d1_peak_m": a["d1_peak_m"] if ok else float("nan"), "d1_t_peak_s": a["d1_t_peak_s"],
                    "d1_status": "ok" if ok else "failed", "d1_nfev": a["d1_nfev"]})
    except Exception as exc:  # recorded, never silently dropped
        out.update({"d1_peak_m": float("nan"), "d1_t_peak_s": float("nan"), "d1_status": f"failed: {exc!r}", "d1_nfev": 0})
    for key, fn in (("d1u", lambda: unsaturated_jacobian_estimate(cw_sim, law, n, R, t_end)),
                    ("cres", lambda: residual_only_diagnostic(cw_sim, n, R, t_end))):
        try:
            r = fn()
            out.update({f"{key}_peak_m": r["peak_m"], f"{key}_t_peak_s": r["t_peak_s"], f"{key}_final_m": r["final_m"],
                        f"{key}_status": r["status"], f"{key}_nfev": r["nfev"],
                        f"{key}_failure_reason": r.get("failure_reason", "")})
        except Exception as exc:
            out.update({f"{key}_peak_m": float("nan"), f"{key}_t_peak_s": float("nan"), f"{key}_final_m": float("nan"),
                        f"{key}_status": f"failed: {exc!r}", f"{key}_nfev": 0,
                        f"{key}_failure_reason": repr(exc)})
    return out


__all__ = ["unsaturated_jacobian_estimate", "residual_only_diagnostic", "estimator_variants"]
