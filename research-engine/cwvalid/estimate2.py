import math
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import acccw, accnerm
from .estimate import firstorderdisc, valcwcov, valestseg

RTOL = 1e-9
ATOL = 1e-9
DTOUT = 2.0
MAXSTEP = 20.0


def segeval(cwsim):
    segs = cwsim.segments
    t0s = [s.t0 for s in segs]

    def xat(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)
    return xat


def integr(rhs, cwsim, t_end):
    valcwcov(cwsim, t_end)
    breaks = [0.0] + sorted(set(t for t in cwsim.satswitchtimes if 0.0 < t < t_end)) + [t_end]
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
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=RTOL, atol=ATOL, max_step=MAXSTEP, dense_output=True)
        nfev += out.nfev
        try:
            valestseg(out, a, b)
        except ValueError as exc:
            return failed(f"segment [{a}, {b}]: {exc}")
        k = max(2, int(math.ceil((b - a) / DTOUT)) + 1)
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


def unsatjacest(cwsim, law, n, R, t_end):
    xat = segeval(cwsim)
    kp, kd = law.kp, law.kd
    n2 = n * n

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = xat(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = accnerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acccw(n, R, x0, x1, x2, x3, x4, x5)
        w0 = -kp * e0 - kd * e3
        w1 = -kp * e1 - kd * e4
        w2 = -kp * e2 - kd * e5
        return [e3, e4, e5,
                3.0 * n2 * e0 + 2.0 * n * e4 + w0 + (an[0] - ac[0]),
                -2.0 * n * e3 + w1 + (an[1] - ac[1]),
                -n2 * e2 + w2 + (an[2] - ac[2])]
    return integr(rhs, cwsim, t_end)


def residonlydiag(cwsim, n, R, t_end):
    xat = segeval(cwsim)

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = xat(t)[:6].tolist()
        an = accnerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acccw(n, R, x0, x1, x2, x3, x4, x5)
        return [e[3], e[4], e[5], an[0] - ac[0], an[1] - ac[1], an[2] - ac[2]]
    return integr(rhs, cwsim, t_end)


def estvar(cwsim, law, n, R, t_end):
    out = {}
    try:
        a = firstorderdisc(cwsim, law, n, R, t_end)
        ok = np.isfinite(a["d1_peak_m"])
        out.update({"d1_peak_m": a["d1_peak_m"] if ok else float("nan"), "d1_t_peak_s": a["d1_t_peak_s"],
                    "d1_status": "ok" if ok else "failed", "d1_nfev": a["d1_nfev"]})
    except Exception as exc:
        out.update({"d1_peak_m": float("nan"), "d1_t_peak_s": float("nan"), "d1_status": f"failed: {exc!r}", "d1_nfev": 0})
    for key, fn in (("d1u", lambda: unsatjacest(cwsim, law, n, R, t_end)),
                    ("cres", lambda: residonlydiag(cwsim, n, R, t_end))):
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
