"""Per-case metrics for a run HeldBatch, following the production conventions:
monitoring grid 0.5 s plus the horizon end; peak discrepancy refined within one grid step of the
grid maximum (bounded scalar optimisation); terminal-arrival last entry located by Brent's method
(xtol 1e-6 s) on the dense solution. The dense solution is the cubic Hermite interpolant of the
batch steps (HeldBatch.dense). Statuses are explicit: a branch that did not complete the horizon
cannot arrive, and its discrepancy is evaluated over the common horizon up to its end time.
Estimator features are the maximum of |e_pos| on the monitoring grid (the feature identity of the
held-command estimators).

v3.1 (C01): HeldBatch.dense masks a branch after its end time, so the monitoring grid of an exited or
failed branch is NaN after t_end and the pair discrepancy can only read the common valid interval
[0, min(t_end_a, t_end_ref)], which ends at the located event. A non-finite value inside that interval
gives a NaN peak (unresolved downstream) instead of a silently wrong maximum. case_summary also
reports, per floor-checked branch, the step that contained the exit and whether its root bracket
failed.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .sampled import ESTIMATORS

POS_TOL, VEL_TOL, DWELL = 1.0, 0.01, 300.0
DT_MON = 0.5


def monitoring_grid(T, dt=DT_MON):
    k = int(np.floor(T / dt + 1e-9))
    t = np.arange(k + 1) * dt
    if t[-1] < T - 1e-9:
        t = np.append(t, T)
    else:
        t[-1] = T
    return t


class CaseView:
    def __init__(self, batch, i, dt_mon=DT_MON):
        self.b, self.i = batch, i
        self.Fr, self.Fl = batch.case_derivatives(i)
        self.tm = monitoring_grid(batch.T, dt_mon)
        self.Ym = batch.dense(i, self.tm, self.Fr, self.Fl)

    def rel_from_rows(self, branch, Y):
        b = self.b
        if branch == "eci":
            return b.rel_eci(Y)
        if branch in b.rel_names:
            return Y[..., b.rel_slice(branch)]
        return Y[..., b.est_slice(branch)]

    def grid(self, branch):
        return self.rel_from_rows(branch, self.Ym)

    def at(self, branch, t):
        return self.rel_from_rows(branch, self.b.dense(self.i, [t], self.Fr, self.Fl))[0]


def peak_discrepancy(cv, a, ref):
    b = cv.b
    tc = min(b.t_end[a][cv.i], b.t_end[ref][cv.i])
    idx = np.where(cv.tm <= tc + 1e-9)[0]
    tt = cv.tm[idx]
    d = np.linalg.norm(cv.grid(a)[idx, :3] - cv.grid(ref)[idx, :3], axis=1)
    if tc > tt[-1] + 1e-9:
        tt = np.append(tt, tc)
        d = np.append(d, np.linalg.norm(cv.at(a, tc)[:3] - cv.at(ref, tc)[:3]))
    if not np.all(np.isfinite(d)):
        nan = float("nan")
        return {"peak_m": nan, "t_peak_s": nan, "peak_grid_m": nan, "common_end_s": float(tc)}
    k = int(np.argmax(d))
    lo, hi = tt[max(k - 1, 0)], tt[min(k + 1, tt.size - 1)]
    best_t, best = float(tt[k]), float(d[k])
    if hi > lo:
        r = minimize_scalar(lambda s: -float(np.linalg.norm(cv.at(a, s)[:3] - cv.at(ref, s)[:3])), bounds=(lo, hi),
                            method="bounded", options={"xatol": 1e-4})
        if -r.fun > best:
            best_t, best = float(r.x), float(-r.fun)
    return {"peak_m": best, "t_peak_s": best_t, "peak_grid_m": float(d[k]), "common_end_s": float(tc)}


def terminal_arrival(cv, branch):
    b = cv.b
    st = b.status[branch][cv.i]
    out = {"status": st, "t_end_s": float(b.t_end[branch][cv.i])}
    if st != "completed":
        out.update(inside_at_end=False, entry_time_s=float("nan"), dwell_margin_s=float("nan"), arrived=False,
                   final_range_m=float("nan"), final_speed_mps=float("nan"))
        return out
    Sg = cv.grid(branch)
    rng = np.linalg.norm(Sg[:, :3], axis=1)
    spd = np.linalg.norm(Sg[:, 3:], axis=1)
    out["final_range_m"], out["final_speed_mps"] = float(rng[-1]), float(spd[-1])
    g = np.maximum(rng / POS_TOL, spd / VEL_TOL) - 1.0
    inside = g < 0
    if not inside[-1]:
        out.update(inside_at_end=False, entry_time_s=float("nan"), dwell_margin_s=float("nan"), arrived=False)
        return out
    bad = np.where(~inside)[0]
    if bad.size == 0:
        te = 0.0
    else:
        j = int(bad[-1])

        def gf(s):
            x = cv.at(branch, s)
            return float(max(np.linalg.norm(x[:3]) / POS_TOL, np.linalg.norm(x[3:]) / VEL_TOL) - 1.0)
        a0, a1 = cv.tm[j], cv.tm[j + 1]
        f0, f1 = gf(a0), gf(a1)
        if f0 == 0.0:
            te = a0
        elif f1 == 0.0:
            te = a1
        elif f0 * f1 < 0:
            te = float(brentq(gf, a0, a1, xtol=1e-6, rtol=1e-12))
        else:
            te = 0.5 * (a0 + a1)
    dwell = b.T - te
    out.update(inside_at_end=True, entry_time_s=te, dwell_margin_s=dwell - DWELL, arrived=bool(dwell >= DWELL))
    return out


def estimator_feature(cv, est):
    b = cv.b
    if b.status[est][cv.i] != "completed":
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "status": "estimator_failed"}
    if b.status["cw"][cv.i] != "completed":
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "status": "nominal_failed"}
    p = np.linalg.norm(cv.grid(est)[:, :3], axis=1)
    if not np.all(np.isfinite(p)):
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "status": "nonfinite"}
    k = int(np.argmax(p))
    return {"peak_m": float(p[k]), "t_peak_s": float(cv.tm[k]), "status": "ok"}


def saturation_summary(cv):
    b = cv.b
    sat = b.sat_at_update[cv.i]
    near = b.near_switch[cv.i]
    ends = np.append(b.update_times[1:], b.T)
    dur = ends - b.update_times
    return {"sat_time_s": float(np.sum(dur[sat])), "sat_fraction": float(np.sum(dur[sat]) / b.T),
            "n_sat_updates": int(sat.sum()), "n_updates": int(sat.size),
            "n_status_changes": int(np.sum(sat[1:] != sat[:-1])), "n_near_switch_updates": int(near.sum()),
            "saturated_at_start": bool(sat[0]) if sat.size else False}


def case_summary(batch, i, pairs=(("cw", "eci"), ("so", "eci"), ("nerm", "eci"), ("cw", "nerm"), ("so", "nerm")), cv=None):
    cv = CaseView(batch, i) if cv is None else cv
    out = {}
    names = [x for x in batch.branch_names() if x not in ESTIMATORS]
    for br in names:
        for k, v in terminal_arrival(cv, br).items():
            out[f"{br}_{k}"] = v
        if batch.exits(br):
            out[f"{br}_exit_step"] = int(batch.exit_step[br][i])
            out[f"{br}_exit_fallback"] = bool(batch.exit_fallback[br][i])
    for a, r in pairs:
        if a in names and r in names:
            for k, v in peak_discrepancy(cv, a, r).items():
                out[f"{a}_vs_{r}_{k}"] = v
    for e in batch.estimators:
        for k, v in estimator_feature(cv, e).items():
            out[f"{e}_{k}"] = v
    if "cw" in names:
        for k, v in saturation_summary(cv).items():
            out[f"cw_{k}"] = v
    return out, cv


__all__ = ["CaseView", "monitoring_grid", "peak_discrepancy", "terminal_arrival", "estimator_feature",
           "saturation_summary", "case_summary"]
