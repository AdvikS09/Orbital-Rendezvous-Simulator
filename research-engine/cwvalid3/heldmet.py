import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .sampled import ESTIM

POSTOL, VELTOL, DWELL = 1.0, 0.01, 300.0
DTMON = 0.5


def mongrid(T, dt=DTMON):
    k = int(np.floor(T / dt + 1e-9))
    t = np.arange(k + 1) * dt
    if t[-1] < T - 1e-9:
        t = np.append(t, T)
    else:
        t[-1] = T
    return t


class CaseView:
    def __init__(self, batch, i, dtmon=DTMON):
        self.b, self.i = batch, i
        self.Fr, self.Fl = batch.casederivs(i)
        self.tm = mongrid(batch.T, dtmon)
        self.Ym = batch.dense(i, self.tm, self.Fr, self.Fl)

    def relrows(self, branch, Y):
        b = self.b
        if branch == "eci":
            return b.releci(Y)
        if branch in b.relnames:
            return Y[..., b.relslice(branch)]
        return Y[..., b.estslice(branch)]

    def grid(self, branch):
        return self.relrows(branch, self.Ym)

    def at(self, branch, t):
        return self.relrows(branch, self.b.dense(self.i, [t], self.Fr, self.Fl))[0]


def peakdisc(cv, a, ref):
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
    bestt, best = float(tt[k]), float(d[k])
    if hi > lo:
        r = minimize_scalar(lambda s: -float(np.linalg.norm(cv.at(a, s)[:3] - cv.at(ref, s)[:3])), bounds=(lo, hi),
                            method="bounded", options={"xatol": 1e-4})
        if -r.fun > best:
            bestt, best = float(r.x), float(-r.fun)
    return {"peak_m": best, "t_peak_s": bestt, "peak_grid_m": float(d[k]), "common_end_s": float(tc)}


def termarr(cv, branch):
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
    g = np.maximum(rng / POSTOL, spd / VELTOL) - 1.0
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
            return float(max(np.linalg.norm(x[:3]) / POSTOL, np.linalg.norm(x[3:]) / VELTOL) - 1.0)
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


def estimfeat(cv, est):
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


def satsumm(cv):
    b = cv.b
    sat = b.sat_at_update[cv.i]
    near = b.nearswitch[cv.i]
    ends = np.append(b.updatetimes[1:], b.T)
    dur = ends - b.updatetimes
    return {"sat_time_s": float(np.sum(dur[sat])), "sat_fraction": float(np.sum(dur[sat]) / b.T),
            "n_sat_updates": int(sat.sum()), "n_updates": int(sat.size),
            "n_status_changes": int(np.sum(sat[1:] != sat[:-1])), "n_near_switch_updates": int(near.sum()),
            "saturated_at_start": bool(sat[0]) if sat.size else False}


def casesumm(batch, i, pairs=(("cw", "eci"), ("so", "eci"), ("nerm", "eci"), ("cw", "nerm"), ("so", "nerm")), cv=None):
    cv = CaseView(batch, i) if cv is None else cv
    out = {}
    names = [x for x in batch.branchnames() if x not in ESTIM]
    for br in names:
        for k, v in termarr(cv, br).items():
            out[f"{br}_{k}"] = v
        if batch.exits(br):
            out[f"{br}_exit_step"] = int(batch.exit_step[br][i])
            out[f"{br}_exit_fallback"] = bool(batch.exit_fallback[br][i])
    for a, r in pairs:
        if a in names and r in names:
            for k, v in peakdisc(cv, a, r).items():
                out[f"{a}_vs_{r}_{k}"] = v
    for e in batch.ests:
        for k, v in estimfeat(cv, e).items():
            out[f"{e}_{k}"] = v
    if "cw" in names:
        for k, v in satsumm(cv).items():
            out[f"cw_{k}"] = v
    return out, cv
