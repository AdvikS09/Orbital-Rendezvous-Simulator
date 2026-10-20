import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class OutcCrit:
    pos_tol: float = 1.0
    vel_tol: float = 0.01
    dwell: float = 300.0
    legacy_samples: int = 2001

    def asdict2(self):
        return asdict(self)


@dataclass
class Trace:
    t: np.ndarray
    S: np.ndarray
    raw: np.ndarray
    app: np.ndarray
    dv: np.ndarray

    @property
    def pos(self):
        return np.linalg.norm(self.S[:, :3], axis=1)

    @property
    def spd(self):
        return np.linalg.norm(self.S[:, 3:], axis=1)


def commongrid(t_end, dt_eval=0.5):
    k = int(math.floor(t_end / dt_eval))
    t = np.arange(k + 1) * dt_eval
    if t[-1] < t_end - 1e-9:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    return t


def maketrace(sim, t):
    if not sim.segments:
        S = np.asarray(sim.rel0, dtype=float)[None, :]
        tt = np.array([0.0])
        if sim.cmd.law is not None and sim.cmd.mode in ("continuous", "sampled"):
            raw, a = sim.cmd.law.rawvec(S), sim.cmd.law.applvec(S)
        else:
            raw = a = np.zeros((1, 3))
        return Trace(tt, S, raw, a, np.zeros(1))
    tt = t[t <= sim.tfinal + 1e-9]
    if sim.tfinal < t[-1] - 1e-9 and (tt.size == 0 or sim.tfinal > tt[-1] + 1e-9):
        tt = np.append(tt, sim.tfinal)
    S = sim.states(tt)
    raw, a = sim.commands(tt, S)
    dv = sim.dvcum(tt)
    return Trace(tt, S, raw, a, dv)


def refineext(fun, t, i, maximize=True):
    lo = t[max(i - 1, 0)]
    hi = t[min(i + 1, t.size - 1)]
    if hi <= lo:
        return float(t[i]), float(fun(t[i]))
    sgn = -1.0 if maximize else 1.0
    r = minimize_scalar(lambda s: sgn * fun(s), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-4})
    fgrid = float(fun(t[i]))
    fref = float(fun(r.x))
    if (maximize and fref >= fgrid) or (not maximize and fref <= fgrid):
        return float(r.x), fref
    return float(t[i]), fgrid


def root(fun, a, b):
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        return 0.5 * (a + b)
    return brentq(fun, a, b, xtol=1e-6, rtol=1e-12)


def legacygrid(sim, horizon, crit):
    orbits = horizon * sim.n / (2.0 * math.pi)
    k = int(round((crit.legacy_samples - 1) * orbits))
    return np.linspace(0.0, horizon, max(k, 1) + 1)


def legacyarr(sim, horizon, crit):
    tl = legacygrid(sim, horizon, crit)
    if sim.tfinal < horizon - 1e-6:
        return {"legacy_arrived": False, "legacy_conv_time_s": float("nan"),
                "legacy_final_pos_m": float("nan")}
    pos = np.linalg.norm(sim.states(tl)[:, :3], axis=1)
    above = np.where(pos > crit.pos_tol)[0]
    if above.size == 0:
        conv = 0.0
    elif above[-1] >= tl.size - 1:
        conv = float("nan")
    else:
        conv = float(tl[above[-1] + 1])
    return {"legacy_arrived": bool(np.isfinite(conv) and pos[-1] < crit.pos_tol),
            "legacy_conv_time_s": conv, "legacy_final_pos_m": float(pos[-1])}


def lastentry(gvals, t, gfun):
    inside = gvals < 0
    if not inside[-1]:
        return False, float("nan")
    bad = np.where(~inside)[0]
    if bad.size == 0:
        return True, 0.0
    j = bad[-1]
    return True, float(root(gfun, t[j], t[j + 1]))


def branchdiags(sim, tr, horizon, crit):
    t = tr.t
    pos, spd = tr.pos, tr.spd
    out = {}
    complete = sim.status == "ok" and sim.tfinal >= horizon - 1e-6
    out["completed_horizon"] = bool(complete)
    out["numerical_failure"] = sim.status == "failed"

    def st(s):
        return sim.states([s])[0]

    gpos = lambda s: float(np.linalg.norm(st(s)[:3]) - crit.pos_tol)
    if complete:
        ins, tpos = lastentry(pos - crit.pos_tol, t, gpos)
    else:
        ins, tpos = False, float("nan")
    out["pos_only_arrived"] = bool(ins)
    out["pos_only_arrival_time_s"] = tpos

    def gterm(s):
        x = st(s)
        return float(max(np.linalg.norm(x[:3]) / crit.pos_tol, np.linalg.norm(x[3:]) / crit.vel_tol) - 1.0)
    gvals = np.maximum(pos / crit.pos_tol, spd / crit.vel_tol) - 1.0
    if complete:
        inst, tarr = lastentry(gvals, t, gterm)
    else:
        inst, tarr = False, float("nan")
    remain = horizon - tarr if inst else float("nan")
    out["terminal_inside_at_end"] = bool(inst)
    out["terminal_arrival_time_s"] = tarr
    out["terminal_dwell_s"] = remain
    out["terminal_arrived"] = bool(inst and remain >= crit.dwell)
    out["terminal_dwell_margin_s"] = remain - crit.dwell if inst else float("nan")
    if inst:
        sel = t >= tarr
        out["terminal_min_margin"] = float(-gvals[sel].max()) if sel.any() else float("nan")
    else:
        out["terminal_min_margin"] = float("nan")

    out["final_pos_m"] = float(pos[-1])
    out["final_speed_mps"] = float(spd[-1])
    i = int(np.argmin(pos))
    tca, ca = refineext(lambda s: float(np.linalg.norm(st(s)[:3])), t, i, maximize=False)
    out["closest_approach_m"] = ca
    out["closest_approach_time_s"] = tca
    out["max_range_m"] = float(pos.max())
    out["max_range_over_R"] = float(pos.max() / sim.R)
    out["speed_at_first_1m_entry_mps"] = float(spd[np.argmax(pos < crit.pos_tol)]) if (pos < crit.pos_tol).any() else float("nan")

    rmag = np.linalg.norm(tr.raw, axis=1)
    amag = np.linalg.norm(tr.app, axis=1)
    out["peak_raw_acc_mps2"] = float(rmag.max())
    out["peak_applied_acc_mps2"] = float(amag.max())
    out["dv_aug_mps"] = float(tr.dv[-1])
    out["dv_simpson_mps"] = float(simpson(amag, x=t)) if t.size > 2 else float("nan")
    out.update(satstats(sim, tr, rmag))
    from .dynamics import relaccvec
    q = np.linalg.norm(relaccvec("so", sim.n, sim.R, tr.S)
                       - relaccvec("cw", sim.n, sim.R, tr.S), axis=1)
    law = sim.cmd.law
    if law is not None and law.a_max is not None and sim.cmd.mode in ("continuous", "sampled"):
        satmask = rmag > law.a_max
    else:
        satmask = np.zeros_like(q, dtype=bool)
    out["q2_peak_mps2"] = float(q.max())
    out["q2_impulse_all_mps"] = float(simpson(q, x=t)) if t.size > 2 else float("nan")
    out["q2_impulse_sat_mps"] = float(simpson(q * satmask, x=t)) if t.size > 2 else float("nan")
    return out


def satstats(sim, tr, rmag):
    law = sim.cmd.law
    keys = ("sat_total_s", "sat_longest_s", "sat_n_intervals", "sat_initial_s", "sat_fraction")
    if sim.cmd.mode not in ("continuous", "sampled") or law is None or law.a_max is None:
        return {k: 0.0 for k in keys} | {"sat_n_intervals": 0}
    am = law.a_max
    t = tr.t
    if sim.cmd.mode == "sampled":
        tu = np.array([s.t0 for s in sim.segments] + [sim.tfinal])
        Su = sim.states(tu[:-1])
        r = np.linalg.norm(law.rawvec(Su), axis=1)
        sat = r > am
        ivals = []
        k = 0
        while k < sat.size:
            if sat[k]:
                j = k
                while j + 1 < sat.size and sat[j + 1]:
                    j += 1
                ivals.append((tu[k], tu[j + 1]))
                k = j + 1
            else:
                k += 1
    else:
        h = lambda s: float(np.linalg.norm(law.rawvec(sim.states([s]))[0]) - am)
        sat = rmag > am
        ivals = []
        k = 0
        N = sat.size
        while k < N:
            if sat[k]:
                j = k
                while j + 1 < N and sat[j + 1]:
                    j += 1
                t0 = t[k] if k == 0 else root(h, t[k - 1], t[k])
                t1 = t[j] if j == N - 1 else root(h, t[j], t[j + 1])
                ivals.append((t0, t1))
                k = j + 1
            else:
                k += 1
    durs = [b - a for a, b in ivals]
    span = t[-1] - t[0] if t.size > 1 else 1.0
    return {
        "sat_total_s": float(sum(durs)),
        "sat_longest_s": float(max(durs)) if durs else 0.0,
        "sat_n_intervals": len(durs),
        "sat_initial_s": float(ivals[0][1] - ivals[0][0]) if ivals and ivals[0][0] <= t[0] + 1e-9 else 0.0,
        "sat_fraction": float(sum(durs) / span) if span > 0 else 0.0,
    }


def pairdiags(sima, tra, simb, trb, horizon, crit):
    tend = min(tra.t[-1], trb.t[-1])
    short, other, simo = (tra, trb, simb) if tra.t[-1] <= trb.t[-1] else (trb, tra, sima)
    t = short.t[short.t <= tend + 1e-9]
    n = t.size
    Ss = short.S[:n]
    Us = short.app[:n]
    if other.t.size >= n and np.array_equal(other.t[:n], t):
        So, Uo = other.S[:n], other.app[:n]
    else:
        So = simo.states(t)
        Uo = simo.commands(t, So)[1]
    if short is tra:
        Sa, Sb, Ua, Ub = Ss, So, Us, Uo
    else:
        Sa, Sb, Ua, Ub = So, Ss, Uo, Us
    dp = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
    dvel = np.linalg.norm(Sa[:, 3:] - Sb[:, 3:], axis=1)
    du = np.linalg.norm(Ua - Ub, axis=1)
    out = {}
    i = int(np.argmax(dp))
    f = lambda s: float(np.linalg.norm(sima.states([s])[0, :3] - simb.states([s])[0, :3]))
    tpk, pk = refineext(f, t, i, maximize=True)
    out["max_pos_disc_m"] = pk
    out["max_pos_disc_grid_m"] = float(dp[i])
    out["t_max_pos_disc_s"] = tpk
    tl = legacygrid(sima, horizon, crit)
    tl = tl[tl <= min(sima.tfinal, simb.tfinal) + 1e-9]
    dl = np.linalg.norm(sima.states(tl)[:, :3] - simb.states(tl)[:, :3], axis=1)
    out["max_pos_disc_legacy_grid_m"] = float(dl.max())
    j = int(np.argmax(dvel))
    g = lambda s: float(np.linalg.norm(sima.states([s])[0, 3:] - simb.states([s])[0, 3:]))
    tvk, vk = refineext(g, t, j, maximize=True)
    out["max_vel_disc_mps"] = vk
    out["t_max_vel_disc_s"] = tvk
    out["final_pos_disc_m"] = float(dp[-1])
    out["max_cmd_diff_mps2"] = float(du.max())
    out["t_max_cmd_diff_s"] = float(t[int(np.argmax(du))])
    out["int_cmd_diff_mps"] = float(simpson(du, x=t)) if n > 2 else float("nan")
    out["rms_cmd_diff_mps2"] = float(np.sqrt(simpson(du**2, x=t) / (t[-1] - t[0]))) if n > 2 else float("nan")
    out["common_horizon_s"] = float(t[-1])
    return out


def classify(cwsuccess, nlsuccess):
    if cwsuccess and nlsuccess:
        return "both_succeed"
    if cwsuccess and not nlsuccess:
        return "cw_only"
    if not cwsuccess and nlsuccess:
        return "nl_only"
    return "both_fail"
