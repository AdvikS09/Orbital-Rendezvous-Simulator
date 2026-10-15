"""Diagnostics and outcome definitions on a controlled common time basis.

Every branch is evaluated from its solver dense output on one uniform grid
(``dt_eval``, default 0.5 s) plus the exact horizon end. Extremes (peak
discrepancy, closest approach) are refined with a bounded scalar optimiser on
the dense output around the grid extreme; threshold crossings (arrival entry,
saturation switches) are refined with Brent root-finding on the dense output.
Delta-v is taken from the solver-integrated accumulator and cross-checked with
composite Simpson quadrature on the grid.

Outcome definitions (configurable; study assumptions, not docking standards):

* ``legacy position-only``: the manuscript's test, evaluated on the
  manuscript's sample grid (default 2001 samples/orbit): converged if the
  position error is below ``pos_tol`` at the last sample and the last sample
  above tolerance is not the final one (``src.validation.convergence_time``).
* ``position-only (refined)``: same definition on the fine grid with the final
  entry time root-refined.
* ``terminal (primary)``: position < ``pos_tol`` AND relative speed <
  ``vel_tol`` continuously from the arrival time to the end of the horizon,
  with at least ``dwell`` seconds remaining after arrival. Branches that end
  early (altitude floor, solver failure) are not arrivals; numerical failure
  is reported separately from physical non-arrival.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class OutcomeCriteria:
    pos_tol: float = 1.0       # [m]
    vel_tol: float = 0.01      # [m/s]
    dwell: float = 300.0       # [s] minimum remaining time inside the terminal set
    legacy_samples: int = 2001  # manuscript separation-sweep grid (samples per orbit)

    def as_dict(self):
        return asdict(self)


@dataclass
class Trace:
    """A branch sampled on the common grid."""
    t: np.ndarray
    S: np.ndarray       # (N,6) LVLH relative state
    raw: np.ndarray     # (N,3) unsaturated command
    app: np.ndarray     # (N,3) applied command
    dv: np.ndarray      # (N,) cumulative Delta-v (solver accumulator)

    @property
    def pos(self):
        return np.linalg.norm(self.S[:, :3], axis=1)

    @property
    def spd(self):
        return np.linalg.norm(self.S[:, 3:], axis=1)


def common_grid(t_end, dt_eval=0.5):
    k = int(math.floor(t_end / dt_eval))
    t = np.arange(k + 1) * dt_eval
    if t[-1] < t_end - 1e-9:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    return t


def make_trace(sim, t):
    if not sim.segments:   # failed before any accepted step: one-point trace at the initial state
        S = np.asarray(sim.rel0, dtype=float)[None, :]
        tt = np.array([0.0])
        if sim.cmd.law is not None and sim.cmd.mode in ("continuous", "sampled"):
            raw, app = sim.cmd.law.raw_vec(S), sim.cmd.law.applied_vec(S)
        else:
            raw = app = np.zeros((1, 3))
        return Trace(tt, S, raw, app, np.zeros(1))
    tt = t[t <= sim.t_final + 1e-9]
    # a branch that left the physical domain early ends exactly at its exit time
    if sim.t_final < t[-1] - 1e-9 and (tt.size == 0 or sim.t_final > tt[-1] + 1e-9):
        tt = np.append(tt, sim.t_final)
    S = sim.states(tt)
    raw, app = sim.commands(tt, S)
    dv = sim.dv_cumulative(tt)
    return Trace(tt, S, raw, app, dv)


# ---------------------------------------------------------------------------
# small refinement helpers
# ---------------------------------------------------------------------------


def _refine_extreme(fun, t, i, maximize=True):
    """Refine an extreme of scalar fun(t) near grid index i. Returns (t*, f*)."""
    lo = t[max(i - 1, 0)]
    hi = t[min(i + 1, t.size - 1)]
    if hi <= lo:
        return float(t[i]), float(fun(t[i]))
    sgn = -1.0 if maximize else 1.0
    r = minimize_scalar(lambda s: sgn * fun(s), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-4})
    f_grid = float(fun(t[i]))
    f_ref = float(fun(r.x))
    if (maximize and f_ref >= f_grid) or (not maximize and f_ref <= f_grid):
        return float(r.x), f_ref
    return float(t[i]), f_grid


def _root(fun, a, b):
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        return 0.5 * (a + b)
    return brentq(fun, a, b, xtol=1e-6, rtol=1e-12)


# ---------------------------------------------------------------------------
# single-branch diagnostics
# ---------------------------------------------------------------------------


def legacy_grid(sim, horizon, crit):
    """Manuscript grid: (legacy_samples - 1) intervals per orbit over the horizon."""
    orbits = horizon * sim.n / (2.0 * math.pi)
    k = int(round((crit.legacy_samples - 1) * orbits))
    return np.linspace(0.0, horizon, max(k, 1) + 1)


def legacy_arrival(sim, horizon, crit: OutcomeCriteria):
    """Manuscript position-only classification on its own sample grid."""
    tl = legacy_grid(sim, horizon, crit)
    if sim.t_final < horizon - 1e-6:
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


def _last_entry(gvals, t, gfun):
    """Given g on grid (g<0 means inside), return (inside_at_end, t_entry)."""
    inside = gvals < 0
    if not inside[-1]:
        return False, float("nan")
    bad = np.where(~inside)[0]
    if bad.size == 0:
        return True, 0.0
    j = bad[-1]
    return True, float(_root(gfun, t[j], t[j + 1]))


def branch_diagnostics(sim, tr: Trace, horizon, crit: OutcomeCriteria):
    t = tr.t
    pos, spd = tr.pos, tr.spd
    out = {}
    complete = sim.status == "ok" and sim.t_final >= horizon - 1e-6
    out["completed_horizon"] = bool(complete)
    out["numerical_failure"] = sim.status == "failed"

    def st(s):
        return sim.states([s])[0]

    # position-only (refined)
    gpos = lambda s: float(np.linalg.norm(st(s)[:3]) - crit.pos_tol)
    if complete:
        ins, t_pos = _last_entry(pos - crit.pos_tol, t, gpos)
    else:
        ins, t_pos = False, float("nan")
    out["pos_only_arrived"] = bool(ins)
    out["pos_only_arrival_time_s"] = t_pos

    # terminal criterion (primary)
    def gterm(s):
        x = st(s)
        return float(max(np.linalg.norm(x[:3]) / crit.pos_tol, np.linalg.norm(x[3:]) / crit.vel_tol) - 1.0)
    gvals = np.maximum(pos / crit.pos_tol, spd / crit.vel_tol) - 1.0
    if complete:
        ins_t, t_arr = _last_entry(gvals, t, gterm)
    else:
        ins_t, t_arr = False, float("nan")
    remain = horizon - t_arr if ins_t else float("nan")
    out["terminal_inside_at_end"] = bool(ins_t)
    out["terminal_arrival_time_s"] = t_arr
    out["terminal_dwell_s"] = remain
    out["terminal_arrived"] = bool(ins_t and remain >= crit.dwell)
    out["terminal_dwell_margin_s"] = remain - crit.dwell if ins_t else float("nan")
    # how deep inside the terminal set after arrival (normalised; >0 = margin)
    if ins_t:
        sel = t >= t_arr
        out["terminal_min_margin"] = float(-gvals[sel].max()) if sel.any() else float("nan")
    else:
        out["terminal_min_margin"] = float("nan")

    # terminal state, closest approach
    out["final_pos_m"] = float(pos[-1])
    out["final_speed_mps"] = float(spd[-1])
    i = int(np.argmin(pos))
    t_ca, ca = _refine_extreme(lambda s: float(np.linalg.norm(st(s)[:3])), t, i, maximize=False)
    out["closest_approach_m"] = ca
    out["closest_approach_time_s"] = t_ca
    out["max_range_m"] = float(pos.max())
    out["max_range_over_R"] = float(pos.max() / sim.R)
    out["speed_at_first_1m_entry_mps"] = float(spd[np.argmax(pos < crit.pos_tol)]) if (pos < crit.pos_tol).any() else float("nan")

    # commands
    rmag = np.linalg.norm(tr.raw, axis=1)
    amag = np.linalg.norm(tr.app, axis=1)
    out["peak_raw_acc_mps2"] = float(rmag.max())
    out["peak_applied_acc_mps2"] = float(amag.max())
    out["dv_aug_mps"] = float(tr.dv[-1])
    out["dv_simpson_mps"] = float(simpson(amag, x=t)) if t.size > 2 else float("nan")
    out.update(saturation_stats(sim, tr, rmag))
    # Second-order gravity residual (the leading term CW neglects) evaluated on
    # this branch's own trajectory. On the CW branch it needs only the CW run
    # plus an analytic expression, so it is admissible as a screening input.
    from .dynamics import relative_acc_vectorized
    q = np.linalg.norm(relative_acc_vectorized("so", sim.n, sim.R, tr.S)
                       - relative_acc_vectorized("cw", sim.n, sim.R, tr.S), axis=1)
    law = sim.cmd.law
    if law is not None and law.a_max is not None and sim.cmd.mode in ("continuous", "sampled"):
        satmask = rmag > law.a_max   # sampled: rmag is the held raw command
    else:
        satmask = np.zeros_like(q, dtype=bool)
    out["q2_peak_mps2"] = float(q.max())
    out["q2_impulse_all_mps"] = float(simpson(q, x=t)) if t.size > 2 else float("nan")
    out["q2_impulse_sat_mps"] = float(simpson(q * satmask, x=t)) if t.size > 2 else float("nan")
    return out


def saturation_stats(sim, tr: Trace, rmag):
    law = sim.cmd.law
    keys = ("sat_total_s", "sat_longest_s", "sat_n_intervals", "sat_initial_s", "sat_fraction")
    if sim.cmd.mode not in ("continuous", "sampled") or law is None or law.a_max is None:
        return {k: 0.0 for k in keys} | {"sat_n_intervals": 0}
    am = law.a_max
    t = tr.t
    if sim.cmd.mode == "sampled":
        # piecewise-constant: use update instants exactly
        tu = np.array([s.t0 for s in sim.segments] + [sim.t_final])
        Su = sim.states(tu[:-1])
        r = np.linalg.norm(law.raw_vec(Su), axis=1)
        sat = r > am
        intervals = []
        k = 0
        while k < sat.size:
            if sat[k]:
                j = k
                while j + 1 < sat.size and sat[j + 1]:
                    j += 1
                intervals.append((tu[k], tu[j + 1]))
                k = j + 1
            else:
                k += 1
    else:
        h = lambda s: float(np.linalg.norm(law.raw_vec(sim.states([s]))[0]) - am)
        sat = rmag > am
        intervals = []
        k = 0
        N = sat.size
        while k < N:
            if sat[k]:
                j = k
                while j + 1 < N and sat[j + 1]:
                    j += 1
                t0 = t[k] if k == 0 else _root(h, t[k - 1], t[k])
                t1 = t[j] if j == N - 1 else _root(h, t[j], t[j + 1])
                intervals.append((t0, t1))
                k = j + 1
            else:
                k += 1
    durs = [b - a for a, b in intervals]
    span = t[-1] - t[0] if t.size > 1 else 1.0
    return {
        "sat_total_s": float(sum(durs)),
        "sat_longest_s": float(max(durs)) if durs else 0.0,
        "sat_n_intervals": len(durs),
        # duration of the saturated interval that starts at t=0 (0 if unsaturated at release)
        "sat_initial_s": float(intervals[0][1] - intervals[0][0]) if intervals and intervals[0][0] <= t[0] + 1e-9 else 0.0,
        "sat_fraction": float(sum(durs) / span) if span > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# paired diagnostics
# ---------------------------------------------------------------------------


def paired_diagnostics(sim_a, tr_a: Trace, sim_b, tr_b: Trace, horizon, crit: OutcomeCriteria):
    """Discrepancy of branch a (e.g. CW) relative to branch b (reference),
    evaluated over the common horizon (up to the earlier branch end)."""
    t_end = min(tr_a.t[-1], tr_b.t[-1])
    short, other, sim_o = (tr_a, tr_b, sim_b) if tr_a.t[-1] <= tr_b.t[-1] else (tr_b, tr_a, sim_a)
    t = short.t[short.t <= t_end + 1e-9]
    n = t.size
    Ss = short.S[:n]
    Us = short.app[:n]
    if other.t.size >= n and np.array_equal(other.t[:n], t):
        So, Uo = other.S[:n], other.app[:n]
    else:
        So = sim_o.states(t)
        Uo = sim_o.commands(t, So)[1]
    if short is tr_a:
        Sa, Sb, Ua, Ub = Ss, So, Us, Uo
    else:
        Sa, Sb, Ua, Ub = So, Ss, Uo, Us
    dp = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
    dvel = np.linalg.norm(Sa[:, 3:] - Sb[:, 3:], axis=1)
    du = np.linalg.norm(Ua - Ub, axis=1)
    out = {}
    i = int(np.argmax(dp))
    f = lambda s: float(np.linalg.norm(sim_a.states([s])[0, :3] - sim_b.states([s])[0, :3]))
    t_pk, pk = _refine_extreme(f, t, i, maximize=True)
    out["max_pos_disc_m"] = pk
    out["max_pos_disc_grid_m"] = float(dp[i])
    out["t_max_pos_disc_s"] = t_pk
    # legacy-grid peak (manuscript metric on 2001 samples)
    tl = legacy_grid(sim_a, horizon, crit)
    tl = tl[tl <= min(sim_a.t_final, sim_b.t_final) + 1e-9]
    dl = np.linalg.norm(sim_a.states(tl)[:, :3] - sim_b.states(tl)[:, :3], axis=1)
    out["max_pos_disc_legacy_grid_m"] = float(dl.max())
    j = int(np.argmax(dvel))
    g = lambda s: float(np.linalg.norm(sim_a.states([s])[0, 3:] - sim_b.states([s])[0, 3:]))
    t_vk, vk = _refine_extreme(g, t, j, maximize=True)
    out["max_vel_disc_mps"] = vk
    out["t_max_vel_disc_s"] = t_vk
    out["final_pos_disc_m"] = float(dp[-1])
    out["max_cmd_diff_mps2"] = float(du.max())
    out["t_max_cmd_diff_s"] = float(t[int(np.argmax(du))])
    out["int_cmd_diff_mps"] = float(simpson(du, x=t)) if n > 2 else float("nan")
    out["rms_cmd_diff_mps2"] = float(np.sqrt(simpson(du**2, x=t) / (t[-1] - t[0]))) if n > 2 else float("nan")
    out["common_horizon_s"] = float(t[-1])
    return out


def classify(cw_success, nl_success):
    if cw_success and nl_success:
        return "both_succeed"
    if cw_success and not nl_success:
        return "cw_only"        # CW succeeds, nonlinear fails (falsely favourable CW)
    if not cw_success and nl_success:
        return "nl_only"        # CW fails, nonlinear succeeds (falsely unfavourable CW)
    return "both_fail"


__all__ = ["OutcomeCriteria", "Trace", "common_grid", "make_trace", "legacy_arrival",
           "branch_diagnostics", "saturation_stats", "paired_diagnostics", "classify"]
