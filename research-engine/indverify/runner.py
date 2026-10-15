"""run_case: integrate the branches of one scenario with one solver setting
and compute every assessed quantity of SPEC section 8."""

import math
import time
from dataclasses import dataclass

import numpy as np

from ._version import __version__
from .assess import (MaxTracker, SignTracker, assess_arrival, locate_root,
                     refine_max, saturated_time, sign_intervals, time_grid)
from .constants import (COMPLETED, DEFAULT_ALTITUDE, DEFAULT_MAX_STEP,
                        FINE_DT, FINE_ROOT_XTOL, GRID_SNAP, PROD_DT,
                        PROD_ENTRY_XTOL, TERMINAL_POS_M, TERMINAL_VEL_MPS)
from .control import PDController
from .frames import TargetOrbit
from .integrators import integrate_adaptive, integrate_rk4
from .models import BRANCHES, make_model

CONVENTIONS = ("prod", "fine")
SOLVER_NAMES = ("radau", "dop853", "rk4")
_NAN = math.nan


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    sid: str
    rel0: np.ndarray
    wn_ratio: float
    a_max: object
    horizon_orbits: float
    altitude: float

    @classmethod
    def from_dict(cls, d):
        a_max = d.get("a_max")
        a_max = None if a_max is None else float(a_max)
        alt = d.get("altitude")
        return cls(sid=str(d["sid"]),
                   rel0=np.asarray(d["rel0"], dtype=float).reshape(6),
                   wn_ratio=float(d["wn_ratio"]),
                   a_max=a_max,
                   horizon_orbits=float(d["horizon_orbits"]),
                   altitude=DEFAULT_ALTITUDE if alt is None else float(alt))


@dataclass
class SolverSetting:
    """Solver setting.  Keys: solver, rtol (radau/dop853), h (rk4), max_step
    (default 20 s), label, and the optional ``ref_rtol``: relative tolerance
    for the inertial reference branch only (default: rtol).

    Why ``ref_rtol`` exists: with the SPEC section 7 tolerances the reference
    branch's local error is about rtol * |r| ~ rtol * 6.8e6 m per component of
    each spacecraft, and under feedback the target and chaser errors do not
    cancel in rho.  At the coarse rungs of a ladder, ref-derived quantities
    (x_vs_ref, ref switch / entry / dwell times) are then dominated by solver
    error, not model error.  ref_rtol tightens only that branch; it is
    recorded in the summary as ``ref_rtol``.
    """

    solver: str
    rtol: float
    h: float
    max_step: float
    label: str
    ref_rtol: float = _NAN

    @classmethod
    def from_dict(cls, d):
        solver = str(d["solver"]).lower()
        if solver not in SOLVER_NAMES:
            raise ValueError("unknown solver %r (expected one of %s)" % (solver, SOLVER_NAMES))
        ms = d.get("max_step")
        max_step = DEFAULT_MAX_STEP if ms is None else float(ms)
        if solver == "rk4":
            if d.get("ref_rtol") is not None:
                raise ValueError("ref_rtol applies to radau/dop853 only")
            rtol, h, ref_rtol = _NAN, float(d["h"]), _NAN
            default_label = "rk4_%g" % h
        else:
            rtol, h = float(d["rtol"]), _NAN
            rr = d.get("ref_rtol")
            ref_rtol = rtol if rr is None else float(rr)
            if not (rtol > 0.0 and ref_rtol > 0.0):
                raise ValueError("rtol and ref_rtol must be positive")
            default_label = "%s_%g" % (solver, rtol)
            if ref_rtol != rtol:
                default_label += "_ref%g" % ref_rtol
        label = d.get("label") or default_label
        return cls(solver=solver, rtol=rtol, h=h, max_step=max_step, label=str(label),
                   ref_rtol=ref_rtol)

    def rtol_for(self, branch):
        """Relative tolerance used for ``branch`` (adaptive solvers)."""
        return self.ref_rtol if branch == "ref" else self.rtol


def normalise_branches(branches):
    if isinstance(branches, str):
        branches = [b.strip() for b in branches.split(",") if b.strip()]
    out = []
    for b in branches:
        if b not in BRANCHES:
            raise ValueError("unknown branch %r (expected a subset of %s)" % (b, BRANCHES))
        if b not in out:
            out.append(b)
    if not out:
        raise ValueError("no branches requested")
    return tuple(out)


def comparison_pairs(branches):
    """(a, 'ref') for every non-reference branch, and ('cw', 'so') if both present."""
    pairs = []
    if "ref" in branches:
        pairs.extend((a, "ref") for a in branches if a != "ref")
    if "cw" in branches and "so" in branches:
        pairs.append(("cw", "so"))
    return pairs


def integrate_branch(model, setting, T):
    if setting.solver == "rk4":
        return integrate_rk4(model, T, setting.h)
    return integrate_adaptive(model, setting.solver, T, setting.rtol_for(model.name),
                              max_step=setting.max_step)


# ---------------------------------------------------------------------------
# Evaluation of a branch's continuous solution
# ---------------------------------------------------------------------------
def _norm(a):
    """Euclidean norm over the first (component) axis."""
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


class BranchEvaluator:
    """rho, rhodot, applied command, g and g_sat from a branch's dense solution."""

    def __init__(self, model, run):
        self.model = model
        self.run = run
        self.ctrl = model.ctrl
        self.has_sat = model.has_sat

    def rel(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        with np.errstate(all="ignore"):
            return self.model.rel_many(self.run.sol(t))

    def derived(self, rho, rhod):
        with np.errstate(all="ignore"):
            a_raw = self.ctrl.raw(rho, rhod)
            a = self.ctrl.saturate(a_raw)
            g = np.maximum(_norm(rho) / TERMINAL_POS_M, _norm(rhod) / TERMINAL_VEL_MPS) - 1.0
            gsat = _norm(a_raw) - self.ctrl.a_max if self.has_sat else None
            return {"a": a, "anorm": _norm(a), "g": g, "gsat": gsat}

    def point(self, s):
        rho, rhod = self.rel(np.array([float(s)]))
        return rho[:, 0], rhod[:, 0], self.derived(rho, rhod)

    def g_at(self, s):
        return float(self.point(s)[2]["g"][0])

    def gsat_at(self, s):
        return float(self.point(s)[2]["gsat"][0])

    def accn_at(self, s):
        return float(self.point(s)[2]["anorm"][0])


def _pair_functions(ea, eb):
    def dpos(s):
        return float(np.linalg.norm(ea.point(s)[0] - eb.point(s)[0]))

    def dvel(s):
        return float(np.linalg.norm(ea.point(s)[1] - eb.point(s)[1]))

    def dcmd(s):
        return abs(ea.accn_at(s) - eb.accn_at(s))

    return dpos, dvel, dcmd


# ---------------------------------------------------------------------------
# Fine-grid pass (chunked, all branches and pairs at once)
# ---------------------------------------------------------------------------
def _fine_pass(evals, branches, pairs, t_end, t_common, dt, chunk=50_000):
    snap = GRID_SNAP
    t_max = max(t_end[b] for b in branches)
    K = int(math.floor(t_max / dt)) + 2
    base = dt * np.arange(K + 1, dtype=float)
    g_tr = {b: SignTracker() for b in branches}
    s_tr = {b: SignTracker() for b in branches if evals[b].has_sat}
    a_tr = {b: MaxTracker() for b in branches}
    p_tr = {p: MaxTracker() for p in pairs}

    def feed_branch(b, tt):
        rho, rhod = evals[b].rel(tt)
        d = evals[b].derived(rho, rhod)
        g_tr[b].feed(tt, d["g"])
        if b in s_tr:
            s_tr[b].feed(tt, d["gsat"])
        a_tr[b].feed(d["anorm"])
        return rho

    with np.errstate(all="ignore"):
        for i0 in range(0, base.size, chunk):
            tc = base[i0:i0 + chunk]
            if tc[0] >= t_max - snap:
                break
            cache = {}
            for b in branches:
                k = int(np.searchsorted(tc, t_end[b] - snap, side="left"))
                if k:
                    cache[b] = feed_branch(b, tc[:k])
            for p in pairs:
                k = int(np.searchsorted(tc, t_common[p] - snap, side="left"))
                if k:
                    d = cache[p[0]][:, :k] - cache[p[1]][:, :k]
                    p_tr[p].feed(_norm(d))
        # appended end points
        for b in branches:
            feed_branch(b, np.array([t_end[b]]))
        for p in pairs:
            tt = np.array([t_common[p]])
            d = evals[p[0]].rel(tt)[0] - evals[p[1]].rel(tt)[0]
            p_tr[p].feed(_norm(d))
    return g_tr, s_tr, a_tr, p_tr


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------
def run_case(scenario, setting, branches=BRANCHES, fine_dt=FINE_DT, keep_history=False):
    """Integrate and assess one scenario with one solver setting.

    Returns (summary, history); history is None unless keep_history.
    """
    wall0 = time.perf_counter()
    sc = Scenario.from_dict(scenario)
    st = SolverSetting.from_dict(setting)
    branches = normalise_branches(branches)
    pairs = comparison_pairs(branches)
    fine_dt = float(fine_dt)

    orbit = TargetOrbit(sc.altitude)
    ctrl = PDController(orbit.n, sc.wn_ratio, sc.a_max)
    T = sc.horizon_orbits * orbit.period

    models = {b: make_model(b, orbit, ctrl, sc.rel0) for b in branches}
    runs = {b: integrate_branch(models[b], st, T) for b in branches}
    evals = {b: BranchEvaluator(models[b], runs[b]) for b in branches}
    wall_integrate = time.perf_counter() - wall0
    t_end = {b: runs[b].t_end for b in branches}
    t_common = {p: min(t_end[p[0]], t_end[p[1]]) for p in pairs}

    adaptive = st.solver != "rk4"
    summary = {
        "sid": sc.sid,
        "solver": st.solver,
        "setting_label": st.label,
        "rtol": float(st.rtol) if adaptive else _NAN,
        "ref_rtol": float(st.ref_rtol) if adaptive else _NAN,
        "h": float(st.h) if not adaptive else _NAN,
        "max_step": float(st.max_step) if adaptive else _NAN,
        "wall_s": _NAN,
        "indverify_version": __version__,
        "branches": list(branches),
        "horizon_s": float(T),
        "T_orb_s": float(orbit.period),
        "n_radps": float(orbit.n),
        "wn_ratio": float(sc.wn_ratio),
        "a_max": _NAN if sc.a_max is None else float(sc.a_max),
        "fine_dt": fine_dt,
    }
    history = {} if keep_history else None
    if keep_history:
        history["t"] = time_grid(T, PROD_DT)

    # ---------------- production grid: history, terminal state, prod arrival
    prod_cache = {}
    for b in branches:
        run, ev = runs[b], evals[b]
        completed = run.status == COMPLETED
        tb = time_grid(run.t_end, PROD_DT)
        rho, rhod = ev.rel(tb)
        d = ev.derived(rho, rhod)
        prod_cache[b] = (tb, rho, rhod, d["anorm"])
        prod = assess_arrival(sign_intervals(tb, d["g"]), d["g"][-1], ev.g_at, T,
                              completed, PROD_ENTRY_XTOL, locate_all=False)
        summary.update({
            b + "_status": run.status,
            b + "_reason": run.reason,
            b + "_t_end_s": float(run.t_end),
            b + "_events_disabled": bool(run.events_disabled),
            b + "_events_disabled_reason": run.events_disabled_reason,
            b + "_n_steps": int(run.n_steps),
            b + "_nfev": int(run.nfev),
            # switch restarts that could not use the kink-free refinement
            # (integrators._switch_restart); 0 in a healthy run
            b + "_n_restart_fallback": int(sum(v for k, v in run.restart_methods.items()
                                               if k != "refine")),
            b + "_final_range_m": float(_norm(rho[:, -1:])[0]),
            b + "_final_speed_mps": float(_norm(rhod[:, -1:])[0]),
        })
        _put_arrival(summary, b, "prod", prod)
        if keep_history:
            history[b + "_t"] = tb
            history[b + "_rho"] = np.ascontiguousarray(rho.T)
            history[b + "_rhodot"] = np.ascontiguousarray(rhod.T)
            history[b + "_acc"] = np.ascontiguousarray(d["a"].T)
            history[b + "_accnorm"] = np.asarray(d["anorm"])
            history[b + "_t_end"] = np.array(float(run.t_end))
            history[b + "_status"] = np.array(run.status)

    # ---------------- fine grid: fine arrival, saturation, peak command, fine peak
    g_tr, s_tr, a_tr, p_tr = _fine_pass(evals, branches, pairs, t_end, t_common, fine_dt)
    for b in branches:
        run, ev = runs[b], evals[b]
        completed = run.status == COMPLETED
        fine = assess_arrival(g_tr[b].intervals, g_tr[b].last, ev.g_at, T,
                              completed, FINE_ROOT_XTOL, locate_all=True)
        _put_arrival(summary, b, "fine", fine)

        if b in s_tr:
            tr = s_tr[b]
            roots = [locate_root(ev.gsat_at, iv[0], iv[1], FINE_ROOT_XTOL) for iv in tr.intervals]
            sat_start = not (tr.first < 0.0)
            sat_total = saturated_time(roots, sat_start, run.t_end)
        else:
            roots, sat_start, sat_total = [], False, 0.0
        if adaptive:
            switch_times = [float(x) for x in run.switch_times]
        else:
            switch_times = [float(x) for x in roots]
        summary.update({
            b + "_n_switches": int(len(switch_times)),
            b + "_switch_times_s": switch_times,
            b + "_n_switches_fine": int(len(roots)),
            b + "_saturated_at_start": bool(sat_start),
            b + "_sat_total_s": float(sat_total),
        })
        grid_b = time_grid(run.t_end, fine_dt)
        _check_count(a_tr[b].count, grid_b, b)
        peak_cmd, t_peak_cmd = refine_max(ev.accn_at, grid_b, None,
                                          k=a_tr[b].index, value=a_tr[b].value)
        summary[b + "_peak_cmd_mps2"] = float(peak_cmd)
        summary[b + "_t_peak_cmd_s"] = float(t_peak_cmd)
        if keep_history:
            history[b + "_switch_t"] = np.asarray(switch_times, dtype=float)

    # ---------------- trajectory comparison
    for p in pairs:
        a, b = p
        ea, eb = evals[a], evals[b]
        tc_end = t_common[p]
        tc = time_grid(tc_end, PROD_DT)
        ra, rda, aa = _on_prefix_grid(prod_cache[a], ea, tc)
        rb, rdb, ab = _on_prefix_grid(prod_cache[b], eb, tc)
        with np.errstate(all="ignore"):
            dpos_v = _norm(ra - rb)
            dvel_v = _norm(rda - rdb)
            dcmd_v = np.abs(aa - ab)
        f_pos, f_vel, f_cmd = _pair_functions(ea, eb)
        peak_pos, t_peak = refine_max(f_pos, tc, dpos_v)
        peak_vel, t_peak_vel = refine_max(f_vel, tc, dvel_v)
        max_cmd, t_max_cmd = refine_max(f_cmd, tc, dcmd_v)
        grid_c = time_grid(tc_end, fine_dt)
        _check_count(p_tr[p].count, grid_c, "%s/%s" % p)
        peak_fine, t_peak_fine = refine_max(f_pos, grid_c, None,
                                            k=p_tr[p].index, value=p_tr[p].value)
        key = "%s_vs_%s_" % (a, b)
        summary.update({
            key + "peak_pos_m": float(peak_pos),
            key + "t_peak_s": float(t_peak),
            key + "peak_vel_mps": float(peak_vel),
            key + "t_peak_vel_s": float(t_peak_vel),
            key + "max_cmd_diff_mps2": float(max_cmd),
            key + "t_max_cmd_diff_s": float(t_max_cmd),
            key + "common_end_s": float(tc_end),
            key + "peak_pos_fine_m": float(peak_fine),
            key + "t_peak_fine_s": float(t_peak_fine),
        })

    summary["wall_integrate_s"] = float(wall_integrate)
    summary["wall_s"] = float(time.perf_counter() - wall0)
    return summary, history


def _check_count(count, grid, what):
    if count != grid.size:  # pragma: no cover - internal consistency guard
        raise RuntimeError("fine-grid bookkeeping mismatch for %s: %d samples vs grid of %d"
                           % (what, count, grid.size))


def _on_prefix_grid(cache, ev, tc):
    """rho, rhodot, |a| on grid tc, reusing the branch's production-grid values.

    tc = time_grid(t_c, PROD_DT) with t_c <= the branch end time, so all of tc
    except possibly its last point is a prefix of the cached branch grid.
    """
    tb, rho, rhod, anorm = cache
    k = tc.size - 1
    if tb.size == tc.size and tb[-1] == tc[-1]:
        return rho, rhod, anorm
    r1, rd1 = ev.rel(tc[-1:])
    a1 = ev.derived(r1, rd1)["anorm"]
    return (np.concatenate([rho[:, :k], r1], axis=1),
            np.concatenate([rhod[:, :k], rd1], axis=1),
            np.concatenate([anorm[:k], a1]))


def _put_arrival(summary, b, conv, res):
    k = "%s_%s_" % (b, conv)
    summary[k + "inside_at_end"] = bool(res["inside_at_end"])
    summary[k + "entry_time_s"] = float(res["entry_time_s"])
    summary[k + "dwell_s"] = float(res["dwell_s"])
    summary[k + "dwell_margin_s"] = float(res["dwell_margin_s"])
    summary[k + "arrived"] = bool(res["arrived"])
    summary[k + "n_crossings"] = int(res["n_crossings"])
    summary[k + "n_entries"] = int(res["n_entries"])
    summary[k + "n_exits"] = int(res["n_exits"])
