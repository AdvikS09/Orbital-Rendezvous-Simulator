import math
import time
from dataclasses import dataclass

import numpy as np

from .vinfo import __version__
from .assess import (MaxTrack, SignTrack, assessarr, locateroot,
                     refinemax, sattime, signivals, timegrid)
from .constants import (COMPL, DEFALT, DEFMAXSTEP,
                        FINEDT, FINEROOTXTOL, GRIDSNAP, PRODDT,
                        PRODENTRYXTOL, TERMPOSM, TERMVELMPS)
from .control import PDCtrl
from .frames import TgtOrbit
from .integrators import integadapt, integrk4
from .models import BRANCHES, makemodel

CONV = ("prod", "fine")
SOLVERNAMES = ("radau", "dop853", "rk4")
NAN = math.nan


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
        amax = d.get("a_max")
        amax = None if amax is None else float(amax)
        alt = d.get("altitude")
        return cls(sid=str(d["sid"]),
                   rel0=np.asarray(d["rel0"], dtype=float).reshape(6),
                   wn_ratio=float(d["wn_ratio"]),
                   a_max=amax,
                   horizon_orbits=float(d["horizon_orbits"]),
                   altitude=DEFALT if alt is None else float(alt))


@dataclass
class SolverSet:

    solver: str
    rtol: float
    h: float
    max_step: float
    label: str
    ref_rtol: float = NAN

    @classmethod
    def from_dict(cls, d):
        solver = str(d["solver"]).lower()
        if solver not in SOLVERNAMES:
            raise ValueError("unknown solver %r (expected one of %s)" % (solver, SOLVERNAMES))
        ms = d.get("max_step")
        maxstep = DEFMAXSTEP if ms is None else float(ms)
        if solver == "rk4":
            if d.get("ref_rtol") is not None:
                raise ValueError("ref_rtol applies to radau/dop853 only")
            rtol, h, refrtol = NAN, float(d["h"]), NAN
            deflabel = "rk4_%g" % h
        else:
            rtol, h = float(d["rtol"]), NAN
            rr = d.get("ref_rtol")
            refrtol = rtol if rr is None else float(rr)
            if not (rtol > 0.0 and refrtol > 0.0):
                raise ValueError("rtol and ref_rtol must be positive")
            deflabel = "%s_%g" % (solver, rtol)
            if refrtol != rtol:
                deflabel += "_ref%g" % refrtol
        label = d.get("label") or deflabel
        return cls(solver=solver, rtol=rtol, h=h, max_step=maxstep, label=str(label),
                   ref_rtol=refrtol)

    def rtolfor(self, branch):
        return self.ref_rtol if branch == "ref" else self.rtol


def normbranch(branches):
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


def cmppairs(branches):
    pairs = []
    if "ref" in branches:
        pairs.extend((a, "ref") for a in branches if a != "ref")
    if "cw" in branches and "so" in branches:
        pairs.append(("cw", "so"))
    return pairs


def integbranch(model, setting, T):
    if setting.solver == "rk4":
        return integrk4(model, T, setting.h)
    return integadapt(model, setting.solver, T, setting.rtolfor(model.name),
                      max_step=setting.max_step)


def norm(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


class BranchEval:

    def __init__(self, model, run):
        self.model = model
        self.run = run
        self.ctrl = model.ctrl
        self.hassat = model.hassat

    def rel(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        with np.errstate(all="ignore"):
            return self.model.relmany(self.run.sol(t))

    def derived(self, rho, rhod):
        with np.errstate(all="ignore"):
            araw = self.ctrl.raw(rho, rhod)
            a = self.ctrl.saturate(araw)
            g = np.maximum(norm(rho) / TERMPOSM, norm(rhod) / TERMVELMPS) - 1.0
            gsat = norm(araw) - self.ctrl.a_max if self.hassat else None
            return {"a": a, "anorm": norm(a), "g": g, "gsat": gsat}

    def point(self, s):
        rho, rhod = self.rel(np.array([float(s)]))
        return rho[:, 0], rhod[:, 0], self.derived(rho, rhod)

    def gat(self, s):
        return float(self.point(s)[2]["g"][0])

    def gsatat(self, s):
        return float(self.point(s)[2]["gsat"][0])

    def accnat(self, s):
        return float(self.point(s)[2]["anorm"][0])


def pairfns(ea, eb):
    def dpos(s):
        return float(np.linalg.norm(ea.point(s)[0] - eb.point(s)[0]))

    def dvel(s):
        return float(np.linalg.norm(ea.point(s)[1] - eb.point(s)[1]))

    def dcmd(s):
        return abs(ea.accnat(s) - eb.accnat(s))

    return dpos, dvel, dcmd


def finepass(evals, branches, pairs, t_end, tcommon, dt, chunk=50_000):
    snap = GRIDSNAP
    tmax = max(t_end[b] for b in branches)
    K = int(math.floor(tmax / dt)) + 2
    base = dt * np.arange(K + 1, dtype=float)
    gtr = {b: SignTrack() for b in branches}
    s = {b: SignTrack() for b in branches if evals[b].hassat}
    atr = {b: MaxTrack() for b in branches}
    ptr = {p: MaxTrack() for p in pairs}

    def feedbranch(b, tt):
        rho, rhod = evals[b].rel(tt)
        d = evals[b].derived(rho, rhod)
        gtr[b].feed(tt, d["g"])
        if b in s:
            s[b].feed(tt, d["gsat"])
        atr[b].feed(d["anorm"])
        return rho

    with np.errstate(all="ignore"):
        for i0 in range(0, base.size, chunk):
            tc = base[i0:i0 + chunk]
            if tc[0] >= tmax - snap:
                break
            cache = {}
            for b in branches:
                k = int(np.searchsorted(tc, t_end[b] - snap, side="left"))
                if k:
                    cache[b] = feedbranch(b, tc[:k])
            for p in pairs:
                k = int(np.searchsorted(tc, tcommon[p] - snap, side="left"))
                if k:
                    d = cache[p[0]][:, :k] - cache[p[1]][:, :k]
                    ptr[p].feed(norm(d))
        for b in branches:
            feedbranch(b, np.array([t_end[b]]))
        for p in pairs:
            tt = np.array([tcommon[p]])
            d = evals[p[0]].rel(tt)[0] - evals[p[1]].rel(tt)[0]
            ptr[p].feed(norm(d))
    return gtr, s, atr, ptr


def runcase(scenario, setting, branches=BRANCHES, fine_dt=FINEDT, keep_history=False):
    wall0 = time.perf_counter()
    sc = Scenario.from_dict(scenario)
    st = SolverSet.from_dict(setting)
    branches = normbranch(branches)
    pairs = cmppairs(branches)
    fine_dt = float(fine_dt)

    orb = TgtOrbit(sc.altitude)
    ctrl = PDCtrl(orb.n, sc.wn_ratio, sc.a_max)
    T = sc.horizon_orbits * orb.period

    mod = {b: makemodel(b, orb, ctrl, sc.rel0) for b in branches}
    runs = {b: integbranch(mod[b], st, T) for b in branches}
    evals = {b: BranchEval(mod[b], runs[b]) for b in branches}
    wallinteg = time.perf_counter() - wall0
    tend = {b: runs[b].t_end for b in branches}
    tcommon = {p: min(tend[p[0]], tend[p[1]]) for p in pairs}

    adaptive = st.solver != "rk4"
    summary = {
        "sid": sc.sid,
        "solver": st.solver,
        "setting_label": st.label,
        "rtol": float(st.rtol) if adaptive else NAN,
        "ref_rtol": float(st.ref_rtol) if adaptive else NAN,
        "h": float(st.h) if not adaptive else NAN,
        "max_step": float(st.max_step) if adaptive else NAN,
        "wall_s": NAN,
        "indverify_version": __version__,
        "branches": list(branches),
        "horizon_s": float(T),
        "T_orb_s": float(orb.period),
        "n_radps": float(orb.n),
        "wn_ratio": float(sc.wn_ratio),
        "a_max": NAN if sc.a_max is None else float(sc.a_max),
        "fine_dt": fine_dt,
    }
    history = {} if keep_history else None
    if keep_history:
        history["t"] = timegrid(T, PRODDT)

    prodcache = {}
    for b in branches:
        run, ev = runs[b], evals[b]
        compl = run.status == COMPL
        tb = timegrid(run.t_end, PRODDT)
        rho, rhod = ev.rel(tb)
        d = ev.derived(rho, rhod)
        prodcache[b] = (tb, rho, rhod, d["anorm"])
        prod = assessarr(signivals(tb, d["g"]), d["g"][-1], ev.gat, T,
                         compl, PRODENTRYXTOL, locateall=False)
        summary.update({
            b + "_status": run.status,
            b + "_reason": run.reason,
            b + "_t_end_s": float(run.t_end),
            b + "_events_disabled": bool(run.events_disabled),
            b + "_events_disabled_reason": run.events_disabled_reason,
            b + "_n_steps": int(run.n_steps),
            b + "_nfev": int(run.nfev),
            b + "_n_restart_fallback": int(sum(v for k, v in run.restart_methods.items()
                                               if k != "refine")),
            b + "_final_range_m": float(norm(rho[:, -1:])[0]),
            b + "_final_speed_mps": float(norm(rhod[:, -1:])[0]),
        })
        putarr(summary, b, "prod", prod)
        if keep_history:
            history[b + "_t"] = tb
            history[b + "_rho"] = np.ascontiguousarray(rho.T)
            history[b + "_rhodot"] = np.ascontiguousarray(rhod.T)
            history[b + "_acc"] = np.ascontiguousarray(d["a"].T)
            history[b + "_accnorm"] = np.asarray(d["anorm"])
            history[b + "_t_end"] = np.array(float(run.t_end))
            history[b + "_status"] = np.array(run.status)

    gtr, s, atr, ptr = finepass(evals, branches, pairs, tend, tcommon, fine_dt)
    for b in branches:
        run, ev = runs[b], evals[b]
        compl = run.status == COMPL
        fine = assessarr(gtr[b].intervals, gtr[b].last, ev.gat, T,
                         compl, FINEROOTXTOL, locateall=True)
        putarr(summary, b, "fine", fine)

        if b in s:
            tr = s[b]
            roots = [locateroot(ev.gsatat, iv[0], iv[1], FINEROOTXTOL) for iv in tr.intervals]
            satstart = not (tr.first < 0.0)
            sattotal = sattime(roots, satstart, run.t_end)
        else:
            roots, satstart, sattotal = [], False, 0.0
        if adaptive:
            switchtim = [float(x) for x in run.switch_times]
        else:
            switchtim = [float(x) for x in roots]
        summary.update({
            b + "_n_switches": int(len(switchtim)),
            b + "_switch_times_s": switchtim,
            b + "_n_switches_fine": int(len(roots)),
            b + "_saturated_at_start": bool(satstart),
            b + "_sat_total_s": float(sattotal),
        })
        gridb = timegrid(run.t_end, fine_dt)
        checkcount(atr[b].count, gridb, b)
        peakcmd, tpeakcmd = refinemax(ev.accnat, gridb, None,
                                      k=atr[b].index, value=atr[b].value)
        summary[b + "_peak_cmd_mps2"] = float(peakcmd)
        summary[b + "_t_peak_cmd_s"] = float(tpeakcmd)
        if keep_history:
            history[b + "_switch_t"] = np.asarray(switchtim, dtype=float)

    for p in pairs:
        a, b = p
        ea, eb = evals[a], evals[b]
        tcend = tcommon[p]
        tc = timegrid(tcend, PRODDT)
        ra, rda, aa = pfxgrid(prodcache[a], ea, tc)
        rb, rdb, ab = pfxgrid(prodcache[b], eb, tc)
        with np.errstate(all="ignore"):
            dposv = norm(ra - rb)
            dvelv = norm(rda - rdb)
            dcmdv = np.abs(aa - ab)
        fpos, fvel, fcmd = pairfns(ea, eb)
        peakpos, tpeak = refinemax(fpos, tc, dposv)
        peakvel, tpeakvel = refinemax(fvel, tc, dvelv)
        maxcmd, tmaxcmd = refinemax(fcmd, tc, dcmdv)
        gridc = timegrid(tcend, fine_dt)
        checkcount(ptr[p].count, gridc, "%s/%s" % p)
        peakfine, tpeakfine = refinemax(fpos, gridc, None,
                                        k=ptr[p].index, value=ptr[p].value)
        key = "%s_vs_%s_" % (a, b)
        summary.update({
            key + "peak_pos_m": float(peakpos),
            key + "t_peak_s": float(tpeak),
            key + "peak_vel_mps": float(peakvel),
            key + "t_peak_vel_s": float(tpeakvel),
            key + "max_cmd_diff_mps2": float(maxcmd),
            key + "t_max_cmd_diff_s": float(tmaxcmd),
            key + "common_end_s": float(tcend),
            key + "peak_pos_fine_m": float(peakfine),
            key + "t_peak_fine_s": float(tpeakfine),
        })

    summary["wall_integrate_s"] = float(wallinteg)
    summary["wall_s"] = float(time.perf_counter() - wall0)
    return summary, history


def checkcount(count, grid, what):
    if count != grid.size:
        raise RuntimeError("fine-grid bookkeeping mismatch for %s: %d samples vs grid of %d"
                           % (what, count, grid.size))


def pfxgrid(cache, ev, tc):
    tb, rho, rhod, anorm = cache
    k = tc.size - 1
    if tb.size == tc.size and tb[-1] == tc[-1]:
        return rho, rhod, anorm
    r1, rd1 = ev.rel(tc[-1:])
    a1 = ev.derived(r1, rd1)["anorm"]
    return (np.concatenate([rho[:, :k], r1], axis=1),
            np.concatenate([rhod[:, :k], rd1], axis=1),
            np.concatenate([anorm[:k], a1]))


def putarr(summary, b, conv, res):
    k = "%s_%s_" % (b, conv)
    summary[k + "inside_at_end"] = bool(res["inside_at_end"])
    summary[k + "entry_time_s"] = float(res["entry_time_s"])
    summary[k + "dwell_s"] = float(res["dwell_s"])
    summary[k + "dwell_margin_s"] = float(res["dwell_margin_s"])
    summary[k + "arrived"] = bool(res["arrived"])
    summary[k + "n_crossings"] = int(res["n_crossings"])
    summary[k + "n_entries"] = int(res["n_entries"])
    summary[k + "n_exits"] = int(res["n_exits"])
