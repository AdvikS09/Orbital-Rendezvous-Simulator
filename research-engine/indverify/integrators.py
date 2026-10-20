from dataclasses import dataclass, field

import math

import numpy as np
from scipy.integrate import DOP853, Radau
from scipy.optimize import brentq

from .assess import timegrid
from .constants import (COMPL, DEFMAXSTEP, DOMAINEXIT,
                        NOPROGRLIMIT, NOADV,
                        NUMFAIL, SWITCHRESTCAP)

EPS = np.finfo(float).eps
SOLV = {"radau": Radau, "dop853": DOP853}
POLISHMAXITER = 8
POLISHSLOPEDT = 1.0e-2
POLISHOVERSH = 1.25
POLISHACCEPTDT = 1.0e-10


class PwDense:

    def __init__(self, t0, y0):
        self._ts = [float(t0)]
        self.interps = []
        self._y0 = np.array(y0, dtype=float)
        self.tsarr = None

    def append(self, tright, interp):
        self._ts.append(float(tright))
        self.interps.append(interp)
        self.tsarr = None

    def setend(self, tright):
        if self.interps:
            self._ts[-1] = float(tright)
            self.tsarr = None

    @property
    def tlast(self):
        return self._ts[-1]

    @property
    def npieces(self):
        return len(self.interps)

    @property
    def knots(self):
        return np.asarray(self._ts, dtype=float)

    def __call__(self, t):
        if self.tsarr is None:
            self.tsarr = np.asarray(self._ts, dtype=float)
        ts = self.tsarr
        npc = len(self.interps)
        t = np.asarray(t, dtype=float)
        if npc == 0:
            if t.ndim == 0:
                return self._y0.copy()
            return np.repeat(self._y0[:, None], t.size, axis=1)
        if t.ndim == 0:
            i = int(np.searchsorted(ts, t, side="right")) - 1
            i = min(max(i, 0), npc - 1)
            return np.asarray(self.interps[i](t), dtype=float)
        order = None
        if t.size > 1 and np.any(np.diff(t) < 0):
            order = np.argsort(t, kind="stable")
            t = t[order]
        out = np.empty((self._y0.size, t.size))
        if t.size:
            idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, npc - 1)
            cuts = np.flatnonzero(np.diff(idx)) + 1
            starts = np.concatenate(([0], cuts))
            ends = np.concatenate((cuts, [t.size]))
            for s, e in zip(starts.tolist(), ends.tolist()):
                out[:, s:e] = self.interps[idx[s]](t[s:e])
        if order is not None:
            res = np.empty_like(out)
            res[:, order] = out
            out = res
        return out


class HermDense:

    def __init__(self, T, Y, F):
        self.T = np.asarray(T, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.F = np.asarray(F, dtype=float)

    @property
    def tlast(self):
        return float(self.T[-1])

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        K = self.T.size - 1
        if K == 0:
            out = np.repeat(self.Y[:, :1], tt.size, axis=1)
        else:
            idx = np.clip(np.searchsorted(self.T, tt, side="right") - 1, 0, K - 1)
            t0 = self.T[idx]
            h = self.T[idx + 1] - t0
            s = (tt - t0) / h
            s2 = s * s
            s3 = s2 * s
            h00 = 2.0 * s3 - 3.0 * s2 + 1.0
            h10 = s3 - 2.0 * s2 + s
            h01 = -2.0 * s3 + 3.0 * s2
            h11 = s3 - s2
            out = (h00 * self.Y[:, idx] + (h10 * h) * self.F[:, idx]
                   + h01 * self.Y[:, idx + 1] + (h11 * h) * self.F[:, idx + 1])
        return out[:, 0] if scalar else out


@dataclass
class BranchRun:

    sol: object
    t_end: float
    status: str
    reason: str
    switch_times: list = field(default_factory=list)
    events_disabled: bool = False
    events_disabled_reason: str = ""
    n_steps: int = 0
    nfev: int = 0
    n_restarts: int = 0
    restart_methods: dict = field(default_factory=dict)


def locateevent(fun, a, b, fa, fb):
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)

    def f(s):
        if s == a:
            return fa
        if s == b:
            return fb
        return fun(s)

    try:
        return float(brentq(f, a, b, xtol=4 * EPS, rtol=4 * EPS))
    except (ValueError, RuntimeError):
        return float(b)


def initside(model, t, y):
    g0 = model.gsat(t, y)
    if g0 != 0.0:
        return 1 if g0 > 0.0 else -1
    with np.errstate(all="ignore"):
        dt = 1e-6
        g1 = model.gsat(t + dt, np.asarray(y, dtype=float) + dt * model.rhs(t, y))
    return 1 if g1 > 0.0 else -1


def cross(g, side):
    return (g <= 0.0) if side > 0 else (g >= 0.0)


def adv(cls, rhs, t0, y0, tbound, rtol, atol, max_step):
    y0 = np.array(y0, dtype=float)
    if not tbound > t0:
        return [], y0, 0, 0
    solver = cls(rhs, t0, y0, tbound, rtol=rtol, atol=atol, max_step=max_step)
    pieces, n = [], 0
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("re-integration failed: %s" % message)
        n += 1
        pieces.append((float(solver.t), solver.dense_output()))
    yend = np.array(solver.y, dtype=float)
    if not np.isfinite(yend).all():
        raise FloatingPointError("non-finite state in re-integration")
    return pieces, yend, solver.nfev, n


def firstcross(cls, fmodel, t0, y0, tbound, side, g0, rtol, atol, max_step):
    solver = cls(fmodel.rhs, t0, np.array(y0, dtype=float), tbound,
                 rtol=rtol, atol=atol, max_step=max_step)
    pieces, n, gold = [], 0, g0
    while solver.status == "running":
        sold, ysold = float(solver.t), np.array(solver.y, dtype=float)
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("frozen-mode integration failed: %s" % message)
        n += 1
        if not np.isfinite(solver.y).all():
            raise FloatingPointError("non-finite state in frozen-mode integration")
        interp = solver.dense_output()
        gs = fmodel.gsat(float(solver.t), solver.y)
        if cross(gs, side):
            return pieces, (sold, ysold, float(solver.t), gold, gs, interp), solver.nfev, n
        pieces.append((float(solver.t), interp))
        gold = gs
    return pieces, None, solver.nfev, n


def _polish(cls, rhs, gfun, t_lo, ylo, t_hi, side, rtol, atol, max_step):
    pieces, nfev, nst, it = [], 0, 0, 0
    while True:
        glo = gfun(t_lo, ylo)
        if cross(glo, side):
            return pieces, t_lo, ylo, nfev, nst, True
        f = np.asarray(rhs(t_lo, ylo), dtype=float)
        e = POLISHSLOPEDT
        gdot = (gfun(t_lo + e, ylo + e * f) - gfun(t_lo - e, ylo - e * f)) / (2.0 * e)
        nfev += 1
        dt = -glo / gdot if gdot != 0.0 else math.nan
        if 0.0 < dt < POLISHACCEPTDT:
            return pieces, t_lo, ylo, nfev, nst, True
        if it >= POLISHMAXITER:
            return pieces, t_lo, ylo, nfev, nst, False
        it += 1
        cand = t_lo + POLISHOVERSH * dt
        if not (dt > 0.0 and cand < t_hi):
            cand = 0.5 * (t_lo + t_hi)
        if not cand > t_lo:
            return pieces, t_lo, ylo, nfev, nst, False
        fin, ylo, nf, ns = adv(cls, rhs, t_lo, ylo, cand, rtol, atol, max_step)
        pieces.extend(fin)
        nfev, nst, t_lo = nfev + nf, nst + ns, cand


def switchrest(model, cls, t_old, yold, tnew, tr0, side, gold, rtol, atol,
               max_step):
    nfev = nst = 0
    if hasattr(model, "frozen"):
        try:
            fmodel = model.frozen(side > 0)
            pre, hit, nf1, ns1 = firstcross(cls, fmodel, t_old, yold, tnew, side,
                                            gold, rtol, atol, max_step)
            nfev, nst = nfev + nf1, nst + ns1
            if hit is not None:
                sold, ysold, snew, ga, gb, interp = hit
                tr = locateevent(lambda s: fmodel.gsat(s, interp(s)), sold, snew, ga, gb)
                fin, yroot, nf2, ns2 = adv(cls, fmodel.rhs, sold, ysold, tr,
                                           rtol, atol, max_step)
                pol, tr, yroot, nf3, ns3, ok = _polish(cls, fmodel.rhs, fmodel.gsat, tr, yroot,
                                                       snew, side, rtol, atol, max_step)
                if ok:
                    return (pre + fin + pol, tr, yroot, nfev + nf2 + nf3,
                            nst + ns2 + ns3, "refine")
        except Exception:  # fall back to re-integrating to the event
            pass
    try:
        fin, yroot, nf2, ns2 = adv(cls, model.rhs, t_old, yold, tr0, rtol, atol, max_step)
        pol, tr, yroot, nf3, ns3, ok = _polish(cls, model.rhs, model.gsat, tr0, yroot, tnew,
                                               side, rtol, atol, max_step)
        if ok:
            return fin + pol, tr, yroot, nfev + nf2 + nf3, nst + ns2 + ns3, "reintegrate"
    except Exception:  # fall back to the dense interpolant
        pass
    return None, tr0, None, nfev, nst, "interpolant"


SWITCHRESTMETH = ("refine", "interpolant")


def integadapt(model, method, T, rtol, max_step=DEFMAXSTEP,
               restcap=SWITCHRESTCAP, switchevents=True,
               atol=None, switchrestart="refine"):
    method = method.lower()
    cls = SOLV[method]
    if switchrestart not in SWITCHRESTMETH:
        raise ValueError("switch_restart must be one of %s" % (SWITCHRESTMETH,))
    if atol is None:
        atol = model.atol(rtol)
    T = float(T)
    t = 0.0
    y = np.array(model.y0, dtype=float)
    dense = PwDense(t, y)
    switchtim = []
    disabled, offreason = False, ""
    nsteps = nfev = nrestarts = noprogr = 0
    restused = {"refine": 0, "reintegrate": 0, "interpolant": 0}
    ref = switchrestart == "refine"

    def _result(status, t_end, reason):
        return BranchRun(sol=dense, t_end=float(t_end), status=status, reason=reason,
                         switch_times=switchtim, events_disabled=disabled,
                         events_disabled_reason=offreason, n_steps=nsteps,
                         nfev=nfev, n_restarts=nrestarts, restart_methods=restused)

    bad = model.badstate(y)
    if bad:
        return _result(NUMFAIL, 0.0, "initial state: " + bad)
    if model.hasalt and model.alt(t, y) <= 0.0:
        return _result(DOMAINEXIT, 0.0, "chaser altitude below floor at t=0")

    satactive = bool(switchevents and model.hassat)
    side = initside(model, t, y) if satactive else 0

    with np.errstate(all="ignore"):
        while True:
            if T - t <= 1e-12 * max(1.0, T):
                dense.setend(T)
                return _result(COMPL, T, "reached horizon")
            try:
                solver = cls(model.rhs, t, y, T, rtol=rtol, atol=atol, max_step=max_step)
            except Exception as exc:
                return _result(NUMFAIL, t,
                               "%s initialisation failed at t=%.6f: %r" % (method, t, exc))
            segstart = t
            galtold = model.alt(t, y) if model.hasalt else None
            gsatold = None
            if satactive:
                gsatold = side * max(abs(model.gsat(t, y)), 1e-300)
            while True:
                yprev = np.array(solver.y, dtype=float)
                try:
                    message = solver.step()
                except Exception as exc:
                    nfev += solver.nfev
                    return _result(NUMFAIL, dense.tlast,
                                   "exception in %s step after t=%.6f: %r"
                                   % (method, dense.tlast, exc))
                if solver.status == "failed":
                    nfev += solver.nfev
                    return _result(NUMFAIL, dense.tlast,
                                   "%s failed after t=%.6f: %s" % (method, dense.tlast, message))
                told, tnew = float(solver.t_old), float(solver.t)
                ynew = solver.y
                bad = model.badstate(ynew)
                if bad:
                    nfev += solver.nfev
                    return _result(NUMFAIL, told,
                                   "%s after t=%.6f" % (bad, told))
                nsteps += 1
                stepsol = solver.dense_output()
                cands = []
                if model.hasalt:
                    ga = model.alt(tnew, ynew)
                    if galtold > 0.0 and ga <= 0.0:
                        tr = locateevent(lambda s: model.alt(s, stepsol(s)),
                                         told, tnew, galtold, ga)
                        cands.append((tr, 0))
                    galtold = ga
                gold = gsatold
                if satactive:
                    gs = model.gsat(tnew, ynew)
                    if cross(gs, side):
                        tr = locateevent(lambda s: model.gsat(s, stepsol(s)),
                                         told, tnew, gsatold, gs)
                        cands.append((tr, 1))
                    gsatold = gs
                if cands:
                    tr, kind = min(cands)
                    pieces = yroot = None
                    if kind == 1:
                        if ref:
                            pieces, trsw, yroot, nfr, nsr, used = switchrest(
                                model, cls, told, yprev, tnew, tr, side, gold,
                                rtol, atol, max_step)
                        else:
                            pieces, trsw, yroot, nfr, nsr, used = (
                                None, tr, None, 0, 0, "interpolant")
                        nfev += nfr
                        nsteps += nsr
                        altfirst = [c for c in cands if c[1] == 0 and c[0] < trsw]
                        if altfirst:
                            tr, kind, pieces, yroot = altfirst[0][0], 0, None, None
                        else:
                            tr = trsw
                            restused[used] += 1
                    if pieces is not None:
                        for tright, interp in pieces:
                            dense.append(tright, interp)
                    elif tr > told:
                        dense.append(tr, stepsol)
                    if kind == 0:
                        nfev += solver.nfev
                        return _result(DOMAINEXIT, tr,
                                       "chaser altitude fell below 100 km at t=%.6f" % tr)
                    nfev += solver.nfev
                    y = yroot if yroot is not None else np.array(stepsol(tr), dtype=float)
                    t = tr
                    switchtim.append(tr)
                    side = -side
                    nrestarts += 1
                    noprogr = noprogr + 1 if (tr - segstart) < NOADV else 0
                    if noprogr >= NOPROGRLIMIT:
                        satactive = False
                        disabled = True
                        offreason = ("no progress: %d consecutive switch restarts advanced "
                                     "< %g s (t=%.6f)" % (noprogr, NOADV, t))
                    elif nrestarts >= restcap:
                        satactive = False
                        disabled = True
                        offreason = "restart cap %d reached at t=%.6f" % (restcap, t)
                    break
                dense.append(tnew, stepsol)
                if solver.status == "finished":
                    nfev += solver.nfev
                    return _result(COMPL, T, "reached horizon")


def integrk4(model, T, h):
    T = float(T)
    h = float(h)
    times = timegrid(T, h)
    K = times.size - 1
    n = model.nstate
    Y = np.empty((n, K + 1))
    F = np.empty((n, K + 1))
    rhs = model.rhs
    y = np.array(model.y0, dtype=float)
    nfev = 0

    def _result(last, status, t_end, reason):
        sol = HermDense(times[:last + 1], Y[:, :last + 1], F[:, :last + 1])
        return BranchRun(sol=sol, t_end=float(t_end), status=status, reason=reason,
                         switch_times=[], events_disabled=False, n_steps=last, nfev=nfev)

    with np.errstate(all="ignore"):
        Y[:, 0] = y
        try:
            f = rhs(0.0, y)
            nfev += 1
        except Exception as exc:
            F[:, 0] = 0.0
            return _result(0, NUMFAIL, 0.0, "rhs failed at t=0: %r" % (exc,))
        F[:, 0] = f
        bad = model.badstate(y)
        if bad or not np.isfinite(f).all():
            return _result(0, NUMFAIL, 0.0, "initial state: " + (bad or "non-finite derivative"))
        galtold = None
        if model.hasalt:
            galtold = model.alt(0.0, y)
            if galtold <= 0.0:
                return _result(0, DOMAINEXIT, 0.0, "chaser altitude below floor at t=0")
        for k in range(K):
            t = float(times[k])
            t1 = float(times[k + 1])
            hk = t1 - t
            hh = 0.5 * hk
            try:
                k1 = f
                k2 = rhs(t + hh, y + hh * k1)
                k3 = rhs(t + hh, y + hh * k2)
                k4 = rhs(t1, y + hk * k3)
                ynew = y + (hk / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                fnew = rhs(t1, ynew)
                nfev += 4
            except Exception as exc:
                return _result(k, NUMFAIL, t,
                               "exception in rk4 step after t=%.6f: %r" % (t, exc))
            bad = model.badstate(ynew)
            if bad is None and not np.isfinite(fnew).all():
                bad = "non-finite derivative"
            if bad:
                return _result(k, NUMFAIL, t, "%s after t=%.6f" % (bad, t))
            Y[:, k + 1] = ynew
            F[:, k + 1] = fnew
            if model.hasalt:
                ga = model.alt(t1, ynew)
                if ga <= 0.0:
                    herm = HermDense(times[k:k + 2], Y[:, k:k + 2], F[:, k:k + 2])
                    tr = locateevent(lambda s: model.alt(s, herm(s)), t, t1, galtold, ga)
                    return _result(k + 1, DOMAINEXIT, tr,
                                   "chaser altitude fell below 100 km at t=%.6f" % tr)
                galtold = ga
            y, f = ynew, fnew
        return _result(K, COMPL, T, "reached horizon")
