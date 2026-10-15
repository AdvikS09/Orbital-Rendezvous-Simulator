"""Integrators (SPEC sections 6 and 7).

* ``integrate_adaptive``: SciPy Radau / DOP853 with saturation-switch events
  and (reference branch) the altitude-floor event.  SPEC section 7 names
  ``solve_ivp``; this is a deliberate, documented deviation in form only: the
  loop below is the step loop of ``scipy.integrate.solve_ivp`` written out
  over the same SciPy ``OdeSolver`` classes, with the same event detection
  (sign change of the event function between accepted step end points, first
  root estimate with Brent's method on the step's dense output to 4 eps).
  Writing the loop out keeps the dense output of every accepted step when a
  step raises (e.g. a non-finite Jacobian in Radau during a blow-up), so the
  failure time is never lost; on an event-free run it reproduces
  ``solve_ivp`` bit for bit (tested).

  Saturation switches differ from solve_ivp's terminal events on purpose
  (SPEC section 6: no step may be taken across a switch without locating
  it).  The accepted step that reveals a switch was integrated across the
  kink, so its interpolant is contaminated by the law on the far side;
  solve_ivp would take both the root and the restart state from it, which
  in the verification review cost 1-2 orders of magnitude of accuracy in
  the reference branch (non-monotone in rtol).  Here the root is re-located
  on a frozen-mode, kink-free integration from the true step start and the
  restart state is the step-end state of a fresh integration that ends
  exactly at the root (``_switch_restart``; a few extra steps per switch).
  ``switch_restart="interpolant"`` reproduces the solve_ivp behaviour.

  Accuracy note (Radau): every assessed quantity is sampled from dense
  output.  SciPy's Radau dense output is the cubic collocation polynomial,
  1-2 orders of magnitude less accurate between steps than the step-end
  solution (DOP853's 7th-order interpolant is about as accurate as its step
  ends), so Radau ladders converge more slowly than its step-end error;
  reference values should come from DOP853 1e-13 or Radau 1e-12, or a
  smaller ``max_step``.
* ``integrate_rk4``: self-written fixed-step classical RK4 (no events; a step
  may straddle a switch, as SPEC section 7.3 allows) with cubic Hermite
  interpolation between steps.  Because of the straddling, saturated cases do
  not give a clean fourth-order RK4 ladder: the h -> h/2 error ratio of
  switch, entry and saturated-time quantities depends on where switches fall
  inside steps (ratios of 6 to 125 were seen at h = 0.5 -> 0.25 s).

Failure times.  A numerical failure (step failure, non-finite state, or the
divergence guard of the CW / quadratic branches, constants.DIVERGENCE_LIMIT)
is reported at the start of the step that failed, so failure times are
defined only to within one step (h, or up to max_step) and must not be
compared across solvers more finely than that.
"""

from dataclasses import dataclass, field

import math

import numpy as np
from scipy.integrate import DOP853, Radau
from scipy.optimize import brentq

from .assess import time_grid
from .constants import (COMPLETED, DEFAULT_MAX_STEP, DOMAIN_EXIT,
                        NO_PROGRESS_LIMIT, NO_PROGRESS_MIN_ADVANCE,
                        NUMERICAL_FAILURE, SWITCH_RESTART_CAP)

EPS = np.finfo(float).eps
_SOLVERS = {"radau": Radau, "dop853": DOP853}
POLISH_MAX_ITER = 8        # Newton/bisection iterations to put a restart state across the surface
POLISH_SLOPE_DT = 1.0e-2   # [s] central-difference step for dg/dt along the vector field
POLISH_OVERSHOOT = 1.25    # aim this many Newton distances ahead (lands just across the surface)
POLISH_ACCEPT_DT = 1.0e-10 # [s] a crossing closer than this ahead is left to the next step


# ---------------------------------------------------------------------------
# Continuous solutions
# ---------------------------------------------------------------------------
class PiecewiseDense:
    """Continuous solution assembled from per-step SciPy dense outputs.

    Piece i is valid on [ts[i], ts[i+1]].  Calling with a scalar returns shape
    (n,); with a 1-D array returns shape (n, m).
    """

    def __init__(self, t0, y0):
        self._ts = [float(t0)]
        self._interps = []
        self._y0 = np.array(y0, dtype=float)
        self._ts_arr = None

    def append(self, t_right, interp):
        self._ts.append(float(t_right))
        self._interps.append(interp)
        self._ts_arr = None

    def set_end(self, t_right):
        if self._interps:
            self._ts[-1] = float(t_right)
            self._ts_arr = None

    @property
    def t_last(self):
        return self._ts[-1]

    @property
    def n_pieces(self):
        return len(self._interps)

    @property
    def knots(self):
        """Piece boundaries (accepted step ends and event roots)."""
        return np.asarray(self._ts, dtype=float)

    def __call__(self, t):
        if self._ts_arr is None:
            self._ts_arr = np.asarray(self._ts, dtype=float)
        ts = self._ts_arr
        npc = len(self._interps)
        t = np.asarray(t, dtype=float)
        if npc == 0:
            if t.ndim == 0:
                return self._y0.copy()
            return np.repeat(self._y0[:, None], t.size, axis=1)
        if t.ndim == 0:
            i = int(np.searchsorted(ts, t, side="right")) - 1
            i = min(max(i, 0), npc - 1)
            return np.asarray(self._interps[i](t), dtype=float)
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
                out[:, s:e] = self._interps[idx[s]](t[s:e])
        if order is not None:
            res = np.empty_like(out)
            res[:, order] = out
            out = res
        return out


class HermiteDense:
    """Piecewise cubic Hermite interpolant through (t_k, y_k, f_k)."""

    def __init__(self, T, Y, F):
        self.T = np.asarray(T, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.F = np.asarray(F, dtype=float)

    @property
    def t_last(self):
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
    """Result of integrating one branch."""

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


def _locate_event(fun, a, b, fa, fb):
    """Root of an event function on [a, b] given end values of opposite sign."""
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


def _initial_side(model, t, y):
    """Side of the switching surface the trajectory is on at the start: +1
    saturated, -1 unsaturated.  If it starts exactly on the surface the side
    is taken from a tiny explicit look-ahead along the vector field, so that
    leaving the surface at t = 0 is not recorded as a switch."""
    g0 = model.gsat(t, y)
    if g0 != 0.0:
        return 1 if g0 > 0.0 else -1
    with np.errstate(all="ignore"):
        dt = 1e-6
        g1 = model.gsat(t + dt, np.asarray(y, dtype=float) + dt * model.rhs(t, y))
    return 1 if g1 > 0.0 else -1


def _crossed(g, side):
    """True if g is on the far side of the switching surface from ``side``."""
    return (g <= 0.0) if side > 0 else (g >= 0.0)


def _advance(cls, rhs, t0, y0, t_bound, rtol, atol, max_step):
    """Fresh SciPy solver from (t0, y0) to t_bound, no events.

    Returns (pieces, y_end, nfev, n_steps); pieces are (t_right, interp) of
    every accepted step, the last one ending exactly at t_bound.  y_end is the
    solver's step-end state (not an interpolant value).  Raises on failure.
    """
    y0 = np.array(y0, dtype=float)
    if not t_bound > t0:
        return [], y0, 0, 0
    solver = cls(rhs, t0, y0, t_bound, rtol=rtol, atol=atol, max_step=max_step)
    pieces, n = [], 0
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("re-integration failed: %s" % message)
        n += 1
        pieces.append((float(solver.t), solver.dense_output()))
    y_end = np.array(solver.y, dtype=float)
    if not np.isfinite(y_end).all():
        raise FloatingPointError("non-finite state in re-integration")
    return pieces, y_end, solver.nfev, n


def _first_crossing(cls, fmodel, t0, y0, t_bound, side, g0, rtol, atol, max_step):
    """Integrate the frozen-mode model from (t0, y0) towards t_bound and stop
    at the first accepted sub-step whose end is across the switching surface.

    Returns (pieces, hit, nfev, n_steps): ``pieces`` cover [t0, s_old] and
    ``hit`` = (s_old, y_s_old, s_new, g_s_old, g_s_new, interp) for the
    crossing sub-step, or None if g_sat does not cross before t_bound.
    """
    solver = cls(fmodel.rhs, t0, np.array(y0, dtype=float), t_bound,
                 rtol=rtol, atol=atol, max_step=max_step)
    pieces, n, g_old = [], 0, g0
    while solver.status == "running":
        s_old, y_s_old = float(solver.t), np.array(solver.y, dtype=float)
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("frozen-mode integration failed: %s" % message)
        n += 1
        if not np.isfinite(solver.y).all():
            raise FloatingPointError("non-finite state in frozen-mode integration")
        interp = solver.dense_output()
        gs = fmodel.gsat(float(solver.t), solver.y)
        if _crossed(gs, side):
            return pieces, (s_old, y_s_old, float(solver.t), g_old, gs, interp), solver.nfev, n
        pieces.append((float(solver.t), interp))
        g_old = gs
    return pieces, None, solver.nfev, n


def _polish(cls, rhs, gfun, t_lo, y_lo, t_hi, side, rtol, atol, max_step):
    """Advance from a state (t_lo, y_lo) that is still on the current side of
    the switching surface to a point just across it, within (t_lo, t_hi],
    where t_hi is a known step end that is across.

    Safeguarded Newton (slope of g along the vector field by a central
    difference along the tangent) aimed POLISH_OVERSHOOT times the Newton
    distance ahead, bisection if that leaves the bracket.  Every candidate
    state is the step-end state of a fresh integration from the previous one,
    so the result is a true integration state.  A state whose Newton distance
    to the surface is below POLISH_ACCEPT_DT is accepted as is: the first
    step of the restarted segment crosses it.
    Returns (pieces, t, y, nfev, n_steps, ok).
    """
    pieces, nfev, nst, it = [], 0, 0, 0
    while True:
        g_lo = gfun(t_lo, y_lo)
        if _crossed(g_lo, side):
            return pieces, t_lo, y_lo, nfev, nst, True
        f = np.asarray(rhs(t_lo, y_lo), dtype=float)
        e = POLISH_SLOPE_DT
        gdot = (gfun(t_lo + e, y_lo + e * f) - gfun(t_lo - e, y_lo - e * f)) / (2.0 * e)
        nfev += 1
        dt = -g_lo / gdot if gdot != 0.0 else math.nan
        if 0.0 < dt < POLISH_ACCEPT_DT:
            return pieces, t_lo, y_lo, nfev, nst, True
        if it >= POLISH_MAX_ITER:
            return pieces, t_lo, y_lo, nfev, nst, False
        it += 1
        cand = t_lo + POLISH_OVERSHOOT * dt
        if not (dt > 0.0 and cand < t_hi):
            cand = 0.5 * (t_lo + t_hi)
        if not cand > t_lo:
            return pieces, t_lo, y_lo, nfev, nst, False
        fin, y_lo, nf, ns = _advance(cls, rhs, t_lo, y_lo, cand, rtol, atol, max_step)
        pieces.extend(fin)
        nfev, nst, t_lo = nfev + nf, nst + ns, cand


def _switch_restart(model, cls, t_old, y_old, t_new, tr0, side, g_old, rtol, atol,
                    max_step):
    """Root time and restart state for a saturation switch found in the
    accepted step [t_old, t_new] (root estimate tr0 on that step's dense
    output, which straddles the kink and is therefore contaminated by the law
    on the far side of the switch; neither tr0 nor step_sol(tr0) is used).

    1. Integrate the model with the current side's law frozen (a smooth
       vector field that equals the true law up to the switch) from the true
       step start (t_old, y_old) towards t_new; locate the root on the first
       sub-step that crosses (kink-free dense output).
    2. Re-integrate from that sub-step's start (a true step end) exactly to
       the root, so the restart state is a step-end state, not an
       interpolant value.
    3. If that state is still on the current side (tolerance-level root
       error), polish it across the surface (``_polish``); otherwise the
       alternating-direction logic would record a spurious extra switch.
    If the model has no frozen mode, or the frozen continuation does not
    cross before t_new (grazing switch), steps 2-3 are done with the true law
    from (t_old, y_old) to tr0 instead.

    Returns (pieces, tr, y_root, nfev, n_steps, used): pieces are the
    (t_right, interp) dense pieces covering [t_old, tr]; ``used`` is
    "refine", "reintegrate" (fallback) or "interpolant" (both failed: caller
    uses the straddling step, i.e. solve_ivp's terminal-event behaviour).
    """
    nfev = nst = 0
    if hasattr(model, "frozen"):
        try:
            fmodel = model.frozen(side > 0)
            pre, hit, nf1, ns1 = _first_crossing(cls, fmodel, t_old, y_old, t_new, side,
                                                 g_old, rtol, atol, max_step)
            nfev, nst = nfev + nf1, nst + ns1
            if hit is not None:
                s_old, y_s_old, s_new, g_a, g_b, interp = hit
                tr = _locate_event(lambda s: fmodel.gsat(s, interp(s)), s_old, s_new, g_a, g_b)
                fin, y_root, nf2, ns2 = _advance(cls, fmodel.rhs, s_old, y_s_old, tr,
                                                 rtol, atol, max_step)
                pol, tr, y_root, nf3, ns3, ok = _polish(cls, fmodel.rhs, fmodel.gsat, tr, y_root,
                                                        s_new, side, rtol, atol, max_step)
                if ok:
                    return (pre + fin + pol, tr, y_root, nfev + nf2 + nf3,
                            nst + ns2 + ns3, "refine")
        except Exception:  # noqa: BLE001  (fall back to re-integration below)
            pass
    try:
        fin, y_root, nf2, ns2 = _advance(cls, model.rhs, t_old, y_old, tr0, rtol, atol, max_step)
        pol, tr, y_root, nf3, ns3, ok = _polish(cls, model.rhs, model.gsat, tr0, y_root, t_new,
                                                side, rtol, atol, max_step)
        if ok:
            return fin + pol, tr, y_root, nfev + nf2 + nf3, nst + ns2 + ns3, "reintegrate"
    except Exception:  # noqa: BLE001  (fall back to the interpolant below)
        pass
    return None, tr0, None, nfev, nst, "interpolant"


# ---------------------------------------------------------------------------
# Adaptive SciPy integrators with events
# ---------------------------------------------------------------------------
SWITCH_RESTART_METHODS = ("refine", "interpolant")


def integrate_adaptive(model, method, T, rtol, max_step=DEFAULT_MAX_STEP,
                       restart_cap=SWITCH_RESTART_CAP, switch_events=True,
                       atol=None, switch_restart="refine"):
    """Integrate ``model`` on [0, T] with SciPy Radau or DOP853.

    Saturation switches (SPEC section 6): event g_sat = |a_raw| - a_max.  At a
    root the integration is stopped and restarted from the root state.  The
    root is never found again because the sought crossing direction alternates
    (after an upward crossing only a downward one is sought, and vice versa),
    and the first step of every restarted segment is judged against the side
    the trajectory is known to be on, not against the round-off-level value of
    g_sat at the root.  A no-progress guard (NO_PROGRESS_LIMIT restarts in a
    row advancing < NO_PROGRESS_MIN_ADVANCE s) and a cap on the number of
    restarts disable switch events for the rest of the run; this is recorded.

    Root time and restart state (``switch_restart``, see ``_switch_restart``):
    the accepted step that reveals a switch was integrated across the kink, so
    by default ("refine") the root is re-located on a frozen-mode (kink-free)
    integration from the step start and the restart state is the step-end
    state of a fresh integration ending exactly at the root (polished across
    the surface if needed).  ``"interpolant"`` restores the solve_ivp
    terminal-event behaviour (root and restart state taken from the
    straddling step's dense output); it is kept only for comparison.
    """
    method = method.lower()
    cls = _SOLVERS[method]
    if switch_restart not in SWITCH_RESTART_METHODS:
        raise ValueError("switch_restart must be one of %s" % (SWITCH_RESTART_METHODS,))
    if atol is None:
        atol = model.atol(rtol)
    T = float(T)
    t = 0.0
    y = np.array(model.y0, dtype=float)
    dense = PiecewiseDense(t, y)
    switch_times = []
    disabled, disabled_reason = False, ""
    n_steps = nfev = n_restarts = no_progress = 0
    restart_used = {"refine": 0, "reintegrate": 0, "interpolant": 0}
    refine = switch_restart == "refine"

    def _result(status, t_end, reason):
        return BranchRun(sol=dense, t_end=float(t_end), status=status, reason=reason,
                         switch_times=switch_times, events_disabled=disabled,
                         events_disabled_reason=disabled_reason, n_steps=n_steps,
                         nfev=nfev, n_restarts=n_restarts, restart_methods=restart_used)

    bad = model.bad_state(y)
    if bad:
        return _result(NUMERICAL_FAILURE, 0.0, "initial state: " + bad)
    if model.has_alt and model.alt(t, y) <= 0.0:
        return _result(DOMAIN_EXIT, 0.0, "chaser altitude below floor at t=0")

    sat_active = bool(switch_events and model.has_sat)
    side = _initial_side(model, t, y) if sat_active else 0

    with np.errstate(all="ignore"):
        while True:
            if T - t <= 1e-12 * max(1.0, T):
                dense.set_end(T)
                return _result(COMPLETED, T, "reached horizon")
            try:
                solver = cls(model.rhs, t, y, T, rtol=rtol, atol=atol, max_step=max_step)
            except Exception as exc:  # noqa: BLE001
                return _result(NUMERICAL_FAILURE, t,
                               "%s initialisation failed at t=%.6f: %r" % (method, t, exc))
            seg_start = t
            g_alt_old = model.alt(t, y) if model.has_alt else None
            g_sat_old = None
            if sat_active:
                # judged against the known side, see docstring
                g_sat_old = side * max(abs(model.gsat(t, y)), 1e-300)
            while True:
                y_prev = np.array(solver.y, dtype=float)    # true step-start state
                try:
                    message = solver.step()
                except Exception as exc:  # noqa: BLE001
                    nfev += solver.nfev
                    return _result(NUMERICAL_FAILURE, dense.t_last,
                                   "exception in %s step after t=%.6f: %r"
                                   % (method, dense.t_last, exc))
                if solver.status == "failed":
                    nfev += solver.nfev
                    return _result(NUMERICAL_FAILURE, dense.t_last,
                                   "%s failed after t=%.6f: %s" % (method, dense.t_last, message))
                t_old, t_new = float(solver.t_old), float(solver.t)
                y_new = solver.y
                bad = model.bad_state(y_new)
                if bad:
                    nfev += solver.nfev
                    return _result(NUMERICAL_FAILURE, t_old,
                                   "%s after t=%.6f" % (bad, t_old))
                n_steps += 1
                step_sol = solver.dense_output()
                cands = []
                if model.has_alt:
                    ga = model.alt(t_new, y_new)
                    if g_alt_old > 0.0 and ga <= 0.0:
                        tr = _locate_event(lambda s: model.alt(s, step_sol(s)),
                                           t_old, t_new, g_alt_old, ga)
                        cands.append((tr, 0))
                    g_alt_old = ga
                g_sat_step_old = g_sat_old
                if sat_active:
                    gs = model.gsat(t_new, y_new)
                    if _crossed(gs, side):
                        tr = _locate_event(lambda s: model.gsat(s, step_sol(s)),
                                           t_old, t_new, g_sat_old, gs)
                        cands.append((tr, 1))
                    g_sat_old = gs
                if cands:
                    tr, kind = min(cands)
                    pieces = y_root = None
                    if kind == 1:
                        if refine:
                            pieces, tr_sw, y_root, nf_r, ns_r, used = _switch_restart(
                                model, cls, t_old, y_prev, t_new, tr, side, g_sat_step_old,
                                rtol, atol, max_step)
                        else:
                            pieces, tr_sw, y_root, nf_r, ns_r, used = (
                                None, tr, None, 0, 0, "interpolant")
                        nfev += nf_r
                        n_steps += ns_r
                        alt_first = [c for c in cands if c[1] == 0 and c[0] < tr_sw]
                        if alt_first:      # refined switch falls after the altitude root
                            tr, kind, pieces, y_root = alt_first[0][0], 0, None, None
                        else:
                            tr = tr_sw
                            restart_used[used] += 1
                    if pieces is not None:
                        for t_right, interp in pieces:
                            dense.append(t_right, interp)
                    elif tr > t_old:
                        dense.append(tr, step_sol)
                    if kind == 0:
                        nfev += solver.nfev
                        return _result(DOMAIN_EXIT, tr,
                                       "chaser altitude fell below 100 km at t=%.6f" % tr)
                    # saturation switch: restart from the root state
                    nfev += solver.nfev
                    y = y_root if y_root is not None else np.array(step_sol(tr), dtype=float)
                    t = tr
                    switch_times.append(tr)
                    side = -side
                    n_restarts += 1
                    no_progress = no_progress + 1 if (tr - seg_start) < NO_PROGRESS_MIN_ADVANCE else 0
                    if no_progress >= NO_PROGRESS_LIMIT:
                        sat_active = False
                        disabled = True
                        disabled_reason = ("no progress: %d consecutive switch restarts advanced "
                                           "< %g s (t=%.6f)" % (no_progress, NO_PROGRESS_MIN_ADVANCE, t))
                    elif n_restarts >= restart_cap:
                        sat_active = False
                        disabled = True
                        disabled_reason = "restart cap %d reached at t=%.6f" % (restart_cap, t)
                    break
                dense.append(t_new, step_sol)
                if solver.status == "finished":
                    nfev += solver.nfev
                    return _result(COMPLETED, T, "reached horizon")


# ---------------------------------------------------------------------------
# Fixed-step RK4 with cubic Hermite interpolation
# ---------------------------------------------------------------------------
def integrate_rk4(model, T, h):
    """Classical RK4 with fixed step h on [0, T] (last step shortened to hit T).

    No switch events (steps may straddle a saturation switch).  For the
    reference branch the altitude floor is checked at every step end and the
    crossing is located on the step's cubic Hermite interpolant.
    """
    T = float(T)
    h = float(h)
    times = time_grid(T, h)
    K = times.size - 1
    n = model.n_state
    Y = np.empty((n, K + 1))
    F = np.empty((n, K + 1))
    rhs = model.rhs
    y = np.array(model.y0, dtype=float)
    nfev = 0

    def _result(last, status, t_end, reason):
        sol = HermiteDense(times[:last + 1], Y[:, :last + 1], F[:, :last + 1])
        return BranchRun(sol=sol, t_end=float(t_end), status=status, reason=reason,
                         switch_times=[], events_disabled=False, n_steps=last, nfev=nfev)

    with np.errstate(all="ignore"):
        Y[:, 0] = y
        try:
            f = rhs(0.0, y)
            nfev += 1
        except Exception as exc:  # noqa: BLE001
            F[:, 0] = 0.0
            return _result(0, NUMERICAL_FAILURE, 0.0, "rhs failed at t=0: %r" % (exc,))
        F[:, 0] = f
        bad = model.bad_state(y)
        if bad or not np.isfinite(f).all():
            return _result(0, NUMERICAL_FAILURE, 0.0, "initial state: " + (bad or "non-finite derivative"))
        g_alt_old = None
        if model.has_alt:
            g_alt_old = model.alt(0.0, y)
            if g_alt_old <= 0.0:
                return _result(0, DOMAIN_EXIT, 0.0, "chaser altitude below floor at t=0")
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
                y_new = y + (hk / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                f_new = rhs(t1, y_new)
                nfev += 4
            except Exception as exc:  # noqa: BLE001
                return _result(k, NUMERICAL_FAILURE, t,
                               "exception in rk4 step after t=%.6f: %r" % (t, exc))
            bad = model.bad_state(y_new)
            if bad is None and not np.isfinite(f_new).all():
                bad = "non-finite derivative"
            if bad:
                return _result(k, NUMERICAL_FAILURE, t, "%s after t=%.6f" % (bad, t))
            Y[:, k + 1] = y_new
            F[:, k + 1] = f_new
            if model.has_alt:
                ga = model.alt(t1, y_new)
                if ga <= 0.0:
                    herm = HermiteDense(times[k:k + 2], Y[:, k:k + 2], F[:, k:k + 2])
                    tr = _locate_event(lambda s: model.alt(s, herm(s)), t, t1, g_alt_old, ga)
                    return _result(k + 1, DOMAIN_EXIT, tr,
                                   "chaser altitude fell below 100 km at t=%.6f" % tr)
                g_alt_old = ga
            y, f = y_new, f_new
        return _result(K, COMPLETED, T, "reached horizon")
