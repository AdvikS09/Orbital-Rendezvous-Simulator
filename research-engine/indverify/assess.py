"""Assessed quantities (SPEC section 8): grids, terminal arrival under the
production and fine monitoring conventions, peak refinement and saturation
statistics.  Nothing here knows about the dynamics; everything works on
sampled values plus a scalar callable for root finding / refinement, so the
arrival logic can be tested on synthetic g(t) histories.
"""

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .constants import DWELL_REQUIRED_S, GRID_SNAP, PEAK_XATOL


def time_grid(t_end, dt, snap=GRID_SNAP):
    """t_k = dt*k for k = 0..floor(t_end/dt), with t_end appended.

    If the last grid point is within ``snap`` of t_end it is replaced by
    t_end (SPEC section 8, production convention).  Equivalent formulation:
    all dt*k < t_end - snap, followed by t_end.
    """
    t_end = float(t_end)
    dt = float(dt)
    if t_end <= 0.0:
        return np.array([0.0])
    K = int(math.floor(t_end / dt)) + 2
    base = dt * np.arange(K + 1, dtype=float)
    base = base[base < t_end - snap]
    return np.append(base, t_end)


def sign_intervals(t, g):
    """Intervals [t_i, t_i+1] on which the 'inside' flag (g < 0) changes.

    Returns a list of (t_a, t_b, g_a, g_b).  NaN counts as outside.
    """
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    if g.size < 2:
        return []
    inside = g < 0.0
    idx = np.flatnonzero(inside[:-1] != inside[1:])
    return [(float(t[i]), float(t[i + 1]), float(g[i]), float(g[i + 1])) for i in idx]


class SignTracker:
    """Streaming version of :func:`sign_intervals` for chunked sampling."""

    def __init__(self):
        self.intervals = []
        self.first = None
        self.last = None
        self._prev = None

    def feed(self, t, g):
        t = np.asarray(t, dtype=float)
        g = np.asarray(g, dtype=float)
        if t.size == 0:
            return
        if self._prev is None:
            self.first = float(g[0])
        else:
            t = np.concatenate(([self._prev[0]], t))
            g = np.concatenate(([self._prev[1]], g))
        self.intervals.extend(sign_intervals(t, g))
        self._prev = (float(t[-1]), float(g[-1]))
        self.last = float(g[-1])


class MaxTracker:
    """Streaming arg-max over a sequence fed in chunks (NaN ignored)."""

    def __init__(self):
        self.value = -math.inf
        self.index = -1
        self.count = 0

    def feed(self, v):
        v = np.asarray(v, dtype=float)
        if v.size and not np.all(np.isnan(v)):
            k = int(np.nanargmax(v))
            if v[k] > self.value or self.index < 0:
                self.value = float(v[k])
                self.index = self.count + k
        self.count += v.size


def locate_root(fun, ta, tb, xtol):
    """Brent root of ``fun`` on [ta, tb]; midpoint if no sign change."""
    try:
        fa = float(fun(ta))
        fb = float(fun(tb))
    except Exception:  # noqa: BLE001
        return 0.5 * (ta + tb)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return 0.5 * (ta + tb)
    if (fa < 0.0) == (fb < 0.0):
        return 0.5 * (ta + tb)
    if fa == 0.0:
        return float(ta)
    if fb == 0.0:
        return float(tb)
    try:
        return float(brentq(fun, ta, tb, xtol=xtol))
    except (ValueError, RuntimeError):
        return 0.5 * (ta + tb)


def assess_arrival(intervals, g_end, gfun, T, completed, xtol, locate_all):
    """Terminal-arrival assessment (SPEC section 8).

    intervals : sign-change intervals of g on the monitoring grid
    g_end     : g at the last grid point (= T for a completed branch)
    gfun      : scalar callable g(t) on the continuous solution
    locate_all: False -> production convention (only the final entry is
                located); True -> fine convention (every sign change located).
    """
    n_cross = len(intervals)
    n_entries = sum(1 for iv in intervals if iv[3] < 0.0)
    roots = [locate_root(gfun, iv[0], iv[1], xtol) for iv in intervals] if locate_all else []
    out = {"inside_at_end": False, "entry_time_s": math.nan, "dwell_s": math.nan,
           "dwell_margin_s": math.nan, "arrived": False, "n_crossings": int(n_cross),
           "n_entries": int(n_entries), "n_exits": int(n_cross - n_entries), "roots": roots}
    if not completed:
        return out
    inside = bool(g_end < 0.0)
    out["inside_at_end"] = inside
    if not inside:
        return out
    if n_cross == 0:
        entry = 0.0
    else:
        ta, tb, _, _ = intervals[-1]          # last change is an entry (inside at end)
        entry = roots[-1] if locate_all else locate_root(gfun, ta, tb, xtol)
    dwell = float(T) - entry
    out.update(entry_time_s=float(entry), dwell_s=float(dwell),
               dwell_margin_s=float(dwell - DWELL_REQUIRED_S),
               arrived=bool(dwell >= DWELL_REQUIRED_S))
    return out


def arrival_on_grid(gfun, T, dt, xtol, locate_all, completed=True):
    """Sample ``gfun`` (vectorised & scalar) on time_grid(T, dt) and assess."""
    t = time_grid(T, dt)
    g = np.asarray(gfun(t), dtype=float)
    return assess_arrival(sign_intervals(t, g), g[-1], gfun, T, completed, xtol, locate_all)


def refine_max(fun, t, v, xatol=PEAK_XATOL, k=None, value=None):
    """Grid maximum of v(t) refined by bounded scalar maximisation of ``fun``
    within one grid step either side of the grid maximum.

    Returns (peak value, time of peak).  The refined value is accepted only if
    it exceeds the grid value.
    """
    t = np.asarray(t, dtype=float)
    if k is None:
        v = np.asarray(v, dtype=float)
        if v.size == 0 or np.all(np.isnan(v)):
            return math.nan, math.nan
        k = int(np.nanargmax(v))
        value = float(v[k])
    if k < 0:
        return math.nan, math.nan
    best, tbest = float(value), float(t[k])
    if not np.isfinite(best):
        return best, tbest
    lo = float(t[max(k - 1, 0)])
    hi = float(t[min(k + 1, t.size - 1)])
    if hi > lo:
        try:
            with np.errstate(all="ignore"):
                res = minimize_scalar(lambda s: -float(fun(s)), bounds=(lo, hi),
                                      method="bounded", options={"xatol": xatol})
            val = -float(res.fun)
            if np.isfinite(val) and val > best:
                best, tbest = val, float(res.x)
        except Exception:  # noqa: BLE001
            pass
    return best, tbest


def saturated_time(roots, saturated_at_start, t_end):
    """Measure of {t in [0, t_end] : saturated}, given the ordered switch roots."""
    edges = [0.0] + sorted(float(r) for r in roots) + [float(t_end)]
    total = 0.0
    sat = bool(saturated_at_start)
    for a, b in zip(edges[:-1], edges[1:]):
        if sat:
            total += b - a
        sat = not sat
    return total
