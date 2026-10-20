import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .constants import DWELLREQS, GRIDSNAP, PEAKXATOL


def timegrid(t_end, dt, snap=GRIDSNAP):
    t_end = float(t_end)
    dt = float(dt)
    if t_end <= 0.0:
        return np.array([0.0])
    K = int(math.floor(t_end / dt)) + 2
    base = dt * np.arange(K + 1, dtype=float)
    base = base[base < t_end - snap]
    return np.append(base, t_end)


def signivals(t, g):
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    if g.size < 2:
        return []
    inside = g < 0.0
    idx = np.flatnonzero(inside[:-1] != inside[1:])
    return [(float(t[i]), float(t[i + 1]), float(g[i]), float(g[i + 1])) for i in idx]


class SignTrack:

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
        self.intervals.extend(signivals(t, g))
        self._prev = (float(t[-1]), float(g[-1]))
        self.last = float(g[-1])


class MaxTrack:

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


def locateroot(fun, ta, tb, xtol):
    try:
        fa = float(fun(ta))
        fb = float(fun(tb))
    except Exception:  # an end cannot be evaluated: use the midpoint
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


def assessarr(intervals, g_end, gfun, T, completed, xtol, locateall):
    ncross = len(intervals)
    nentries = sum(1 for iv in intervals if iv[3] < 0.0)
    roots = [locateroot(gfun, iv[0], iv[1], xtol) for iv in intervals] if locateall else []
    out = {"inside_at_end": False, "entry_time_s": math.nan, "dwell_s": math.nan,
           "dwell_margin_s": math.nan, "arrived": False, "n_crossings": int(ncross),
           "n_entries": int(nentries), "n_exits": int(ncross - nentries), "roots": roots}
    if not completed:
        return out
    inside = bool(g_end < 0.0)
    out["inside_at_end"] = inside
    if not inside:
        return out
    if ncross == 0:
        entry = 0.0
    else:
        ta, tb, _, _ = intervals[-1]
        entry = roots[-1] if locateall else locateroot(gfun, ta, tb, xtol)
    dwell = float(T) - entry
    out.update(entry_time_s=float(entry), dwell_s=float(dwell),
               dwell_margin_s=float(dwell - DWELLREQS),
               arrived=bool(dwell >= DWELLREQS))
    return out


def arrgrid(gfun, T, dt, xtol, locateall, completed=True):
    t = timegrid(T, dt)
    g = np.asarray(gfun(t), dtype=float)
    return assessarr(signivals(t, g), g[-1], gfun, T, completed, xtol, locateall)


def refinemax(fun, t, v, xatol=PEAKXATOL, k=None, value=None):
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
        except Exception:  # keep the best grid point if the bounded search fails
            pass
    return best, tbest


def sattime(roots, saturated_at_start, t_end):
    edges = [0.0] + sorted(float(r) for r in roots) + [float(t_end)]
    total = 0.0
    sat = bool(saturated_at_start)
    for a, b in zip(edges[:-1], edges[1:]):
        if sat:
            total += b - a
        sat = not sat
    return total
