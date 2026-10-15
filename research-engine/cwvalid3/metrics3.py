"""Independent checker for the trajectory metrics (v3 numerical-assurance work package P02).

The production routines (``cwvalid/metrics.py`` and ``cwvalid3/held_metrics.py``) sample each branch
on the 0.5 s monitoring grid plus the horizon end, refine the peak discrepancy only inside the two
grid intervals next to the grid argmax (bounded scalar optimisation, xatol 1e-4 s), and locate the
last terminal-set entry with Brent's method (xtol 1e-6 s) in the last grid interval that shows a
sign change. This module recomputes the same quantities with more care so the two can be compared
case by case. It does not import, call or change the production code.

Every routine takes plain callables (a dense solution evaluated at a time, returning a float or an
array) plus metadata, so it works with any engine: SciPy dense output, the cubic Hermite interpolant
of the held-command batch, or a closed-form test function. Pass ``vectorized=True`` when the
callable accepts a 1-D array of times and returns one value per time; the grid is then evaluated in
one call.

Definitions (``v3_extension/methods/V3_CONVENTIONS.md``, "Targets and monitoring"):

* d(t) = |rho_a(t) - rho_b(t)| (same-time position difference); the peak is the maximum of d over
  the common completed horizon [0, t_common].
* g(t) = max(|rho| / 1 m, |rho_dot| / 0.01 m/s) - 1; the chaser is inside the terminal set iff
  g < 0.
* arrived iff the branch completed the horizon, is inside at T, and T - t_entry >= 300 s, where
  t_entry is the last entry time (0 if the branch is inside throughout).

What sampling can and cannot do. Every routine starts from samples on a grid and then refines
locally. A feature narrower than the grid spacing that leaves no trace in the samples (a short spike
of d between two samples, a brief exit from the terminal set between two samples) cannot be ruled
out by any amount of local refinement. Nothing here is a certificate. The routines report the grid
spacing they relied on (``resolution_assumption_s``), repeat the peak search on successively halved
grids and say whether the answers agree, and return explicit 'unresolved' statuses when a decision
depends on quantities smaller than the stated tolerances.

Routines: ``robust_peak``, ``crossings``, ``arrival_label``, ``threshold_label``, ``coverage_check``,
plus the builders ``discrepancy_fun``, ``terminal_g_fun``, ``peak_discrepancy`` and ``build_grid``.
See ``v3_extension/methods/metrics3_notes.md``.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

EPS = float(np.finfo(float).eps)
_INVPHI = (math.sqrt(5.0) - 1.0) / 2.0
_MAX_DEPTH = 6          # recursion limit when splitting brackets around hidden excursions

POS_TOL = 1.0           # [m]
VEL_TOL = 0.01          # [m/s]
DWELL = 300.0           # [s]

ARRIVAL_STATUSES = ("arrived", "not_arrived", "unresolved_tangency", "unresolved_near_deadline",
                    "incomplete_horizon")
THRESHOLD_LABELS = ("ok", "violation", "unresolved")

RESOLUTION_NOTE = ("Sample-based result: an excursion narrower than the grid spacing that leaves no trace in "
                   "the samples cannot be excluded. Agreement between refinement levels is evidence, not proof.")


# ---------------------------------------------------------------------------
# grid and evaluation helpers
# ---------------------------------------------------------------------------


def build_grid(t0, t1, dt, extra_times=()):
    """Monitoring grid t0 + k*dt (k = 0, 1, ...) plus t1, merged with ``extra_times``.

    Matches the production grid when t0 = 0 (``held_metrics.monitoring_grid``): the last regular
    point is snapped to t1 when it lies within 1e-9 s of it, otherwise t1 is appended. Extra times
    outside [t0, t1] are ignored. Duplicates, and points closer than 4 eps max(|t0|, |t1|, 1), are
    merged so that repeated update or switch times are counted once. The end points are exact.
    """
    t0, t1, dt = float(t0), float(t1), float(dt)
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise ValueError("t0 and t1 must be finite")
    if t1 < t0:
        raise ValueError(f"empty interval: t1 = {t1!r} < t0 = {t0!r}")
    if not (dt > 0.0 and math.isfinite(dt)):
        raise ValueError(f"grid spacing must be positive and finite, got {dt!r}")
    k = int(math.floor((t1 - t0) / dt + 1e-9))
    base = t0 + np.arange(k + 1) * dt
    if base[-1] < t1 - 1e-9:
        base = np.append(base, t1)
    else:
        base[-1] = t1
    ex = np.asarray(extra_times, dtype=float).ravel()
    if ex.size and not np.all(np.isfinite(ex)):
        raise ValueError("extra_times must be finite")
    ex = ex[(ex >= t0) & (ex <= t1)]
    allt = np.sort(np.concatenate((base, ex)))
    merge = 4.0 * EPS * max(abs(t0), abs(t1), 1.0)
    keep = np.ones(allt.size, dtype=bool)
    keep[1:] = np.diff(allt) > merge
    t = allt[keep]
    t[0], t[-1] = t0, t1
    return t


def _scalar(fun, vectorized):
    if vectorized:
        return lambda s: float(np.asarray(fun(np.array([s], dtype=float)), dtype=float).reshape(-1)[0])
    return lambda s: float(fun(s))


def _sample(fun, t, vectorized):
    if vectorized:
        v = np.asarray(fun(t), dtype=float).reshape(-1)
        if v.size != t.size:
            raise ValueError("vectorized callable must return one value per time")
        return v
    return np.fromiter((float(fun(s)) for s in t), dtype=float, count=t.size)


def _check_finite(t, v, what):
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"non-finite {what} at t = {float(t[bad][0])!r}")


def _local_max_runs(v):
    """Local maxima of a sampled sequence, end points included. A run of equal values counts once.
    Returns (starts, ends): index ranges of the runs that exceed both outside neighbours."""
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [n - 1]))
    vals = v[starts]
    left = np.ones(starts.size, dtype=bool)
    right = np.ones(starts.size, dtype=bool)
    left[1:] = vals[:-1] < vals[1:]
    right[:-1] = vals[1:] < vals[:-1]
    keep = left & right
    return starts[keep], ends[keep]


def _golden_max(f, a, b, xtol, maxiter=200):
    """Golden-section search for a maximum of f on [a, b]. Returns the best point evaluated."""
    c = b - _INVPHI * (b - a)
    d = a + _INVPHI * (b - a)
    fc, fd = f(c), f(d)
    best_t, best_f = (c, fc) if fc >= fd else (d, fd)
    it = 0
    while (b - a) > xtol and it < maxiter:
        it += 1
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - _INVPHI * (b - a)
            fc = f(c)
            if fc > best_f:
                best_t, best_f = c, fc
        else:
            a, c, fc = c, d, fd
            d = a + _INVPHI * (b - a)
            fd = f(d)
            if fd > best_f:
                best_t, best_f = d, fd
    return best_t, best_f


def _refine_max(f, a, b, xtol):
    """Maximise f on [a, b] twice: SciPy's bounded Brent method and an independent golden-section
    search. The bounded search runs in the shifted variable u = t - a, because SciPy adds a relative
    tolerance sqrt(eps)*|x| that would otherwise be about 1e-4 s at t ~ 5000 s.
    The bracket ends are candidates too. Returns (t_best, f_best, gap), where gap is the difference
    between the two searches after each is combined with the end values; it is near zero when both
    reach the same maximum and large when they settle on different local maxima of one bracket."""
    w = b - a
    r = minimize_scalar(lambda u: -f(a + u), bounds=(0.0, w), method="bounded",
                        options={"xatol": xtol, "maxiter": 500})
    tb, fb = a + float(r.x), -float(r.fun)
    tg, fg = _golden_max(f, a, b, xtol)
    fa_end, fb_end = f(a), f(b)
    t_end, f_end = (a, fa_end) if fa_end >= fb_end else (b, fb_end)
    cands = [(tb, fb), (tg, fg), (t_end, f_end)]
    finite = [c for c in cands if math.isfinite(c[1])]
    if not finite:
        return a, float("nan"), float("inf")
    t_best, f_best = max(finite, key=lambda c: c[1])
    if math.isfinite(fb) and math.isfinite(fg):
        ends = f_end if math.isfinite(f_end) else -math.inf
        gap = abs(max(fb, ends) - max(fg, ends))
    else:
        gap = float("inf")
    return t_best, f_best, gap


# ---------------------------------------------------------------------------
# 1. peak of a scalar function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeakLevel:
    """Peak search on one grid."""
    dt: float               # nominal grid spacing [s]
    n_samples: int          # distinct sample times (extra times included, duplicates merged)
    max_gap_s: float        # largest spacing between consecutive samples
    grid_only: float        # largest sampled value
    t_grid_only: float
    peak: float             # best value after refining every candidate
    t_peak: float
    n_candidates: int       # local-maximum runs refined
    method_gap: float       # largest |bounded - golden| disagreement over the candidates


@dataclass(frozen=True)
class PeakResult:
    peak: float                     # best value found on any level
    t_peak: float
    n_candidates: int               # candidates refined on the primary grid
    grid_only: float                # primary-grid sampled maximum, no refinement
    t_grid_only: float
    peak_primary: float             # refined value on the primary grid
    t_peak_primary: float
    refinement_dts: tuple           # monitoring refinement grid spacings
    refinement_values: tuple        # refined peak on each monitoring refinement
    converged: bool                 # last two refinements agree within convergence_tol
    last_change: float              # |peak(last) - peak(second last)|, nan if fewer than two levels
    convergence_tol: float          # max(abs_tol, rel_tol*|peak|)
    uncertainty: float              # max(convergence_tol, last_change, method gap); inf if no sequence
    resolution_assumption_s: float  # finest grid spacing used: narrower unseen features are not excluded
    levels: tuple                   # PeakLevel for the primary grid and each refinement
    t0: float
    t1: float
    note: str = RESOLUTION_NOTE

    def as_dict(self):
        return asdict(self)


def _peak_level(fun, f, t0, t1, dt, extra_times, rel_band, top_k, xtol, vectorized):
    t = build_grid(t0, t1, dt, extra_times)
    v = _sample(fun, t, vectorized)
    _check_finite(t, v, "function value")
    ig = int(np.argmax(v))
    gmax = float(v[ig])
    starts, ends = _local_max_runs(v)
    vals = v[starts]
    band = gmax - (1.0 - rel_band) * abs(gmax)      # = rel_band*gmax for gmax >= 0
    sel = vals >= band
    if top_k:
        sel[np.argsort(-vals, kind="stable")[:top_k]] = True
    best_t, best = float(t[ig]), gmax
    gap_max = 0.0
    for k in np.flatnonzero(sel):
        lo = float(t[max(starts[k] - 1, 0)])
        hi = float(t[min(ends[k] + 1, t.size - 1)])
        if hi > lo:
            tr, fr, gap = _refine_max(f, lo, hi, xtol)
            gap_max = max(gap_max, gap)
            if fr > best:
                best_t, best = float(tr), float(fr)
    max_gap = float(np.max(np.diff(t))) if t.size > 1 else 0.0
    return PeakLevel(float(dt), int(t.size), max_gap, gmax, float(t[ig]), best, best_t, int(sel.sum()), gap_max)


def robust_peak(fun, t0, t1, grid_dt=0.5, extra_times=(), rel_band=0.5, refine_xtol=1e-6,
                monitor_refinements=(0.5, 0.25, 0.125), *, top_k=5, abs_tol=1e-9, rel_tol=1e-9,
                vectorized=False):
    """Maximum of a scalar function fun(t) on [t0, t1], found by sampling and local refinement.

    On each grid (spacing ``grid_dt`` for the primary result, then each spacing in
    ``monitor_refinements``; extra times such as update and saturation-switch instants are added to
    every grid) the function is sampled, and every local maximum of the sampled sequence (end points
    included, a run of equal samples counted once) whose value is at least ``rel_band`` times the
    sampled maximum is refined, together with the ``top_k`` largest local maxima whatever their
    value. Each candidate is refined between its neighbouring samples by bounded scalar
    minimisation of -fun and by an independent golden-section search; the better of the two (and of
    the samples) is kept.

    ``converged`` is True iff the refined peak changes by less than max(abs_tol, rel_tol*|peak|)
    between the last two monitoring refinements. It is False when fewer than two refinements are
    requested. A True flag does not prove that no narrower excursion exists: a feature narrower
    than the finest grid that leaves no trace in its samples is invisible to every level. The grid
    spacing relied on is returned as ``resolution_assumption_s``.

    Returns a PeakResult. ``peak`` is the best value found on any level; every value reported is an
    actual evaluation of fun, so it is a lower bound on the true maximum.
    """
    if not 0.0 <= rel_band <= 1.0:
        raise ValueError("rel_band must lie in [0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not refine_xtol > 0.0:
        raise ValueError("refine_xtol must be positive")
    f = _scalar(fun, vectorized)
    t0, t1 = float(t0), float(t1)
    cache = {}

    def level(dt):
        key = float(dt)
        if key not in cache:
            cache[key] = _peak_level(fun, f, t0, t1, key, extra_times, rel_band, int(top_k), refine_xtol, vectorized)
        return cache[key]

    prim = level(grid_dt)
    mon = tuple(level(dt) for dt in monitor_refinements)
    best = prim
    for lv in mon:
        if lv.peak > best.peak:
            best = lv
    tol = max(abs_tol, rel_tol * abs(best.peak))
    if len(mon) >= 2:
        change = abs(mon[-1].peak - mon[-2].peak)
        converged = bool(change < tol)
        unc = max(tol, change, mon[-1].method_gap, mon[-2].method_gap)
    else:
        change, converged, unc = float("nan"), False, float("inf")
    levels = (prim,) + tuple(lv for lv in mon if lv is not prim)
    return PeakResult(
        peak=best.peak, t_peak=best.t_peak, n_candidates=prim.n_candidates,
        grid_only=prim.grid_only, t_grid_only=prim.t_grid_only,
        peak_primary=prim.peak, t_peak_primary=prim.t_peak,
        refinement_dts=tuple(lv.dt for lv in mon), refinement_values=tuple(lv.peak for lv in mon),
        converged=converged, last_change=change, convergence_tol=tol, uncertainty=unc,
        resolution_assumption_s=min(lv.dt for lv in levels), levels=levels, t0=t0, t1=t1)


# ---------------------------------------------------------------------------
# 2. sign changes of g (terminal-set boundary crossings)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossingEvent:
    kind: str       # "entry" (g goes from >= 0 to < 0), "exit", or "tangency_unresolved"
    t: float        # located time (midpoint of [t_lo, t_hi] for clustered or boundary events)
    t_lo: float     # interval known to contain the event
    t_hi: float
    basis: str      # "grid_point" | "bracketed" | "refinement" | "cluster" | "near_tangency"
    g_scale: float  # tangency: closest approach of g to zero, or depth of the merged excursions
    n_roots: int    # located zeros of g making up the event (0 for a near-tangency)

    @property
    def width(self):
        return self.t_hi - self.t_lo


@dataclass(frozen=True)
class CrossingResult:
    events: tuple
    t0: float
    t1: float
    g_start: float
    g_end: float
    start_on_boundary: bool         # |g(t0)| <= tangency_tol
    end_on_boundary: bool           # |g(t1)| <= tangency_tol
    n_samples: int
    resolution_assumption_s: float  # largest sample spacing: where no event is reported, absence is sample-based
    tangency_tol: float
    xtol: float
    absence_basis: str = "sampled"
    note: str = RESOLUTION_NOTE

    @property
    def entries(self):
        return tuple(e for e in self.events if e.kind == "entry")

    @property
    def exits(self):
        return tuple(e for e in self.events if e.kind == "exit")

    @property
    def tangencies(self):
        return tuple(e for e in self.events if e.kind == "tangency_unresolved")

    @property
    def inside_at_end(self):
        return self.g_end < 0.0

    @property
    def n_bracketed(self):
        """Entries and exits established by a sign change between samples (or a zero at a sample)."""
        return sum(e.kind != "tangency_unresolved" and e.basis in ("grid_point", "bracketed") for e in self.events)

    @property
    def n_from_refinement(self):
        """Entries and exits the samples alone missed, found by refining near a sampled minimum of |g|."""
        return sum(e.kind != "tangency_unresolved" and e.basis == "refinement" for e in self.events)

    @property
    def n_clustered(self):
        return sum(e.kind != "tangency_unresolved" and e.basis == "cluster" for e in self.events)

    def as_dict(self):
        out = asdict(self)
        out.update(n_entries=len(self.entries), n_exits=len(self.exits), n_tangencies=len(self.tangencies),
                   n_bracketed=self.n_bracketed, n_from_refinement=self.n_from_refinement,
                   n_clustered=self.n_clustered)
        return out


@dataclass(frozen=True)
class _Root:
    t: float
    lo: float
    hi: float
    basis: str


@dataclass(frozen=True)
class _Exc:
    sign: float
    depth: float
    empty: bool


def _locate(g, a, b, ga, gb, xtol):
    """Zero of g in [a, b] (ga, gb nonzero with opposite signs) by brentq, plus a checked bracket.
    Returns (r, lo, hi, g(lo), g(hi)) with a sign change on [lo, hi], or lo = hi = r if g(r) == 0."""
    r = float(brentq(g, a, b, xtol=xtol, rtol=4.0 * EPS, maxiter=500))
    if g(r) == 0.0:
        return r, r, r, 0.0, 0.0
    delta = xtol
    for _ in range(200):
        lo, hi = max(a, r - delta), min(b, r + delta)
        glo = ga if lo == a else g(lo)
        ghi = gb if hi == b else g(hi)
        if glo == 0.0:
            return lo, lo, lo, 0.0, 0.0
        if ghi == 0.0:
            return hi, hi, hi, 0.0, 0.0
        if (glo > 0.0) != (ghi > 0.0):
            return r, lo, hi, glo, ghi
        if lo == a and hi == b:
            break
        delta *= 2.0
    return r, a, b, ga, gb


def _resolve_sign_change(g, a, b, ga, gb, xtol, basis, out, depth=0):
    """Locate the zero in a bracket with a sign change, then look for further hidden zeros on each
    side of it (an odd number of zeros can hide in one bracket)."""
    r, lo, hi, glo, ghi = _locate(g, a, b, ga, gb, xtol)
    out.append(_Root(r, lo, hi, basis))
    if depth >= _MAX_DEPTH:
        return
    for u, v, gu, gv in ((a, lo, ga, glo), (hi, b, ghi, gb)):
        if not v > u or (gu == 0.0 and gv == 0.0):
            continue
        if gu != 0.0 and gv != 0.0 and (gu > 0.0) != (gv > 0.0):
            _resolve_sign_change(g, u, v, gu, gv, xtol, "refinement", out, depth + 1)
        else:
            sigma = 1.0 if (gu > 0.0 or gv > 0.0) else -1.0
            _resolve_same_sign(g, u, v, gu, gv, sigma, xtol, out, depth + 1)


def _resolve_same_sign(g, a, b, ga, gb, sigma, xtol, out, depth=0):
    """sigma*g >= 0 at a and b. Minimise sigma*g on [a, b]; if it goes negative, an excursion of the
    opposite sign hides between the samples and its ends are located. Returns (t_min, min sigma*g)."""
    tm, negm, _ = _refine_max(lambda s: -sigma * g(s), a, b, xtol)
    m = -negm
    if m < 0.0 and depth < _MAX_DEPTH:
        gm = g(tm)
        if ga != 0.0:
            _resolve_sign_change(g, a, tm, ga, gm, xtol, "refinement", out, depth + 1)
        if gb != 0.0:
            _resolve_sign_change(g, tm, b, gm, gb, xtol, "refinement", out, depth + 1)
    return float(tm), float(m)


def _dedupe_roots(roots, xtol):
    rank = {"grid_point": 0, "bracketed": 1, "refinement": 2}
    out = []
    for r in sorted(roots, key=lambda x: x.t):
        if out and (r.t - out[-1].t <= 2.0 * xtol or r.lo <= out[-1].hi):
            p = out[-1]
            keep = p if (rank[p.basis], p.hi - p.lo) <= (rank[r.basis], r.hi - r.lo) else r
            out[-1] = _Root(keep.t, min(p.lo, r.lo), max(p.hi, r.hi), keep.basis)
        else:
            out.append(r)
    return out


def _excursion(g, t, G, a, b, inc_a, inc_b, tol, xtol):
    """Sign and depth (largest |g|) of g between two consecutive zeros (or a zero and an end)."""
    if b < a:
        return _Exc(0.0, 0.0, True)
    mask = (t > a) & (t < b)
    if inc_a:
        mask |= t == a
    if inc_b:
        mask |= t == b
    vals = G[mask]
    vals = vals[vals != 0.0]
    if vals.size:
        k = int(np.argmax(np.abs(vals)))
        sign, depth = math.copysign(1.0, vals[k]), abs(float(vals[k]))
    else:
        sign, depth = 0.0, 0.0
    if depth < tol and b > a:
        if sign == 0.0:
            gm = g(0.5 * (a + b))
            sign = 0.0 if gm == 0.0 else math.copysign(1.0, gm)
            depth = abs(gm)
        if sign != 0.0:
            _, big, _ = _refine_max(lambda s: sign * g(s), a, b, xtol)
            depth = max(depth, big)
    return _Exc(sign, depth, False)


def _cluster_event(roots, p, q, exc, resolved, t0, t1):
    sb = exc[p].sign if resolved[p] else 0.0
    sa = exc[q + 1].sign if resolved[q + 1] else 0.0
    lo, hi = roots[p].lo, roots[q].hi
    if sb == 0.0:
        lo = t0
    if sa == 0.0:
        hi = t1
    single = p == q and lo == roots[p].lo and hi == roots[p].hi
    t_ev = roots[p].t if single else 0.5 * (lo + hi)
    basis = roots[p].basis if single else "cluster"
    inner = [exc[k].depth for k in range(p + 1, q + 1)]
    scale = max(inner) if inner else 0.0
    if sb != 0.0 and sa != 0.0 and sb != sa:
        kind = "entry" if sa < 0.0 else "exit"
    elif sb == 0.0 and sa < 0.0:
        kind = "entry"           # on the boundary at the start (or within tolerance of it), then inside
    elif sa == 0.0 and sb < 0.0:
        kind = "exit"            # inside, then on the boundary at the end
    else:
        kind = "tangency_unresolved"
    return CrossingEvent(kind, float(t_ev), float(lo), float(hi), basis, float(scale), q - p + 1)


def crossings(gfun, t0, t1, grid_dt=0.5, extra_times=(), tangency_tol=1e-9, *, xtol=1e-9, vectorized=False):
    """Entries into and exits from {g < 0} on [t0, t1], with near-tangencies reported, not dropped.

    1. Sample g on the grid (``grid_dt`` plus t1 plus ``extra_times``, duplicates merged).
    2. A zero at a sample is a located zero (basis "grid_point"); this covers zeros exactly at t0,
       at t1 and at update times. Each sign change between consecutive samples is bracketed and
       located by brentq (xtol, default 1e-9 s); the bracket is then shrunk to about 2*xtol and
       checked to still contain a sign change (basis "bracketed").
    3. Around every sampled local minimum of |g| without a sign change, sigma*g (sigma the local
       sign) is minimised between the neighbouring samples. A negative minimum means an excursion
       the samples missed; its two ends are located (basis "refinement"). A minimum in
       [0, tangency_tol) is a near-tangency. The two sub-brackets next to every located zero are
       searched the same way.
    4. The zeros split [t0, t1] into excursions of one sign. An interior excursion whose largest
       |g| is below ``tangency_tol`` cannot be told apart from a touch, so the zeros on either side
       are merged into one cluster. A cluster (or single zero) with different resolved signs on the
       two sides is an entry or exit; with the same sign on both sides it is reported as
       "tangency_unresolved" and is not counted as a crossing. An end of the horizon where
       |g| <= tangency_tol is flagged (start_on_boundary / end_on_boundary).

    Choose ``tangency_tol`` at or above the error of g evaluated on the dense output. Where no event
    is reported, absence rests on the samples and the local refinements (``absence_basis`` =
    "sampled"); ``resolution_assumption_s`` is the largest sample spacing.
    """
    if not tangency_tol >= 0.0:
        raise ValueError("tangency_tol must be >= 0")
    if not xtol > 0.0:
        raise ValueError("xtol must be positive")
    g = _scalar(gfun, vectorized)
    t = build_grid(t0, t1, grid_dt, extra_times)
    G = _sample(gfun, t, vectorized)
    _check_finite(t, G, "g")
    n = t.size
    s = np.sign(G)
    tol = float(tangency_tol)
    roots, near = [], []

    # zeros exactly at samples, and hidden excursions next to them
    for i in np.flatnonzero(G == 0.0):
        ti = float(t[i])
        roots.append(_Root(ti, ti, ti, "grid_point"))
        for j in (i - 1, i + 1):
            if 0 <= j < n and G[j] != 0.0:
                if j < i:
                    _resolve_same_sign(g, float(t[j]), ti, float(G[j]), 0.0, float(s[j]), xtol, roots, 1)
                else:
                    _resolve_same_sign(g, ti, float(t[j]), 0.0, float(G[j]), float(s[j]), xtol, roots, 1)

    # sign changes between consecutive nonzero samples
    for i in np.flatnonzero(s[:-1] * s[1:] < 0.0):
        _resolve_sign_change(g, float(t[i]), float(t[i + 1]), float(G[i]), float(G[i + 1]), xtol, "bracketed", roots)

    # sampled local minima of |g| with no sign change around them
    if n > 1:
        starts, ends = _local_max_runs(-np.abs(G))
        for i, j in zip(starts, ends):
            a_i, b_j = max(i - 1, 0), min(j + 1, n - 1)
            nb = s[a_i:b_j + 1]
            if nb[0] == 0.0 or np.any(nb != nb[0]):
                continue
            sigma = float(nb[0])
            tm, m = _resolve_same_sign(g, float(t[a_i]), float(t[b_j]), float(G[a_i]), float(G[b_j]), sigma,
                                       xtol, roots)
            run = sigma * G[i:j + 1]
            k = int(np.argmin(run))
            if run[k] < m:
                tm, m = float(t[i + k]), float(run[k])
            if 0.0 <= m < tol:
                near.append((tm, m))

    roots = _dedupe_roots(roots, xtol)
    nr = len(roots)
    t_first, t_last = float(t[0]), float(t[-1])
    events = []
    if nr == 0:
        e0 = _excursion(g, t, G, t_first, t_last, True, True, tol, xtol)
        if e0.depth < tol:
            events.append(CrossingEvent("tangency_unresolved", 0.5 * (t_first + t_last), t_first, t_last,
                                        "cluster", e0.depth, 0))
    else:
        exc = []
        for k in range(nr + 1):
            if k == 0:
                exc.append(_Exc(0.0, 0.0, True) if roots[0].lo <= t_first else
                           _excursion(g, t, G, t_first, roots[0].lo, True, False, tol, xtol))
            elif k == nr:
                exc.append(_Exc(0.0, 0.0, True) if roots[-1].hi >= t_last else
                           _excursion(g, t, G, roots[-1].hi, t_last, False, True, tol, xtol))
            else:
                exc.append(_excursion(g, t, G, roots[k - 1].hi, roots[k].lo, False, False, tol, xtol))
        resolved = [(not e.empty) and e.depth >= tol and e.sign != 0.0 for e in exc]
        p = 0
        for k in range(1, nr + 1):
            if k == nr or resolved[k]:
                events.append(_cluster_event(roots, p, k - 1, exc, resolved, t_first, t_last))
                p = k

    spans = [(e.t_lo, e.t_hi) for e in events]
    last_near = None
    for tm, m in sorted(near):
        if any(lo - 2.0 * xtol <= tm <= hi + 2.0 * xtol for lo, hi in spans):
            continue
        if last_near is not None and tm - last_near <= 2.0 * xtol:
            continue
        events.append(CrossingEvent("tangency_unresolved", tm, tm, tm, "near_tangency", m, 0))
        last_near = tm
    events.sort(key=lambda e: (e.t, e.t_lo))

    max_gap = float(np.max(np.diff(t))) if n > 1 else 0.0
    return CrossingResult(events=tuple(events), t0=t_first, t1=t_last, g_start=float(G[0]), g_end=float(G[-1]),
                          start_on_boundary=bool(abs(G[0]) <= tol), end_on_boundary=bool(abs(G[-1]) <= tol),
                          n_samples=int(n), resolution_assumption_s=max_gap, tangency_tol=tol, xtol=float(xtol))


# ---------------------------------------------------------------------------
# 5. dense-output coverage (used by arrival_label)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    ok: bool
    first_gap: object       # (start, end) of the first uncovered interval, or None
    covered_until: float    # end of the contiguous coverage that starts at t_start
    n_segments: int
    tol: float

    def as_dict(self):
        return asdict(self)


def coverage_check(segments, T, *, t_start=0.0):
    """Check that dense-output segments [(t0, t1), ...] cover [t_start, T] without gaps.

    Segments may be unsorted and may overlap. Two pieces count as joined when the gap between them
    is at most 32*eps*|T|. Returns a CoverageResult with the first gap found (or None).
    """
    T, t_start = float(T), float(t_start)
    tol = 32.0 * EPS * abs(T)
    segs = []
    for seg in segments:
        a, b = float(seg[0]), float(seg[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"non-finite segment {seg!r}")
        if b < a:
            raise ValueError(f"reversed segment {seg!r}")
        segs.append((a, b))
    segs.sort()
    reach = t_start
    for a, b in segs:
        if a > reach + tol:
            if reach < T - tol:
                return CoverageResult(False, (reach, min(a, T)), reach, len(segs), tol)
            break
        reach = max(reach, b)
    if reach < T - tol:
        return CoverageResult(False, (reach, T), reach, len(segs), tol)
    return CoverageResult(True, None, reach, len(segs), tol)


# ---------------------------------------------------------------------------
# 3. terminal-arrival label
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrivalResult:
    status: str                 # one of ARRIVAL_STATUSES
    arrived: object             # True / False, or None when unresolved
    nominal_arrived: bool       # the plain definition applied to the located times, ignoring uncertainty
    reason: str
    T: float
    t_end_completed: float
    inside_at_end: object       # bool, or None when g was not evaluated
    g_end: float
    t_entry: float              # last entry (t0 if inside throughout)
    t_entry_lo: float
    t_entry_hi: float
    dwell_s: float              # T - t_entry
    dwell_margin_s: float       # dwell_s - dwell
    decision_margin_s: float    # |dwell_margin_s| below this -> unresolved_near_deadline
    n_entries: int
    n_exits: int
    n_tangencies: int
    n_tangencies_after_entry: int
    crossings: object = None    # CrossingResult
    coverage: object = None     # CoverageResult

    def as_dict(self):
        return asdict(self)


def arrival_label(g_fun, t_end_completed, T, dwell=DWELL, decision_margin_s=None, *, time_uncertainty_s=1e-3,
                  t0=0.0, grid_dt=0.5, extra_times=(), tangency_tol=1e-9, xtol=1e-9, segments=None,
                  vectorized=False):
    """Terminal-arrival status of one branch with explicit unresolved outcomes.

    Order of decisions:
      incomplete_horizon        t_end_completed < T (or ``segments`` leave a gap before T); g is
                                not evaluated.
      unresolved_tangency       |g(T)| <= tangency_tol: inside-at-T cannot be decided.
      not_arrived               g(T) > 0.
      unresolved_near_deadline  |dwell margin| < decision margin, where dwell margin = (T - t_entry)
                                - dwell and the decision margin is ``decision_margin_s`` if given,
                                else max(time_uncertainty_s, width of the located entry interval).
      not_arrived               dwell margin <= -decision margin.
      unresolved_tangency       a tangency after the last entry could be a brief exit; if re-entry
                                at its latest possible time would leave less than dwell plus the
                                decision margin, the label is unresolved.
      arrived                   otherwise.
    t_entry is the last "entry" event from ``crossings`` (t0 if there is none and g < 0 throughout).
    """
    T, t0 = float(T), float(t0)
    t_done = float(t_end_completed)
    cov = None
    if segments is not None:
        cov = coverage_check(segments, T, t_start=t0)
        if not cov.ok:
            t_done = min(t_done, float(cov.covered_until))
    nan = float("nan")
    base = dict(T=T, t_end_completed=t_done, coverage=cov)
    if not t_done >= T - 32.0 * EPS * abs(T):
        return ArrivalResult(status="incomplete_horizon", arrived=False, nominal_arrived=False,
                             reason="dense output does not reach T", inside_at_end=None, g_end=nan, t_entry=nan,
                             t_entry_lo=nan, t_entry_hi=nan, dwell_s=nan, dwell_margin_s=nan,
                             decision_margin_s=nan, n_entries=0, n_exits=0, n_tangencies=0,
                             n_tangencies_after_entry=0, **base)

    cr = crossings(g_fun, t0, T, grid_dt, extra_times, tangency_tol, xtol=xtol, vectorized=vectorized)
    ents, exts, tans = cr.entries, cr.exits, cr.tangencies
    if ents:
        e = ents[-1]
        t_entry, lo, hi = e.t, e.t_lo, e.t_hi
    else:
        t_entry = lo = hi = t0
    dwell_s = T - t_entry
    margin = dwell_s - float(dwell)
    u = float(decision_margin_s) if decision_margin_s is not None else max(float(time_uncertainty_s), hi - lo)
    later = [ev for ev in tans if ev.t_hi > t_entry]
    nominal = bool(cr.g_end < 0.0 and margin >= 0.0)
    common = dict(nominal_arrived=nominal, inside_at_end=bool(cr.g_end < 0.0), g_end=cr.g_end, t_entry=t_entry,
                  t_entry_lo=lo, t_entry_hi=hi, dwell_s=dwell_s, dwell_margin_s=margin, decision_margin_s=u,
                  n_entries=len(ents), n_exits=len(exts), n_tangencies=len(tans),
                  n_tangencies_after_entry=len(later), crossings=cr, **base)

    if cr.end_on_boundary:
        return ArrivalResult(status="unresolved_tangency", arrived=None,
                             reason="g(T) within tangency_tol of zero: inside at T cannot be decided", **common)
    if cr.g_end > 0.0:
        return ArrivalResult(status="not_arrived", arrived=False, reason="outside the terminal set at T", **common)
    if abs(margin) < u:
        return ArrivalResult(status="unresolved_near_deadline", arrived=None,
                             reason="dwell margin smaller than the entry-time uncertainty", **common)
    if margin < 0.0:
        return ArrivalResult(status="not_arrived", arrived=False, reason="dwell shorter than required", **common)
    if later:
        t_alt = max(ev.t_hi for ev in later)
        if (T - t_alt) - float(dwell) < u:
            return ArrivalResult(status="unresolved_tangency", arrived=None,
                                 reason=f"tangency at t = {t_alt:.9g} s after the last entry could be a brief exit "
                                        "that would cut the dwell below the requirement", **common)
    return ArrivalResult(status="arrived", arrived=True, reason="inside at T with sufficient dwell", **common)


# ---------------------------------------------------------------------------
# 4. threshold label
# ---------------------------------------------------------------------------


def threshold_label(peak_value, peak_uncertainty, threshold):
    """'unresolved' iff |peak - threshold| <= peak_uncertainty (also for a NaN peak or uncertainty);
    otherwise 'violation' if peak > threshold, else 'ok'."""
    p, u, th = float(peak_value), float(peak_uncertainty), float(threshold)
    if u < 0.0:
        raise ValueError("peak_uncertainty must be >= 0")
    if math.isnan(th):
        raise ValueError("threshold must not be NaN")
    if math.isnan(p) or math.isnan(u) or abs(p - th) <= u:
        return "unresolved"
    return "violation" if p > th else "ok"


# ---------------------------------------------------------------------------
# callable builders for engine dense outputs
# ---------------------------------------------------------------------------


def discrepancy_fun(rho_a, rho_b):
    """d(t) = |rho_a(t)[:3] - rho_b(t)[:3]| from two dense callables returning position or state."""
    def d(t):
        a = np.asarray(rho_a(t), dtype=float).reshape(-1)[:3]
        b = np.asarray(rho_b(t), dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(a - b))
    return d


def terminal_g_fun(state_fun, pos_tol=POS_TOL, vel_tol=VEL_TOL):
    """g(t) = max(|rho|/pos_tol, |rho_dot|/vel_tol) - 1 from a dense callable returning the 6-state."""
    def g(t):
        x = np.asarray(state_fun(t), dtype=float).reshape(-1)
        return float(max(np.linalg.norm(x[:3]) / pos_tol, np.linalg.norm(x[3:6]) / vel_tol) - 1.0)
    return g


def peak_discrepancy(rho_a, rho_b, t_end_a, t_end_b, T=None, **kwargs):
    """robust_peak of d(t) over the common completed horizon [0, min(t_end_a, t_end_b[, T])]."""
    t_common = min(float(t_end_a), float(t_end_b))
    if T is not None:
        t_common = min(t_common, float(T))
    return robust_peak(discrepancy_fun(rho_a, rho_b), 0.0, t_common, **kwargs)


__all__ = ["ARRIVAL_STATUSES", "THRESHOLD_LABELS", "RESOLUTION_NOTE", "PeakLevel", "PeakResult", "CrossingEvent",
           "CrossingResult", "CoverageResult", "ArrivalResult", "build_grid", "robust_peak", "crossings",
           "arrival_label", "threshold_label", "coverage_check", "discrepancy_fun", "terminal_g_fun",
           "peak_discrepancy"]
