import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

EPS = float(np.finfo(float).eps)
INVPHI = (math.sqrt(5.0) - 1.0) / 2.0
MAXDEPTH = 6

POSTOL = 1.0
VELTOL = 0.01
DWELL = 300.0

ARRSTATUSES = ("arrived", "not_arrived", "unresolved_tangency", "unresolved_near_deadline",
               "incomplete_horizon")
THRLABELS = ("ok", "violation", "unresolved")

RESNOTE = ("Sample-based result: an excursion narrower than the grid spacing that leaves no trace in "
           "the samples cannot be excluded. Agreement between refinement levels is evidence, not proof.")


def buildgrid(t0, t1, dt, extratimes=()):
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
    ex = np.asarray(extratimes, dtype=float).ravel()
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


def scalar(fun, vec):
    if vec:
        return lambda s: float(np.asarray(fun(np.array([s], dtype=float)), dtype=float).reshape(-1)[0])
    return lambda s: float(fun(s))


def samp(fun, t, vec):
    if vec:
        v = np.asarray(fun(t), dtype=float).reshape(-1)
        if v.size != t.size:
            raise ValueError("vectorized callable must return one value per time")
        return v
    return np.fromiter((float(fun(s)) for s in t), dtype=float, count=t.size)


def checkfinite(t, v, what):
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"non-finite {what} at t = {float(t[bad][0])!r}")


def localmaxruns(v):
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


def goldenmax(f, a, b, xtol, maxiter=200):
    c = b - INVPHI * (b - a)
    d = a + INVPHI * (b - a)
    fc, fd = f(c), f(d)
    bestt, bestf = (c, fc) if fc >= fd else (d, fd)
    it = 0
    while (b - a) > xtol and it < maxiter:
        it += 1
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - INVPHI * (b - a)
            fc = f(c)
            if fc > bestf:
                bestt, bestf = c, fc
        else:
            a, c, fc = c, d, fd
            d = a + INVPHI * (b - a)
            fd = f(d)
            if fd > bestf:
                bestt, bestf = d, fd
    return bestt, bestf


# Maximise twice (bounded Brent and an independent golden-section search). Brent runs in u = t - a
# because SciPy's relative tolerance sqrt(eps)|x| would be about 1e-4 s at t ~ 5000 s.
def refinemax(f, a, b, xtol):
    w = b - a
    r = minimize_scalar(lambda u: -f(a + u), bounds=(0.0, w), method="bounded",
                        options={"xatol": xtol, "maxiter": 500})
    tb, fb = a + float(r.x), -float(r.fun)
    tg, fg = goldenmax(f, a, b, xtol)
    faend, fbend = f(a), f(b)
    tend, fend = (a, faend) if faend >= fbend else (b, fbend)
    cands = [(tb, fb), (tg, fg), (tend, fend)]
    finite = [c for c in cands if math.isfinite(c[1])]
    if not finite:
        return a, float("nan"), float("inf")
    tbest, fbest = max(finite, key=lambda c: c[1])
    if math.isfinite(fb) and math.isfinite(fg):
        ends = fend if math.isfinite(fend) else -math.inf
        gap = abs(max(fb, ends) - max(fg, ends))
    else:
        gap = float("inf")
    return tbest, fbest, gap


@dataclass(frozen=True)
class PeakLev:
    dt: float
    n_samples: int
    max_gap_s: float
    grid_only: float
    t_grid_only: float
    peak: float
    t_peak: float
    n_candidates: int
    method_gap: float


@dataclass(frozen=True)
class PeakRes:
    peak: float
    t_peak: float
    n_candidates: int
    grid_only: float
    t_grid_only: float
    peak_primary: float
    t_peak_primary: float
    refinement_dts: tuple
    refinement_values: tuple
    converged: bool
    last_change: float
    convergence_tol: float
    uncertainty: float
    resolution_assumption_s: float
    levels: tuple
    t0: float
    t1: float
    note: str = RESNOTE

    def asdict2(self):
        return asdict(self)


def peaklevel(fun, f, t0, t1, dt, extratimes, relband, topk, xtol, vec):
    t = buildgrid(t0, t1, dt, extratimes)
    v = samp(fun, t, vec)
    checkfinite(t, v, "function value")
    ig = int(np.argmax(v))
    gmax = float(v[ig])
    starts, ends = localmaxruns(v)
    vals = v[starts]
    band = gmax - (1.0 - relband) * abs(gmax)
    sel = vals >= band
    if topk:
        sel[np.argsort(-vals, kind="stable")[:topk]] = True
    bestt, best = float(t[ig]), gmax
    gapmax = 0.0
    for k in np.flatnonzero(sel):
        lo = float(t[max(starts[k] - 1, 0)])
        hi = float(t[min(ends[k] + 1, t.size - 1)])
        if hi > lo:
            tr, fr, gap = refinemax(f, lo, hi, xtol)
            gapmax = max(gapmax, gap)
            if fr > best:
                bestt, best = float(tr), float(fr)
    maxgap = float(np.max(np.diff(t))) if t.size > 1 else 0.0
    return PeakLev(float(dt), int(t.size), maxgap, gmax, float(t[ig]), best, bestt, int(sel.sum()), gapmax)


def robustpeak(fun, t0, t1, griddt=0.5, extratimes=(), relband=0.5, refinextol=1e-6,
               monrefs=(0.5, 0.25, 0.125), *, topk=5, abs_tol=1e-9, rel_tol=1e-9,
               vec=False):
    if not 0.0 <= relband <= 1.0:
        raise ValueError("rel_band must lie in [0, 1]")
    if topk < 0:
        raise ValueError("top_k must be >= 0")
    if not refinextol > 0.0:
        raise ValueError("refine_xtol must be positive")
    f = scalar(fun, vec)
    t0, t1 = float(t0), float(t1)
    cache = {}

    def level(dt):
        key = float(dt)
        if key not in cache:
            cache[key] = peaklevel(fun, f, t0, t1, key, extratimes, relband, int(topk), refinextol, vec)
        return cache[key]

    prim = level(griddt)
    mon = tuple(level(dt) for dt in monrefs)
    best = prim
    for lv in mon:
        if lv.peak > best.peak:
            best = lv
    tol = max(abs_tol, rel_tol * abs(best.peak))
    if len(mon) >= 2:
        change = abs(mon[-1].peak - mon[-2].peak)
        conv = bool(change < tol)
        unc = max(tol, change, mon[-1].method_gap, mon[-2].method_gap)
    else:
        change, conv, unc = float("nan"), False, float("inf")
    levels = (prim,) + tuple(lv for lv in mon if lv is not prim)
    return PeakRes(
        peak=best.peak, t_peak=best.t_peak, n_candidates=prim.n_candidates,
        grid_only=prim.grid_only, t_grid_only=prim.t_grid_only,
        peak_primary=prim.peak, t_peak_primary=prim.t_peak,
        refinement_dts=tuple(lv.dt for lv in mon), refinement_values=tuple(lv.peak for lv in mon),
        converged=conv, last_change=change, convergence_tol=tol, uncertainty=unc,
        resolution_assumption_s=min(lv.dt for lv in levels), levels=levels, t0=t0, t1=t1)


@dataclass(frozen=True)
class CrossEvent:
    kind: str
    t: float
    t_lo: float
    t_hi: float
    basis: str
    g_scale: float
    n_roots: int

    @property
    def width(self):
        return self.t_hi - self.t_lo


@dataclass(frozen=True)
class CrossRes:
    events: tuple
    t0: float
    t1: float
    g_start: float
    g_end: float
    start_on_boundary: bool
    end_on_boundary: bool
    n_samples: int
    resolution_assumption_s: float
    tangency_tol: float
    xtol: float
    absence_basis: str = "sampled"
    note: str = RESNOTE

    @property
    def entries(self):
        return tuple(e for e in self.events if e.kind == "entry")

    @property
    def exits(self):
        return tuple(e for e in self.events if e.kind == "exit")

    @property
    def tang(self):
        return tuple(e for e in self.events if e.kind == "tangency_unresolved")

    @property
    def inside_at_end(self):
        return self.g_end < 0.0

    @property
    def n_bracketed(self):
        return sum(e.kind != "tangency_unresolved" and e.basis in ("grid_point", "bracketed") for e in self.events)

    @property
    def n_from_refinement(self):
        return sum(e.kind != "tangency_unresolved" and e.basis == "refinement" for e in self.events)

    @property
    def n_clustered(self):
        return sum(e.kind != "tangency_unresolved" and e.basis == "cluster" for e in self.events)

    def asdict2(self):
        out = asdict(self)
        out.update(n_entries=len(self.entries), n_exits=len(self.exits), n_tangencies=len(self.tang),
                   n_bracketed=self.n_bracketed, n_from_refinement=self.n_from_refinement,
                   n_clustered=self.n_clustered)
        return out


@dataclass(frozen=True)
class Root:
    t: float
    lo: float
    hi: float
    basis: str


@dataclass(frozen=True)
class Exc:
    sign: float
    depth: float
    empty: bool


def locate(g, a, b, ga, gb, xtol):
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


def ressignchange(g, a, b, ga, gb, xtol, basis, out, depth=0):
    r, lo, hi, glo, ghi = locate(g, a, b, ga, gb, xtol)
    out.append(Root(r, lo, hi, basis))
    if depth >= MAXDEPTH:
        return
    for u, v, gu, gv in ((a, lo, ga, glo), (hi, b, ghi, gb)):
        if not v > u or (gu == 0.0 and gv == 0.0):
            continue
        if gu != 0.0 and gv != 0.0 and (gu > 0.0) != (gv > 0.0):
            ressignchange(g, u, v, gu, gv, xtol, "refinement", out, depth + 1)
        else:
            sigma = 1.0 if (gu > 0.0 or gv > 0.0) else -1.0
            ressamesign(g, u, v, gu, gv, sigma, xtol, out, depth + 1)


# Same sign at both ends: minimise sigma g on [a, b] to catch an excursion hidden between samples
def ressamesign(g, a, b, ga, gb, sigma, xtol, out, depth=0):
    tm, negm, _ = refinemax(lambda s: -sigma * g(s), a, b, xtol)
    m = -negm
    if m < 0.0 and depth < MAXDEPTH:
        gm = g(tm)
        if ga != 0.0:
            ressignchange(g, a, tm, ga, gm, xtol, "refinement", out, depth + 1)
        if gb != 0.0:
            ressignchange(g, tm, b, gm, gb, xtol, "refinement", out, depth + 1)
    return float(tm), float(m)


def deduproots(roots, xtol):
    rank = {"grid_point": 0, "bracketed": 1, "refinement": 2}
    out = []
    for r in sorted(roots, key=lambda x: x.t):
        if out and (r.t - out[-1].t <= 2.0 * xtol or r.lo <= out[-1].hi):
            p = out[-1]
            keep = p if (rank[p.basis], p.hi - p.lo) <= (rank[r.basis], r.hi - r.lo) else r
            out[-1] = Root(keep.t, min(p.lo, r.lo), max(p.hi, r.hi), keep.basis)
        else:
            out.append(r)
    return out


def excurs(g, t, G, a, b, inca, incb, tol, xtol):
    if b < a:
        return Exc(0.0, 0.0, True)
    mask = (t > a) & (t < b)
    if inca:
        mask |= t == a
    if incb:
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
            _, big, _ = refinemax(lambda s: sign * g(s), a, b, xtol)
            depth = max(depth, big)
    return Exc(sign, depth, False)


def clustevent(roots, p, q, exc, resolved, t0, t1):
    sb = exc[p].sign if resolved[p] else 0.0
    sa = exc[q + 1].sign if resolved[q + 1] else 0.0
    lo, hi = roots[p].lo, roots[q].hi
    if sb == 0.0:
        lo = t0
    if sa == 0.0:
        hi = t1
    single = p == q and lo == roots[p].lo and hi == roots[p].hi
    tev = roots[p].t if single else 0.5 * (lo + hi)
    basis = roots[p].basis if single else "cluster"
    inner = [exc[k].depth for k in range(p + 1, q + 1)]
    scale = max(inner) if inner else 0.0
    if sb != 0.0 and sa != 0.0 and sb != sa:
        kind = "entry" if sa < 0.0 else "exit"
    elif sb == 0.0 and sa < 0.0:
        kind = "entry"
    elif sa == 0.0 and sb < 0.0:
        kind = "exit"
    else:
        kind = "tangency_unresolved"
    return CrossEvent(kind, float(tev), float(lo), float(hi), basis, float(scale), q - p + 1)


# Near-tangencies are reported, not dropped
def crossings(gfun, t0, t1, griddt=0.5, extratimes=(), tangency_tol=1e-9, *, xtol=1e-9, vec=False):
    if not tangency_tol >= 0.0:
        raise ValueError("tangency_tol must be >= 0")
    if not xtol > 0.0:
        raise ValueError("xtol must be positive")
    g = scalar(gfun, vec)
    t = buildgrid(t0, t1, griddt, extratimes)
    G = samp(gfun, t, vec)
    checkfinite(t, G, "g")
    n = t.size
    s = np.sign(G)
    tol = float(tangency_tol)
    roots, near = [], []

    for i in np.flatnonzero(G == 0.0):
        ti = float(t[i])
        roots.append(Root(ti, ti, ti, "grid_point"))
        for j in (i - 1, i + 1):
            if 0 <= j < n and G[j] != 0.0:
                if j < i:
                    ressamesign(g, float(t[j]), ti, float(G[j]), 0.0, float(s[j]), xtol, roots, 1)
                else:
                    ressamesign(g, ti, float(t[j]), 0.0, float(G[j]), float(s[j]), xtol, roots, 1)

    for i in np.flatnonzero(s[:-1] * s[1:] < 0.0):
        ressignchange(g, float(t[i]), float(t[i + 1]), float(G[i]), float(G[i + 1]), xtol, "bracketed", roots)

    if n > 1:
        starts, ends = localmaxruns(-np.abs(G))
        for i, j in zip(starts, ends):
            ai, bj = max(i - 1, 0), min(j + 1, n - 1)
            nb = s[ai:bj + 1]
            if nb[0] == 0.0 or np.any(nb != nb[0]):
                continue
            sigma = float(nb[0])
            tm, m = ressamesign(g, float(t[ai]), float(t[bj]), float(G[ai]), float(G[bj]), sigma,
                                xtol, roots)
            run = sigma * G[i:j + 1]
            k = int(np.argmin(run))
            if run[k] < m:
                tm, m = float(t[i + k]), float(run[k])
            if 0.0 <= m < tol:
                near.append((tm, m))

    roots = deduproots(roots, xtol)
    nr = len(roots)
    tfirst, tlast = float(t[0]), float(t[-1])
    events = []
    if nr == 0:
        e0 = excurs(g, t, G, tfirst, tlast, True, True, tol, xtol)
        if e0.depth < tol:
            events.append(CrossEvent("tangency_unresolved", 0.5 * (tfirst + tlast), tfirst, tlast,
                                     "cluster", e0.depth, 0))
    else:
        exc = []
        for k in range(nr + 1):
            if k == 0:
                exc.append(Exc(0.0, 0.0, True) if roots[0].lo <= tfirst else
                           excurs(g, t, G, tfirst, roots[0].lo, True, False, tol, xtol))
            elif k == nr:
                exc.append(Exc(0.0, 0.0, True) if roots[-1].hi >= tlast else
                           excurs(g, t, G, roots[-1].hi, tlast, False, True, tol, xtol))
            else:
                exc.append(excurs(g, t, G, roots[k - 1].hi, roots[k].lo, False, False, tol, xtol))
        resolved = [(not e.empty) and e.depth >= tol and e.sign != 0.0 for e in exc]
        p = 0
        for k in range(1, nr + 1):
            if k == nr or resolved[k]:
                events.append(clustevent(roots, p, k - 1, exc, resolved, tfirst, tlast))
                p = k

    spans = [(e.t_lo, e.t_hi) for e in events]
    lastnear = None
    for tm, m in sorted(near):
        if any(lo - 2.0 * xtol <= tm <= hi + 2.0 * xtol for lo, hi in spans):
            continue
        if lastnear is not None and tm - lastnear <= 2.0 * xtol:
            continue
        events.append(CrossEvent("tangency_unresolved", tm, tm, tm, "near_tangency", m, 0))
        lastnear = tm
    events.sort(key=lambda e: (e.t, e.t_lo))

    maxgap = float(np.max(np.diff(t))) if n > 1 else 0.0
    return CrossRes(events=tuple(events), t0=tfirst, t1=tlast, g_start=float(G[0]), g_end=float(G[-1]),
                    start_on_boundary=bool(abs(G[0]) <= tol), end_on_boundary=bool(abs(G[-1]) <= tol),
                    n_samples=int(n), resolution_assumption_s=maxgap, tangency_tol=tol, xtol=float(xtol))


@dataclass(frozen=True)
class CovRes:
    ok: bool
    first_gap: object
    covered_until: float
    n_segments: int
    tol: float

    def asdict2(self):
        return asdict(self)


# Pieces count as joined when the gap between them is at most 32 eps |T|
def covcheck(segments, T, *, tstart=0.0):
    T, tstart = float(T), float(tstart)
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
    reach = tstart
    for a, b in segs:
        if a > reach + tol:
            if reach < T - tol:
                return CovRes(False, (reach, min(a, T)), reach, len(segs), tol)
            break
        reach = max(reach, b)
    if reach < T - tol:
        return CovRes(False, (reach, T), reach, len(segs), tol)
    return CovRes(True, None, reach, len(segs), tol)


@dataclass(frozen=True)
class ArrRes:
    status: str
    arrived: object
    nominal_arrived: bool
    reason: str
    T: float
    t_end_completed: float
    inside_at_end: object
    g_end: float
    t_entry: float
    t_entry_lo: float
    t_entry_hi: float
    dwell_s: float
    dwell_margin_s: float
    decision_margin_s: float
    n_entries: int
    n_exits: int
    n_tangencies: int
    n_tangencies_after_entry: int
    crossings: object = None
    coverage: object = None

    def asdict2(self):
        return asdict(self)


def arrlabel(gfun2, t_end_completed, T, dwell=DWELL, decision_margin_s=None, *, timeuncs=1e-3,
             t0=0.0, griddt=0.5, extratimes=(), tangency_tol=1e-9, xtol=1e-9, segments=None,
             vec=False):
    T, t0 = float(T), float(t0)
    tdone = float(t_end_completed)
    cov = None
    if segments is not None:
        cov = covcheck(segments, T, tstart=t0)
        if not cov.ok:
            tdone = min(tdone, float(cov.covered_until))
    nan = float("nan")
    base = dict(T=T, t_end_completed=tdone, coverage=cov)
    if not tdone >= T - 32.0 * EPS * abs(T):
        return ArrRes(status="incomplete_horizon", arrived=False, nominal_arrived=False,
                      reason="dense output does not reach T", inside_at_end=None, g_end=nan, t_entry=nan,
                      t_entry_lo=nan, t_entry_hi=nan, dwell_s=nan, dwell_margin_s=nan,
                      decision_margin_s=nan, n_entries=0, n_exits=0, n_tangencies=0,
                      n_tangencies_after_entry=0, **base)

    cr = crossings(gfun2, t0, T, griddt, extratimes, tangency_tol, xtol=xtol, vec=vec)
    ents, exts, tans = cr.entries, cr.exits, cr.tang
    if ents:
        e = ents[-1]
        tentry, lo, hi = e.t, e.t_lo, e.t_hi
    else:
        tentry = lo = hi = t0
    dwells = T - tentry
    margin = dwells - float(dwell)
    u = float(decision_margin_s) if decision_margin_s is not None else max(float(timeuncs), hi - lo)
    later = [ev for ev in tans if ev.t_hi > tentry]
    nominal = bool(cr.g_end < 0.0 and margin >= 0.0)
    common = dict(nominal_arrived=nominal, inside_at_end=bool(cr.g_end < 0.0), g_end=cr.g_end, t_entry=tentry,
                  t_entry_lo=lo, t_entry_hi=hi, dwell_s=dwells, dwell_margin_s=margin, decision_margin_s=u,
                  n_entries=len(ents), n_exits=len(exts), n_tangencies=len(tans),
                  n_tangencies_after_entry=len(later), crossings=cr, **base)

    if cr.end_on_boundary:
        return ArrRes(status="unresolved_tangency", arrived=None,
                      reason="g(T) within tangency_tol of zero: inside at T cannot be decided", **common)
    if cr.g_end > 0.0:
        return ArrRes(status="not_arrived", arrived=False, reason="outside the terminal set at T", **common)
    if abs(margin) < u:
        return ArrRes(status="unresolved_near_deadline", arrived=None,
                      reason="dwell margin smaller than the entry-time uncertainty", **common)
    if margin < 0.0:
        return ArrRes(status="not_arrived", arrived=False, reason="dwell shorter than required", **common)
    if later:
        talt = max(ev.t_hi for ev in later)
        if (T - talt) - float(dwell) < u:
            return ArrRes(status="unresolved_tangency", arrived=None,
                          reason=f"tangency at t = {talt:.9g} s after the last entry could be a brief exit "
                                 "that would cut the dwell below the requirement", **common)
    return ArrRes(status="arrived", arrived=True, reason="inside at T with sufficient dwell", **common)


def thrlabel(peakvalue, peakunc, threshold):
    p, u, th = float(peakvalue), float(peakunc), float(threshold)
    if u < 0.0:
        raise ValueError("peak_uncertainty must be >= 0")
    if math.isnan(th):
        raise ValueError("threshold must not be NaN")
    if math.isnan(p) or math.isnan(u) or abs(p - th) <= u:
        return "unresolved"
    return "violation" if p > th else "ok"


def discfun(rhoa, rhob):
    def d(t):
        a = np.asarray(rhoa(t), dtype=float).reshape(-1)[:3]
        b = np.asarray(rhob(t), dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(a - b))
    return d


def termgfun(statefun, pos_tol=POSTOL, vel_tol=VELTOL):
    def g(t):
        x = np.asarray(statefun(t), dtype=float).reshape(-1)
        return float(max(np.linalg.norm(x[:3]) / pos_tol, np.linalg.norm(x[3:6]) / vel_tol) - 1.0)
    return g


def peakdisc(rhoa, rhob, tenda, tendb, T=None, **kwargs):
    tcommon = min(float(tenda), float(tendb))
    if T is not None:
        tcommon = min(tcommon, float(T))
    return robustpeak(discfun(rhoa, rhob), 0.0, tcommon, **kwargs)
