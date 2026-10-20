import math
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cwvalid.metrics import refineext, commongrid
from cwvalid3.metrics3 import (
    arrlabel, buildgrid, covcheck, crossings, discfun, peakdisc, robustpeak,
    termgfun, thrlabel)


def prodpeak(fun, t_end, dt=0.5):
    t = commongrid(t_end, dt)
    v = np.array([fun(s) for s in t])
    return refineext(fun, t, int(np.argmax(v)), maximize=True)


def localtruth(fun, a, b):
    r = minimize_scalar(lambda u: -fun(a + u), bounds=(0.0, b - a), method="bounded",
                        options={"xatol": 1e-13, "maxiter": 2000})
    return a + r.x, -r.fun


BUMPT, BUMPSIGMA, BUMPA = 10.125, 0.05, 1.5


def bumpfun(t):
    return math.sin(2.0 * math.pi * t / 10.0) + BUMPA * math.exp(-0.5 * ((t - BUMPT) / BUMPSIGMA) ** 2)


def testshortexcmissed():
    ttrue, ftrue = localtruth(bumpfun, 10.0, 10.25)
    assert ftrue > 1.5

    _, fprod = prodpeak(bumpfun, 20.0)
    assert fprod == pytest.approx(1.0, abs=1e-9)

    res = robustpeak(bumpfun, 0.0, 20.0)
    assert res.grid_only == pytest.approx(1.0, abs=1e-12)
    assert res.peak_primary == pytest.approx(1.0, abs=1e-9)
    v05, v025, v0125 = res.refinement_values
    assert v05 == pytest.approx(1.0, abs=1e-9)
    assert v025 == pytest.approx(1.0, abs=1e-9)
    assert v0125 == pytest.approx(ftrue, abs=1e-9)
    assert res.converged is False
    assert res.last_change > 0.5
    assert res.uncertainty >= res.last_change
    assert res.peak == pytest.approx(ftrue, abs=1e-9)
    assert res.t_peak == pytest.approx(ttrue, abs=1e-5)
    assert res.resolution_assumption_s == 0.125

    res2 = robustpeak(bumpfun, 0.0, 20.0, monrefs=(0.5, 0.25, 0.125, 0.0625))
    assert res2.converged is True
    assert res2.peak == pytest.approx(ftrue, abs=1e-9)
    assert res2.uncertainty < 1e-8

    assert thrlabel(res.peak, res.uncertainty, 1.2) == "unresolved"
    assert thrlabel(res2.peak, res2.uncertainty, 1.2) == "violation"


def testagrbetwlevels():
    ts, sig = 10.046875, 0.002

    def f(t):
        return math.sin(2.0 * math.pi * t / 10.0) + 1.5 * math.exp(-0.5 * ((t - ts) / sig) ** 2)

    _, ftrue = localtruth(f, ts - 0.01, ts + 0.01)
    res = robustpeak(f, 0.0, 20.0)
    assert ftrue > 1.5
    assert res.converged is True
    assert res.peak == pytest.approx(1.0, abs=1e-9)
    assert res.resolution_assumption_s == 0.125
    assert "cannot be excluded" in res.note


def twopeaks(t):
    a = 1.000 * math.exp(-((t - 10.2) / 0.40) ** 2)
    b = 1.001 * math.exp(-((t - 12.2) / 0.25) ** 2)
    decoy = 0.7 * math.exp(-((t - 20.0) / 0.40) ** 2)
    return a + b + decoy


def testbclosemaxima():
    t = buildgrid(0.0, 30.0, 0.5)
    v = np.array([twopeaks(s) for s in t])
    assert t[int(np.argmax(v))] == pytest.approx(10.0)

    _, fprod = prodpeak(twopeaks, 30.0)
    assert fprod == pytest.approx(1.000, abs=1e-6)

    res = robustpeak(twopeaks, 0.0, 30.0)
    assert res.peak == pytest.approx(1.001, abs=1e-9)
    assert res.t_peak == pytest.approx(12.2, abs=1e-3)
    assert res.peak_primary == pytest.approx(1.001, abs=1e-9)
    assert res.n_candidates == 3
    assert res.converged is True

    resk = robustpeak(twopeaks, 0.0, 30.0, relband=1.0, topk=5)
    assert resk.peak == pytest.approx(1.001, abs=1e-9)
    resoff = robustpeak(twopeaks, 0.0, 30.0, relband=1.0, topk=0)
    assert resoff.peak == pytest.approx(1.000, abs=1e-6)
    assert resoff.n_candidates == 1


@pytest.mark.parametrize("t0", [0.0, 0.2])
def testcexacttouch(t0):
    cr = crossings(lambda t: (t - 5.0) ** 2, t0, 10.0)
    assert len(cr.entries) == 0 and len(cr.exits) == 0
    assert len(cr.tang) == 1
    ev = cr.tang[0]
    assert ev.t_lo - 1e-6 <= 5.0 <= ev.t_hi + 1e-6
    assert ev.g_scale < 1e-9
    assert ev.basis == ("grid_point" if t0 == 0.0 else "near_tangency")


@pytest.mark.parametrize("t0", [0.0, 0.2])
def testcdipbelow(t0):
    cr = crossings(lambda t: (t - 5.0) ** 2 - 1e-12, t0, 10.0)
    assert len(cr.entries) == 0 and len(cr.exits) == 0
    assert len(cr.tang) == 1
    ev = cr.tang[0]
    assert ev.n_roots == 2
    assert ev.t_lo <= 5.0 <= ev.t_hi
    assert ev.g_scale == pytest.approx(1e-12, rel=1e-3)


def testcresdip():
    cr = crossings(lambda t: (t - 5.0) ** 2 - 1e-4, 0.2, 10.0)
    assert [e.kind for e in cr.events] == ["entry", "exit"]
    assert cr.entries[0].t == pytest.approx(4.99, abs=1e-8)
    assert cr.exits[0].t == pytest.approx(5.01, abs=1e-8)
    assert cr.n_from_refinement == 2 and cr.n_bracketed == 0
    assert cr.absence_basis == "sampled"


def testctangentry():
    T = 1000.0
    lab = arrlabel(lambda t: -((t - 800.0) ** 2) / 1e4, T, T)
    assert lab.status == "unresolved_tangency" and lab.arrived is None
    assert lab.nominal_arrived is True
    lab2 = arrlabel(lambda t: -((t - 500.0) ** 2) / 1e4, T, T)
    assert lab2.status == "arrived" and lab2.n_tangencies_after_entry == 1


def testdthrplat():
    thr = 2.5

    def d(t):
        return min(thr, 1.5 * thr * math.exp(-((t - 5.0) / 2.0) ** 2))

    res = robustpeak(d, 0.0, 10.0)
    assert res.peak == thr
    assert 3.7 <= res.t_peak <= 6.3
    assert res.n_candidates == 1
    assert thrlabel(res.peak, res.uncertainty, thr) == "unresolved"
    assert thrlabel(res.peak, 0.0, thr) == "unresolved"
    u = 1e-6
    assert thrlabel(thr + 0.5 * u, u, thr) == "unresolved"
    assert thrlabel(thr + 2.0 * u, u, thr) == "violation"
    assert thrlabel(thr - 2.0 * u, u, thr) == "ok"
    assert thrlabel(float("nan"), u, thr) == "unresolved"
    with pytest.raises(ValueError):
        thrlabel(thr, -1.0, thr)


def testeentryexactly():
    T = 1000.0
    cr = crossings(lambda t: -t, 0.0, T)
    assert len(cr.entries) == 1 and len(cr.exits) == 0
    e = cr.entries[0]
    assert e.t == 0.0 and e.width == 0.0 and e.basis == "grid_point"
    assert cr.start_on_boundary
    lab = arrlabel(lambda t: -t, T, T)
    assert lab.status == "arrived" and lab.t_entry == 0.0 and lab.dwell_s == T


def testeboundcont():
    cr = crossings(lambda t: t, 0.0, 10.0)
    assert len(cr.entries) == 0 and len(cr.tang) == 1
    assert cr.tang[0].t == 0.0


@pytest.mark.parametrize("T", [1000.0, 1000.3])
def testeexitexactly(T):
    cr = crossings(lambda t: t - T, 0.0, T)
    assert len(cr.exits) == 1 and len(cr.entries) == 0
    assert cr.exits[0].t == T and cr.end_on_boundary
    lab = arrlabel(lambda t: t - T, T, T)
    assert lab.status == "unresolved_tangency"
    assert lab.arrived is None


def testecrossjust():
    T = 1000.0
    epst = 1e-7
    cr = crossings(lambda t: t - epst, 0.0, T)
    assert len(cr.exits) == 1 and cr.exits[0].t == pytest.approx(epst, abs=2e-9)
    assert cr.exits[0].basis == "bracketed"
    g = lambda t: (T - epst) - t
    cr2 = crossings(g, 0.0, T)
    assert len(cr2.entries) == 1 and cr2.entries[0].t == pytest.approx(T - epst, abs=2e-9)
    lab = arrlabel(g, T, T)
    assert lab.status == "not_arrived" and lab.dwell_s < 1e-6


def decstate(te2, tau=200.0):
    def x(t):
        r = math.exp(-(t - te2) / tau)
        return np.array([r, 0.0, 0.0, -r / tau, 0.0, 0.0])
    return x


@pytest.mark.parametrize("offset, status", [(+0.01, "arrived"), (-0.01, "not_arrived"),
                                            (+1e-4, "unresolved_near_deadline"),
                                            (-1e-4, "unresolved_near_deadline")])
def testfdwellnear(offset, status):
    T, dwell = 1000.0, 300.0
    te = T - dwell - offset
    for g in (termgfun(decstate(te)), lambda t: 0.01 * (te - t)):
        lab = arrlabel(g, T, T, dwell)
        assert lab.status == status
        assert lab.t_entry == pytest.approx(te, abs=1e-8)
        assert lab.dwell_margin_s == pytest.approx(offset, abs=1e-8)
        assert lab.nominal_arrived is (offset >= 0)
        assert lab.decision_margin_s == pytest.approx(1e-3)
    lab = arrlabel(lambda t: 0.01 * (te - t), T, T, dwell, decision_margin_s=0.02)
    assert lab.status == "unresolved_near_deadline"


def testflastentry():
    T = 1000.0
    g = lambda t: -(t - 100.0) * (t - 400.0) * (t - 600.0) / 1e7
    lab = arrlabel(g, T, T)
    assert lab.n_entries == 2 and lab.n_exits == 1
    assert lab.t_entry == pytest.approx(600.0, abs=1e-8)
    assert lab.status == "arrived" and lab.dwell_margin_s == pytest.approx(100.0, abs=1e-8)
    assert arrlabel(g, T, 850.0).status == "not_arrived"


def testgdupextra():
    extra = [3.3, 3.3, 3.3, 7.0, 7.0, 2.5, 3.3 + 1e-15, -1.0, 11.0]
    t = buildgrid(0.0, 10.0, 0.5, extra)
    assert t.size == 22
    assert np.all(np.diff(t) > 0)
    assert np.count_nonzero(np.isclose(t, 3.3, atol=1e-12)) == 1

    f = lambda s: 1.0 - (s - 3.3) ** 2
    r1 = robustpeak(f, 0.0, 10.0, extratimes=extra)
    r0 = robustpeak(f, 0.0, 10.0, extratimes=[3.3])
    assert r1.peak == 1.0 and r1.t_peak == 3.3
    assert r1.n_candidates == r0.n_candidates == 1
    assert [lv.n_samples for lv in r1.levels] == [lv.n_samples for lv in r0.levels]

    g = lambda s: 3.3 - s
    c1 = crossings(g, 0.0, 10.0, extratimes=extra)
    c0 = crossings(g, 0.0, 10.0, extratimes=[3.3])
    assert len(c1.entries) == 1 and len(c1.events) == 1
    assert c1.entries[0].t == 3.3 and c1.entries[0].basis == "grid_point"
    assert c1.events == c0.events
    lab = arrlabel(g, 10.0, 10.0, dwell=5.0, extratimes=extra)
    assert lab.status == "arrived" and lab.n_entries == 1


def testhincdense():
    T = 1000.0
    segs = [(0.0, 400.0), (400.0, T - 1.0)]
    cov = covcheck(segs, T)
    assert cov.ok is False and cov.first_gap == (T - 1.0, T) and cov.covered_until == T - 1.0

    def gnever(t):
        raise AssertionError("g must not be evaluated beyond the completed horizon")

    lab = arrlabel(gnever, T, T, segments=segs)
    assert lab.status == "incomplete_horizon" and lab.arrived is False
    assert lab.coverage.first_gap == (T - 1.0, T)
    lab2 = arrlabel(gnever, T - 1.0, T)
    assert lab2.status == "incomplete_horizon"

    gap = covcheck([(0.0, 100.0), (100.5, T)], T)
    assert gap.ok is False and gap.first_gap == (100.0, 100.5)
    assert covcheck([(0.0, 100.0), (100.0 + 1e-13, T)], T).ok
    assert covcheck([(500.0, T), (0.0, 600.0), (100.0, 200.0)], T).ok
    assert covcheck([(1e-3, T)], T).first_gap == (0.0, 1e-3)
    with pytest.raises(ValueError):
        covcheck([(0.0, T), (5.0, 4.0)], T)

    full = arrlabel(lambda t: -1.0, T, T, segments=[(0.0, 500.0), (500.0, T)])
    assert full.status == "arrived" and full.coverage.ok


def smoothdisc(t):
    return 2.0 + 0.5 * math.sin(2.0 * math.pi * t / 5400.0 + 0.3) + 0.1 * math.cos(2.0 * math.pi * t / 900.0)


@pytest.mark.parametrize("t_end", [3000.0, 3000.3])
def testimatchprod(t_end):
    tprod, fprod = prodpeak(smoothdisc, t_end)
    res = robustpeak(smoothdisc, 0.0, t_end)
    assert abs(res.peak - fprod) <= 1e-9
    assert abs(res.peak_primary - fprod) <= 1e-9
    assert res.t_peak == pytest.approx(tprod, abs=0.05)
    assert res.converged is True
    assert np.array_equal(buildgrid(0.0, t_end, 0.5), commongrid(t_end, 0.5))


def testigridmatch():
    from cwvalid3.heldmet import mongrid
    for T in (5400.0, 5400.25, 1234.5678):
        assert np.array_equal(buildgrid(0.0, T, 0.5), mongrid(T))


def testpeakdisctwo():
    rhoa = lambda t: np.array([t, 0.0, 0.0, 1.0, 0.0, 0.0])
    rhob = lambda t: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = peakdisc(rhoa, rhob, 800.0, 650.0, T=1000.0)
    assert res.t1 == 650.0 and res.peak == 650.0 and res.t_peak == 650.0
    assert discfun(rhoa, rhob)(3.0) == 3.0


def testvecmatchscalar():
    fv = np.vectorize(bumpfun)
    a = robustpeak(bumpfun, 0.0, 20.0)
    b = robustpeak(fv, 0.0, 20.0, vec=True)
    assert a.refinement_values == pytest.approx(b.refinement_values, abs=1e-12)
    gv = lambda t: 0.01 * (700.0 - np.asarray(t))
    assert crossings(gv, 0.0, 1000.0, vec=True).entries[0].t == pytest.approx(700.0, abs=1e-9)


def testnonfinsampsraise():
    with pytest.raises(ValueError):
        robustpeak(lambda t: float("nan") if t > 5 else 1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        crossings(lambda t: float("inf"), 0.0, 10.0)
