import ast
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import indverify
from indverify import IMPL, __version__, runcase
from indverify.assess import (arrgrid, assessarr, refinemax,
                              sattime, signivals, timegrid)
from indverify.constants import MU
from indverify.control import PDCtrl
from indverify.frames import (TgtOrbit, dcmecilvlh, inertrel,
                              relinert)
from indverify.integrators import (HermDense, integadapt,
                                   integrk4)
from indverify.models import CWModel, RefModel, SOModel

ORBIT = TgtOrbit(400000.0)
N = ORBIT.n


def cwclosedform(rel0, t):
    x0, y0, z0, u0, v0, w0 = rel0
    n = N
    c, s = np.cos(n * t), np.sin(n * t)
    x = (4 - 3 * c) * x0 + s / n * u0 + 2 / n * (1 - c) * v0
    y = 6 * (s - n * t) * x0 + y0 - 2 / n * (1 - c) * u0 + (4 * s - 3 * n * t) / n * v0
    z = c * z0 + s / n * w0
    xd = 3 * n * s * x0 + c * u0 + 2 * s * v0
    yd = -6 * n * (1 - c) * x0 - 2 * s * u0 + (4 * c - 3) * v0
    zd = -n * s * z0 + c * w0
    return np.array([x, y, z, xd, yd, zd])


def closedloopmat(wn_ratio):
    n = N
    wn = wn_ratio * n
    kp, kd = wn ** 2, 2 * wn
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = np.diag([3 * n * n, 0.0, -n * n]) - kp * np.eye(3)
    A[3:6, 3:6] = np.array([[0, 2 * n, 0], [-2 * n, 0, 0], [0, 0, 0]]) - kd * np.eye(3)
    return A


def max_rel_diff(eva, evb, t):
    ra = eva.model.relmany(eva.sol(t))[0]
    rb = evb.model.relmany(evb.sol(t))[0]
    return float(np.max(np.linalg.norm(ra - rb, axis=0)))


class _Ev:
    def __init__(self, model, run):
        self.model = model
        self.sol = run.sol
        self.run = run


@pytest.mark.parametrize("theta", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("scale", [1.0, 1.0e3, 5.0e4])
def testframeroundtrip(theta, scale):
    rt, vt = ORBIT.state(theta / N)
    rng = np.random.default_rng(int(theta * 10 + scale))
    for _ in range(5):
        rel0 = rng.normal(size=6) * np.array([scale] * 3 + [scale * 1e-3] * 3)
        rc, vc = relinert(rt, vt, rel0[:3], rel0[3:])
        rho, rhod = inertrel(rt, vt, rc, vc)
        assert np.allclose(rho, rel0[:3], rtol=0, atol=1e-8)
        assert np.allclose(rhod, rel0[3:], rtol=0, atol=1e-10)


def testlvlhframeprop():
    rt, vt = ORBIT.state(1234.5)
    C, w = dcmecilvlh(rt, vt)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(C) - 1.0) < 1e-14
    assert np.allclose(C @ rt, [ORBIT.R, 0, 0], rtol=1e-14, atol=1e-6)
    assert np.allclose(C @ vt, [0, N * ORBIT.R, 0], rtol=1e-14, atol=1e-9)
    assert abs(w - N) < 1e-15 * N * 10


def testrefscalarvec():
    model = RefModel(ORBIT, PDCtrl(N, 6.0, 0.05), [500, -700, 300, 0.4, -0.1, 0.2])
    rng = np.random.default_rng(3)
    Y = []
    for t in rng.uniform(0, 5000, size=6):
        rt, vt = ORBIT.state(t)
        rel = rng.normal(size=6) * np.array([1e3, 1e3, 1e3, 1, 1, 1])
        rc, vc = relinert(rt, vt, rel[:3], rel[3:])
        Y.append(np.concatenate([rt, vt, rc, vc]))
    Y = np.array(Y).T
    rho, rhod = model.relmany(Y)
    for k in range(Y.shape[1]):
        p = np.array(model.relscalar(Y[:, k]))
        assert np.allclose(p[:3], rho[:, k], rtol=0, atol=1e-9)
        assert np.allclose(p[3:], rhod[:, k], rtol=0, atol=1e-12)


def testtransptermmatch():
    r, v = ORBIT.initstate()
    rel0 = np.array([800.0, -1500.0, 400.0, 0.3, -0.4, 0.2])
    rc, vc = relinert(r, v, rel0[:3], rel0[3:])

    def twobody(t, y):
        out = np.empty(12)
        for k in (0, 6):
            r = y[k:k + 3]
            out[k:k + 3] = y[k + 3:k + 6]
            out[k + 3:k + 6] = -MU * r / np.linalg.norm(r) ** 3
        return out

    y0 = np.concatenate([r, v, rc, vc])
    atol = np.array([1e-7] * 3 + [1e-10] * 3 + [1e-7] * 3 + [1e-10] * 3)
    sol = solve_ivp(twobody, (0.0, 3000.0), y0, method="DOP853", rtol=1e-13,
                    atol=atol, dense_output=True)

    def rhoat(s):
        y = sol.sol(s)
        return inertrel(y[0:3], y[3:6], y[6:9], y[9:12])[0]

    dt = 0.5
    for t in (0.0 + 2 * dt, 500.0, 1234.5, 2500.0):
        fd = (-rhoat(t + 2 * dt) + 8 * rhoat(t + dt)
              - 8 * rhoat(t - dt) + rhoat(t - 2 * dt)) / (12 * dt)
        y = sol.sol(t)
        _, rhod = inertrel(y[0:3], y[3:6], y[6:9], y[9:12])
        assert np.max(np.abs(fd - rhod)) < 1e-6
        C, _ = dcmecilvlh(y[0:3], y[3:6])
        naive = C @ (y[9:12] - y[3:6])
        assert np.max(np.abs(naive - fd)) > 0.5


@pytest.mark.parametrize("solver,kw,tol_pos,tol_vel", [
    ("radau", {"rtol": 1e-11}, 1e-6, 1e-9),
    ("dop853", {"rtol": 1e-12}, 1e-6, 1e-9),
    ("rk4", {"h": 0.5}, 1e-6, 1e-9),
])
def testfreecwmatch(solver, kw, tol_pos, tol_vel):
    rel0 = np.array([100.0, 200.0, -50.0, 0.1, -0.2, 0.05])
    model = CWModel(ORBIT, PDCtrl(N, 0.0, None), rel0)
    T = ORBIT.period
    if solver == "rk4":
        run = integrk4(model, T, kw["h"])
    else:
        run = integadapt(model, solver, T, kw["rtol"])
    assert run.status == "completed" and run.t_end == T
    t = np.linspace(0.0, T, 401)
    num = run.sol(t)
    exact = np.array([cwclosedform(rel0, tk) for tk in t]).T
    assert np.max(np.abs(num[:3] - exact[:3])) < tol_pos
    assert np.max(np.abs(num[3:] - exact[3:])) < tol_vel


@pytest.mark.parametrize("a_max", [None, 10.0])
@pytest.mark.parametrize("solver,kw", [("radau", {"rtol": 1e-11}),
                                       ("dop853", {"rtol": 1e-12}),
                                       ("rk4", {"h": 0.5})])
def testcontrcwmatch(a_max, solver, kw):
    wnratio = 6.0
    rel0 = np.array([500.0, -800.0, 300.0, 0.2, 0.1, -0.3])
    model = CWModel(ORBIT, PDCtrl(N, wnratio, a_max), rel0)
    T = ORBIT.period
    if solver == "rk4":
        run = integrk4(model, T, kw["h"])
    else:
        run = integadapt(model, solver, T, kw["rtol"])
    assert run.status == "completed"
    assert run.switch_times == []
    A = closedloopmat(wnratio)
    t = np.linspace(0.0, T, 301)
    exact = np.array([expm(A * tk) @ rel0 for tk in t]).T
    num = run.sol(t)
    assert np.max(np.abs(num[:3] - exact[:3])) < 1e-6
    assert np.max(np.abs(num[3:] - exact[3:])) < 1e-8


def testsatdircap():
    amax = 0.05
    ctrl = PDCtrl(N, 6.0, amax)
    rng = np.random.default_rng(7)
    rho = rng.normal(size=(3, 500)) * rng.choice([1.0, 100.0, 5000.0], size=500)
    rhod = rng.normal(size=(3, 500)) * 0.5
    raw = ctrl.raw(rho, rhod)
    a = ctrl.saturate(raw)
    nraw = np.linalg.norm(raw, axis=0)
    napp = np.linalg.norm(a, axis=0)
    over = nraw > amax
    assert over.any() and (~over).any()
    assert np.all(napp <= amax * (1 + 1e-14))
    assert np.allclose(napp[over], amax, rtol=1e-14)
    assert np.array_equal(a[:, ~over], raw[:, ~over])
    cross = np.cross(raw.T, a.T)
    assert np.max(np.linalg.norm(cross, axis=1) / (nraw * napp)) < 1e-14
    assert np.all(np.sum(raw * a, axis=0) > 0)
    for k in range(50):
        s = ctrl.applscalar(*rho[:, k], *rhod[:, k])
        assert np.allclose(s, a[:, k], rtol=1e-14, atol=1e-18)
    ctrlfree = PDCtrl(N, 6.0, None)
    assert np.array_equal(ctrlfree.saturate(raw), raw)
    assert ctrl.kp == pytest.approx((6.0 * N) ** 2) and ctrl.kd == pytest.approx(12.0 * N)


def _both(gfun, T):
    prod = arrgrid(gfun, T, 0.5, 1e-6, locateall=False)
    fine = arrgrid(gfun, T, 0.01, 1e-9, locateall=True)
    return prod, fine


def testarrsingleentry():
    T = 2000.0
    te = 1000.123456

    def g(t):
        return (te - np.asarray(t, dtype=float)) / 100.0

    prod, fine = _both(g, T)
    for res, tol in ((prod, 2e-6), (fine, 1e-8)):
        assert res["inside_at_end"] and res["arrived"]
        assert abs(res["entry_time_s"] - te) < tol
        assert abs(res["dwell_s"] - (T - te)) < tol
        assert abs(res["dwell_margin_s"] - (T - te - 300.0)) < tol
        assert res["n_crossings"] == 1


def testarrinsidethr():
    T = 1500.0
    prod, fine = _both(lambda t: -0.5 + 0.0 * np.asarray(t, dtype=float), T)
    for res in (prod, fine):
        assert res["entry_time_s"] == 0.0
        assert res["dwell_s"] == T
        assert res["arrived"] and res["n_crossings"] == 0


def testarroutsend():
    T = 2000.0
    prod, fine = _both(lambda t: (np.asarray(t, dtype=float) - 1500.0) / 100.0, T)
    for res in (prod, fine):
        assert not res["inside_at_end"] and not res["arrived"]
        assert math.isnan(res["entry_time_s"]) and math.isnan(res["dwell_s"])
        assert res["n_crossings"] == 1


def testarrdwellthr():
    T = 2000.0
    for dwell, expected in ((299.9, False), (300.1, True)):
        te = T - dwell

        def g(t, te2=te):
            return (te2 - np.asarray(t, dtype=float)) / 100.0

        prod, fine = _both(g, T)
        assert prod["arrived"] is expected and fine["arrived"] is expected
        assert abs(fine["dwell_margin_s"] - (dwell - 300.0)) < 1e-8


def testarrexcbetw():
    T = 2200.0
    tc, width = 2000.25, 0.03

    def g(t):
        t = np.asarray(t, dtype=float)
        return np.maximum((1000.0 - t) / 100.0, -0.5) + 0.6 * np.exp(-((t - tc) / width) ** 2)

    half = width * math.sqrt(math.log(1.2))
    prod, fine = _both(g, T)
    assert prod["arrived"] and abs(prod["entry_time_s"] - 1000.0) < 2e-6
    assert prod["n_crossings"] == 1
    assert fine["n_crossings"] == 3 and fine["n_entries"] == 2 and fine["n_exits"] == 1
    assert abs(fine["entry_time_s"] - (tc + half)) < 1e-8
    assert not fine["arrived"]
    assert abs(fine["dwell_s"] - (T - tc - half)) < 1e-8


def testarrreqcompl():
    res = assessarr([], -1.0, lambda t: -1.0, 1000.0, False, 1e-6, False)
    assert not res["arrived"] and not res["inside_at_end"]
    assert math.isnan(res["entry_time_s"])


def testtimegridend():
    g = timegrid(10.3, 0.5)
    assert g[0] == 0.0 and g[-2] == 10.0 and g[-1] == 10.3 and g.size == 22
    g = timegrid(10.0 + 1e-10, 0.5)
    assert g.size == 21 and g[-1] == 10.0 + 1e-10 and g[-2] == 9.5
    g = timegrid(10.0, 0.5)
    assert g.size == 21 and g[-1] == 10.0
    assert np.all(np.diff(timegrid(5553.6, 0.01)) > 0)


def testsattimesign():
    assert sattime([10.0, 30.0], True, 100.0) == pytest.approx(10.0 + 70.0)
    assert sattime([10.0, 30.0], False, 100.0) == pytest.approx(20.0)
    assert sattime([], False, 100.0) == 0.0
    iv = signivals([0, 1, 2, 3], [1.0, -1.0, -2.0, 0.0])
    assert [(a, b) for a, b, _, _ in iv] == [(0.0, 1.0), (2.0, 3.0)]


def testrefinemaxfinds():
    t = np.arange(0.0, 10.0, 0.5)
    f = lambda s: 1.0 - (s - 3.3) ** 2
    v, tp = refinemax(f, t, f(t))
    assert abs(tp - 3.3) < 1e-5 and abs(v - 1.0) < 1e-10


def peakbranchdiff(rel0, branchb, wn_ratio=6.0, horizon=0.5):
    ctrl = PDCtrl(N, wn_ratio, None)
    T = horizon * ORBIT.period
    ref = RefModel(ORBIT, ctrl, rel0)
    other = (CWModel if branchb == "cw" else SOModel)(ORBIT, ctrl, rel0)
    runr = integadapt(ref, "dop853", T, 1e-13)
    runo = integadapt(other, "dop853", T, 1e-13)
    t = np.linspace(0.0, T, 3001)
    return max_rel_diff(_Ev(ref, runr), _Ev(other, runo), t)


def testrefcwsecond():
    d = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    d /= np.linalg.norm(d[:3])
    p1 = peakbranchdiff(400.0 * d, "cw")
    p2 = peakbranchdiff(200.0 * d, "cw")
    ratio = p1 / p2
    assert 3.8 < ratio < 4.2, (p1, p2, ratio)


def testrefquadthird():
    d = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    d /= np.linalg.norm(d[:3])
    p1 = peakbranchdiff(4000.0 * d, "so")
    p2 = peakbranchdiff(2000.0 * d, "so")
    ratio = p1 / p2
    assert 7.0 < ratio < 9.0, (p1, p2, ratio)


@pytest.mark.parametrize("method", ["Radau", "DOP853"])
def testadaptdrvrepr(method):
    rel0 = np.array([300.0, -500.0, 100.0, 0.1, 0.0, -0.05])
    model = CWModel(ORBIT, PDCtrl(N, 4.0, None), rel0)
    T, rtol = 2000.0, 1e-10
    run = integadapt(model, method.lower(), T, rtol, max_step=20.0)
    ref = solve_ivp(model.rhs, (0.0, T), model.y0, method=method, rtol=rtol,
                    atol=model.atol(rtol), max_step=20.0)
    assert run.n_steps == ref.t.size - 1
    assert np.array_equal(run.sol.knots, ref.t)
    scale = np.abs(ref.y).max(axis=1, keepdims=True)
    assert np.max(np.abs(run.sol(ref.t) - ref.y) / scale) < 1e-13


def testswitcheventmatch():
    amax, wnratio, z0 = 0.05, 6.0, -3000.0
    ctrl = PDCtrl(N, wnratio, amax)
    model = CWModel(ORBIT, ctrl, [0, 0, z0, 0, 0, 0])
    zc = amax / N ** 2

    def gsw(t):
        z = zc + (z0 - zc) * math.cos(N * t)
        zd = -N * (z0 - zc) * math.sin(N * t)
        return abs(ctrl.kp * z + ctrl.kd * zd) - amax

    texact = brentq(gsw, 1.0, 150.0, xtol=1e-14)
    for rtol, tol in ((1e-11, 1e-11), (1e-9, 1e-10)):
        run = integadapt(model, "radau", ORBIT.period, rtol)
        assert run.status == "completed" and len(run.switch_times) == 1
        assert run.restart_methods["refine"] == 1
        ts = run.switch_times[0]
        assert abs(ts - texact) < tol
        assert abs(model.gsat(ts, run.sol(ts))) < 1e-12
    runi = integadapt(model, "radau", ORBIT.period, 1e-11, switchrestart="interpolant")
    ev = lambda t, y: model.gsat(t, y)
    ev.terminal = True
    sol = solve_ivp(model.rhs, (0.0, 500.0), model.y0, method="Radau", rtol=1e-11,
                    atol=model.atol(1e-11), max_step=20.0, events=ev)
    assert abs(sol.t_events[0][0] - runi.switch_times[0]) < 1e-12


def testswitchrestsat():
    sc = {"sid": "sat", "rel0": [2000.0, -3000.0, 1000.0, 0.5, 0.0, -0.3],
          "wn_ratio": 6.0, "a_max": 0.03, "horizon_orbits": 0.3, "altitude": 400000.0}
    sr, _ = runcase(sc, {"solver": "radau", "rtol": 1e-11, "label": "r"}, branches=("cw", "so"))
    sk, _ = runcase(sc, {"solver": "rk4", "h": 0.25, "label": "k"}, branches=("cw", "so"))
    for b in ("cw", "so"):
        assert sr[b + "_n_switches"] >= 1
        assert sr[b + "_n_switches"] == sr[b + "_n_switches_fine"]
        assert not sr[b + "_events_disabled"]
        assert sr[b + "_n_switches"] == sk[b + "_n_switches"]
        assert np.allclose(sr[b + "_switch_times_s"], sk[b + "_switch_times_s"], atol=1e-3)
        assert abs(sr[b + "_sat_total_s"] - sk[b + "_sat_total_s"]) < 1e-3
        assert sr[b + "_peak_cmd_mps2"] == pytest.approx(0.03, rel=1e-12)
        assert sr[b + "_saturated_at_start"]


class SineSwitchModel:

    name = "toy"
    nstate = 1
    hasalt = False
    hassat = True
    y0 = np.array([0.0])

    def atol(self, rtol):
        return np.array([rtol])

    def rhs(self, t, y):
        return np.array([1.0])

    def gsat(self, t, y):
        return math.sin(float(y[0]))

    def badstate(self, y):
        return None


@pytest.mark.parametrize("method", ["radau", "dop853"])
def testswitchrestfind(method):
    run = integadapt(SineSwitchModel(), method, 100.0, 1e-10, max_step=1.0)
    expected = [k * math.pi for k in range(1, 32)]
    assert run.status == "completed" and not run.events_disabled
    assert len(run.switch_times) == len(expected)
    assert np.max(np.abs(np.array(run.switch_times) - expected)) < 1e-9


def testnoprogrguard(monkeypatch):
    import indverify.integrators as integ
    monkeypatch.setattr(integ, "NOADV", 1.0e3)
    run = integadapt(SineSwitchModel(), "radau", 100.0, 1e-10, max_step=1.0)
    assert run.events_disabled and "no progress" in run.events_disabled_reason
    assert len(run.switch_times) == integ.NOPROGRLIMIT
    assert run.status == "completed" and run.t_end == 100.0


def testrestcaprec():
    model = CWModel(ORBIT, PDCtrl(N, 6.0, 0.03), [2000.0, -3000.0, 1000.0, 0.5, 0.0, -0.3])
    run = integadapt(model, "radau", 0.3 * ORBIT.period, 1e-10, restcap=1)
    assert run.events_disabled and "cap" in run.events_disabled_reason
    assert len(run.switch_times) == 1 and run.status == "completed"


RND33 = dict(rel0=[-103.7, 253.1, -582.1, -0.106, -0.023, -0.042], wn_ratio=8.0,
             a_max=0.005169, horizon_orbits=1, altitude=400000.0)


def frozentruth(model, T, rtol=1e-13, max_step=2.0):
    t, y = 0.0, np.array(model.y0, dtype=float)
    side = 1 if model.gsat(0.0, y) > 0 else -1
    segs, switches = [], []
    while T - t > 1e-12 * T:
        fm = model.frozen(side > 0)

        def ev(tt, yy, fm=fm):
            return fm.gsat(tt, yy)
        ev.terminal = True
        ev.direction = -1.0 if side > 0 else 1.0
        r = solve_ivp(fm.rhs, (t, T), y, method="DOP853", rtol=rtol, atol=model.atol(rtol),
                      max_step=max_step, events=ev, dense_output=True)
        segs.append((float(r.t[-1]), r.sol))
        if r.status != 1:
            break
        te = float(r.t_events[0][0])
        k = int(np.searchsorted(r.t, te)) - 1
        r2 = solve_ivp(fm.rhs, (float(r.t[k]), te), r.y[:, k], method="DOP853", rtol=rtol,
                       atol=model.atol(rtol), max_step=max_step)
        t, y, side = te, r2.y[:, -1], -side
        switches.append(te)

    def sol(tt):
        out = np.empty((model.nstate, tt.size))
        ends = np.array([e for e, _ in segs])
        idx = np.minimum(np.searchsorted(ends, tt, side="left"), len(segs) - 1)
        for i in np.unique(idx):
            out[:, idx == i] = segs[i][1](tt[idx == i])
        return out
    return sol, switches


def testfrozenmodectrl():
    ctrl = PDCtrl(N, 6.0, 0.05)
    sat, uns = ctrl.frozen(True), ctrl.frozen(False)
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = rng.normal(size=6) * np.array([300, 300, 300, 0.3, 0.3, 0.3])
        raw = np.array(ctrl.rawscalar(*p))
        nr = np.linalg.norm(raw)
        true = np.array(ctrl.applscalar(*p))
        assert np.allclose(sat.applscalar(*p) if nr > 0.05 else uns.applscalar(*p),
                           true, rtol=1e-15, atol=0)
        assert np.allclose(np.linalg.norm(sat.applscalar(*p)), 0.05, rtol=1e-14)
        assert np.allclose(uns.applscalar(*p), raw, rtol=0, atol=0)
    a = rng.normal(size=(3, 50))
    assert np.allclose(np.linalg.norm(sat.saturate(a), axis=0), 0.05, rtol=1e-14)
    assert np.array_equal(uns.saturate(a), a)
    model = RefModel(ORBIT, ctrl, [500.0, 0, 0, 0, 0, 0])
    fm = model.frozen(True)
    assert fm.ctrl.saturated and model.ctrl is ctrl and fm.y0 is model.y0


def testswitchrestregr():
    ctrl = PDCtrl(N, RND33["wn_ratio"], RND33["a_max"])
    T = ORBIT.period
    tg = np.arange(0.0, T, 0.5)
    for branch, method, rtol, tolpos, tolsw in (("ref", "radau", 1e-8, 1.5e-3, 1e-3),
                                                ("ref", "radau", 1e-11, 5e-6, 2e-6),
                                                ("ref", "dop853", 1e-12, 5e-7, 1e-6),
                                                ("cw", "radau", 1e-12, 3e-10, 1e-9)):
        model = (RefModel if branch == "ref" else CWModel)(ORBIT, ctrl, RND33["rel0"])
        tsol, tsw = frozentruth(model, T)
        assert len(tsw) == 3
        rhot = model.relmany(tsol(tg))[0]
        errs = {}
        for how in ("refine", "interpolant"):
            run = integadapt(model, method, T, rtol, switchrestart=how)
            assert run.status == "completed" and len(run.switch_times) == 3
            rho = model.relmany(run.sol(tg))[0]
            errs[how] = np.max(np.linalg.norm(rho - rhot, axis=0))
            if how == "refine":
                assert run.restart_methods == {"refine": 3, "reintegrate": 0, "interpolant": 0}
                assert np.max(np.abs(np.array(run.switch_times) - tsw)) < tolsw, (branch, rtol)
        assert errs["refine"] < tolpos, (branch, method, rtol, errs)
        assert errs["interpolant"] > 5.0 * errs["refine"], (branch, method, rtol, errs)


def testswitchcountsmatch():
    s, _ = runcase(dict(RND33, sid="rnd33"), {"solver": "radau", "rtol": 1e-8, "label": "r8"})
    for b in ("ref", "cw", "so"):
        assert s[b + "_n_switches"] == s[b + "_n_switches_fine"] == 3, b
        assert s[b + "_n_restart_fallback"] == 0 and not s[b + "_events_disabled"]
        assert np.all(np.diff(s[b + "_switch_times_s"]) > 1.0)


def testpolishputsrest():
    from scipy.integrate import Radau
    from indverify.integrators import _polish, switchrest
    toy = SineSwitchModel()
    rhs, gfun = toy.rhs, toy.gsat
    pieces, t, y, _, _, ok = _polish(Radau, rhs, gfun, 3.0, np.array([3.0]), 3.5, 1,
                                     1e-10, np.array([1e-10]), 1.0)
    assert ok and math.pi < t < math.pi + 0.3 * (math.pi - 3.0) and gfun(t, y) < 0
    assert abs(y[0] - t) < 1e-9 and pieces and pieces[-1][0] == t
    for t0 in (3.2, math.pi - 1e-12):
        pieces, t, y, _, _, ok = _polish(Radau, rhs, gfun, t0, np.array([t0]), 3.5, 1,
                                         1e-10, np.array([1e-10]), 1.0)
        assert ok and t == t0 and not pieces
    pieces, tr, yroot, _, _, used = switchrest(toy, Radau, 3.0, np.array([3.0]), 3.5, 3.1,
                                               1, math.sin(3.0), 1e-10,
                                               np.array([1e-10]), 1.0)
    assert used == "reintegrate" and math.pi < tr < math.pi + 0.3 * (math.pi - 3.1)
    assert abs(yroot[0] - tr) < 1e-9 and pieces[-1][0] == tr


def testherminterpexact():
    T = np.array([0.0, 0.7, 2.0, 2.5])
    p = lambda t: 1.5 - 2.0 * t + 0.3 * t ** 2 - 0.7 * t ** 3
    dp = lambda t: -2.0 + 0.6 * t - 2.1 * t ** 2
    H = HermDense(T, p(T)[None, :], dp(T)[None, :])
    tt = np.linspace(0.0, 2.5, 97)
    assert np.allclose(H(tt)[0], p(tt), rtol=0, atol=1e-13)
    assert abs(H(1.3)[0] - p(1.3)) < 1e-13


def testrk4fourthorder():
    rel0 = np.array([100.0, 200.0, -50.0, 0.1, -0.2, 0.05])
    model = CWModel(ORBIT, PDCtrl(N, 20.0, None), rel0)
    T = 600.0
    A = closedloopmat(20.0)
    exact = expm(A * T) @ rel0
    errs = [np.max(np.abs(integrk4(model, T, h).sol(T) - exact)) for h in (4.0, 2.0)]
    assert 13.0 < errs[0] / errs[1] < 19.0


@pytest.mark.parametrize("setting", [{"solver": "radau", "rtol": 1e-10, "label": "r"},
                                     {"solver": "rk4", "h": 0.5, "label": "k"}])
def testquadblowup(setting):
    sc = {"sid": "blow", "rel0": [-2.0e6, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
          "horizon_orbits": 1, "altitude": 400000.0}
    s, h = runcase(sc, setting, branches=("cw", "so"), keep_history=True)
    T = s["horizon_s"]
    assert s["cw_status"] == "completed"
    assert s["so_status"] == "numerical_failure"
    assert 0.0 < s["so_t_end_s"] < T
    assert s["so_reason"]
    for c in ("prod", "fine"):
        assert s["so_%s_arrived" % c] is False
        assert s["so_%s_inside_at_end" % c] is False
        assert math.isnan(s["so_%s_entry_time_s" % c])
    assert s["cw_vs_so_common_end_s"] == pytest.approx(s["so_t_end_s"])
    assert h["so_t"][-1] == pytest.approx(s["so_t_end_s"])
    assert h["so_t"].size < h["t"].size


@pytest.mark.parametrize("setting", [{"solver": "radau", "rtol": 1e-10, "label": "r"},
                                     {"solver": "dop853", "rtol": 1e-11, "label": "d"},
                                     {"solver": "rk4", "h": 0.5, "label": "k"}])
def testaltfloordomain(setting):
    sc = {"sid": "low", "rel0": [-250e3, 0, 0, -150.0, 0, 0], "wn_ratio": 6.0, "a_max": 0.01,
          "horizon_orbits": 1, "altitude": 400000.0}
    s, _ = runcase(sc, setting, branches=("ref",))
    assert s["ref_status"] == "domain_exit"
    texit = s["ref_t_end_s"]
    assert 0.0 < texit < s["horizon_s"]
    assert not s["ref_prod_arrived"] and not s["ref_fine_arrived"]
    model = RefModel(ORBIT, PDCtrl(N, 6.0, 0.01), sc["rel0"])
    from indverify.runner import SolverSet, integbranch
    run = integbranch(model, SolverSet.from_dict(setting), s["horizon_s"])
    assert abs(model.alt(texit, run.sol(texit))) < 1e-3
    assert abs(run.t_end - texit) < 1e-9


def testaltexittimes():
    sc = {"sid": "low", "rel0": [-250e3, 0, 0, -150.0, 0, 0], "wn_ratio": 6.0, "a_max": 0.01,
          "horizon_orbits": 1, "altitude": 400000.0}
    t = [runcase(sc, st, branches=("ref",))[0]["ref_t_end_s"]
         for st in ({"solver": "radau", "rtol": 1e-11}, {"solver": "rk4", "h": 0.25})]
    assert abs(t[0] - t[1]) < 1e-4


BRANCHKEYS = ["status", "reason", "t_end_s", "n_switches", "events_disabled", "sat_total_s",
              "final_range_m", "final_speed_mps", "peak_cmd_mps2"]
CONVKEYS = ["inside_at_end", "entry_time_s", "dwell_s", "dwell_margin_s", "arrived",
            "n_crossings"]
PAIRKEYS = ["peak_pos_m", "t_peak_s", "peak_vel_mps", "max_cmd_diff_mps2", "common_end_s",
            "peak_pos_fine_m"]


def testruncaseint():
    sc = {"sid": "pR1000", "rel0": [1000.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
          "horizon_orbits": 1, "altitude": 400000.0}
    s1, h1 = runcase(sc, {"solver": "dop853", "rtol": 1e-12, "label": "d12"}, keep_history=True)
    s2, h2 = runcase(sc, {"solver": "rk4", "h": 0.5, "label": "k05"})
    assert h2 is None
    for k in ("sid", "solver", "setting_label", "rtol", "h", "max_step", "wall_s"):
        assert k in s1
    assert s1["sid"] == "pR1000" and s1["solver"] == "dop853" and s1["setting_label"] == "d12"
    assert math.isnan(s1["h"]) and math.isnan(s2["rtol"]) and math.isnan(s2["max_step"])
    assert s1["max_step"] == 20.0 and s2["h"] == 0.5
    for b in ("ref", "cw", "so"):
        for k in BRANCHKEYS:
            assert b + "_" + k in s1
        for c in ("prod", "fine"):
            for k in CONVKEYS:
                assert "%s_%s_%s" % (b, c, k) in s1
        assert s1[b + "_status"] == "completed"
        assert isinstance(s1[b + "_n_switches"], int)
        assert isinstance(s1[b + "_events_disabled"], bool)
        assert isinstance(s1[b + "_prod_arrived"], bool)
        assert isinstance(s1[b + "_fine_n_crossings"], int)
        assert isinstance(s1[b + "_t_end_s"], float)
        assert s1[b + "_prod_arrived"] and s1[b + "_fine_arrived"]
        assert s1[b + "_prod_entry_time_s"] < s1["T_orb_s"]
        assert abs(s1[b + "_prod_entry_time_s"] - s1[b + "_fine_entry_time_s"]) < 2e-6
        assert abs(s1[b + "_fine_entry_time_s"] - s2[b + "_fine_entry_time_s"]) < 1e-4
        n = h1["t"].size
        assert h1[b + "_t"].size == n
        for k, shape in (("rho", (n, 3)), ("rhodot", (n, 3)), ("acc", (n, 3)), ("accnorm", (n,))):
            assert h1[b + "_" + k].shape == shape
        assert np.allclose(np.linalg.norm(h1[b + "_acc"], axis=1), h1[b + "_accnorm"])
        assert np.allclose(h1[b + "_rho"][0], sc["rel0"][:3], atol=1e-8)
    for a, b in (("cw", "ref"), ("so", "ref"), ("cw", "so")):
        for k in PAIRKEYS:
            assert "%s_vs_%s_%s" % (a, b, k) in s1
        assert s1["%s_vs_%s_common_end_s" % (a, b)] == s1["horizon_s"]
    assert 1e-3 < s1["cw_vs_ref_peak_pos_m"] < 2e-2
    assert abs(s1["cw_vs_ref_peak_pos_m"] - s2["cw_vs_ref_peak_pos_m"]) < 1e-7
    assert abs(s1["cw_vs_ref_peak_pos_m"] - s1["cw_vs_ref_peak_pos_fine_m"]) < 1e-8
    assert s1["so_vs_ref_peak_pos_m"] < 1e-2 * s1["cw_vs_ref_peak_pos_m"]
    assert json.loads(json.dumps(s1))["sid"] == "pR1000"


def testbranchsubsetval():
    sc = {"sid": "x", "rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": None,
          "horizon_orbits": 0.1, "altitude": 400000.0}
    s, _ = runcase(sc, {"solver": "dop853", "rtol": 1e-10}, branches=("cw",))
    assert "cw_status" in s and "ref_status" not in s and not any("_vs_" in k for k in s)
    assert s["cw_sat_total_s"] == 0.0 and s["cw_n_switches"] == 0
    with pytest.raises(ValueError):
        runcase(sc, {"solver": "euler", "h": 1.0})
    with pytest.raises(ValueError):
        runcase(sc, {"solver": "rk4", "h": 1.0}, branches=("cw", "bogus"))


def testcliwritesres(tmp_path):
    cas = [
        {"sid": "c1", "rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
         "horizon_orbits": 0.05, "altitude": 400000.0},
        {"sid": "c2", "rel0": [0, 0, -300.0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.005,
         "horizon_orbits": 0.05, "altitude": 400000.0},
        {"sid": "bad", "wn_ratio": 6.0, "a_max": 0.05, "horizon_orbits": 0.05},
    ]
    settings = [{"solver": "dop853", "rtol": 1e-10, "label": "d10"},
                {"solver": "rk4", "h": 1.0, "label": "k1"}]
    cpath, spath, out = tmp_path / "cases.json", tmp_path / "settings.json", tmp_path / "out"
    cpath.write_text(json.dumps(cas))
    spath.write_text(json.dumps(settings))
    cmd = [sys.executable, "-m", "indverify", "--cases", str(cpath), "--settings", str(spath),
           "--out", str(out), "--workers", "2", "--history-setting", "d10"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (out / "indverifyresults.jsonl").read_text().splitlines()]
    assert len(lines) == 6
    pairs = {(x["sid"], x["setting_label"]) for x in lines}
    assert len(pairs) == 6
    errs = [x for x in lines if "error" in x]
    assert len(errs) == 2 and all(x["sid"] == "bad" for x in errs)
    assert all("Traceback" in x["error"] for x in errs)
    ok = [x for x in lines if "error" not in x]
    assert all(x["cw_status"] == "completed" for x in ok)
    for sid in ("c1", "c2"):
        with np.load(out / "histories" / ("%s.npz" % sid)) as z:
            assert {"t", "ref_t", "ref_rho", "cw_rhodot", "so_acc", "so_accnorm"} <= set(z.files)
    assert not (out / "histories" / "bad.npz").exists()
    r2 = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr
    assert "2 task(s) to run, 4 already present" in r2.stdout
    lines2 = [json.loads(x) for x in (out / "indverifyresults.jsonl").read_text().splitlines()]
    assert len(lines2) == 6 and len({(x["sid"], x["setting_label"]) for x in lines2}) == 6
    assert len((out / "indverify_results.bad").read_text().splitlines()) == 2
    r3 = subprocess.run(cmd + ["--no-retry-errors"], cwd=str(ROOT), env=env, capture_output=True,
                        text=True, timeout=600)
    assert r3.returncode == 0, r3.stderr
    assert "0 task(s) to run" in r3.stdout
    assert len((out / "indverifyresults.jsonl").read_text().splitlines()) == 6


def testclimalfcases(tmp_path):
    good = {"rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
            "horizon_orbits": 0.02, "altitude": 400000.0}
    cas = [dict(good, sid="x/y"), dict(good, sid="x_y"), dict(good),
           "not a scenario"]
    settings = [{"solver": "dop853", "rtol": 1e-9, "label": "d9"},
                {"solver": "euler", "h": 1.0, "label": "eul"},
                {"solver": "rk4"}]
    cpath, spath, out = tmp_path / "cases.json", tmp_path / "settings.json", tmp_path / "out"
    cpath.write_text(json.dumps(cas))
    spath.write_text(json.dumps(settings))
    cmd = [sys.executable, "-m", "indverify", "--cases", str(cpath), "--settings", str(spath),
           "--out", str(out), "--workers", "2", "--branches", "cw", "--history-setting", "d9"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (out / "indverifyresults.jsonl").read_text().splitlines()]
    pairs = {(x["sid"], x["setting_label"]) for x in lines}
    assert pairs == {(s, l) for s in ("x/y", "x_y", "#2", "#3") for l in ("d9", "eul", "#setting2")}
    ok = {(x["sid"], x["setting_label"]) for x in lines if "error" not in x}
    assert ok == {("x/y", "d9"), ("x_y", "d9")}
    assert all("Traceback" in x["error"] for x in lines if "error" in x)
    hist = sorted(p.name for p in (out / "histories").iterdir())
    assert len(hist) == 2 and "x_y.npz" not in hist
    from indverify.cli import histfnames
    names = histfnames(["x/y", "x_y", "c1"])
    assert names["c1"] == "c1.npz" and set(hist) == {names["x/y"], names["x_y"]}
    ci = histfnames(["Case", "case", "other"])
    assert len({v.lower() for v in ci.values()}) == 3 and ci["other"] == "other.npz"
    cpath.write_text(json.dumps([dict(good, sid="d"), dict(good, sid="d")]))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "unique" in r.stderr


def testrescompresume(tmp_path):
    from indverify.cli import compactres
    path, bad = str(tmp_path / "res.jsonl"), str(tmp_path / "res.bad")
    rec = lambda sid, lab, **kw: json.dumps(dict(sid=sid, setting_label=lab, **kw))
    lines = [rec("A", "s", v=1),
             rec("B", "s", error="Traceback: worker crashed"),
             rec("C", "s", error="Traceback: BrokenProcessPool"),
             rec("C", "s", v=3),
             rec("A", "s", v=2),
             rec("D", "s", v=4)[:25]]
    pathlib.Path(path).write_text("\n".join(lines))
    keeperr = str(tmp_path / "keep.jsonl")
    pathlib.Path(keeperr).write_text("\n".join(lines))
    done, st = compactres(path, bad, retry_errors=True)
    assert done == {("A", "s"), ("C", "s")}
    kept = [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]
    assert [(r["sid"], r["v"]) for r in kept] == [("C", 3), ("A", 2)]
    moved = pathlib.Path(bad).read_text().splitlines()
    assert len(moved) == 4 and moved[0] == lines[-1]
    assert st["unparseable"] == 1 and st["superseded"] == 2 and st["errors_to_retry"] == 1
    done2, st2 = compactres(path, bad, retry_errors=True)
    assert done2 == done and not st2["rewritten"]
    done3, st3 = compactres(keeperr, str(tmp_path / "keep.bad"), retry_errors=False)
    assert done3 == {("A", "s"), ("B", "s"), ("C", "s")} and st3["errors_kept"] == 1


def testloadreskeeps(tmp_path):
    from indverify.convergence import refderived, ladderchang, loadres
    path = tmp_path / "res.jsonl"
    base = dict(sid="s", solver="radau", cw_vs_ref_peak_pos_m=0.0)
    path.write_text("\n".join([
        json.dumps(dict(base, setting_label="r8", rtol=1e-8, cw_vs_ref_peak_pos_m=1.0)),
        json.dumps(dict(sid="s", setting_label="r10", error="Traceback: crashed")),
        json.dumps(dict(base, setting_label="r10", rtol=1e-10, cw_vs_ref_peak_pos_m=2.0)),
        json.dumps(dict(base, setting_label="r8", rtol=1e-8, cw_vs_ref_peak_pos_m=1.5)),
        json.dumps(dict(sid="t", setting_label="r8", error="Traceback: crashed")),
        '{"sid": "s", "setting_lab',
    ]) + "\n")
    st = {}
    recs = loadres(str(path), st)
    assert [(r["setting_label"], r["cw_vs_ref_peak_pos_m"]) for r in recs] == [("r8", 1.5), ("r10", 2.0)]
    assert st == {"lines": 6, "unparseable": 1, "dropped": 3, "error_only_pairs": 1}
    rows = ladderchang(recs)
    row = [r for r in rows if r["quantity"] == "cw_vs_ref_peak_pos_m"][0]
    assert row["change"] == pytest.approx(0.5) and row["ref_derived"] is True
    assert refderived("ref_prod_entry_time_s") and not refderived("cw_vs_so_peak_pos_m")


def testrefrtoltight():
    sc = dict(RND33, sid="rnd33", horizon_orbits=0.3)
    smix, _ = runcase(sc, {"solver": "radau", "rtol": 1e-8, "ref_rtol": 1e-11},
                      branches=("ref", "cw"))
    s, _ = runcase(sc, {"solver": "radau", "rtol": 1e-8}, branches=("ref", "cw"))
    s11, _ = runcase(sc, {"solver": "radau", "rtol": 1e-11}, branches=("ref", "cw"))
    assert smix["setting_label"] == "radau_1e-08_ref1e-11"
    assert smix["rtol"] == 1e-8 and smix["ref_rtol"] == 1e-11 and s["ref_rtol"] == 1e-8
    for k in ("_t_end_s", "_final_range_m", "_prod_entry_time_s", "_n_steps"):
        assert smix["ref" + k] == s11["ref" + k] or (math.isnan(smix["ref" + k])
                                                      and math.isnan(s11["ref" + k]))
        assert smix["cw" + k] == s["cw" + k] or (math.isnan(smix["cw" + k])
                                                 and math.isnan(s["cw" + k]))
    assert smix["ref_switch_times_s"] == s11["ref_switch_times_s"]
    with pytest.raises(ValueError):
        runcase(sc, {"solver": "rk4", "h": 1.0, "ref_rtol": 1e-11}, branches=("cw",))


def testconvladderrep():
    from indverify.convergence import changref, ladderchang, summar
    base = {"sid": "s", "cw_status": "completed", "cw_prod_arrived": True,
            "cw_switch_times_s": [10.0, 20.0], "cw_vs_ref_peak_pos_m": 0.01,
            "cw_prod_entry_time_s": float("nan"), "wall_s": 1.0}
    recs = []
    for label, rtol, d in (("r8", 1e-8, 1e-5), ("r10", 1e-10, 1e-7), ("r12", 1e-12, 0.0)):
        r = dict(base, solver="radau", rtol=rtol, setting_label=label,
                 cw_vs_ref_peak_pos_m=0.01 + d, cw_switch_times_s=[10.0 + d, 20.0])
        recs.append(r)
    rows = ladderchang(list(reversed(recs)))
    peak = [r for r in rows if r["quantity"] == "cw_vs_ref_peak_pos_m"]
    assert [(r["from_label"], r["to_label"]) for r in peak] == [("r8", "r10"), ("r10", "r12")]
    assert peak[0]["abs_change"] == pytest.approx(1e-5 - 1e-7)
    assert not any(r["quantity"] == "wall_s" for r in rows)
    sw = [r for r in rows if r["quantity"] == "cw_switch_times_s"]
    assert sw[1]["abs_change"] == pytest.approx(1e-7)
    nanrow = [r for r in rows if r["quantity"] == "cw_prod_entry_time_s"][0]
    assert nanrow["change"] == 0.0 and nanrow["note"] == "both NaN"
    vs = changref(recs, "r12")
    assert {r["from_label"] for r in vs} == {"r12"} and {r["to_label"] for r in vs} == {"r8", "r10"}
    summ = summar(rows)
    assert any(x["quantity"] == "cw_vs_ref_peak_pos_m" for x in summ)


FORB = ("cwvalid", "src", "experiments")


def testindnoprod():
    pkg = pathlib.Path(indverify.__file__).resolve().parent
    files = sorted(pkg.glob("*.py"))
    assert len(files) >= 5
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 1:
                    pytest.fail("%s: relative import escapes the package" % f.name)
                if node.level == 0:
                    mods = [node.module or ""]
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name in ("__import__", "import_module") and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        mods = [arg.value]
            for m in mods:
                top = m.split(".")[0]
                assert top not in FORB, "%s imports %s" % (f.name, m)
                allowed = top in {"indverify", "numpy", "scipy"} or top in sys.stdlib_module_names
                assert allowed, "%s imports non-allowed module %s" % (f.name, m)


def testmet():
    assert IMPL == "indverify/SPEC.md"
    assert isinstance(__version__, str) and __version__
