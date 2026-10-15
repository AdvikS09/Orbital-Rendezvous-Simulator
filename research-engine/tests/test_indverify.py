"""Unit tests for the independent closed-loop verifier (indverify/SPEC.md).

Covers every self-check of SPEC section 10 plus checks of the integrators,
event handling, statuses, the run_case interface and the command line.
"""

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

import indverify  # noqa: E402
from indverify import IMPLEMENTS, __version__, run_case  # noqa: E402
from indverify.assess import (arrival_on_grid, assess_arrival, refine_max,  # noqa: E402
                              saturated_time, sign_intervals, time_grid)
from indverify.constants import MU, R_E  # noqa: E402
from indverify.control import PDController  # noqa: E402
from indverify.frames import (TargetOrbit, dcm_eci_to_lvlh, inertial_to_rel,  # noqa: E402
                              rel_to_inertial)
from indverify.integrators import (HermiteDense, integrate_adaptive,  # noqa: E402
                                   integrate_rk4)
from indverify.models import CWModel, RefModel, SOModel  # noqa: E402

ORBIT = TargetOrbit(400000.0)
N = ORBIT.n


def cw_closed_form(rel0, t):
    """Closed-form solution of the uncontrolled CW equations."""
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


def closed_loop_matrix(wn_ratio):
    n = N
    wn = wn_ratio * n
    kp, kd = wn ** 2, 2 * wn
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = np.diag([3 * n * n, 0.0, -n * n]) - kp * np.eye(3)
    A[3:6, 3:6] = np.array([[0, 2 * n, 0], [-2 * n, 0, 0], [0, 0, 0]]) - kd * np.eye(3)
    return A


def max_rel_diff(ev_a, ev_b, t):
    ra = ev_a.model.rel_many(ev_a.sol(t))[0]
    rb = ev_b.model.rel_many(ev_b.sol(t))[0]
    return float(np.max(np.linalg.norm(ra - rb, axis=0)))


class _Ev:
    def __init__(self, model, run):
        self.model = model
        self.sol = run.sol
        self.run = run


# ---------------------------------------------------------------------------
# SPEC 10: frame round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("scale", [1.0, 1.0e3, 5.0e4])
def test_frame_round_trip(theta, scale):
    r_t, v_t = ORBIT.state(theta / N)
    rng = np.random.default_rng(int(theta * 10 + scale))
    for _ in range(5):
        rel0 = rng.normal(size=6) * np.array([scale] * 3 + [scale * 1e-3] * 3)
        r_c, v_c = rel_to_inertial(r_t, v_t, rel0[:3], rel0[3:])
        rho, rhod = inertial_to_rel(r_t, v_t, r_c, v_c)
        assert np.allclose(rho, rel0[:3], rtol=0, atol=1e-8)
        assert np.allclose(rhod, rel0[3:], rtol=0, atol=1e-10)


def test_lvlh_frame_properties():
    r_t, v_t = ORBIT.state(1234.5)
    C, w = dcm_eci_to_lvlh(r_t, v_t)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(C) - 1.0) < 1e-14
    assert np.allclose(C @ r_t, [ORBIT.R, 0, 0], rtol=1e-14, atol=1e-6)
    # circular orbit: velocity is purely along-track and w = n
    assert np.allclose(C @ v_t, [0, N * ORBIT.R, 0], rtol=1e-14, atol=1e-9)
    assert abs(w - N) < 1e-15 * N * 10


def test_reference_scalar_and_vector_relative_state_agree():
    model = RefModel(ORBIT, PDController(N, 6.0, 0.05), [500, -700, 300, 0.4, -0.1, 0.2])
    rng = np.random.default_rng(3)
    Y = []
    for t in rng.uniform(0, 5000, size=6):
        r_t, v_t = ORBIT.state(t)
        rel = rng.normal(size=6) * np.array([1e3, 1e3, 1e3, 1, 1, 1])
        r_c, v_c = rel_to_inertial(r_t, v_t, rel[:3], rel[3:])
        Y.append(np.concatenate([r_t, v_t, r_c, v_c]))
    Y = np.array(Y).T
    rho, rhod = model.rel_many(Y)
    for k in range(Y.shape[1]):
        p = np.array(model.rel_scalar(Y[:, k]))
        assert np.allclose(p[:3], rho[:, k], rtol=0, atol=1e-9)
        assert np.allclose(p[3:], rhod[:, k], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# SPEC 10: transport term vs finite difference along two-body propagation
# ---------------------------------------------------------------------------
def test_transport_term_matches_finite_difference():
    r_t0, v_t0 = ORBIT.initial_state()
    rel0 = np.array([800.0, -1500.0, 400.0, 0.3, -0.4, 0.2])
    r_c0, v_c0 = rel_to_inertial(r_t0, v_t0, rel0[:3], rel0[3:])

    def two_body(t, y):
        out = np.empty(12)
        for k in (0, 6):
            r = y[k:k + 3]
            out[k:k + 3] = y[k + 3:k + 6]
            out[k + 3:k + 6] = -MU * r / np.linalg.norm(r) ** 3
        return out

    y0 = np.concatenate([r_t0, v_t0, r_c0, v_c0])
    atol = np.array([1e-7] * 3 + [1e-10] * 3 + [1e-7] * 3 + [1e-10] * 3)
    sol = solve_ivp(two_body, (0.0, 3000.0), y0, method="DOP853", rtol=1e-13,
                    atol=atol, dense_output=True)

    def rho_at(s):
        y = sol.sol(s)
        return inertial_to_rel(y[0:3], y[3:6], y[6:9], y[9:12])[0]

    dt = 0.5
    for t in (0.0 + 2 * dt, 500.0, 1234.5, 2500.0):
        fd = (-rho_at(t + 2 * dt) + 8 * rho_at(t + dt)
              - 8 * rho_at(t - dt) + rho_at(t - 2 * dt)) / (12 * dt)
        y = sol.sol(t)
        _, rhod = inertial_to_rel(y[0:3], y[3:6], y[6:9], y[9:12])
        assert np.max(np.abs(fd - rhod)) < 1e-6
        # without the transport term the mismatch is O(n |rho|) ~ 1 m/s
        C, _ = dcm_eci_to_lvlh(y[0:3], y[3:6])
        naive = C @ (y[9:12] - y[3:6])
        assert np.max(np.abs(naive - fd)) > 0.5


# ---------------------------------------------------------------------------
# SPEC 10: uncontrolled CW vs closed form over one orbit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver,kw,tol_pos,tol_vel", [
    ("radau", {"rtol": 1e-11}, 1e-6, 1e-9),
    ("dop853", {"rtol": 1e-12}, 1e-6, 1e-9),
    ("rk4", {"h": 0.5}, 1e-6, 1e-9),
])
def test_uncontrolled_cw_matches_closed_form(solver, kw, tol_pos, tol_vel):
    rel0 = np.array([100.0, 200.0, -50.0, 0.1, -0.2, 0.05])
    model = CWModel(ORBIT, PDController(N, 0.0, None), rel0)
    T = ORBIT.period
    if solver == "rk4":
        run = integrate_rk4(model, T, kw["h"])
    else:
        run = integrate_adaptive(model, solver, T, kw["rtol"])
    assert run.status == "completed" and run.t_end == T
    t = np.linspace(0.0, T, 401)
    num = run.sol(t)
    exact = np.array([cw_closed_form(rel0, tk) for tk in t]).T
    assert np.max(np.abs(num[:3] - exact[:3])) < tol_pos
    assert np.max(np.abs(num[3:] - exact[3:])) < tol_vel


# ---------------------------------------------------------------------------
# SPEC 10: unsaturated controlled CW vs matrix exponential
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a_max", [None, 10.0])
@pytest.mark.parametrize("solver,kw", [("radau", {"rtol": 1e-11}),
                                       ("dop853", {"rtol": 1e-12}),
                                       ("rk4", {"h": 0.5})])
def test_controlled_cw_matches_matrix_exponential(a_max, solver, kw):
    wn_ratio = 6.0
    rel0 = np.array([500.0, -800.0, 300.0, 0.2, 0.1, -0.3])
    model = CWModel(ORBIT, PDController(N, wn_ratio, a_max), rel0)
    T = ORBIT.period
    if solver == "rk4":
        run = integrate_rk4(model, T, kw["h"])
    else:
        run = integrate_adaptive(model, solver, T, kw["rtol"])
    assert run.status == "completed"
    assert run.switch_times == []
    A = closed_loop_matrix(wn_ratio)
    t = np.linspace(0.0, T, 301)
    exact = np.array([expm(A * tk) @ rel0 for tk in t]).T
    num = run.sol(t)
    assert np.max(np.abs(num[:3] - exact[:3])) < 1e-6
    assert np.max(np.abs(num[3:] - exact[3:])) < 1e-8


# ---------------------------------------------------------------------------
# SPEC 10: saturation preserves direction and caps magnitude
# ---------------------------------------------------------------------------
def test_saturation_direction_and_cap():
    a_max = 0.05
    ctrl = PDController(N, 6.0, a_max)
    rng = np.random.default_rng(7)
    rho = rng.normal(size=(3, 500)) * rng.choice([1.0, 100.0, 5000.0], size=500)
    rhod = rng.normal(size=(3, 500)) * 0.5
    raw = ctrl.raw(rho, rhod)
    app = ctrl.saturate(raw)
    nraw = np.linalg.norm(raw, axis=0)
    napp = np.linalg.norm(app, axis=0)
    over = nraw > a_max
    assert over.any() and (~over).any()
    assert np.all(napp <= a_max * (1 + 1e-14))
    assert np.allclose(napp[over], a_max, rtol=1e-14)
    assert np.array_equal(app[:, ~over], raw[:, ~over])
    # direction preserved: parallel and same sense
    cross = np.cross(raw.T, app.T)
    assert np.max(np.linalg.norm(cross, axis=1) / (nraw * napp)) < 1e-14
    assert np.all(np.sum(raw * app, axis=0) > 0)
    # scalar path agrees with vector path
    for k in range(50):
        s = ctrl.applied_scalar(*rho[:, k], *rhod[:, k])
        assert np.allclose(s, app[:, k], rtol=1e-14, atol=1e-18)
    # no cap -> unchanged
    ctrl_free = PDController(N, 6.0, None)
    assert np.array_equal(ctrl_free.saturate(raw), raw)
    assert ctrl.kp == pytest.approx((6.0 * N) ** 2) and ctrl.kd == pytest.approx(12.0 * N)


# ---------------------------------------------------------------------------
# SPEC 10: arrival assessment on synthetic g(t)
# ---------------------------------------------------------------------------
def _both(gfun, T):
    prod = arrival_on_grid(gfun, T, 0.5, 1e-6, locate_all=False)
    fine = arrival_on_grid(gfun, T, 0.01, 1e-9, locate_all=True)
    return prod, fine


def test_arrival_single_entry():
    T = 2000.0
    t_e = 1000.123456

    def g(t):
        return (t_e - np.asarray(t, dtype=float)) / 100.0

    prod, fine = _both(g, T)
    for res, tol in ((prod, 2e-6), (fine, 1e-8)):
        assert res["inside_at_end"] and res["arrived"]
        assert abs(res["entry_time_s"] - t_e) < tol
        assert abs(res["dwell_s"] - (T - t_e)) < tol
        assert abs(res["dwell_margin_s"] - (T - t_e - 300.0)) < tol
        assert res["n_crossings"] == 1


def test_arrival_inside_throughout():
    T = 1500.0
    prod, fine = _both(lambda t: -0.5 + 0.0 * np.asarray(t, dtype=float), T)
    for res in (prod, fine):
        assert res["entry_time_s"] == 0.0
        assert res["dwell_s"] == T
        assert res["arrived"] and res["n_crossings"] == 0


def test_arrival_outside_at_end():
    T = 2000.0
    prod, fine = _both(lambda t: (np.asarray(t, dtype=float) - 1500.0) / 100.0, T)
    # g < 0 before 1500 s, then leaves the set and stays out
    for res in (prod, fine):
        assert not res["inside_at_end"] and not res["arrived"]
        assert math.isnan(res["entry_time_s"]) and math.isnan(res["dwell_s"])
        assert res["n_crossings"] == 1


def test_arrival_dwell_threshold():
    T = 2000.0
    for dwell, expected in ((299.9, False), (300.1, True)):
        t_e = T - dwell

        def g(t, t_e=t_e):
            return (t_e - np.asarray(t, dtype=float)) / 100.0

        prod, fine = _both(g, T)
        assert prod["arrived"] is expected and fine["arrived"] is expected
        assert abs(fine["dwell_margin_s"] - (dwell - 300.0)) < 1e-8


def test_arrival_excursion_between_production_samples():
    """A brief exit between two 0.5 s samples: only the fine convention sees it."""
    T = 2200.0
    tc, width = 2000.25, 0.03

    def g(t):
        t = np.asarray(t, dtype=float)
        return np.maximum((1000.0 - t) / 100.0, -0.5) + 0.6 * np.exp(-((t - tc) / width) ** 2)

    half = width * math.sqrt(math.log(1.2))       # g >= 0 on [tc - half, tc + half]
    prod, fine = _both(g, T)
    assert prod["arrived"] and abs(prod["entry_time_s"] - 1000.0) < 2e-6
    assert prod["n_crossings"] == 1
    assert fine["n_crossings"] == 3 and fine["n_entries"] == 2 and fine["n_exits"] == 1
    assert abs(fine["entry_time_s"] - (tc + half)) < 1e-8
    assert not fine["arrived"]
    assert abs(fine["dwell_s"] - (T - tc - half)) < 1e-8


def test_arrival_requires_completed_run():
    res = assess_arrival([], -1.0, lambda t: -1.0, 1000.0, False, 1e-6, False)
    assert not res["arrived"] and not res["inside_at_end"]
    assert math.isnan(res["entry_time_s"])


def test_time_grid_end_handling():
    g = time_grid(10.3, 0.5)
    assert g[0] == 0.0 and g[-2] == 10.0 and g[-1] == 10.3 and g.size == 22
    g = time_grid(10.0 + 1e-10, 0.5)              # last point within 1e-9 s of T -> replaced
    assert g.size == 21 and g[-1] == 10.0 + 1e-10 and g[-2] == 9.5
    g = time_grid(10.0, 0.5)
    assert g.size == 21 and g[-1] == 10.0
    assert np.all(np.diff(time_grid(5553.6, 0.01)) > 0)


def test_saturated_time_and_sign_intervals():
    assert saturated_time([10.0, 30.0], True, 100.0) == pytest.approx(10.0 + 70.0)
    assert saturated_time([10.0, 30.0], False, 100.0) == pytest.approx(20.0)
    assert saturated_time([], False, 100.0) == 0.0
    iv = sign_intervals([0, 1, 2, 3], [1.0, -1.0, -2.0, 0.0])
    assert [(a, b) for a, b, _, _ in iv] == [(0.0, 1.0), (2.0, 3.0)]


def test_refine_max_finds_peak_between_samples():
    t = np.arange(0.0, 10.0, 0.5)
    f = lambda s: 1.0 - (s - 3.3) ** 2  # noqa: E731
    v, tp = refine_max(f, t, f(t))
    assert abs(tp - 3.3) < 1e-5 and abs(v - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# SPEC 10: second-order agreement of reference and CW at small separation
# ---------------------------------------------------------------------------
def _peak_branch_difference(rel0, branch_b, wn_ratio=6.0, horizon=0.5):
    ctrl = PDController(N, wn_ratio, None)
    T = horizon * ORBIT.period
    ref = RefModel(ORBIT, ctrl, rel0)
    other = (CWModel if branch_b == "cw" else SOModel)(ORBIT, ctrl, rel0)
    run_r = integrate_adaptive(ref, "dop853", T, 1e-13)
    run_o = integrate_adaptive(other, "dop853", T, 1e-13)
    t = np.linspace(0.0, T, 3001)
    return max_rel_diff(_Ev(ref, run_r), _Ev(other, run_o), t)


def test_reference_vs_cw_second_order_in_separation():
    d = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    d /= np.linalg.norm(d[:3])
    p1 = _peak_branch_difference(400.0 * d, "cw")
    p2 = _peak_branch_difference(200.0 * d, "cw")
    ratio = p1 / p2
    assert 3.8 < ratio < 4.2, (p1, p2, ratio)


def test_reference_vs_quadratic_third_order_in_separation():
    d = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    d /= np.linalg.norm(d[:3])
    p1 = _peak_branch_difference(4000.0 * d, "so")
    p2 = _peak_branch_difference(2000.0 * d, "so")
    ratio = p1 / p2
    assert 7.0 < ratio < 9.0, (p1, p2, ratio)


# ---------------------------------------------------------------------------
# Integrators and events
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["Radau", "DOP853"])
def test_adaptive_driver_reproduces_solve_ivp(method):
    rel0 = np.array([300.0, -500.0, 100.0, 0.1, 0.0, -0.05])
    model = CWModel(ORBIT, PDController(N, 4.0, None), rel0)
    T, rtol = 2000.0, 1e-10
    run = integrate_adaptive(model, method.lower(), T, rtol, max_step=20.0)
    ref = solve_ivp(model.rhs, (0.0, T), model.y0, method=method, rtol=rtol,
                    atol=model.atol(rtol), max_step=20.0)
    # identical step sequence; states at the steps agree to round-off (the
    # driver evaluates them from the step's dense output)
    assert run.n_steps == ref.t.size - 1
    assert np.array_equal(run.sol.knots, ref.t)
    scale = np.abs(ref.y).max(axis=1, keepdims=True)
    assert np.max(np.abs(run.sol(ref.t) - ref.y) / scale) < 1e-13


def test_switch_event_matches_solve_ivp_and_analytic():
    """Cross-track only CW case: saturated at start, one switch, analytic solution."""
    a_max, wn_ratio, z0 = 0.05, 6.0, -3000.0
    ctrl = PDController(N, wn_ratio, a_max)
    model = CWModel(ORBIT, ctrl, [0, 0, z0, 0, 0, 0])
    # analytic: z'' = -n^2 z + a_max until |Kp z + Kd z'| = a_max
    zc = a_max / N ** 2

    def gsw(t):
        z = zc + (z0 - zc) * math.cos(N * t)
        zd = -N * (z0 - zc) * math.sin(N * t)
        return abs(ctrl.kp * z + ctrl.kd * zd) - a_max

    t_exact = brentq(gsw, 1.0, 150.0, xtol=1e-14)
    # default restart: root re-located on a kink-free (frozen-mode) integration
    for rtol, tol in ((1e-11, 1e-11), (1e-9, 1e-10)):
        run = integrate_adaptive(model, "radau", ORBIT.period, rtol)
        assert run.status == "completed" and len(run.switch_times) == 1
        assert run.restart_methods["refine"] == 1
        ts = run.switch_times[0]
        assert abs(ts - t_exact) < tol
        assert abs(model.gsat(ts, run.sol(ts))) < 1e-12
    # the legacy "interpolant" restart is solve_ivp's terminal-event behaviour:
    # same root as solve_ivp's own event location (which is ~1e3 x less accurate)
    run_i = integrate_adaptive(model, "radau", ORBIT.period, 1e-11, switch_restart="interpolant")
    ev = lambda t, y: model.gsat(t, y)  # noqa: E731
    ev.terminal = True
    sol = solve_ivp(model.rhs, (0.0, 500.0), model.y0, method="Radau", rtol=1e-11,
                    atol=model.atol(1e-11), max_step=20.0, events=ev)
    assert abs(sol.t_events[0][0] - run_i.switch_times[0]) < 1e-12


def test_switch_restarts_and_saturation_statistics_consistent():
    sc = {"sid": "sat", "rel0": [2000.0, -3000.0, 1000.0, 0.5, 0.0, -0.3],
          "wn_ratio": 6.0, "a_max": 0.03, "horizon_orbits": 0.3, "altitude": 400000.0}
    s_r, _ = run_case(sc, {"solver": "radau", "rtol": 1e-11, "label": "r"}, branches=("cw", "so"))
    s_k, _ = run_case(sc, {"solver": "rk4", "h": 0.25, "label": "k"}, branches=("cw", "so"))
    for b in ("cw", "so"):
        assert s_r[b + "_n_switches"] >= 1
        assert s_r[b + "_n_switches"] == s_r[b + "_n_switches_fine"]
        assert not s_r[b + "_events_disabled"]
        assert s_r[b + "_n_switches"] == s_k[b + "_n_switches"]
        assert np.allclose(s_r[b + "_switch_times_s"], s_k[b + "_switch_times_s"], atol=1e-3)
        assert abs(s_r[b + "_sat_total_s"] - s_k[b + "_sat_total_s"]) < 1e-3
        assert s_r[b + "_peak_cmd_mps2"] == pytest.approx(0.03, rel=1e-12)
        # switch times alternate sides; saturated at start -> first is a down-crossing
        assert s_r[b + "_saturated_at_start"]


class _SineSwitchModel:
    """Toy model: y' = 1 with switch function sin(t); roots at k*pi alternate in direction."""

    name = "toy"
    n_state = 1
    has_alt = False
    has_sat = True
    y0 = np.array([0.0])

    def atol(self, rtol):
        return np.array([rtol])

    def rhs(self, t, y):
        return np.array([1.0])

    def gsat(self, t, y):
        return math.sin(float(y[0]))

    def bad_state(self, y):
        return None


@pytest.mark.parametrize("method", ["radau", "dop853"])
def test_switch_restarts_find_every_root_once(method):
    run = integrate_adaptive(_SineSwitchModel(), method, 100.0, 1e-10, max_step=1.0)
    expected = [k * math.pi for k in range(1, 32)]
    assert run.status == "completed" and not run.events_disabled
    assert len(run.switch_times) == len(expected)
    assert np.max(np.abs(np.array(run.switch_times) - expected)) < 1e-9


def test_no_progress_guard_disables_switch_events(monkeypatch):
    import indverify.integrators as integ
    monkeypatch.setattr(integ, "NO_PROGRESS_MIN_ADVANCE", 1.0e3)   # every restart "stalls"
    run = integrate_adaptive(_SineSwitchModel(), "radau", 100.0, 1e-10, max_step=1.0)
    assert run.events_disabled and "no progress" in run.events_disabled_reason
    assert len(run.switch_times) == integ.NO_PROGRESS_LIMIT
    assert run.status == "completed" and run.t_end == 100.0


def test_restart_cap_is_recorded():
    model = CWModel(ORBIT, PDController(N, 6.0, 0.03), [2000.0, -3000.0, 1000.0, 0.5, 0.0, -0.3])
    run = integrate_adaptive(model, "radau", 0.3 * ORBIT.period, 1e-10, restart_cap=1)
    assert run.events_disabled and "cap" in run.events_disabled_reason
    assert len(run.switch_times) == 1 and run.status == "completed"


# ---------------------------------------------------------------------------
# Switch restarts: kink-free root and step-end restart state (review finding 1)
# ---------------------------------------------------------------------------
RND33 = dict(rel0=[-103.7, 253.1, -582.1, -0.106, -0.023, -0.042], wn_ratio=8.0,
             a_max=0.005169, horizon_orbits=1, altitude=400000.0)


def _frozen_truth(model, T, rtol=1e-13, max_step=2.0):
    """Independent truth driver: solve_ivp DOP853 with the current side's law
    frozen, terminal switch event, and restart state from a fresh integration
    ending exactly at the event (never from a step that straddled the kink)."""
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
        out = np.empty((model.n_state, tt.size))
        ends = np.array([e for e, _ in segs])
        idx = np.minimum(np.searchsorted(ends, tt, side="left"), len(segs) - 1)
        for i in np.unique(idx):
            out[:, idx == i] = segs[i][1](tt[idx == i])
        return out
    return sol, switches


def test_frozen_mode_controller_continues_each_side_smoothly():
    ctrl = PDController(N, 6.0, 0.05)
    sat, uns = ctrl.frozen(True), ctrl.frozen(False)
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = rng.normal(size=6) * np.array([300, 300, 300, 0.3, 0.3, 0.3])
        raw = np.array(ctrl.raw_scalar(*p))
        nr = np.linalg.norm(raw)
        true = np.array(ctrl.applied_scalar(*p))
        # each frozen law equals the true law on its own side of the surface
        assert np.allclose(sat.applied_scalar(*p) if nr > 0.05 else uns.applied_scalar(*p),
                           true, rtol=1e-15, atol=0)
        # and continues its own formula on the other side
        assert np.allclose(np.linalg.norm(sat.applied_scalar(*p)), 0.05, rtol=1e-14)
        assert np.allclose(uns.applied_scalar(*p), raw, rtol=0, atol=0)
    a = rng.normal(size=(3, 50))
    assert np.allclose(np.linalg.norm(sat.saturate(a), axis=0), 0.05, rtol=1e-14)
    assert np.array_equal(uns.saturate(a), a)
    model = RefModel(ORBIT, ctrl, [500.0, 0, 0, 0, 0, 0])
    fm = model.frozen(True)
    assert fm.ctrl.saturated and model.ctrl is ctrl and fm.y0 is model.y0


def test_switch_restart_regression_vs_independent_truth():
    """Saturated case with 3 switches (rnd33).  Restarting from the straddling
    step's interpolant (solve_ivp's terminal-event behaviour) costs 1-2 orders
    of magnitude of accuracy in the reference branch and is non-monotone in
    rtol; the default restart is not."""
    ctrl = PDController(N, RND33["wn_ratio"], RND33["a_max"])
    T = ORBIT.period
    tg = np.arange(0.0, T, 0.5)
    for branch, method, rtol, tol_pos, tol_sw in (("ref", "radau", 1e-8, 1.5e-3, 1e-3),
                                                  ("ref", "radau", 1e-11, 5e-6, 2e-6),
                                                  ("ref", "dop853", 1e-12, 5e-7, 1e-6),
                                                  ("cw", "radau", 1e-12, 3e-10, 1e-9)):
        model = (RefModel if branch == "ref" else CWModel)(ORBIT, ctrl, RND33["rel0"])
        tsol, tsw = _frozen_truth(model, T)
        assert len(tsw) == 3
        rho_t = model.rel_many(tsol(tg))[0]
        errs = {}
        for how in ("refine", "interpolant"):
            run = integrate_adaptive(model, method, T, rtol, switch_restart=how)
            assert run.status == "completed" and len(run.switch_times) == 3
            rho = model.rel_many(run.sol(tg))[0]
            errs[how] = np.max(np.linalg.norm(rho - rho_t, axis=0))
            if how == "refine":
                assert run.restart_methods == {"refine": 3, "reintegrate": 0, "interpolant": 0}
                assert np.max(np.abs(np.array(run.switch_times) - tsw)) < tol_sw, (branch, rtol)
        assert errs["refine"] < tol_pos, (branch, method, rtol, errs)
        assert errs["interpolant"] > 5.0 * errs["refine"], (branch, method, rtol, errs)


def test_switch_counts_match_fine_grid_after_restart_fix():
    """Keeping the interpolant root but re-integrating to it can leave the
    restart state on the old side and record spurious duplicate switches
    (5 instead of 3 for this case at Radau 1e-8); the default restart polishes
    the state across the surface."""
    s, _ = run_case(dict(RND33, sid="rnd33"), {"solver": "radau", "rtol": 1e-8, "label": "r8"})
    for b in ("ref", "cw", "so"):
        assert s[b + "_n_switches"] == s[b + "_n_switches_fine"] == 3, b
        assert s[b + "_n_restart_fallback"] == 0 and not s[b + "_events_disabled"]
        assert np.all(np.diff(s[b + "_switch_times_s"]) > 1.0)


def test_polish_puts_restart_state_across_the_surface():
    from scipy.integrate import Radau
    from indverify.integrators import _polish, _switch_restart
    toy = _SineSwitchModel()                    # y' = 1, g = sin(y): root at pi, g > 0 before
    rhs, gfun = toy.rhs, toy.gsat
    # state well before the root (as from a contaminated interpolant root)
    pieces, t, y, _, _, ok = _polish(Radau, rhs, gfun, 3.0, np.array([3.0]), 3.5, 1,
                                     1e-10, np.array([1e-10]), 1.0)
    assert ok and math.pi < t < math.pi + 0.3 * (math.pi - 3.0) and gfun(t, y) < 0
    assert abs(y[0] - t) < 1e-9 and pieces and pieces[-1][0] == t
    # a state already across, or within POLISH_ACCEPT_DT of the surface, is kept
    for t0 in (3.2, math.pi - 1e-12):
        pieces, t, y, _, _, ok = _polish(Radau, rhs, gfun, t0, np.array([t0]), 3.5, 1,
                                         1e-10, np.array([1e-10]), 1.0)
        assert ok and t == t0 and not pieces
    # model without a frozen mode: re-integration fallback from the true step
    # start to a (deliberately early) interpolant root, then polished across
    pieces, tr, y_root, _, _, used = _switch_restart(toy, Radau, 3.0, np.array([3.0]), 3.5, 3.1,
                                                     1, math.sin(3.0), 1e-10,
                                                     np.array([1e-10]), 1.0)
    assert used == "reintegrate" and math.pi < tr < math.pi + 0.3 * (math.pi - 3.1)
    assert abs(y_root[0] - tr) < 1e-9 and pieces[-1][0] == tr


def test_hermite_interpolation_exact_for_cubics():
    T = np.array([0.0, 0.7, 2.0, 2.5])
    p = lambda t: 1.5 - 2.0 * t + 0.3 * t ** 2 - 0.7 * t ** 3  # noqa: E731
    dp = lambda t: -2.0 + 0.6 * t - 2.1 * t ** 2  # noqa: E731
    H = HermiteDense(T, p(T)[None, :], dp(T)[None, :])
    tt = np.linspace(0.0, 2.5, 97)
    assert np.allclose(H(tt)[0], p(tt), rtol=0, atol=1e-13)
    assert abs(H(1.3)[0] - p(1.3)) < 1e-13


def test_rk4_fourth_order_convergence():
    rel0 = np.array([100.0, 200.0, -50.0, 0.1, -0.2, 0.05])
    model = CWModel(ORBIT, PDController(N, 20.0, None), rel0)
    T = 600.0
    A = closed_loop_matrix(20.0)
    exact = expm(A * T) @ rel0
    errs = [np.max(np.abs(integrate_rk4(model, T, h).sol(T) - exact)) for h in (4.0, 2.0)]
    assert 13.0 < errs[0] / errs[1] < 19.0


# ---------------------------------------------------------------------------
# Statuses: numerical failure of the quadratic branch, altitude-floor exit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("setting", [{"solver": "radau", "rtol": 1e-10, "label": "r"},
                                     {"solver": "rk4", "h": 0.5, "label": "k"}])
def test_quadratic_blow_up_is_numerical_failure(setting):
    sc = {"sid": "blow", "rel0": [-2.0e6, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
          "horizon_orbits": 1, "altitude": 400000.0}
    s, h = run_case(sc, setting, branches=("cw", "so"), keep_history=True)
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
def test_altitude_floor_is_domain_exit(setting):
    sc = {"sid": "low", "rel0": [-250e3, 0, 0, -150.0, 0, 0], "wn_ratio": 6.0, "a_max": 0.01,
          "horizon_orbits": 1, "altitude": 400000.0}
    s, _ = run_case(sc, setting, branches=("ref",))
    assert s["ref_status"] == "domain_exit"
    t_exit = s["ref_t_end_s"]
    assert 0.0 < t_exit < s["horizon_s"]
    assert not s["ref_prod_arrived"] and not s["ref_fine_arrived"]
    # altitude at the exit time is the floor
    model = RefModel(ORBIT, PDController(N, 6.0, 0.01), sc["rel0"])
    from indverify.runner import SolverSetting, integrate_branch
    run = integrate_branch(model, SolverSetting.from_dict(setting), s["horizon_s"])
    assert abs(model.alt(t_exit, run.sol(t_exit))) < 1e-3
    assert abs(run.t_end - t_exit) < 1e-9


def test_altitude_exit_times_agree_between_solvers():
    sc = {"sid": "low", "rel0": [-250e3, 0, 0, -150.0, 0, 0], "wn_ratio": 6.0, "a_max": 0.01,
          "horizon_orbits": 1, "altitude": 400000.0}
    t = [run_case(sc, st, branches=("ref",))[0]["ref_t_end_s"]
         for st in ({"solver": "radau", "rtol": 1e-11}, {"solver": "rk4", "h": 0.25})]
    assert abs(t[0] - t[1]) < 1e-4


# ---------------------------------------------------------------------------
# run_case interface
# ---------------------------------------------------------------------------
BRANCH_KEYS = ["status", "reason", "t_end_s", "n_switches", "events_disabled", "sat_total_s",
               "final_range_m", "final_speed_mps", "peak_cmd_mps2"]
CONV_KEYS = ["inside_at_end", "entry_time_s", "dwell_s", "dwell_margin_s", "arrived",
             "n_crossings"]
PAIR_KEYS = ["peak_pos_m", "t_peak_s", "peak_vel_mps", "max_cmd_diff_mps2", "common_end_s",
             "peak_pos_fine_m"]


def test_run_case_interface_and_cross_solver_agreement():
    sc = {"sid": "pR1000", "rel0": [1000.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
          "horizon_orbits": 1, "altitude": 400000.0}
    s1, h1 = run_case(sc, {"solver": "dop853", "rtol": 1e-12, "label": "d12"}, keep_history=True)
    s2, h2 = run_case(sc, {"solver": "rk4", "h": 0.5, "label": "k05"})
    assert h2 is None
    for k in ("sid", "solver", "setting_label", "rtol", "h", "max_step", "wall_s"):
        assert k in s1
    assert s1["sid"] == "pR1000" and s1["solver"] == "dop853" and s1["setting_label"] == "d12"
    assert math.isnan(s1["h"]) and math.isnan(s2["rtol"]) and math.isnan(s2["max_step"])
    assert s1["max_step"] == 20.0 and s2["h"] == 0.5
    for b in ("ref", "cw", "so"):
        for k in BRANCH_KEYS:
            assert b + "_" + k in s1
        for c in ("prod", "fine"):
            for k in CONV_KEYS:
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
        # history
        n = h1["t"].size
        assert h1[b + "_t"].size == n
        for k, shape in (("rho", (n, 3)), ("rhodot", (n, 3)), ("acc", (n, 3)), ("accnorm", (n,))):
            assert h1[b + "_" + k].shape == shape
        assert np.allclose(np.linalg.norm(h1[b + "_acc"], axis=1), h1[b + "_accnorm"])
        assert np.allclose(h1[b + "_rho"][0], sc["rel0"][:3], atol=1e-8)
    for a, b in (("cw", "ref"), ("so", "ref"), ("cw", "so")):
        for k in PAIR_KEYS:
            assert "%s_vs_%s_%s" % (a, b, k) in s1
        assert s1["%s_vs_%s_common_end_s" % (a, b)] == s1["horizon_s"]
    # CW vs reference differs by millimetres here; solvers agree far below that
    assert 1e-3 < s1["cw_vs_ref_peak_pos_m"] < 2e-2
    assert abs(s1["cw_vs_ref_peak_pos_m"] - s2["cw_vs_ref_peak_pos_m"]) < 1e-7
    assert abs(s1["cw_vs_ref_peak_pos_m"] - s1["cw_vs_ref_peak_pos_fine_m"]) < 1e-8
    assert s1["so_vs_ref_peak_pos_m"] < 1e-2 * s1["cw_vs_ref_peak_pos_m"]
    assert json.loads(json.dumps(s1))["sid"] == "pR1000"


def test_branch_subset_and_validation():
    sc = {"sid": "x", "rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": None,
          "horizon_orbits": 0.1, "altitude": 400000.0}
    s, _ = run_case(sc, {"solver": "dop853", "rtol": 1e-10}, branches=("cw",))
    assert "cw_status" in s and "ref_status" not in s and not any("_vs_" in k for k in s)
    assert s["cw_sat_total_s"] == 0.0 and s["cw_n_switches"] == 0
    with pytest.raises(ValueError):
        run_case(sc, {"solver": "euler", "h": 1.0})
    with pytest.raises(ValueError):
        run_case(sc, {"solver": "rk4", "h": 1.0}, branches=("cw", "bogus"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_cli_writes_resumes_and_records_errors(tmp_path):
    cases = [
        {"sid": "c1", "rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
         "horizon_orbits": 0.05, "altitude": 400000.0},
        {"sid": "c2", "rel0": [0, 0, -300.0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.005,
         "horizon_orbits": 0.05, "altitude": 400000.0},
        {"sid": "bad", "wn_ratio": 6.0, "a_max": 0.05, "horizon_orbits": 0.05},  # no rel0
    ]
    settings = [{"solver": "dop853", "rtol": 1e-10, "label": "d10"},
                {"solver": "rk4", "h": 1.0, "label": "k1"}]
    cpath, spath, out = tmp_path / "cases.json", tmp_path / "settings.json", tmp_path / "out"
    cpath.write_text(json.dumps(cases))
    spath.write_text(json.dumps(settings))
    cmd = [sys.executable, "-m", "indverify", "--cases", str(cpath), "--settings", str(spath),
           "--out", str(out), "--workers", "2", "--history-setting", "d10"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (out / "indverify_results.jsonl").read_text().splitlines()]
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
    # resume: successful pairs are skipped, error-only pairs are re-run by
    # default; their old error lines move to the .bad file (one record per pair)
    r2 = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr
    assert "2 task(s) to run, 4 already present" in r2.stdout
    lines2 = [json.loads(x) for x in (out / "indverify_results.jsonl").read_text().splitlines()]
    assert len(lines2) == 6 and len({(x["sid"], x["setting_label"]) for x in lines2}) == 6
    assert len((out / "indverify_results.bad").read_text().splitlines()) == 2
    # --no-retry-errors: nothing new is run or written
    r3 = subprocess.run(cmd + ["--no-retry-errors"], cwd=str(ROOT), env=env, capture_output=True,
                        text=True, timeout=600)
    assert r3.returncode == 0, r3.stderr
    assert "0 task(s) to run" in r3.stdout
    assert len((out / "indverify_results.jsonl").read_text().splitlines()) == 6


def test_cli_malformed_cases_settings_and_colliding_history_names(tmp_path):
    """Malformed cases/settings become per-pair error lines instead of aborting
    the run; sids that sanitise to the same file name keep separate histories."""
    good = {"rel0": [100.0, 0, 0, 0, 0, 0], "wn_ratio": 6.0, "a_max": 0.05,
            "horizon_orbits": 0.02, "altitude": 400000.0}
    cases = [dict(good, sid="x/y"), dict(good, sid="x_y"), dict(good),   # 3rd: no sid
             "not a scenario"]
    settings = [{"solver": "dop853", "rtol": 1e-9, "label": "d9"},
                {"solver": "euler", "h": 1.0, "label": "eul"},          # unknown solver
                {"solver": "rk4"}]                                       # no h, no label
    cpath, spath, out = tmp_path / "cases.json", tmp_path / "settings.json", tmp_path / "out"
    cpath.write_text(json.dumps(cases))
    spath.write_text(json.dumps(settings))
    cmd = [sys.executable, "-m", "indverify", "--cases", str(cpath), "--settings", str(spath),
           "--out", str(out), "--workers", "2", "--branches", "cw", "--history-setting", "d9"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (out / "indverify_results.jsonl").read_text().splitlines()]
    pairs = {(x["sid"], x["setting_label"]) for x in lines}
    assert pairs == {(s, l) for s in ("x/y", "x_y", "#2", "#3") for l in ("d9", "eul", "#setting2")}
    ok = {(x["sid"], x["setting_label"]) for x in lines if "error" not in x}
    assert ok == {("x/y", "d9"), ("x_y", "d9")}
    assert all("Traceback" in x["error"] for x in lines if "error" in x)
    hist = sorted(p.name for p in (out / "histories").iterdir())
    assert len(hist) == 2 and "x_y.npz" not in hist          # both names carry a hash
    from indverify.cli import history_filenames
    names = history_filenames(["x/y", "x_y", "c1"])
    assert names["c1"] == "c1.npz" and set(hist) == {names["x/y"], names["x_y"]}
    # case-insensitive collisions (Windows file systems) are resolved too
    ci = history_filenames(["Case", "case", "other"])
    assert len({v.lower() for v in ci.values()}) == 3 and ci["other"] == "other.npz"
    # duplicate sids still abort the whole run
    cpath.write_text(json.dumps([dict(good, sid="d"), dict(good, sid="d")]))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "unique" in r.stderr


def test_results_compaction_on_resume(tmp_path):
    from indverify.cli import compact_results
    path, bad = str(tmp_path / "res.jsonl"), str(tmp_path / "res.bad")
    rec = lambda sid, lab, **kw: json.dumps(dict(sid=sid, setting_label=lab, **kw))  # noqa: E731
    lines = [rec("A", "s", v=1),
             rec("B", "s", error="Traceback: worker crashed"),
             rec("C", "s", error="Traceback: BrokenProcessPool"),
             rec("C", "s", v=3),                              # later success beats the error
             rec("A", "s", v=2),                              # duplicate success: last wins
             rec("D", "s", v=4)[:25]]                         # truncated by a crash mid-write
    pathlib.Path(path).write_text("\n".join(lines))           # no trailing newline
    keep_err = str(tmp_path / "keep.jsonl")
    pathlib.Path(keep_err).write_text("\n".join(lines))
    done, st = compact_results(path, bad, retry_errors=True)
    assert done == {("A", "s"), ("C", "s")}
    kept = [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]
    assert [(r["sid"], r["v"]) for r in kept] == [("C", 3), ("A", 2)]
    moved = pathlib.Path(bad).read_text().splitlines()
    assert len(moved) == 4 and moved[0] == lines[-1]
    assert st["unparseable"] == 1 and st["superseded"] == 2 and st["errors_to_retry"] == 1
    # second compaction is a no-op
    done2, st2 = compact_results(path, bad, retry_errors=True)
    assert done2 == done and not st2["rewritten"]
    # --no-retry-errors keeps the error-only pair as done
    done3, st3 = compact_results(keep_err, str(tmp_path / "keep.bad"), retry_errors=False)
    assert done3 == {("A", "s"), ("B", "s"), ("C", "s")} and st3["errors_kept"] == 1


def test_load_results_keeps_latest_success_per_pair(tmp_path):
    from indverify.convergence import is_ref_derived, ladder_changes, load_results
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
    recs = load_results(str(path), st)
    assert [(r["setting_label"], r["cw_vs_ref_peak_pos_m"]) for r in recs] == [("r8", 1.5), ("r10", 2.0)]
    assert st == {"lines": 6, "unparseable": 1, "dropped": 3, "error_only_pairs": 1}
    rows = ladder_changes(recs)
    row = [r for r in rows if r["quantity"] == "cw_vs_ref_peak_pos_m"][0]
    assert row["change"] == pytest.approx(0.5) and row["ref_derived"] is True
    assert is_ref_derived("ref_prod_entry_time_s") and not is_ref_derived("cw_vs_so_peak_pos_m")


def test_ref_rtol_tightens_only_the_reference_branch():
    sc = dict(RND33, sid="rnd33", horizon_orbits=0.3)
    s_mix, _ = run_case(sc, {"solver": "radau", "rtol": 1e-8, "ref_rtol": 1e-11},
                        branches=("ref", "cw"))
    s_8, _ = run_case(sc, {"solver": "radau", "rtol": 1e-8}, branches=("ref", "cw"))
    s_11, _ = run_case(sc, {"solver": "radau", "rtol": 1e-11}, branches=("ref", "cw"))
    assert s_mix["setting_label"] == "radau_1e-08_ref1e-11"
    assert s_mix["rtol"] == 1e-8 and s_mix["ref_rtol"] == 1e-11 and s_8["ref_rtol"] == 1e-8
    for k in ("_t_end_s", "_final_range_m", "_prod_entry_time_s", "_n_steps"):
        assert s_mix["ref" + k] == s_11["ref" + k] or (math.isnan(s_mix["ref" + k])
                                                        and math.isnan(s_11["ref" + k]))
        assert s_mix["cw" + k] == s_8["cw" + k] or (math.isnan(s_mix["cw" + k])
                                                    and math.isnan(s_8["cw" + k]))
    assert s_mix["ref_switch_times_s"] == s_11["ref_switch_times_s"]
    with pytest.raises(ValueError):
        run_case(sc, {"solver": "rk4", "h": 1.0, "ref_rtol": 1e-11}, branches=("cw",))


def test_convergence_ladder_report():
    from indverify.convergence import changes_vs_reference, ladder_changes, summarise
    base = {"sid": "s", "cw_status": "completed", "cw_prod_arrived": True,
            "cw_switch_times_s": [10.0, 20.0], "cw_vs_ref_peak_pos_m": 0.01,
            "cw_prod_entry_time_s": float("nan"), "wall_s": 1.0}
    recs = []
    for label, rtol, d in (("r8", 1e-8, 1e-5), ("r10", 1e-10, 1e-7), ("r12", 1e-12, 0.0)):
        r = dict(base, solver="radau", rtol=rtol, setting_label=label,
                 cw_vs_ref_peak_pos_m=0.01 + d, cw_switch_times_s=[10.0 + d, 20.0])
        recs.append(r)
    rows = ladder_changes(list(reversed(recs)))
    peak = [r for r in rows if r["quantity"] == "cw_vs_ref_peak_pos_m"]
    assert [(r["from_label"], r["to_label"]) for r in peak] == [("r8", "r10"), ("r10", "r12")]
    assert peak[0]["abs_change"] == pytest.approx(1e-5 - 1e-7)
    assert not any(r["quantity"] == "wall_s" for r in rows)
    sw = [r for r in rows if r["quantity"] == "cw_switch_times_s"]
    assert sw[1]["abs_change"] == pytest.approx(1e-7)
    nanrow = [r for r in rows if r["quantity"] == "cw_prod_entry_time_s"][0]
    assert nanrow["change"] == 0.0 and nanrow["note"] == "both NaN"
    vs = changes_vs_reference(recs, "r12")
    assert {r["from_label"] for r in vs} == {"r12"} and {r["to_label"] for r in vs} == {"r8", "r10"}
    summ = summarise(rows)
    assert any(x["quantity"] == "cw_vs_ref_peak_pos_m" for x in summ)


# ---------------------------------------------------------------------------
# SPEC 10: independence
# ---------------------------------------------------------------------------
FORBIDDEN = ("cwvalid", "src", "experiments")


def test_independence_no_production_imports():
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
                assert top not in FORBIDDEN, "%s imports %s" % (f.name, m)
                # only the standard library, NumPy and SciPy (task rules)
                allowed = top in {"indverify", "numpy", "scipy"} or top in sys.stdlib_module_names
                assert allowed, "%s imports non-allowed module %s" % (f.name, m)


def test_metadata():
    assert IMPLEMENTS == "indverify/SPEC.md"
    assert isinstance(__version__, str) and __version__
    assert (ROOT / IMPLEMENTS).exists()
