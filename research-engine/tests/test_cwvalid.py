"""Tests for the cwvalid research-extension package."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from cwvalid import dynamics as dyn
from cwvalid.constants import DEFAULT_RADIUS, MU_EARTH, mean_motion, period
from cwvalid.control import CommandSpec, PDLaw
from cwvalid.experiment import RunConfig, run_scenario
from cwvalid.scenario import (Scenario, axis_scenario, vel_exact_energy_matched_radial,
                              vel_first_order_energy_matched)
from cwvalid.simulate import EciConfig, SolverConfig, simulate
from src import cw as src_cw
from src import lvlh as src_lvlh
from src import orbit as src_orbit
from src import perturbations as src_pert
from src import validation as src_val
from src.controller import PDController

R = DEFAULT_RADIUS
N = mean_motion(R)
T = period(R)
RNG = np.random.default_rng(12345)


def _rand_states(k, scale):
    pos = RNG.normal(size=(k, 3)) * scale
    vel = RNG.normal(size=(k, 3)) * N * scale
    return np.hstack((pos, vel))


# --- dynamics ---------------------------------------------------------------

def test_nerm_stable_form_matches_direct_form():
    for s in _rand_states(200, 20_000.0):
        a = np.array(dyn.acc_nerm(N, R, *s))
        b = np.array(dyn.acc_nerm_naive(N, R, *s))
        assert np.allclose(a, b, rtol=1e-8, atol=1e-13)


def test_nerm_reduces_to_cw_quadratically_and_so_cubically():
    u = np.array([0.6, -0.5, 0.62])
    u /= np.linalg.norm(u)
    errs_cw, errs_so = [], []
    for rho in (100.0, 200.0, 400.0, 800.0):
        s = np.concatenate((u * rho, [0.0, 0.0, 0.0]))
        a_n = np.array(dyn.acc_nerm(N, R, *s))
        errs_cw.append(np.linalg.norm(a_n - np.array(dyn.acc_cw(N, R, *s))))
        errs_so.append(np.linalg.norm(a_n - np.array(dyn.acc_so(N, R, *s))))
    r_cw = np.array(errs_cw[1:]) / np.array(errs_cw[:-1])
    r_so = np.array(errs_so[1:]) / np.array(errs_so[:-1])
    assert np.allclose(r_cw, 4.0, rtol=0.02)   # O(rho^2)
    assert np.allclose(r_so, 8.0, rtol=0.05)   # O(rho^3)
    # leading CW residual magnitude is of order n^2 rho^2 / R
    assert errs_cw[0] < 3.0 * N * N * 100.0**2 / R


def test_vectorised_relative_acc_matches_scalar():
    S = _rand_states(50, 5000.0)
    for m in ("cw", "so", "nerm"):
        V = dyn.relative_acc_vectorized(m, N, R, S)
        for i, s in enumerate(S):
            assert np.allclose(V[i], dyn.RELATIVE_MODELS[m](N, R, *s), rtol=1e-12, atol=1e-15)


def test_inertial_force_matches_src_force_model():
    fm = src_pert.ForceModel.j2_drag()
    f = dyn.InertialForce(use_j2=True, use_drag=True)
    tgt = src_orbit.circular_orbit_state(R, MU_EARTH, math.radians(51.6))
    for k in range(20):
        st = tgt + np.concatenate((RNG.normal(size=3) * 3e4, RNG.normal(size=3) * 30))
        a_src = fm.acceleration(st)
        a_new = np.array(f.acceleration(*st))
        assert np.allclose(a_src, a_new, rtol=1e-12, atol=1e-18)


def test_lvlh_roundtrip_and_agreement_with_src():
    tgt = src_orbit.circular_orbit_state(R, MU_EARTH, math.radians(51.6))
    for rel in _rand_states(20, 3000.0):
        ch = dyn.lvlh_to_eci(tgt, rel)
        back, _ = dyn.eci_to_lvlh_scalar(tuple(tgt), tuple(ch))
        assert np.allclose(back, rel, rtol=1e-9, atol=1e-9)
        assert np.allclose(src_lvlh.inertial_to_lvlh(tgt, ch), back, rtol=1e-12, atol=1e-9)


def test_exact_frame_rate_matches_finite_difference_under_j2():
    """With J2, d(rho)/dt from finite differences must match rho_dot only when the
    LVLH frame rate includes the out-of-plane component wx = r a_h / h."""
    fm = src_pert.ForceModel.j2_only()
    tgt0 = src_orbit.circular_orbit_state(R, MU_EARTH, math.radians(51.6))
    rel0 = np.array([0.0, 0.0, 20_000.0, 0.0, 0.0, 0.0])
    ch0 = src_lvlh.lvlh_to_inertial(tgt0, rel0)
    tt = np.linspace(0.0, 1500.0, 15001)
    tg = src_orbit.propagate(fm.derivatives, tgt0, tt)
    cg = src_orbit.propagate(fm.derivatives, ch0, tt)
    f = dyn.InertialForce(use_j2=True)
    ap = np.array([f.perturbation(*s) for s in tg])
    rel_exact, _ = dyn.eci_to_lvlh_vectorized(tg, cg, ap)
    rel_base, _ = dyn.eci_to_lvlh_vectorized(tg, cg, None)
    h = tt[1] - tt[0]
    fd = (rel_exact[2:, :3] - rel_exact[:-2, :3]) / (2 * h)
    e_exact = np.abs(fd - rel_exact[1:-1, 3:]).max()
    e_base = np.abs(fd - rel_base[1:-1, 3:]).max()
    assert e_exact < 1e-6
    assert e_base > 100 * e_exact


# --- control ----------------------------------------------------------------

def test_saturation_preserves_direction_and_caps_magnitude():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    for s in _rand_states(100, 50_000.0):
        raw = np.array(law.raw(*s))
        app = np.array(law.command(*s))
        if np.linalg.norm(raw) > 0.05:
            assert abs(np.linalg.norm(app) - 0.05) < 1e-15
            assert np.allclose(app / np.linalg.norm(app), raw / np.linalg.norm(raw), atol=1e-14)
        else:
            assert np.array_equal(app, raw)
    # same law as the engine's PDController
    ctrl = PDController.critically_damped(6 * N, max_accel=0.05)
    s = np.array([3000.0, -2000.0, 500.0, 1.0, -0.5, 0.2])
    assert np.allclose(ctrl(s), law.command(*s), rtol=1e-15)


def test_saturation_distance_is_amax_over_kp():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    assert law.saturation_distance == pytest.approx(0.05 / (36 * N * N))
    assert 1080 < law.saturation_distance < 1090


# --- energy matching ---------------------------------------------------------

@pytest.mark.parametrize("x0", [-20_000.0, -500.0, 500.0, 5_000.0, 20_000.0])
def test_exact_energy_matching_equalises_specific_energy(x0):
    tgt = src_orbit.circular_orbit_state(R, MU_EARTH, 0.0)
    ydot = vel_exact_energy_matched_radial(x0, R)
    ch = dyn.lvlh_to_eci(tgt, [x0, 0, 0, 0, ydot, 0])
    e_t = src_orbit.specific_energy(tgt)
    e_c = src_orbit.specific_energy(ch)
    assert abs(e_c - e_t) < 1e-9 * abs(e_t)
    # first-order expansion -2 n x0 + n x0^2 /(2R)
    approx = -2 * N * x0 + N * x0**2 / (2 * R)
    assert abs(ydot - approx) < 5 * N * abs(x0) ** 3 / R**2 + 1e-12
    assert abs(vel_first_order_energy_matched(x0, N) - (-2 * N * x0)) == 0.0


def test_exact_energy_matching_domain():
    with pytest.raises(ValueError):
        vel_exact_energy_matched_radial(R * 1.01, R)


# --- propagation ---------------------------------------------------------------

def test_uncontrolled_cw_matches_state_transition_matrix():
    s0 = np.array([300.0, -800.0, 200.0, 0.1, -0.2, 0.05])
    sim = simulate("cw", s0, CommandSpec("none"), T, n=N, R=R)
    t = np.linspace(0, T, 101)
    ref = src_cw.propagate_cw_analytic(s0, t, N)
    assert np.abs(sim.states(t) - ref)[:, :3].max() < 1e-7


def test_unsaturated_closed_loop_cw_matches_matrix_exponential():
    law = PDLaw.from_bandwidth(6 * N, None)
    A, B = src_cw.cw_system_matrices(N)
    K = np.hstack((law.kp * np.eye(3), law.kd * np.eye(3)))
    Acl = A - B @ K
    s0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
    sim = simulate("cw", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    for t in (100.0, 1000.0, 3000.0):
        assert np.abs(sim.states([t])[0] - expm(Acl * t) @ s0).max() < 1e-7


def _rk4_node_errors(sep, ks):
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([sep, 0, 0, 0, 0, 0])
    cmd = CommandSpec("continuous", law=law)
    ref = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverConfig(rtol=1e-13, max_step=5.0))
    tn = np.linspace(0, T, 1001)
    out = []
    for k in ks:
        r4 = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverConfig(method="RK4", rk4_dt=T / k))
        out.append(np.abs(r4.segments[0].sol.y[::k // 1000, :3] - ref.states(tn)[:, :3]).max())
    return out


def test_rk4_fourth_order_when_unsaturated():
    e = _rk4_node_errors(1000.0, (1000, 2000, 4000))
    assert 12 < e[0] / e[1] < 20 and 12 < e[1] / e[2] < 20


def test_rk4_reduced_order_when_saturated_documented():
    """Saturation makes the RHS non-smooth; fixed-step RK4 then loses its
    fourth-order behaviour (documented verification finding). At the
    manuscript sweep step (T/2000) the error is still at the mm level."""
    e = _rk4_node_errors(5000.0, (1000, 2000))
    assert e[0] / e[1] < 4          # far from the factor 16 of 4th order
    assert e[1] < 5e-3


def test_eci_branch_matches_original_engine_closed_loop():
    """New scalar ECI branch vs src.validation.high_fidelity_relative (RK4, fine step)."""
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    sim = simulate("eci", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    tgt0 = src_orbit.circular_orbit_state(R, MU_EARTH, math.radians(51.6))
    t = np.linspace(0, T, 8001)
    ctrl = PDController.critically_damped(6 * N, max_accel=0.05)
    rel, _, _, _ = src_val.high_fidelity_relative(tgt0, s0, t, src_pert.ForceModel.two_body(), ctrl)
    assert np.abs(sim.states(t)[:, :3] - rel[:, :3]).max() < 1e-5


def test_nerm_and_eci_agree_closed_loop_saturated():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([-8000.0, 3000.0, 2000.0, 0.5, -1.0, 0.2])
    cmd = CommandSpec("continuous", law=law)
    a = simulate("nerm", s0, cmd, T, n=N, R=R)
    b = simulate("eci", s0, cmd, T, n=N, R=R)
    t = np.linspace(0, T, 2001)
    assert np.abs(a.states(t)[:, :3] - b.states(t)[:, :3]).max() < 1e-4


def test_dv_accumulator_matches_quadrature():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([3000.0, 0, 0, 0, 0, 0])
    sim = simulate("cw", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    t = np.linspace(0, T, 200_001)
    _, app = sim.commands(t)
    from scipy.integrate import simpson
    assert abs(sim.dv_final - simpson(np.linalg.norm(app, axis=1), x=t)) < 1e-6


def test_sampled_control_holds_commands_and_converges_to_continuous():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([2000.0, 0, 0, 0, 0, 0])
    cont = simulate("cw", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    samp = simulate("cw", s0, CommandSpec("sampled", law=law, sample_period=10.0), T, n=N, R=R)
    assert len(samp.segments) == math.ceil(T / 10.0)
    # held command constant inside a segment
    _, app = samp.commands(np.array([20.0, 25.0, 29.999]))
    assert np.allclose(app[0], app[1]) and np.allclose(app[1], app[2])
    fine = simulate("cw", s0, CommandSpec("sampled", law=law, sample_period=0.25), T, n=N, R=R)
    t = np.linspace(0, T, 501)
    e10 = np.abs(samp.states(t)[:, :3] - cont.states(t)[:, :3]).max()
    e025 = np.abs(fine.states(t)[:, :3] - cont.states(t)[:, :3]).max()
    assert e025 < e10 / 10


def test_prescribed_command_replay_reproduces_cw_loop_in_cw():
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    s0 = np.array([4000.0, 0, 0, 0, 0, 0])
    cl = simulate("cw", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    pres = lambda t: tuple(law.command(*cl.states([t])[0]))
    pres_vec = lambda tt: law.applied_vec(cl.states(tt))
    rep = simulate("cw", s0, CommandSpec("prescribed", prescribed=pres, prescribed_vec=pres_vec), T, n=N, R=R)
    t = np.linspace(0, T, 401)
    assert np.abs(rep.states(t)[:, :3] - cl.states(t)[:, :3]).max() < 1e-4


# --- scenario / experiment ------------------------------------------------------

def test_scenario_ids_are_deterministic_and_distinct():
    a = axis_scenario("main", "+R", 1000.0, 6.0, 0.05)
    b = axis_scenario("main", "+R", 1000.0, 6.0, 0.05)
    c = axis_scenario("main", "-R", 1000.0, 6.0, 0.05)
    d = axis_scenario("main", "+R", 1000.0, 6.0, 0.05, velocity="em1")
    assert a.sid == b.sid
    assert len({a.sid, c.sid, d.sid}) == 3
    assert Scenario.from_dict(a.to_dict()).sid == a.sid


def test_paired_run_reproduces_manuscript_1km_along_track_metric():
    """Manuscript Table 4: 1 km along-track closed-loop peak difference 2.263 mm."""
    s = axis_scenario("test", "+V", 1000.0, 6.0, 0.05)
    row, _ = run_scenario(s, RunConfig(models=("cw", "nerm"), reference="nerm", crosscheck=None))
    assert row["cw_vs_nerm_max_pos_disc_m"] == pytest.approx(2.263e-3, rel=1e-3)
    assert row["outcome_terminal_cw"] == "both_succeed"


def test_terminal_arrival_time_is_a_boundary_crossing():
    s = axis_scenario("test", "+R", 3000.0, 6.0, 0.05)
    cfg = RunConfig(models=("cw", "nerm"), reference="nerm", crosscheck=None)
    from cwvalid.experiment import run_branches
    sims, _ = run_branches(s, cfg)
    row, _ = run_scenario(s, cfg)
    ta = row["nerm_terminal_arrival_time_s"]
    x = sims["nerm"].states([ta])[0]
    g = max(np.linalg.norm(x[:3]) / 1.0, np.linalg.norm(x[3:]) / 0.01)
    assert abs(g - 1.0) < 1e-6


def test_saturation_switch_events_lie_on_the_switching_surface():
    law = PDLaw.from_bandwidth(4 * N, 0.01)
    s0 = np.array([0.0, 0.0, 14471.330624, 0.0, 0.0, 0.0])
    sim = simulate("nerm", s0, CommandSpec("continuous", law=law), T, n=N, R=R)
    assert len(sim.sat_switch_times) >= 2
    S = sim.states(np.array(sim.sat_switch_times))
    g = np.linalg.norm(law.raw_vec(S), axis=1) - law.a_max
    assert np.abs(g).max() < 1e-9


def test_event_restarts_bring_formulations_together():
    """Production settings (events on) keep NERM and ECI within 1e-5 m on a
    multi-switch sustained-saturation case."""
    law = PDLaw.from_bandwidth(4 * N, 0.01)
    s0 = np.array([0.0, 0.0, 14471.330624, 0.0, 0.0, 0.0])
    cmd = CommandSpec("continuous", law=law)
    a = simulate("nerm", s0, cmd, T, n=N, R=R)
    b = simulate("eci", s0, cmd, T, n=N, R=R)
    t = np.linspace(0, T, 4001)
    assert np.abs(a.states(t)[:, :3] - b.states(t)[:, :3]).max() < 1e-5


# --- regression tests for issues found in the independent code review ---------

def test_rk4_branch_stops_at_altitude_floor():
    law = PDLaw.from_bandwidth(8 * N, 0.05)
    s0 = np.array([-17275.536473, 0.0, 0.0, 0.0, 0.0, 0.0])
    cmd = CommandSpec("continuous", law=law)
    dop = simulate("nerm", s0, cmd, T, n=N, R=R)
    rk4 = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverConfig(method="RK4", rk4_dt=T / 2000))
    assert dop.termination == "altitude_floor" and rk4.termination == "altitude_floor"
    assert abs(rk4.t_final - dop.t_final) <= T / 2000 + 1e-9


def test_trace_for_branch_without_segments():
    from cwvalid import metrics as M
    from cwvalid.simulate import SimResult
    law = PDLaw.from_bandwidth(6 * N, 0.05)
    sim = SimResult("cw", CommandSpec("continuous", law=law), N, R, T, SolverConfig(), None, [1000.0, 0, 0, 0, 0, 0])
    sim.status, sim.termination = "failed", "solver_failure"
    tr = M.make_trace(sim, M.common_grid(T))
    d = M.branch_diagnostics(sim, tr, T, M.OutcomeCriteria())
    assert tr.t.size == 1 and d["numerical_failure"] and not d["terminal_arrived"]


def test_legacy_grid_is_per_orbit():
    from cwvalid import metrics as M
    sim = simulate("cw", np.array([100.0, 0, 0, 0, 0, 0]), CommandSpec("none"), 2 * T, n=N, R=R)
    g = M.legacy_grid(sim, 2 * T, M.OutcomeCriteria())
    assert g.size == 4001 and abs(g[1] - T / 2000) < 1e-9


def test_runner_refuses_mixed_configuration_and_tolerates_partial_line(tmp_path):
    from cwvalid.runner import _read_jsonl, run_campaign
    s = axis_scenario("t", "+V", 200.0, 6.0, 0.05)
    cfg = {"models": ["cw", "nerm"], "reference": "nerm", "crosscheck": None}
    run_campaign(str(tmp_path), [s], cfg, workers=1, keep_hist=False, log=lambda *a: None)
    with open(tmp_path / "results.jsonl", "a") as fh:
        fh.write('{"sid": "partial')            # interrupted write
    assert len(_read_jsonl(str(tmp_path / "results.jsonl"), log=lambda *a: None)) == 1
    with pytest.raises(RuntimeError):
        run_campaign(str(tmp_path), [s], dict(cfg, dt_eval=1.0), workers=1, keep_hist=False, log=lambda *a: None)
