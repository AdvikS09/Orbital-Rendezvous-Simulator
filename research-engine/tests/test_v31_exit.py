"""v3.1 C01 regressions: storage of the held-engine step that contains a physical-domain exit.

v3.0 overwrote the exiting branch's step end with the start-of-step state, so the dense state at the
located exit time was up to 565 m above the 100 km floor and exit-case peak discrepancies were wrong.
These tests pin the corrected behaviour: the dense state at t_end lies on the floor and matches an
independent adaptive solution (state and event time), it is stable under step halving (also with a
saturated nonzero gain), values after t_end are masked, rows that do not exit are unaffected, pair
discrepancies use the common valid interval, a numerically failing branch is frozen at its last good
state, and a failed root bracket is flagged.
"""

import json
import math
import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cwvalid.constants import MU_EARTH
from cwvalid3.held_metrics import CaseView, case_summary, peak_discrepancy
import cwvalid3.sampled as SA
from cwvalid3.sampled import EXIT_AT_T0, HeldBatch, pd_matrix, rel_to_eci

from . import v31fx as FX

K0 = np.zeros((3, 6))
INTERIOR = [-290e3, 0.0, 0.0, -30.0, 0.0, 0.0]      # nerm/eci exit in step 53, hom:3 in step 52 (h = 2 s)
FIRST_STEP = [-299.95e3, 0.0, 0.0, -100.0, 0.0, 0.0]
BELOW = [-300.5e3, 0.0, 0.0, -10.0, 0.0, 0.0]        # starts below the floor
COMPANION = [100.0, -50.0, 20.0, 0.01, 0.0, -0.02]   # stays near the target
FLOOR_TOL_M = 1e-6


def batch(rel0s, h=2.0, Ts=2.0, lambdas=(3.0,), horizon=0.05, run=True, K=None, amax=0.05):
    N = len(rel0s)
    K = np.broadcast_to(K0 if K is None else K, (N, 3, 6))
    hb = HeldBatch(rel0s, K, [amax] * N, 400e3, Ts, horizon_orbits=horizon, h=h, models=("cw", "nerm", "eci"),
                   estimators=("A_H", "B_H"), lambdas=lambdas)
    return hb.run() if run else hb


def floor_gap(hb, i, b, t):
    y = hb.dense(i, [t])
    return float(hb.radius(y, b)[0] - hb.r_floor)


@pytest.fixture(scope="module")
def interior():
    return batch([INTERIOR, COMPANION])


def test_interior_exit_dense_state_on_floor(interior):
    hb = interior
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit"
        j = hb.exit_step[b][0]
        te = hb.t_end[b][0]
        assert hb.t[j] < te < hb.t[j + 1]
        assert not hb.exit_fallback[b][0]
        assert abs(floor_gap(hb, 0, b, te)) < FLOOR_TOL_M
        # the stored step end is the trial endpoint (below the floor), not the start-of-step state
        cols = hb.cols(b)
        assert not np.array_equal(hb.Y[j + 1, 0, cols], hb.Y[j, 0, cols])
        assert hb.radius(hb.Y[j + 1, 0], b) < hb.r_floor
        # frozen afterwards
        assert np.array_equal(hb.Y[j + 2, 0, cols], hb.Y[j + 1, 0, cols])


def test_dense_masked_after_event_and_finite_before(interior):
    hb = interior
    for b in ("nerm", "eci", "hom:3"):
        te = hb.t_end[b][0]
        cols = hb.cols(b)
        y = hb.dense(0, [te - 0.25, te, te + 1e-6, te + 3.0, hb.T])
        assert np.all(np.isfinite(y[:2][:, cols]))
        assert np.all(np.isnan(y[2:][:, cols]))
    # the cw branch and the estimators completed: never masked
    y = hb.dense(0, np.linspace(0.0, hb.T, 101))
    assert np.all(np.isfinite(y[:, hb.cols("cw")]))
    assert np.all(np.isfinite(y[:, hb.cols("A_H")]))
    # the unmasked view still exists for diagnostics
    assert np.all(np.isfinite(hb.dense(0, [hb.T], mask=False)))


def test_exit_time_matches_adaptive_reference(interior):
    """With zero gain the eci chaser is a two-body orbit: locate the floor crossing independently."""
    hb = interior
    t0 = hb.Y[0, 0, hb.c_eci:hb.c_eci + 6]
    rc, vc = rel_to_eci(t0[None, :3], t0[None, 3:], np.asarray(INTERIOR)[None, :])

    def f(t, y):
        r = y[:3]
        return np.r_[y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3]

    def ev(t, y):
        return np.linalg.norm(y[:3]) - hb.r_floor
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(f, (0.0, hb.T), np.r_[rc[0], vc[0]], method="DOP853", rtol=1e-13, atol=1e-9, events=ev)
    t_ref = float(sol.t_events[0][0])
    assert abs(hb.t_end["eci"][0] - t_ref) < 1e-7


def test_exit_in_first_step():
    hb = batch([FIRST_STEP])
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit"
        assert hb.exit_step[b][0] == 0 and 0.0 < hb.t_end[b][0] < hb.t[1]
        assert abs(floor_gap(hb, 0, b, hb.t_end[b][0])) < FLOOR_TOL_M


def test_initial_state_below_floor_is_an_exit_at_t0():
    hb = batch([BELOW, COMPANION])
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit" and hb.t_end[b][0] == 0.0
        assert hb.exit_step[b][0] == EXIT_AT_T0 and not hb.exit_fallback[b][0]
        assert np.all(np.isnan(hb.dense(0, [0.5, hb.T])[:, hb.cols(b)]))
        assert hb.status[b][1] == "completed"
    pk = peak_discrepancy(CaseView(hb, 0), "cw", "eci")
    assert pk["common_end_s"] == 0.0 and math.isfinite(pk["peak_m"])


def test_exit_at_a_step_node():
    """Set the floor to the stored radius at a node: the event is located exactly at that node."""
    ref = batch([INTERIOR], lambdas=())
    js = 40
    assert ref.exit_step["nerm"][0] > js
    rn = float(ref.radius(ref.Y[js, 0], "nerm"))
    hb = batch([INTERIOR], lambdas=(), run=False)
    hb.r_floor = rn
    hb.run()
    assert hb.status["nerm"][0] == "domain_exit"
    assert hb.t_end["nerm"][0] == hb.t[js] and hb.exit_step["nerm"][0] == js
    assert not hb.exit_fallback["nerm"][0]
    assert np.array_equal(hb.Y[js, 0], ref.Y[js, 0])
    assert abs(floor_gap(hb, 0, "nerm", hb.t[js])) <= 1e-9


def test_branches_exiting_in_different_steps(interior):
    hb = interior
    jh, jn = hb.exit_step["hom:3"][0], hb.exit_step["nerm"][0]
    assert jh < jn
    th, tn = hb.t_end["hom:3"][0], hb.t_end["nerm"][0]
    # the later branch is still valid after the earlier one has ended
    y = hb.dense(0, [0.5 * (th + tn)])
    assert np.all(np.isnan(y[:, hb.cols("hom:3")])) and np.all(np.isfinite(y[:, hb.cols("nerm")]))
    cv = CaseView(hb, 0)
    pk = peak_discrepancy(cv, "hom:3", "nerm")
    assert pk["common_end_s"] == th and pk["t_peak_s"] <= th + 1e-9 and math.isfinite(pk["peak_m"])


def test_completed_companion_unaffected_by_exiting_row(interior):
    """Freezing is row-wise: a companion row is bit-identical whether its neighbour exits or not.
    (A batch of a different size may differ in the last bits through BLAS blocking: compare same shapes.)"""
    other = batch([[-250e3, 0.0, 0.0, -10.0, 0.0, 0.0], COMPANION])   # neighbour completes
    hb = interior
    assert all(other.status[b][0] == "completed" for b in other.branch_names())
    for b in hb.branch_names():
        assert hb.status[b][1] == "completed" and hb.t_end[b][1] == hb.T and hb.exit_step[b][1] == -1
    assert np.array_equal(hb.Y[:, 1, :], other.Y[:, 1, :])
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    s_b, _ = case_summary(hb, 1, pairs=pairs)
    s_o, _ = case_summary(other, 1, pairs=pairs)
    assert json.dumps(s_b, sort_keys=True) == json.dumps(s_o, sort_keys=True)
    alone = batch([COMPANION])
    assert np.allclose(hb.Y[:, 1, :], alone.Y[:, 0, :], rtol=1e-14, atol=1e-18)


def test_step_halving_agrees_at_the_event():
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    s1, _ = case_summary(batch([INTERIOR], h=2.0), 0, pairs=pairs)
    s2, _ = case_summary(batch([INTERIOR], h=1.0), 0, pairs=pairs)
    for b in ("nerm", "eci"):
        assert abs(s1[f"{b}_t_end_s"] - s2[f"{b}_t_end_s"]) < 1e-7
    for p in ("cw_vs_eci", "cw_vs_nerm"):
        assert abs(s1[f"{p}_peak_m"] - s2[f"{p}_peak_m"]) < 1e-5
        assert s1[f"{p}_t_peak_s"] == s1[f"{p}_common_end_s"]
    assert s1["nerm_vs_eci_peak_m"] < 1e-4 and s2["nerm_vs_eci_peak_m"] < 1e-4


def test_common_interval_discrepancy_matches_brute_force(interior):
    hb = interior
    cv = CaseView(hb, 0)
    pk = peak_discrepancy(cv, "cw", "eci")
    tc = min(hb.t_end["cw"][0], hb.t_end["eci"][0])
    assert pk["common_end_s"] == tc and pk["t_peak_s"] <= tc + 1e-9
    tt = np.linspace(0.0, tc, 20001)
    d = np.linalg.norm(cv.rel_from_rows("cw", hb.dense(0, tt))[:, :3] - cv.rel_from_rows("eci", hb.dense(0, tt))[:, :3],
                       axis=1)
    assert np.all(np.isfinite(d))
    assert pk["peak_m"] >= d.max() - 1e-6 and pk["peak_m"] <= d.max() + 1e-3
    # the monitoring grid of the exited reference is masked after the event
    g = cv.grid("eci")
    assert np.all(np.isnan(g[cv.tm > tc + 1e-9])) and np.all(np.isfinite(g[cv.tm <= tc]))


def _campaign_record(cell, split, index):
    """Planned case and stored v3.0 record, as exact fixture lines of the original files (v3.1.1 R04; never skipped)."""
    man = FX.planned(f"{cell}_{split}", index)
    return man, FX.result_for(f"{cell}_{split}", man["run_id"])


def test_reproducer_D4_dev_56():
    """Audit reproducer: D4 dev index 56 (Cbfad11507250fd59), exit inside step [3916, 3918] s."""
    man, res = _campaign_record("D4", "dev", 56)
    assert man["case_id"] == "Cbfad11507250fd59"
    hb = HeldBatch([man["scenario"]["rel0"]], np.array([man["K"]]), [man["scenario"]["a_max"]], 400e3, 10.0,
                   horizon_orbits=1.0, h=2.0, models=("cw", "nerm", "eci"), estimators=("A_H", "B_H", "C_H")).run()
    for b in ("nerm", "eci"):
        assert hb.status[b][0] == res[f"{b}_status"] == "domain_exit"
        assert abs(hb.t_end[b][0] - res[f"{b}_t_end_s"]) < 1e-8
        assert hb.t[hb.exit_step[b][0]] == 3916.0
        assert abs(floor_gap(hb, 0, b, hb.t_end[b][0])) < FLOOR_TOL_M
    sm, _ = case_summary(hb, 0, pairs=(("cw", "eci"), ("cw", "nerm"), ("nerm", "eci")))
    # stored v3.0 value 55164.70 m (start-of-step endpoint); corrected value from the audit replay
    assert abs(res["cw_vs_eci_peak_m"] - 55164.7034834116) < 1e-6
    assert abs(sm["cw_vs_eci_peak_m"] - 55166.86759277691) < 1e-3
    assert sm["nerm_vs_eci_peak_m"] < 1e-5
    for k in ("A_H_peak_m", "B_H_peak_m", "C_H_peak_m", "cw_final_range_m"):
        assert abs(sm[k] - res[k]) <= 1e-9 * max(1.0, abs(res[k]))


def test_dense_state_at_event_matches_adaptive_solution(interior):
    """The chaser's dense ECI state at the located exit equals an independent adaptive solution at that time
    (v3.0: the dense curve on the exit step was built on the start-of-step state, metres off)."""
    hb = interior
    t0 = hb.Y[0, 0, hb.c_eci:hb.c_eci + 6]
    rc, vc = rel_to_eci(t0[None, :3], t0[None, 3:], np.asarray(INTERIOR)[None, :])

    def f(t, y):
        r = y[:3]
        return np.r_[y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3]
    te = hb.t_end["eci"][0]
    sol = solve_ivp(f, (0.0, te), np.r_[rc[0], vc[0]], method="DOP853", rtol=1e-13, atol=1e-9, dense_output=True)
    y_ref = sol.sol(te)
    y = hb.dense(0, [te])[0]
    c = hb.c_eci
    assert np.linalg.norm(y[c + 6:c + 9] - y_ref[:3]) < 1e-4
    assert np.linalg.norm(y[c + 9:c + 12] - y_ref[3:]) < 1e-6


def test_numerical_failure_frozen_at_last_good_state():
    """A discretely unstable held loop (omega_n = 0.5 rad/s, Ts = 10 s): the failing branch is frozen at the
    last good state, t_end is that step start, it is masked afterwards and it is not a domain exit."""
    hb = HeldBatch([[1000.0, 0, 0, 0, 0, 0]], pd_matrix(0.5)[None], [None], 400e3, 10.0, h=2.0, models=("cw", "nerm"),
                   estimators=("A_H",)).run()
    assert hb.status["nerm"][0] in ("domain_exit", "numerical_failure")   # nerm leaves the domain first here
    for b in ("cw",):
        assert hb.status[b][0] == "numerical_failure"
        te = hb.t_end[b][0]
        j = int(np.searchsorted(hb.t, te))
        assert hb.t[j] == te
        cols = hb.cols(b)
        assert np.array_equal(hb.Y[j + 1, 0, cols], hb.Y[j, 0, cols])
        assert np.all(np.isfinite(hb.Y[j, 0, cols]))
        assert np.all(np.isnan(hb.dense(0, [te + 1.0])[:, cols]))
        assert hb.exit_step[b][0] == -1


def test_failed_root_bracket_is_flagged(monkeypatch):
    def no_bracket(*a, **k):
        raise ValueError("f(a) and f(b) must have different signs")
    monkeypatch.setattr(SA, "brentq", no_bracket)
    hb = batch([INTERIOR], lambdas=())
    for b in ("nerm", "eci"):
        j = hb.exit_step[b][0]
        assert hb.status[b][0] == "domain_exit" and hb.exit_fallback[b][0]
        assert hb.t_end[b][0] == hb.t[j + 1]


def test_saturated_gain_exit_is_stable_under_step_halving():
    """Held-command feedback active inside the exit step (PD gain, 0.05 m/s^2 cap, saturated)."""
    K = pd_matrix(6 * 0.00113)
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    s1, _ = case_summary(batch([INTERIOR], h=2.0, lambdas=(), K=K), 0, pairs=pairs)
    hb2 = batch([INTERIOR], h=1.0, lambdas=(), K=K)
    s2, _ = case_summary(hb2, 0, pairs=pairs)
    assert hb2.status["eci"][0] == "domain_exit" and hb2.sat_at_update[0].any()
    for b in ("nerm", "eci"):
        assert abs(floor_gap(hb2, 0, b, hb2.t_end[b][0])) < FLOOR_TOL_M
        assert abs(s1[f"{b}_t_end_s"] - s2[f"{b}_t_end_s"]) < 1e-7
    assert abs(s1["cw_vs_eci_peak_m"] - s2["cw_vs_eci_peak_m"]) < 1e-5
