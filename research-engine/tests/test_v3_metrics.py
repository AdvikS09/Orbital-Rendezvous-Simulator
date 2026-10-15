"""Tests for cwvalid3.metrics3 (P02 numerical assurance) on synthetic trajectories with known answers.

Each test names the property it checks (a)-(i) from the work-package brief. The production
refinement (cwvalid.metrics) is used only as a comparator; nothing in it is changed.
"""

import math
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cwvalid.metrics import _refine_extreme, common_grid  # noqa: E402  (production comparator)
from cwvalid3.metrics3 import (  # noqa: E402
    arrival_label, build_grid, coverage_check, crossings, discrepancy_fun, peak_discrepancy, robust_peak,
    terminal_g_fun, threshold_label)


def production_peak(fun, t_end, dt=0.5):
    """The production peak: 0.5 s grid plus t_end, refined only around the grid argmax."""
    t = common_grid(t_end, dt)
    v = np.array([fun(s) for s in t])
    return _refine_extreme(fun, t, int(np.argmax(v)), maximize=True)


def local_truth(fun, a, b):
    """Reference maximum of fun on a short interval known to contain only the target peak."""
    r = minimize_scalar(lambda u: -fun(a + u), bounds=(0.0, b - a), method="bounded",
                        options={"xatol": 1e-13, "maxiter": 2000})
    return a + r.x, -r.fun


# ---------------------------------------------------------------------------
# (a) a short excursion between 0.5 s samples
# ---------------------------------------------------------------------------

BUMP_T, BUMP_SIGMA, BUMP_A = 10.125, 0.05, 1.5   # odd multiple of 0.125 s: off the 0.5 s and 0.25 s grids


def bump_fun(t):
    return math.sin(2.0 * math.pi * t / 10.0) + BUMP_A * math.exp(-0.5 * ((t - BUMP_T) / BUMP_SIGMA) ** 2)


def test_a_short_excursion_missed_by_coarse_grid_and_flagged():
    t_true, f_true = local_truth(bump_fun, 10.0, 10.25)
    assert f_true > 1.5                      # the bump dominates the base maximum of 1.0

    # the production procedure on the 0.5 s grid never sees it
    _, f_prod = production_peak(bump_fun, 20.0)
    assert f_prod == pytest.approx(1.0, abs=1e-9)

    res = robust_peak(bump_fun, 0.0, 20.0)   # grid 0.5 s, refinements 0.5, 0.25, 0.125 s
    assert res.grid_only == pytest.approx(1.0, abs=1e-12)
    assert res.peak_primary == pytest.approx(1.0, abs=1e-9)
    v05, v025, v0125 = res.refinement_values
    assert v05 == pytest.approx(1.0, abs=1e-9)
    assert v025 == pytest.approx(1.0, abs=1e-9)      # 0.25 s grid still blind to it
    assert v0125 == pytest.approx(f_true, abs=1e-9)  # 0.125 s grid samples the bump
    # the last two refinements disagree, so the routine reports the risk
    assert res.converged is False
    assert res.last_change > 0.5
    assert res.uncertainty >= res.last_change
    # the best value found is reported, with its time
    assert res.peak == pytest.approx(f_true, abs=1e-9)
    assert res.t_peak == pytest.approx(t_true, abs=1e-5)
    assert res.resolution_assumption_s == 0.125

    # one more halving: the last two levels now agree on the bump
    res2 = robust_peak(bump_fun, 0.0, 20.0, monitor_refinements=(0.5, 0.25, 0.125, 0.0625))
    assert res2.converged is True
    assert res2.peak == pytest.approx(f_true, abs=1e-9)
    assert res2.uncertainty < 1e-8

    # the threshold decision inherits the flagged uncertainty
    assert threshold_label(res.peak, res.uncertainty, 1.2) == "unresolved"
    assert threshold_label(res2.peak, res2.uncertainty, 1.2) == "violation"


def test_a_agreement_between_levels_is_not_proof():
    """A spike narrower than every grid leaves no trace in any sample: all levels agree and the flag
    is True, yet the peak is wrong. This is the documented limit of a sampled checker."""
    ts, sig = 10.046875, 0.002          # 3/64 s past a 0.125 s sample

    def f(t):
        return math.sin(2.0 * math.pi * t / 10.0) + 1.5 * math.exp(-0.5 * ((t - ts) / sig) ** 2)

    _, f_true = local_truth(f, ts - 0.01, ts + 0.01)
    res = robust_peak(f, 0.0, 20.0)
    assert f_true > 1.5
    assert res.converged is True
    assert res.peak == pytest.approx(1.0, abs=1e-9)
    assert res.resolution_assumption_s == 0.125     # the assumption the result rests on is stated
    assert "cannot be excluded" in res.note


# ---------------------------------------------------------------------------
# (b) close local maxima
# ---------------------------------------------------------------------------


def two_peaks(t):
    a = 1.000 * math.exp(-((t - 10.2) / 0.40) ** 2)   # sampled near its top at 10.0 s
    b = 1.001 * math.exp(-((t - 12.2) / 0.25) ** 2)   # higher but narrower: poorly sampled
    decoy = 0.7 * math.exp(-((t - 20.0) / 0.40) ** 2)
    return a + b + decoy


def test_b_close_maxima_true_global_max_returned():
    t = build_grid(0.0, 30.0, 0.5)
    v = np.array([two_peaks(s) for s in t])
    assert t[int(np.argmax(v))] == pytest.approx(10.0)   # grid argmax sits on the lower peak

    _, f_prod = production_peak(two_peaks, 30.0)
    assert f_prod == pytest.approx(1.000, abs=1e-6)     # production refines the wrong peak

    res = robust_peak(two_peaks, 0.0, 30.0)
    assert res.peak == pytest.approx(1.001, abs=1e-9)
    assert res.t_peak == pytest.approx(12.2, abs=1e-3)
    assert res.peak_primary == pytest.approx(1.001, abs=1e-9)  # found on the 0.5 s grid already
    assert res.n_candidates == 3                         # peak A, peak B and the decoy
    assert res.converged is True

    # the top-k rule alone also catches it when the relative band admits only the grid maximum
    res_k = robust_peak(two_peaks, 0.0, 30.0, rel_band=1.0, top_k=5)
    assert res_k.peak == pytest.approx(1.001, abs=1e-9)
    # with both safeguards switched off the routine behaves like production and misses it
    res_off = robust_peak(two_peaks, 0.0, 30.0, rel_band=1.0, top_k=0)
    assert res_off.peak == pytest.approx(1.000, abs=1e-6)
    assert res_off.n_candidates == 1


# ---------------------------------------------------------------------------
# (c) tangency without a sign change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t0", [0.0, 0.2])   # 5 s on the grid, and off it
def test_c_exact_touch_is_tangency_not_crossing(t0):
    cr = crossings(lambda t: (t - 5.0) ** 2, t0, 10.0)
    assert len(cr.entries) == 0 and len(cr.exits) == 0
    assert len(cr.tangencies) == 1
    ev = cr.tangencies[0]
    assert ev.t_lo - 1e-6 <= 5.0 <= ev.t_hi + 1e-6
    assert ev.g_scale < 1e-9
    assert ev.basis == ("grid_point" if t0 == 0.0 else "near_tangency")


@pytest.mark.parametrize("t0", [0.0, 0.2])
def test_c_dip_below_tolerance_is_tangency_not_crossing(t0):
    cr = crossings(lambda t: (t - 5.0) ** 2 - 1e-12, t0, 10.0)
    assert len(cr.entries) == 0 and len(cr.exits) == 0
    assert len(cr.tangencies) == 1
    ev = cr.tangencies[0]
    assert ev.n_roots == 2                      # both zeros were located, then merged
    assert ev.t_lo <= 5.0 <= ev.t_hi
    assert ev.g_scale == pytest.approx(1e-12, rel=1e-3)


def test_c_resolved_dip_between_samples_found_by_refinement():
    """Contrast: a dip of depth 1e-4 is a genuine entry/exit pair. Off the grid the samples show no
    sign change; the refinement finds it and says so through the basis."""
    cr = crossings(lambda t: (t - 5.0) ** 2 - 1e-4, 0.2, 10.0)
    assert [e.kind for e in cr.events] == ["entry", "exit"]
    assert cr.entries[0].t == pytest.approx(4.99, abs=1e-8)
    assert cr.exits[0].t == pytest.approx(5.01, abs=1e-8)
    assert cr.n_from_refinement == 2 and cr.n_bracketed == 0
    assert cr.absence_basis == "sampled"


def test_c_tangency_after_entry_unresolved_only_if_it_matters():
    T = 1000.0
    # inside throughout, touching the boundary from inside at 800 s: a brief exit there would leave
    # 200 s < 300 s of dwell, so the label cannot be decided
    lab = arrival_label(lambda t: -((t - 800.0) ** 2) / 1e4, T, T)
    assert lab.status == "unresolved_tangency" and lab.arrived is None
    assert lab.nominal_arrived is True
    # the same touch at 500 s cannot change the label
    lab2 = arrival_label(lambda t: -((t - 500.0) ** 2) / 1e4, T, T)
    assert lab2.status == "arrived" and lab2.n_tangencies_after_entry == 1


# ---------------------------------------------------------------------------
# (d) threshold plateau
# ---------------------------------------------------------------------------


def test_d_threshold_plateau_unresolved():
    thr = 2.5

    def d(t):
        return min(thr, 1.5 * thr * math.exp(-((t - 5.0) / 2.0) ** 2))   # equals thr on ~[3.73, 6.27]

    res = robust_peak(d, 0.0, 10.0)
    assert res.peak == thr
    assert 3.7 <= res.t_peak <= 6.3
    assert res.n_candidates == 1           # the plateau is one candidate, not one per sample
    assert threshold_label(res.peak, res.uncertainty, thr) == "unresolved"
    assert threshold_label(res.peak, 0.0, thr) == "unresolved"
    u = 1e-6
    assert threshold_label(thr + 0.5 * u, u, thr) == "unresolved"
    assert threshold_label(thr + 2.0 * u, u, thr) == "violation"
    assert threshold_label(thr - 2.0 * u, u, thr) == "ok"
    assert threshold_label(float("nan"), u, thr) == "unresolved"
    with pytest.raises(ValueError):
        threshold_label(thr, -1.0, thr)


# ---------------------------------------------------------------------------
# (e) crossings at t = 0 and at t = T
# ---------------------------------------------------------------------------


def test_e_entry_exactly_at_start():
    T = 1000.0
    cr = crossings(lambda t: -t, 0.0, T)
    assert len(cr.entries) == 1 and len(cr.exits) == 0
    e = cr.entries[0]
    assert e.t == 0.0 and e.width == 0.0 and e.basis == "grid_point"
    assert cr.start_on_boundary
    lab = arrival_label(lambda t: -t, T, T)
    assert lab.status == "arrived" and lab.t_entry == 0.0 and lab.dwell_s == T


def test_e_boundary_contact_at_start_without_entry():
    cr = crossings(lambda t: t, 0.0, 10.0)      # on the boundary at t0, then outside
    assert len(cr.entries) == 0 and len(cr.tangencies) == 1
    assert cr.tangencies[0].t == 0.0


@pytest.mark.parametrize("T", [1000.0, 1000.3])   # T on and off the 0.5 s grid
def test_e_exit_exactly_at_end(T):
    cr = crossings(lambda t: t - T, 0.0, T)
    assert len(cr.exits) == 1 and len(cr.entries) == 0
    assert cr.exits[0].t == T and cr.end_on_boundary
    lab = arrival_label(lambda t: t - T, T, T)
    assert lab.status == "unresolved_tangency"     # g(T) = 0: inside at T is undecidable
    assert lab.arrived is None


def test_e_crossings_just_inside_the_ends():
    T = 1000.0
    eps_t = 1e-7
    cr = crossings(lambda t: t - eps_t, 0.0, T)          # inside at t0, leaves after 1e-7 s
    assert len(cr.exits) == 1 and cr.exits[0].t == pytest.approx(eps_t, abs=2e-9)
    assert cr.exits[0].basis == "bracketed"
    g = lambda t: (T - eps_t) - t                         # enters 1e-7 s before T
    cr2 = crossings(g, 0.0, T)
    assert len(cr2.entries) == 1 and cr2.entries[0].t == pytest.approx(T - eps_t, abs=2e-9)
    lab = arrival_label(g, T, T)
    assert lab.status == "not_arrived" and lab.dwell_s < 1e-6


# ---------------------------------------------------------------------------
# (f) dwell just above and below the 300 s deadline
# ---------------------------------------------------------------------------


def decaying_state(t_e, tau=200.0):
    """Radial approach r(t) = exp(-(t - t_e)/tau): speed r/tau = 0.005 m/s at the 1 m boundary,
    so g = r - 1 and the entry is exactly t_e."""
    def x(t):
        r = math.exp(-(t - t_e) / tau)
        return np.array([r, 0.0, 0.0, -r / tau, 0.0, 0.0])
    return x


@pytest.mark.parametrize("offset, status", [(+0.01, "arrived"), (-0.01, "not_arrived"),
                                            (+1e-4, "unresolved_near_deadline"),
                                            (-1e-4, "unresolved_near_deadline")])
def test_f_dwell_near_deadline(offset, status):
    T, dwell = 1000.0, 300.0
    t_e = T - dwell - offset          # dwell = 300 + offset
    for g in (terminal_g_fun(decaying_state(t_e)), lambda t: 0.01 * (t_e - t)):
        lab = arrival_label(g, T, T, dwell)
        assert lab.status == status
        assert lab.t_entry == pytest.approx(t_e, abs=1e-8)
        assert lab.dwell_margin_s == pytest.approx(offset, abs=1e-8)
        assert lab.nominal_arrived is (offset >= 0)
        assert lab.decision_margin_s == pytest.approx(1e-3)
    # an explicit, wider decision margin turns the +/-0.01 s cases into unresolved ones too
    lab = arrival_label(lambda t: 0.01 * (t_e - t), T, T, dwell, decision_margin_s=0.02)
    assert lab.status == "unresolved_near_deadline"


def test_f_last_entry_is_used():
    T = 1000.0
    g = lambda t: -(t - 100.0) * (t - 400.0) * (t - 600.0) / 1e7   # in at 100, out at 400, in at 600
    lab = arrival_label(g, T, T)
    assert lab.n_entries == 2 and lab.n_exits == 1
    assert lab.t_entry == pytest.approx(600.0, abs=1e-8)
    assert lab.status == "arrived" and lab.dwell_margin_s == pytest.approx(100.0, abs=1e-8)
    assert arrival_label(g, T, 850.0).status == "not_arrived"


# ---------------------------------------------------------------------------
# (g) repeated switch/update times
# ---------------------------------------------------------------------------


def test_g_duplicate_extra_times_counted_once():
    extra = [3.3, 3.3, 3.3, 7.0, 7.0, 2.5, 3.3 + 1e-15, -1.0, 11.0]
    t = build_grid(0.0, 10.0, 0.5, extra)
    assert t.size == 22                          # 21 grid points + 3.3; the rest merge or lie outside
    assert np.all(np.diff(t) > 0)
    assert np.count_nonzero(np.isclose(t, 3.3, atol=1e-12)) == 1

    f = lambda s: 1.0 - (s - 3.3) ** 2
    r1 = robust_peak(f, 0.0, 10.0, extra_times=extra)
    r0 = robust_peak(f, 0.0, 10.0, extra_times=[3.3])
    assert r1.peak == 1.0 and r1.t_peak == 3.3
    assert r1.n_candidates == r0.n_candidates == 1
    assert [lv.n_samples for lv in r1.levels] == [lv.n_samples for lv in r0.levels]

    g = lambda s: 3.3 - s
    c1 = crossings(g, 0.0, 10.0, extra_times=extra)
    c0 = crossings(g, 0.0, 10.0, extra_times=[3.3])
    assert len(c1.entries) == 1 and len(c1.events) == 1
    assert c1.entries[0].t == 3.3 and c1.entries[0].basis == "grid_point"
    assert c1.events == c0.events
    lab = arrival_label(g, 10.0, 10.0, dwell=5.0, extra_times=extra)
    assert lab.status == "arrived" and lab.n_entries == 1


# ---------------------------------------------------------------------------
# (h) incomplete dense output
# ---------------------------------------------------------------------------


def test_h_incomplete_dense_output():
    T = 1000.0
    segs = [(0.0, 400.0), (400.0, T - 1.0)]
    cov = coverage_check(segs, T)
    assert cov.ok is False and cov.first_gap == (T - 1.0, T) and cov.covered_until == T - 1.0

    def g_never(t):
        raise AssertionError("g must not be evaluated beyond the completed horizon")

    lab = arrival_label(g_never, T, T, segments=segs)    # claimed complete, but the segments say otherwise
    assert lab.status == "incomplete_horizon" and lab.arrived is False
    assert lab.coverage.first_gap == (T - 1.0, T)
    lab2 = arrival_label(g_never, T - 1.0, T)
    assert lab2.status == "incomplete_horizon"

    gap = coverage_check([(0.0, 100.0), (100.5, T)], T)
    assert gap.ok is False and gap.first_gap == (100.0, 100.5)
    assert coverage_check([(0.0, 100.0), (100.0 + 1e-13, T)], T).ok          # within 32 eps T
    assert coverage_check([(500.0, T), (0.0, 600.0), (100.0, 200.0)], T).ok   # unsorted, overlapping
    assert coverage_check([(1e-3, T)], T).first_gap == (0.0, 1e-3)
    with pytest.raises(ValueError):
        coverage_check([(0.0, T), (5.0, 4.0)], T)

    full = arrival_label(lambda t: -1.0, T, T, segments=[(0.0, 500.0), (500.0, T)])
    assert full.status == "arrived" and full.coverage.ok


# ---------------------------------------------------------------------------
# (i) agreement with the production refinement on a smooth case
# ---------------------------------------------------------------------------


def smooth_disc(t):
    return 2.0 + 0.5 * math.sin(2.0 * math.pi * t / 5400.0 + 0.3) + 0.1 * math.cos(2.0 * math.pi * t / 900.0)


@pytest.mark.parametrize("t_end", [3000.0, 3000.3])
def test_i_matches_production_on_smooth_case(t_end):
    t_prod, f_prod = production_peak(smooth_disc, t_end)
    res = robust_peak(smooth_disc, 0.0, t_end)
    assert abs(res.peak - f_prod) <= 1e-9
    assert abs(res.peak_primary - f_prod) <= 1e-9
    assert res.t_peak == pytest.approx(t_prod, abs=0.05)
    assert res.converged is True
    # same primary grid as production
    assert np.array_equal(build_grid(0.0, t_end, 0.5), common_grid(t_end, 0.5))


def test_i_grid_matches_held_metrics_monitoring_grid():
    from cwvalid3.held_metrics import monitoring_grid
    for T in (5400.0, 5400.25, 1234.5678):
        assert np.array_equal(build_grid(0.0, T, 0.5), monitoring_grid(T))


# ---------------------------------------------------------------------------
# builders and input checks
# ---------------------------------------------------------------------------


def test_peak_discrepancy_from_two_dense_states_uses_common_horizon():
    rho_a = lambda t: np.array([t, 0.0, 0.0, 1.0, 0.0, 0.0])
    rho_b = lambda t: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = peak_discrepancy(rho_a, rho_b, 800.0, 650.0, T=1000.0)
    assert res.t1 == 650.0 and res.peak == 650.0 and res.t_peak == 650.0
    assert discrepancy_fun(rho_a, rho_b)(3.0) == 3.0


def test_vectorized_matches_scalar():
    fv = np.vectorize(bump_fun)
    a = robust_peak(bump_fun, 0.0, 20.0)
    b = robust_peak(fv, 0.0, 20.0, vectorized=True)
    assert a.refinement_values == pytest.approx(b.refinement_values, abs=1e-12)
    gv = lambda t: 0.01 * (700.0 - np.asarray(t))
    assert crossings(gv, 0.0, 1000.0, vectorized=True).entries[0].t == pytest.approx(700.0, abs=1e-9)


def test_nonfinite_samples_raise():
    with pytest.raises(ValueError):
        robust_peak(lambda t: float("nan") if t > 5 else 1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        crossings(lambda t: float("inf"), 0.0, 10.0)
