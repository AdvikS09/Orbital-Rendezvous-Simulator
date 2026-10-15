"""Tests for the v2 extension: estimator variants, fitting edge cases, freeze hashes,
development/test isolation and failure-aware accounting."""

import json
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "v2"))

from cwvalid.estimate import first_order_discrepancy  # noqa: E402
from cwvalid.estimate_v2 import estimator_variants, residual_only_diagnostic, unsaturated_jacobian_estimate  # noqa: E402
from cwvalid.experiment import RunConfig, make_law, run_branches  # noqa: E402
from cwvalid.scenario import axis_scenario  # noqa: E402

RES = os.path.join(os.path.dirname(REPO), "results")


def _cw(axis, sep, wn=6.0, am=0.05):
    scn = axis_scenario("t", axis, sep, wn, am)
    sims, _ = run_branches(scn, RunConfig(models=("cw",), reference="cw", crosscheck=None))
    return scn, sims["cw"]


def test_variant_b_equals_a_when_never_saturated():
    scn, sim = _cw("+R", 300.0)
    assert not sim.sat_switch_times
    law = make_law(scn)
    a = first_order_discrepancy(sim, law, scn.n, scn.R, scn.horizon)["d1_peak_m"]
    b = unsaturated_jacobian_estimate(sim, law, scn.n, scn.R, scn.horizon)["peak_m"]
    assert b == pytest.approx(a, rel=1e-12)


def test_variant_b_differs_from_a_when_saturated():
    scn, sim = _cw("+R", 3000.0)
    assert sim.sat_switch_times
    law = make_law(scn)
    a = first_order_discrepancy(sim, law, scn.n, scn.R, scn.horizon)["d1_peak_m"]
    b = unsaturated_jacobian_estimate(sim, law, scn.n, scn.R, scn.horizon)["peak_m"]
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(b / a - 1.0) > 1e-3


def test_variant_c_matches_duhamel_quadrature():
    """C solves e'' = r(t), e(0) = e'(0) = 0, i.e. e(t) = int_0^t (t - s) r(s) ds."""
    from cwvalid.dynamics import acc_cw, acc_nerm
    scn, sim = _cw("+V", 2000.0)
    c = residual_only_diagnostic(sim, scn.n, scn.R, scn.horizon)
    t = np.linspace(0.0, scn.horizon, 40001)
    X = sim.states(t)
    r = np.array([np.subtract(acc_nerm(scn.n, scn.R, *x), acc_cw(scn.n, scn.R, *x)) for x in X])
    v = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t)[:, None], axis=0)])
    e = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t)[:, None], axis=0)])
    peak = np.linalg.norm(e, axis=1).max()
    assert c["peak_m"] == pytest.approx(peak, rel=1e-4)


def test_variant_c_scales_quadratically_with_separation():
    s1 = _cw("+H", 100.0)
    s2 = _cw("+H", 200.0)
    c1 = residual_only_diagnostic(s1[1], s1[0].n, s1[0].R, s1[0].horizon)["peak_m"]
    c2 = residual_only_diagnostic(s2[1], s2[0].n, s2[0].R, s2[0].horizon)["peak_m"]
    assert c2 / c1 == pytest.approx(4.0, rel=0.02)


def test_nonfinite_trajectory_gives_failed_status(monkeypatch):
    scn, sim = _cw("+R", 300.0)
    import cwvalid.estimate_v2 as E

    def bad(*a, **k):
        return (float("nan"),) * 3
    monkeypatch.setattr(E, "acc_nerm", bad)
    r = E.residual_only_diagnostic(sim, scn.n, scn.R, scn.horizon)
    assert r["status"] == "failed" and math.isnan(r["peak_m"])
    out = estimator_variants(sim, make_law(scn), scn.n, scn.R, scn.horizon)
    assert out["cres_status"] == "failed" and math.isnan(out["cres_peak_m"])


def test_fit_edge_cases():
    import estimator_fit as F
    t, st = F.upper(np.array([1.0, 2.0, np.nan]), np.array([False, False, True]))
    assert t is None and st.startswith("not_calibratable")
    t, st = F.upper(np.array([1.0, 2.0, 5.0]), np.array([False, True, True]))
    assert st == "ok" and t == pytest.approx(0.8 * 2.0)
    vals = np.array([3.0, 1.0, np.nan, 2.0])
    t, st = F.acc_quantile(vals, 0.5)
    assert st == "ok" and t == 2.0 and np.mean(np.isfinite(vals) & (vals <= t)) >= 0.5
    t, st = F.acc_quantile(vals, 1.0)
    assert t is None and st.startswith("not_calibratable")


def test_nan_features_are_always_referred():
    import estimator_fit as F
    import pandas as pd
    D = pd.DataFrame({"d1": [0.1, np.nan, 5.0], "cw_arrived": [True, True, True], "cw_margin_s": [1000.0, 1000.0, np.nan]})
    acc = F.rule_accept(D, {"feature": "d1", "threshold": 1.0, "status": "ok", "combined": False})
    assert acc.tolist() == [True, False, False]
    acc = F.rule_accept(D, {"feature": "d1", "threshold": 10.0, "status": "ok", "combined": True, "m_min_s": 380.0})
    assert acc.tolist() == [True, False, False]
    assert F.rule_accept(D, {"feature": "d1", "threshold": None, "status": "not_calibratable"}) is None


def test_failure_aware_metrics():
    import estimator_evaluate as EV
    import pandas as pd
    L = pd.DataFrame({"traj_ok_1": [True, False, True, True], "adequate": [True, False, True, True],
                      "false_fav": [False, False, False, False], "false_unfav": [False, False, True, False],
                      "label_valid": [True, True, True, False], "ref_termination": ["horizon", "altitude_floor", "horizon", "horizon"],
                      "ref_status": ["ok", "terminated", "ok", "failed"], "disc_m": [0.1, 5.0, 0.2, np.nan]})
    acc = np.array([True, True, False, True])
    m = EV.metrics(acc, L, "traj_ok_1", feature_finite=np.array([True, True, True, True]))
    assert m["n_total"] == 4 and m["n_label_invalid"] == 1 and m["n_evaluated"] == 3
    assert m["accepted"] == 2 and m["incorrect_accepted"] == 1
    assert m["referred"] == 1 and m["unnecessary_referrals"] == 1
    assert m["ref_domain_exits"] == 1 and m["ref_domain_exits_accepted"] == 1
    z = EV.metrics(np.zeros(4, bool), L, "traj_ok_1", feature_finite=np.ones(4, bool))
    assert z["accepted"] == 0 and math.isnan(z["incorrect_rate"])


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "frozen_v2", "freeze_record.json")), reason="v2 not frozen yet")
def test_v2_freeze_hashes_match():
    sys.path.insert(0, os.path.join(REPO, "experiments", "v2"))
    import v2_common as C
    rec = json.load(open(os.path.join(C.FROZEN_V2_DIR, "freeze_record.json")))
    assert C.sha256(C.FROZEN_V2) == rec["estimator_v2_frozen_sha256"]
    fr = json.load(open(C.FROZEN_V2))
    assert fr["v1_frozen_sha256"] == C.sha256(C.FROZEN_V1)
    assert fr["A_threshold_equals_v1_frozen"] is True


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "fresh_test", "set_definition.json")), reason="fresh test not generated yet")
def test_fresh_test_isolated_from_development_and_hashed():
    import v2_common as C
    sd = json.load(open(os.path.join(C.RAW, "fresh_test", "set_definition.json")))
    assert sd["overlap_with_archive"] == []
    assert C.sha256(os.path.join(C.RAW, "fresh_test", "scenarios.jsonl")) == sd["scenarios_sha256"]
    assert sd["estimator_v2_frozen_sha256"] == C.sha256(C.FROZEN_V2)
    fr = json.load(open(C.FROZEN_V2))
    assert fr["frozen_at"] <= sd["written"]
    fresh = {json.loads(l)["sid"] for l in open(os.path.join(C.RAW, "fresh_test", "scenarios.jsonl")) if l.strip()}
    for c in ("main_grid/grid", "main_grid/refine", "main_grid/refine_bands", "screening/dev"):
        dev = {json.loads(l)["sid"] for l in open(os.path.join(C.RESULTS, c, "scenarios.jsonl")) if l.strip()}
        assert not (fresh & dev)


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "boundary_families", "families_record.json")), reason="families not declared")
def test_boundary_families_declaration_unchanged():
    import v2_common as C
    rec = json.load(open(os.path.join(C.RAW, "boundary_families", "families_record.json")))
    assert C.sha256(os.path.join(C.RAW, "boundary_families", "families.json")) == rec["families_sha256"]


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "verifier", "selection_record.json")), reason="verifier cases not selected")
def test_verifier_selection_unchanged():
    import v2_common as C
    rec = json.load(open(os.path.join(C.RAW, "verifier", "selection_record.json")))
    assert C.sha256(os.path.join(C.RAW, "verifier", "cases.json")) == rec["cases_sha256"]
