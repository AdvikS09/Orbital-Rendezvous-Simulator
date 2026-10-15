"""v3.1 C02 regressions: conservative record contracts, label partition, independent feature
availability, operational interruption, and fault injection into the staged campaign workers."""

import copy
import json
import math
import os
import sys

import numpy as np
import pytest

from cwvalid3 import calibration as CAL
from cwvalid3 import contracts as C

from . import v31fx as FX

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "v3"))
T = 5553.6
N_RAD = 2.0 * math.pi / T          # planned one-orbit horizon consistent with T


def cont_record(**kw):
    r = {"cell": "D0", "worker_status": "ok", "horizon_s": T, "n_radps": N_RAD}
    for b in ("cw", "eci", "nerm"):
        r.update({f"{b}_status": "ok", f"{b}_completed_horizon": True, f"{b}_t_final_s": T,
                  f"{b}_terminal_arrived": True, f"{b}_terminal_dwell_margin_s": 900.0})
    r.update({"cw_vs_eci_max_pos_disc_m": 0.4, "cw_vs_nerm_max_pos_disc_m": 0.4 + 1e-6,
              "cw_vs_eci_common_horizon_s": T, "cw_vs_nerm_common_horizon_s": T,
              "d1_peak_m": 0.3, "d1_status": "ok", "d1u_peak_m": 0.2, "d1u_status": "ok", "cres_peak_m": 0.1,
              "cres_status": "ok", "sep_m": 900.0, "feature_sat_fraction": 0.25})
    r.update(kw)
    return r


def held_record(**kw):
    r = {"cell": "D3", "worker_status": "ok", "horizon_s": T, "n_radps": N_RAD}
    for b in ("cw", "eci", "nerm"):
        r.update({f"{b}_status": "completed", f"{b}_t_end_s": T, f"{b}_arrived": False, f"{b}_dwell_margin_s": None})
    r.update({"cw_vs_eci_peak_m": 3.0, "cw_vs_nerm_peak_m": 3.0 + 1e-7, "cw_vs_eci_common_end_s": T,
              "cw_vs_nerm_common_end_s": T, "A_H_peak_m": 2.0, "A_H_status": "ok", "B_H_peak_m": 1.0, "B_H_status": "ok",
              "A_naive_peak_m": 2.0, "A_naive_status": "ok", "C_H_peak_m": 1.5, "C_H_status": "ok", "sep_m": 4000.0,
              "feature_sat_fraction": 1.0})
    r.update(kw)
    return r


# ---------------- contract ----------------------------------------------------------------

def test_valid_records_and_labels():
    tr = C.truth_record(cont_record())
    assert tr["truth_status"] == "completed_valid"
    assert C.criterion_label(tr, 1.0) == "ok" and C.criterion_label(tr, 0.1) == "violation"
    assert C.arrival_label(cont_record(), tr)["arrival_status"] == "arrived"
    th = C.truth_record(held_record())
    assert th["truth_status"] == "completed_valid" and C.criterion_label(th, 1.0) == "violation"
    assert C.arrival_label(held_record(), th)["arrival_status"] == "completed_non_arrival"


def test_missing_arrival_boolean_is_unresolved_not_non_arrival():
    r = cont_record()
    del r["eci_terminal_arrived"], r["cw_terminal_arrived"], r["nerm_terminal_arrived"]
    tr = C.truth_record(r)
    assert C.arrival_label(r, tr)["arrival_status"] == "unresolved"
    assert CAL.arrival_label(r, CAL.truth_record(r))["arrival_status"] == "completed_non_arrival"   # v3.0 defect
    r = cont_record(eci_terminal_arrived=1)
    assert C.arrival_label(r, C.truth_record(r))["arrival_status"] == "unresolved"
    r = cont_record(cw_terminal_dwell_margin_s=None)
    assert C.arrival_label(r, C.truth_record(r))["arrival_status"] == "unresolved"


@pytest.mark.parametrize("field,value", [("eci_status", "okay"), ("eci_status", None), ("nerm_status", 3),
                                         ("cw_status", "completed")])
def test_unknown_status_is_unresolved(field, value):
    tr = C.truth_record(cont_record(**{field: value}))
    assert tr["truth_status"] == "unresolved"
    assert C.criterion_label(tr, 1.0) == "unresolved" and C.is_adverse("unresolved")


def test_short_reference_or_pair_coverage_is_unresolved():
    assert C.truth_record(cont_record(eci_t_final_s=T - 10.0))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(eci_completed_horizon=None))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(cw_vs_eci_common_horizon_s=T - 1.0))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(cw_vs_nerm_common_horizon_s=None))["truth_status"] == "unresolved"
    assert C.truth_record(held_record(eci_t_end_s=T - 2.0))["truth_status"] == "unresolved"
    assert C.truth_record(held_record(cw_vs_eci_common_end_s=T - 0.5))["truth_status"] == "unresolved"
    assert C.truth_record(held_record(horizon_s=None))["truth_status"] == "unresolved"


def test_negative_or_missing_discrepancy_is_unresolved():
    for k, v in (("cw_vs_eci_max_pos_disc_m", -0.1), ("cw_vs_eci_max_pos_disc_m", None),
                 ("cw_vs_nerm_max_pos_disc_m", float("nan")), ("cw_vs_eci_max_pos_disc_m", True)):
        assert C.truth_record(cont_record(**{k: v}))["truth_status"] == "unresolved"


def test_exit_failure_and_disagreement():
    tr = C.truth_record(cont_record(eci_status="terminated", nerm_status="terminated", eci_completed_horizon=False,
                                    nerm_completed_horizon=False, eci_t_final_s=3000.0, nerm_t_final_s=3000.0))
    assert tr["truth_status"] == "physical_domain_exit" and C.criterion_label(tr, 1.0) == "exit"
    assert C.arrival_label(cont_record(), tr)["arrival_status"] == "not_applicable"
    tr = C.truth_record(held_record(eci_status="domain_exit", eci_t_end_s=3000.0))
    assert tr["truth_status"] == "unresolved"       # cross-check completed, reference exited
    tr = C.truth_record(held_record(nerm_status="numerical_failure", nerm_t_end_s=100.0))
    assert tr["truth_status"] == "numerical_failure" and C.criterion_label(tr, 1.0) == "failure"


@pytest.mark.parametrize("ws", ["worker_crash", "missing"])
def test_operational_interruption_is_its_own_status(ws):
    r = {"cell": "D1", "worker_status": ws, "run_id": "R0"}
    tr = C.truth_record(r)
    assert tr["truth_status"] == "interrupted" and C.criterion_label(tr, 1.0) == "interrupted"
    assert C.is_adverse("interrupted")
    assert C.feature_availability(r, "A")[1] == "unavailable"
    assert CAL.truth_record(r)["truth_status"] == "numerical_failure"          # v3.0 label


def test_exception_and_unknown_worker_status():
    assert C.truth_record({"cell": "D0", "worker_status": "exception"})["truth_status"] == "numerical_failure"
    assert C.truth_record({"cell": "D0", "worker_status": "weird"})["truth_status"] == "unresolved"
    assert C.truth_record({"cell": "D0"})["truth_status"] == "unresolved"


def test_feature_availability_is_independent_of_the_reference():
    r = cont_record(worker_status="reference_failed", eci_status=None, nerm_status=None, reference_error="boom")
    v, st, _ = C.feature_availability(r, "A")
    assert st == "ok" and v == 0.3
    assert C.truth_record(r)["truth_status"] == "unresolved"          # plan: accepted with unresolved truth
    r = held_record(eci_status="numerical_failure", eci_t_end_s=10.0)
    assert C.feature_availability(r, "A_H")[1] == "ok"


def test_feature_availability_rules():
    assert C.feature_availability(cont_record(cw_t_final_s=T - 5.0), "A")[1] == "unavailable"
    assert C.feature_availability(cont_record(cw_completed_horizon=False), "A")[1] == "unavailable"
    assert C.feature_availability(cont_record(d1_status="failed"), "A")[1] == "unavailable"
    assert C.feature_availability(cont_record(d1_peak_m=-1.0), "A")[1] == "unavailable"
    assert C.feature_availability(cont_record(d1_peak_m=float("inf")), "A")[1] == "unavailable"
    assert C.feature_availability(cont_record(worker_status="exception"), "A")[1] == "unavailable"
    assert C.feature_availability(held_record(feature_sat_fraction=1.0), "sat") == (1.0, "ok", "")
    assert C.feature_availability(held_record(feature_sat_fraction=0.0), "sat")[1] == "ok"
    assert C.feature_availability(held_record(feature_sat_fraction=1.0 + 1e-12), "sat")[1] == "unavailable"
    assert C.feature_availability(held_record(cw_t_end_s=T - 1.0), "A_H")[1] == "unavailable"
    with pytest.raises(ValueError):
        C.feature_availability(held_record(), "A")


def test_explicit_kind_mismatch_is_unresolved():
    assert C.truth_record(held_record(), kind="continuous")["truth_status"] == "unresolved"
    assert C.truth_record(held_record(), kind="held")["truth_status"] == "completed_valid"
    with pytest.raises(ValueError):
        C.truth_record(held_record(), kind="sampled")


def test_label_partition_identity():
    labels = ["ok", "violation", "exit", "failure", "unresolved", "interrupted", "ok"]
    acc = [True, True, False, True, True, True, False]
    cnt = C.label_counts(acc, labels)
    assert sum(cnt[f"acc_{x}"] for x in C.LABELS) == sum(acc)
    assert sum(cnt.values()) == len(labels)
    with pytest.raises(ValueError):
        C.label_counts([True], ["completed"])
    with pytest.raises(ValueError):
        C.check_label_partition({"acc_ok": 2}, 3)
    with pytest.raises(ValueError):
        C.is_adverse("bogus")
    assert set(C.ADVERSE_LABELS) == set(C.LABELS) - {"ok"}


@pytest.mark.parametrize("cell", ["D0", "D1", "D3", "D4", "D5"])
def test_contract_matches_v30_on_stored_records(cell):
    """Exact stored calibration records (fixture: the first 40 lines of each file, plus every D1 worker_crash record, the
    first exit record of D0 and D3, the first 10 exits of D1 and D4 and the first 10 D1 1 m violations; the full-archive
    scan of all 47,983 records is v3_1_correction/results/existing_record_contract_impact.csv)."""
    kind = C.kind_of({"cell": cell})
    recs = FX.results(f"{cell}_cal")
    assert len(recs) >= 40
    if cell == "D1":
        assert sum(r.get("worker_status") == "worker_crash" for r in recs) == 43
    for r in recs:
        t0, t1 = CAL.truth_record(r), C.truth_record(r, kind)
        if r.get("worker_status") == "worker_crash":
            assert t1["truth_status"] == "interrupted"
            continue
        assert t0["truth_status"] == t1["truth_status"]
        for tol in (1.0, 0.1):
            assert CAL.criterion_label(t0, tol) == C.criterion_label(t1, tol)
        meths = ("A", "B", "C", "sep", "sat") if kind == "continuous" else ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")
        for m in meths:
            a, b = CAL.feature_value(r, m, kind), C.feature_value(r, m, kind)
            assert (math.isnan(a) and math.isnan(b)) or a == b


# ---------------- fault injection into the staged workers -------------------------------------

def _manifest(cell, split, n):
    """The first n planned cases, as exact fixture lines of the original manifest (never skipped)."""
    return FX.manifest(f"{cell}_{split}", n)


@pytest.fixture(scope="module")
def CP():
    import campaign
    return campaign


def test_continuous_reference_failure_keeps_features(CP, monkeypatch):
    m = _manifest("D0", "test", 1)[0]
    real = CP._propagate

    def boom(scn, cfg, law, models):
        if "eci" in models:
            raise RuntimeError("injected reference failure")
        return real(scn, cfg, law, models)
    monkeypatch.setattr(CP, "_propagate", boom)
    o = CP.work_continuous(m)[0]
    assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "reference"
    assert "injected reference failure" in o["reference_error"] and o["feature_stage"] == "complete"
    for meth in ("A", "B", "C", "sep", "sat"):
        assert C.feature_availability(o, meth, "continuous")[1] == "ok", meth
    tr = C.truth_record(o, "continuous")
    assert tr["truth_status"] == "unresolved" and C.criterion_label(tr, 1.0) == "unresolved"
    # v3.0 semantics would have referred this case (feature unavailable)
    assert math.isnan(CAL.feature_value({**o, "worker_status": "exception"}, "A", "continuous"))


def test_continuous_nominal_failure_makes_features_unavailable(CP, monkeypatch):
    m = _manifest("D0", "test", 1)[0]

    def boom(scn, cfg, law, models):
        raise RuntimeError("injected nominal failure")
    monkeypatch.setattr(CP, "_propagate", boom)
    o = CP.work_continuous(m)[0]
    assert o["worker_status"] == "exception" and o["failed_stage"] == "nominal"
    assert C.feature_availability(o, "A", "continuous")[1] == "unavailable"
    assert C.truth_record(o, "continuous")["truth_status"] == "numerical_failure"


def test_continuous_metric_failure_after_reference(CP, monkeypatch):
    import cwvalid3.continuous as CONT
    m = _manifest("D0", "test", 1)[0]

    def boom(*a, **k):
        raise ValueError("injected metric failure")
    monkeypatch.setattr(CONT, "evaluate_general", boom)
    o = CP.work_continuous(m)[0]
    assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "metrics"
    assert C.feature_availability(o, "A", "continuous")[1] == "ok"


def test_held_reference_metric_failure_keeps_features(CP, monkeypatch):
    import cwvalid3.held_metrics as HM
    recs = _manifest("D4", "test", 2)

    def boom(*a, **k):
        raise RuntimeError("injected held metric failure")
    # v3.1.1: the worker computes the reference metrics with peak_discrepancy (no longer via case_summary, which also
    # re-extracted the features); the injection point moved with it
    monkeypatch.setattr(HM, "peak_discrepancy", boom)
    outs = CP.work_held(recs)
    for o in outs:
        assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "reference"
        for meth in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"):
            assert C.feature_availability(o, meth, "held")[1] == "ok", meth
        assert C.truth_record(o, "held")["truth_status"] == "unresolved"


def test_held_batch_exception_is_retried_case_by_case(CP, monkeypatch):
    import cwvalid3.sampled as SA
    recs = _manifest("D4", "test", 2)
    real = SA.HeldBatch.run
    bad_id = recs[1]["scenario"]["rel0"]

    def run(self):
        if self.N > 1:
            raise FloatingPointError("injected batch failure")
        if np.allclose(self.rel0[0], bad_id):
            raise FloatingPointError("injected single-case failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    outs = CP.work_held(recs)
    assert [o["case_id"] for o in outs] == [r["case_id"] for r in recs]
    assert outs[0]["worker_status"] == "ok" and "case by case" in outs[0]["batch_retry"]
    # the bad case also fails in its CW + estimator fallback: no feature, numerical failure
    assert outs[1]["worker_status"] == "exception" and outs[1]["failed_stage"] == "features"
    assert C.truth_record(outs[1], "held")["truth_status"] == "numerical_failure"


def test_campaign_refuses_to_write_into_the_v30_tree(CP):
    for sub in ("raw/results", "raw/manifests", "", "reports"):
        with pytest.raises(SystemExit):
            CP.out_paths(os.path.join(CP.V3, sub))
    assert CP.out_paths(os.path.join(CP.V3, "v3_1_correction", "raw", "campaign"))["res"].endswith("results")


def test_held_reference_failure_keeps_features_via_cw_only_batch(CP, monkeypatch):
    """The combined held batch (cw + nerm + eci) raises: the features come from a CW + estimator batch, the case is
    reference_failed with unresolved truth and available features."""
    import cwvalid3.sampled as SA
    recs = _manifest("D4", "test", 1)
    real = SA.HeldBatch.run

    def run(self):
        if self.has_eci:
            raise FloatingPointError("injected reference-branch failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    o = CP.work_held(recs)[0]
    assert o["worker_status"] == "reference_failed" and o["feature_source"] == "cw_estimator_only_fallback"
    for meth in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"):
        assert C.feature_availability(o, meth, "held")[1] == "ok", meth
    assert C.truth_record(o, "held")["truth_status"] == "unresolved"


def test_held_naive_pass_failure_removes_only_a_naive(CP, monkeypatch):
    import cwvalid3.sampled as SA
    recs = _manifest("D4", "test", 1)
    real = SA.HeldBatch.run

    def run(self):
        if "A_naive" in self.estimators:
            raise FloatingPointError("injected naive-pass failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    o = CP.work_held(recs)[0]
    assert o["worker_status"] == "ok" and o["A_naive_status"] == "naive_pass_failed"
    assert C.feature_availability(o, "A_naive", "held")[1] == "unavailable"
    assert C.feature_availability(o, "A_H", "held")[1] == "ok"
    assert C.truth_record(o, "held")["truth_status"] in ("completed_valid", "physical_domain_exit")


def test_feature_checkpoint_written_and_recovered_after_a_crash(CP, monkeypatch, tmp_path):
    from cwvalid3 import serialization as SER
    m = _manifest("D0", "test", 1)[0]
    monkeypatch.setenv(CP.CHECKPOINT_ENV, str(tmp_path))
    o = CP.work_continuous(m)[0]
    ck = tmp_path / f"{m['run_id']}.json"
    assert ck.exists()
    stored = json.loads(ck.read_text(encoding="utf-8"))
    assert stored["feature_stage"] == "complete" and stored["d1_peak_m"] == o["d1_peak_m"]
    assert "eci_status" not in stored                       # written before the reference stage
    crash = CP._crash_record(m, RuntimeError("BrokenProcessPool"), str(tmp_path))
    assert crash["worker_status"] == "worker_crash" and crash["feature_checkpoint_recovered"] is True
    assert C.truth_record(crash, "continuous")["truth_status"] == "interrupted"
    assert C.feature_value(crash, "A", "continuous") == o["d1_peak_m"]
    no_ck = CP._crash_record({**m, "run_id": "Rnone"}, RuntimeError("x"), str(tmp_path))
    assert C.feature_availability(no_ck, "A", "continuous")[1] == "unavailable"
    assert SER.is_volatile("batch_members_sha256")


def test_horizon_and_inconsistent_ok_records_are_unresolved():
    for kw in ({"horizon_s": 0.0}, {"horizon_s": -5.0}, {"horizon_s": 10.0}, {"n_radps": None}):
        r = cont_record(**kw)
        assert C.truth_record(r)["truth_status"] == "unresolved", kw
        assert C.feature_availability(r, "A")[1] == "unavailable", kw
    assert C.truth_record(held_record(horizon_s=10.0))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(failed_stage="reference"))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(reference_error="boom"))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(cw_numerical_failure=True))["truth_status"] == "unresolved"
    assert C.truth_record(cont_record(cw_numerical_failure=False))["truth_status"] == "completed_valid"


def test_v31_evaluate_rule_counts_every_label_and_partitions():
    out = C.evaluate_rule([0.1, 0.1, 0.1, 0.1, 5.0, float("nan")], ["ok", "interrupted", "unresolved", "ok", "violation", "ok"], 1.0)
    assert out["n_accepted"] == 4 and out["conservative_adverse_accepted"] == 2
    assert sum(out[f"acc_{x}"] for x in C.LABELS) == out["n_accepted"]
    assert sum(out[f"ref_{x}"] for x in C.LABELS) == out["n_referred"] == 2
    with pytest.raises(ValueError):
        C.evaluate_rule([0.1], ["bogus"], 1.0)
    # the frozen v3.0 function is kept as the definition behind the frozen results and does not count 'interrupted'
    assert CAL.evaluate_rule([0.1, 0.1], ["ok", "interrupted"], 1.0)["conservative_adverse_accepted"] == 0


def test_v31_loader_keeps_feature_of_a_reference_failure(tmp_path):
    sys.path.insert(0, os.path.join(REPO, "experiments", "v3_1"))
    import v31data
    from cwvalid3 import serialization as SER
    man = _manifest("D0", "test", 3)
    stored = {r["run_id"]: r for r in FX.results("D0_test") if r["run_id"] in {m["run_id"] for m in man}}
    assert len(stored) == 3
    (tmp_path / "manifests").mkdir()
    (tmp_path / "results").mkdir()
    with open(tmp_path / "manifests" / "D0_t1.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for m in man:
            SER.write_record(fh, m, schema=None)
    a, b = stored[man[0]["run_id"]], stored[man[1]["run_id"]]
    ref_failed = {k: v for k, v in a.items() if not k.startswith(("eci_", "nerm_", "cw_vs_", "nerm_vs_"))}
    ref_failed.update(worker_status="reference_failed", failed_stage="reference", reference_error="injected")
    with open(tmp_path / "results" / "D0_t1.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        SER.write_record(fh, ref_failed)
        SER.write_record(fh, b)
    df, rc = v31data.load_split("D0", "t1", root=str(tmp_path))
    assert rc["n_missing"] == 1
    r0, r1, r2 = (df[df.run_id == m["run_id"]].iloc[0] for m in man)
    assert r0.truth_status == "unresolved" and r0.f_A_status == "ok" and r0.f_A == a["d1_peak_m"]
    assert r2.worker_status == "missing" and r2.truth_status == "interrupted" and r2.label_1m == "interrupted"
    assert r1.truth_status == C.truth_record(b, "continuous")["truth_status"]
    ev = v31data.evaluate(df, "A", "1m", 1e9)
    assert ev["n_accepted"] == 2 and ev["acc_unresolved"] == 1


@pytest.mark.parametrize("variant", ["lower", "upper", "mixed"])
def test_guards_are_case_insensitive_on_windows(CP, variant):
    if os.name != "nt":
        pytest.skip("case-insensitive filesystem check")
    sys.path.insert(0, os.path.join(REPO, "experiments", "v3_1"))
    import v31paths
    target = os.path.join(CP.V3, "raw", "results")
    t = {"lower": target.lower(), "upper": target.upper(), "mixed": target.replace("v3_extension", "V3_Extension")}[variant]
    with pytest.raises(SystemExit):
        CP.v30_tree_guard(t)
    with pytest.raises(PermissionError):
        v31paths.guard_v30(os.path.join(t, "x.csv"))
    ok = os.path.join(CP.V3, "v3_1_correction", "raw").upper()
    assert CP.v30_tree_guard(ok)
