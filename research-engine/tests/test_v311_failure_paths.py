"""v3.1.1 regressions for the two reproduced C02 failure-path gaps (follow-up brief of 14 Oct 2026).

R01  a failure of the optional A_naive comparator (its pass, its case view, its feature extraction or the naive-pass
     diagnostic) must not discard valid A_H/B_H/C_H/separation/saturation features or the reference metrics; a genuine
     CW or primary-estimator failure must still refer conservatively.
R02  a held worker process lost after its nominal work but before any checkpoint: the task is re-executed alone in a
     fresh process; if it is lost again, the driver-level nominal-only recovery (fresh process, no reference branches)
     restores the screening features, the case keeps 'interrupted' truth and counts as adverse if accepted. A task that
     only failed because another task broke the pool completes normally on re-execution. A failed recovery, a stale or
     wrong-run checkpoint, failed checkpoint writes and the normal path are tested too. Process loss is real: the worker
     calls os._exit (CWVALID3_FAULT_INJECT, for every task or for one case with point@run_id).
Normal-path comparisons with the stored records allow round-off (RTOL) and require identical frozen decisions: the
brief asks not to demand bit identity across platforms or batch shapes (exact old/new identity on one machine is
experiments/v3_1_1/c11b_compare_normal_path.py).
All inputs are exact fixture lines of the original manifests (tests/fixtures/v31); the counterexample is D4 test
index 0 (Caf93a5f0b849d73a), the frozen cutoffs come from the fixture copy of calibration_freeze.json.
"""

import copy
import hashlib
import json
import math
import os
import pickle
import sys

import pytest

from cwvalid3 import contracts as C
from cwvalid3 import serialization as SER

from . import v31fx as FX

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments", "v3"))
HELD_METHODS = ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")
COUNTEREXAMPLE = "Caf93a5f0b849d73a"


@pytest.fixture(scope="module")
def CP():
    import campaign
    return campaign


# ---------------------------------------------------------------------------------------------------- helpers

_RUNS = {}


@pytest.fixture
def cached_batches(monkeypatch):
    """Replace HeldBatch by a subclass that memoises run() by its constructor arguments: the propagation of a given
    batch is computed once (really) and later calls get a deep copy of that result. Tests then inject failures only
    after the propagation, which is all the R01 paths need. A post hook can alter the copied result (true CW failure)."""
    import cwvalid3.sampled as SA
    real = SA.HeldBatch

    class Cached(real):
        post = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._ckey = hashlib.sha256(pickle.dumps((a, sorted(k.items())), protocol=4)).hexdigest()

        def run(self):
            if self._ckey not in _RUNS:
                real.run(self)
                _RUNS[self._ckey] = copy.deepcopy(self.__dict__)
            else:
                self.__dict__.update(copy.deepcopy(_RUNS[self._ckey]))
            if type(self).post is not None:
                type(self).post(self)
            return self
    monkeypatch.setattr(SA, "HeldBatch", Cached)
    return Cached


def counterexample():
    m = FX.manifest("D4_test", 1)[0]
    assert m["case_id"] == COUNTEREXAMPLE
    return m


def decisions(o):
    """Accept (True) / refer (False) for every frozen D4 selection with a threshold."""
    out = {}
    for key, s in FX.freeze()["selection"].items():
        cell, meth, crit = key.split("|")
        if cell != o["cell"] or s.get("threshold") is None:
            continue
        v, st, _ = C.feature_availability(o, meth, "held")
        out[key] = bool(st == "ok" and v <= s["threshold"])
    return out


RTOL = 1e-9          # round-off allowance (platform, BLAS, batch shape); bit identity is checked by c11b on one machine


def _close(x, y):
    if isinstance(x, bool) or isinstance(y, bool) or not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(_close(p, q) for p, q in zip(x, y))
        return x == y
    return math.isclose(x, y, rel_tol=RTOL, abs_tol=1e-12)


def compare_to_stored(o, stored):
    """Fields that differ between a new record and the stored v3.0 record beyond round-off (canonical values; statuses,
    Booleans, integers and strings exactly, floats to RTOL; volatile and v3.1-only fields excluded, as in
    experiments/v3_1/c02_worker_equivalence.py), and the stored fields the new record lacks."""
    a, b = SER.normalize(o), SER.normalize(stored)
    keys = {k for k in a if not SER.is_volatile(k) and k not in SER.V31_ONLY_FIELDS} & {k for k in b if not SER.is_volatile(k)}
    return sorted(k for k in keys if not _close(a[k], b[k])), sorted({k for k in b if not SER.is_volatile(k)} - set(a))


def same_decisions(o, ref, kind="held"):
    sel = FX.freeze()["selection"]
    for key, s in sel.items():
        cell, meth, crit = key.split("|")
        if cell != o["cell"] or s.get("threshold") is None:
            continue
        va, sa, _ = C.feature_availability(o, meth, kind)
        vb, sb, _ = C.feature_availability(ref, meth, kind)
        assert (sa == "ok" and va <= s["threshold"]) == (sb == "ok" and vb <= s["threshold"]), key


@pytest.fixture
def normal(CP, cached_batches):
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "feature_errors" not in o
    return o


def check_survivors(o, normal, affected, status=None):
    """Every method not in `affected` keeps its normal value and decision; the affected ones are unavailable (marked by
    their status field); truth and the reference metrics are those of the normal run."""
    for meth in HELD_METHODS:
        v, st, why = C.feature_availability(o, meth, "held")
        if meth in affected:
            assert st == "unavailable", meth
        else:
            assert st == "ok" and v == C.feature_value(normal, meth, "held"), (meth, why)
    d0, d1 = decisions(normal), decisions(o)
    for key in d0:
        if key.split("|")[1] not in affected:
            assert d1[key] == d0[key], key
    if status:
        for meth, st in status.items():
            assert o[f"{meth}_status"] == st
    t0, t1 = C.truth_record(normal, "held"), C.truth_record(o, "held")
    assert t1["truth_status"] == t0["truth_status"] == "completed_valid"
    assert t1["disc_m"] == t0["disc_m"] and t1["u_num_m"] == t0["u_num_m"]
    for k in ("cw_vs_eci_peak_m", "cw_vs_nerm_peak_m", "nerm_vs_eci_peak_m", "eci_status", "eci_arrived"):
        assert o[k] == normal[k], k


# ---------------------------------------------------------------------------------------------------- R01

def test_r01_normal_path_matches_the_stored_record(normal):
    """Normal path of the restructured worker on the counterexample = stored v3.0 record (all compared fields);
    A_H 3.17e-4 m is accepted at the frozen D4 A_H 1 m cutoff; truth completed_valid."""
    stored = FX.result_for("D4_test", normal["run_id"])
    diff, missing = compare_to_stored(normal, stored)
    assert diff == [] and missing == []
    assert math.isclose(normal["A_H_peak_m"], stored["A_H_peak_m"], rel_tol=RTOL)        # stored 3.1723376174226604e-4 m
    same_decisions(normal, stored)
    assert decisions(normal)["D4|A_H|1m"] is True
    assert C.truth_record(normal, "held")["truth_status"] == "completed_valid"


def test_r01_cached_propagation_is_exact(CP, normal, cached_batches, monkeypatch):
    """The memoised propagation used by the R01 tests gives the same record as a fresh, uncached propagation."""
    import cwvalid3.sampled as SA
    monkeypatch.setattr(SA, "HeldBatch", cached_batches.__mro__[1])        # the real HeldBatch
    fresh = CP.work_held([counterexample()])[0]
    assert SER.physics_sha256(fresh) == SER.physics_sha256(normal)


def test_r01_naive_pass_run_failure(CP, normal, cached_batches, monkeypatch):
    cached_run = cached_batches.run

    def run(self):
        if "A_naive" in self.estimators:
            raise FloatingPointError("injected naive-pass failure")
        return cached_run(self)
    monkeypatch.setattr(cached_batches, "run", run)
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "injected naive-pass failure" in o["naive_pass_error"]
    assert "A_naive" in o["feature_errors"]
    check_survivors(o, normal, {"A_naive"}, {"A_naive": "naive_pass_failed"})


def test_r01_naive_case_view_failure(CP, normal, monkeypatch):
    import cwvalid3.held_metrics as HM
    real = HM.CaseView

    def view(batch, i, *a, **k):
        if "A_naive" in batch.estimators:
            raise IndexError("injected A_naive case-view failure")
        return real(batch, i, *a, **k)
    monkeypatch.setattr(HM, "CaseView", view)
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "failed_stage" not in o and "error" not in o
    assert "injected A_naive case-view failure" in o["feature_errors"]["A_naive"]
    check_survivors(o, normal, {"A_naive"}, {"A_naive": "naive_feature_failed"})


def test_r01_naive_estimator_feature_failure(CP, normal, monkeypatch):
    """The auditor's counterexample: RuntimeError only for estimator_feature(cv, 'A_naive')."""
    import cwvalid3.held_metrics as HM
    real = HM.estimator_feature

    def feat(cv, est):
        if est == "A_naive":
            raise RuntimeError("injected A_naive extraction failure")
        return real(cv, est)
    monkeypatch.setattr(HM, "estimator_feature", feat)
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "failed_stage" not in o
    assert "injected A_naive extraction failure" in o["feature_errors"]["A_naive"]
    check_survivors(o, normal, {"A_naive"}, {"A_naive": "naive_feature_failed"})
    assert decisions(o)["D4|A_H|1m"] is True                       # the frozen A_H decision stands


def test_r01_naive_diagnostic_failure(CP, normal, monkeypatch):
    def boom(*a, **k):
        raise ValueError("injected naive-pass diagnostic failure")
    monkeypatch.setattr(CP, "_naive_pass_diagnostic", boom)
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok"
    assert o["cw_naive_pass_max_state_diff_m"] != o["cw_naive_pass_max_state_diff_m"]           # NaN
    assert "injected naive-pass diagnostic failure" in o["feature_errors"]["cw_naive_pass_diagnostic"]
    check_survivors(o, normal, set(), {"A_naive": "ok"})            # A_naive itself is unaffected
    assert o["A_naive_peak_m"] == normal["A_naive_peak_m"]


def test_r01_saturation_failure_removes_only_sat(CP, normal, monkeypatch):
    import cwvalid3.held_metrics as HM

    def boom(cv):
        raise RuntimeError("injected saturation-summary failure")
    monkeypatch.setattr(HM, "saturation_summary", boom)
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "saturation" in o["feature_errors"]
    check_survivors(o, normal, {"sat"})


def test_r01_primary_estimator_failure_refers_conservatively(CP, normal, monkeypatch):
    """A genuine A_H extraction failure: A_H is unavailable (the accepted case becomes a referral), every other
    feature and the truth stand."""
    import cwvalid3.held_metrics as HM
    real = HM.estimator_feature

    def feat(cv, est):
        if est == "A_H":
            raise RuntimeError("injected A_H extraction failure")
        return real(cv, est)
    monkeypatch.setattr(HM, "estimator_feature", feat)
    o = CP.work_held([counterexample()])[0]
    check_survivors(o, normal, {"A_H"}, {"A_H": "feature_extraction_failed"})
    assert decisions(normal)["D4|A_H|1m"] is True and decisions(o)["D4|A_H|1m"] is False


def test_r01_nominal_cw_failure_refers_every_feature(CP, cached_batches, monkeypatch):
    """A genuine nominal CW branch failure in the combined batch: every feature is unavailable and the truth is a
    numerical failure (conservative referral)."""
    def mark(self):
        if self.has_eci:
            self.status["cw"][0] = "numerical_failure"
            self.t_end["cw"][0] = 1000.0
    monkeypatch.setattr(cached_batches, "post", staticmethod(mark))
    o = CP.work_held([counterexample()])[0]
    for meth in HELD_METHODS:
        assert C.feature_availability(o, meth, "held")[1] == "unavailable", meth
    assert C.truth_record(o, "held")["truth_status"] == "numerical_failure"
    assert not any(decisions(o).values())


def test_r01_checkpoint_write_failure_never_fails_the_case(CP, normal, monkeypatch, tmp_path):
    """Both held checkpoint writes fail (the checkpoint 'directory' is a file): the error is recorded, the case stays ok
    with every feature, the reference metrics and its truth (review fix: an unguarded rewrite took down the batch)."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(CP.CHECKPOINT_ENV, str(blocker / "ck"))
    o = CP.work_held([counterexample()])[0]
    assert o["worker_status"] == "ok" and "error" not in o and o["feature_checkpoint_error"]
    check_survivors(o, normal, set(), {"A_H": "ok", "A_naive": "ok"})
    assert SER.physics_sha256(o) == SER.physics_sha256(normal)          # feature_checkpoint_error is volatile


def test_r01_continuous_checkpoint_write_failure_never_fails_the_case(CP, monkeypatch, tmp_path):
    m = FX.manifest("D0_test", 1)[0]
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(CP.CHECKPOINT_ENV, str(blocker / "ck"))
    o = CP.work_continuous(m)[0]
    assert o["worker_status"] == "ok" and o["feature_checkpoint_error"]
    for meth in ("A", "B", "C", "sep", "sat"):
        assert C.feature_availability(o, meth, "continuous")[1] == "ok", meth
    diff, missing = compare_to_stored(o, FX.result_for("D0_test", m["run_id"]))
    assert diff == [] and missing == []


def test_r01_feature_errors_do_not_mark_the_record_failed(normal):
    """feature_errors is diagnostic: an ok record with a feature error keeps the truth of its reference (the ok-with-
    error rule of the contract applies to error, reference_error and failed_stage only)."""
    r = {**normal, "feature_errors": {"A_naive": "x"}, "A_naive_status": "naive_feature_failed"}
    assert C.truth_record(r, "held")["truth_status"] == "completed_valid"
    assert C.feature_availability(r, "A_naive", "held")[1] == "unavailable"
    assert C.truth_record({**normal, "error": "x"}, "held")["truth_status"] == "unresolved"


# ---------------------------------------------------------------------------------------------------- R02

def _out(tmp_path, cs, n):
    """A v3.1 output root whose manifest holds the exact fixture lines of the first n planned cases."""
    (tmp_path / "manifests").mkdir()
    lines = FX.manifest_lines(cs)[:n]
    assert [json.loads(b)["index"] for b in lines] == [m["index"] for m in FX.manifest(cs, n)]
    (tmp_path / "manifests" / f"{cs}.jsonl").write_bytes(b"".join(b + b"\n" for b in lines))
    return str(tmp_path)


def _run(CP, monkeypatch, out, cell, split, batch, fault=None, workers=2):
    for k in (CP.CHECKPOINT_ENV, CP.SESSION_ENV, CP.FAULT_ENV):
        monkeypatch.setenv(k, "")       # registers the variable with monkeypatch, which removes it after the test; an
    monkeypatch.setenv(CP.FAULT_ENV, fault or "")    # empty value disables checkpoints and faults until cmd_run sets them
    CP.cmd_run(cell, split, workers=workers, batch=batch, out=out)
    res = [json.loads(x) for x in open(os.path.join(out, "results", f"{cell}_{split}.jsonl"), encoding="utf-8")]
    led = [json.loads(x) for x in open(os.path.join(out, "RUN_LEDGER.jsonl"), encoding="utf-8")]
    return {r["index"]: r for r in res}, led


def test_r02_process_loss_after_the_combined_batch_recovers_nominal_features(CP, monkeypatch, tmp_path):
    """Real process loss right after the combined batch (nominal work done, nothing checkpointed). Case 1 also has a
    stale checkpoint from another run session holding a wrong A_H; it must be ignored."""
    out = _out(tmp_path, "D4_test", 2)
    m0, m1 = FX.manifest("D4_test", 2)
    ckdir = tmp_path / "results" / "feature_checkpoints" / "D4_test"
    ckdir.mkdir(parents=True)
    stale = {**{k: m1[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")},
             "feature_stage": "complete", "A_H_peak_m": 123.0, "A_H_status": "ok", "checkpoint_session": "an-earlier-run",
             "checkpoint_engine_sha": CP.engine_version()}
    (ckdir / f"{m1['run_id']}.json").write_text(json.dumps(stale), encoding="utf-8")
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault="loss_held_after_combined")
    assert sorted(res) == [0, 1]
    stored = {i: FX.result_for("D4_test", m["run_id"]) for i, m in ((0, m0), (1, m1))}
    for i, o in res.items():
        assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "driver_nominal_recovery"
        assert o["task_retry"] == "lost again when re-executed alone in a fresh process"
        assert "feature_checkpoint_recovered" not in o and "eci_status" not in o and "cw_vs_eci_peak_m" not in o
        for meth in HELD_METHODS:
            v, st, why = C.feature_availability(o, meth, "held")
            assert st == "ok", (meth, why)
            ref = C.feature_value(stored[i], meth, "held")
            assert abs(v - ref) <= 1e-9 * max(abs(ref), 1e-12), (meth, v, ref)    # round-off only (batch shape)
        tr = C.truth_record(o, "held")
        assert tr["truth_status"] == "interrupted" and C.criterion_label(tr, 1.0) == "interrupted"
    assert res[1]["stale_checkpoint_ignored"] == "session" and res[1]["A_H_peak_m"] != 123.0
    # the recovered, accepted counterexample counts as adverse at the frozen cutoff
    thr = FX.freeze()["selection"]["D4|A_H|1m"]["threshold"]
    ev = C.evaluate_rule([C.feature_value(res[0], "A_H", "held")], ["interrupted"], thr)
    assert ev["n_accepted"] == 1 and ev["acc_interrupted"] == 1 and ev["conservative_adverse_accepted"] == 1
    ev = [e for e in led if e.get("event") == "completed"]
    assert len(ev) == 2 and all(e["feature_recovery"] == "driver_nominal_recovery" for e in ev)


def test_r02_tasks_broken_by_another_tasks_loss_complete_on_reexecution(CP, monkeypatch, tmp_path):
    """Only case 0's task loses its process (fault point@run_id); the pool break also fails case 1's task, which never
    lost a process. After the main pool, case 1 is re-executed alone and completes normally (task_retry, volatile), so it
    is not recorded as interrupted; case 0 is lost again and gets the nominal-only recovery (review fix)."""
    out = _out(tmp_path, "D4_test", 2)
    m0, m1 = FX.manifest("D4_test", 2)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault=f"loss_held_after_combined@{m0['run_id']}", workers=1)
    assert res[0]["worker_status"] == "worker_crash" and res[0]["feature_recovery"] == "driver_nominal_recovery"
    o = res[1]
    assert o["worker_status"] == "ok" and "feature_recovery" not in o and o["task_retry"].startswith("re-executed alone")
    diff, missing = compare_to_stored(o, FX.result_for("D4_test", m1["run_id"]))
    assert diff == [] and missing == []
    assert C.truth_record(o, "held")["truth_status"] == C.truth_record(FX.result_for("D4_test", m1["run_id"]), "held")["truth_status"]
    assert sum(e.get("event") == "completed" for e in led) == 2


def test_r02_process_loss_after_primary_features_uses_this_runs_checkpoint(CP, monkeypatch, tmp_path):
    """Loss after the primary features were checkpointed (before the optional A_naive pass): the checkpoint of this
    run session is merged; A_naive, never computed, is unavailable; truth stays interrupted."""
    out = _out(tmp_path, "D4_test", 2)
    res, _ = _run(CP, monkeypatch, out, "D4", "test", batch=2, fault="loss_held_after_features")
    for o in res.values():
        assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "checkpoint"
        assert o["feature_checkpoint_recovered"] is True and "checkpoint_session" not in o
        for meth in ("A_H", "B_H", "C_H", "sep", "sat"):
            assert C.feature_availability(o, meth, "held")[1] == "ok", meth
        assert C.feature_availability(o, "A_naive", "held")[1] == "unavailable"
        assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def test_r02_genuine_recovery_failure_leaves_features_unavailable(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4_test", 1)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault="loss_held_after_combined,fail_recovery")
    o = res[0]
    assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "failed"
    assert "injected fault at fail_recovery" in o["feature_recovery_error"]
    for meth in HELD_METHODS:
        assert C.feature_availability(o, meth, "held")[1] == "unavailable"
    assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def test_r02_recovered_nominal_failure_is_not_a_feature(CP, cached_batches, monkeypatch):
    """The recovery itself finds the nominal CW branch failed: the merged record has no available feature."""
    m = counterexample()

    def mark(self):
        self.status["cw"][0] = "numerical_failure"
    monkeypatch.setattr(cached_batches, "post", staticmethod(mark))
    got = CP.recover_features([m])[m["run_id"]]
    o = CP._merge_features(CP._crash_record(m, RuntimeError("lost"), None), got, "driver_nominal_recovery")
    assert o["A_H_status"] == "nominal_failed"
    for meth in HELD_METHODS:
        assert C.feature_availability(o, meth, "held")[1] == "unavailable", meth
    assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def test_r02_stale_or_wrong_run_checkpoints_are_never_merged(CP, tmp_path):
    m0, m1 = FX.manifest("D4_test", 2)
    good = {**{k: m0[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")}, "feature_stage": "complete",
            "A_H_peak_m": 0.5, "A_H_status": "ok", "checkpoint_session": "S1", "checkpoint_engine_sha": CP.engine_version()}
    cases = {"run_id": {**good, "run_id": m1["run_id"]}, "incomplete": {**good, "feature_stage": "partial"},
             "engine": {**good, "checkpoint_engine_sha": "0" * 16}, "session": {**good, "checkpoint_session": "S0"}}
    for why, ck in cases.items():
        (tmp_path / f"{m0['run_id']}.json").write_text(json.dumps(ck), encoding="utf-8")
        o = CP._crash_record(m0, RuntimeError("lost"), str(tmp_path), session="S1")
        assert o["stale_checkpoint_ignored"] == why and "A_H_peak_m" not in o
        assert C.feature_availability(o, "A_H", "held")[1] == "unavailable"
    (tmp_path / f"{m0['run_id']}.json").write_text("{not json", encoding="utf-8")
    assert CP._crash_record(m0, RuntimeError("lost"), str(tmp_path), session="S1")["stale_checkpoint_ignored"].startswith("unreadable")
    (tmp_path / f"{m0['run_id']}.json").write_text(json.dumps(good), encoding="utf-8")
    o = CP._crash_record(m0, RuntimeError("lost"), str(tmp_path), session="S1")
    assert o["feature_recovery"] == "checkpoint" and o["A_H_peak_m"] == 0.5 and "checkpoint_session" not in o


def test_r02_continuous_loss_before_features_recovers_identical_features(CP, monkeypatch, tmp_path):
    """Continuous worker lost after the nominal propagation, before its features: the driver recovery recomputes the
    same features (same computation, same values as the stored record)."""
    out = _out(tmp_path, "D0_test", 1)
    res, _ = _run(CP, monkeypatch, out, "D0", "test", batch=1, fault="loss_continuous_before_features")
    o = res[0]
    stored = FX.result_for("D0_test", o["run_id"])
    assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "driver_nominal_recovery"
    for meth in ("A", "B", "C", "sep", "sat"):
        v, st, _ = C.feature_availability(o, meth, "continuous")
        assert st == "ok" and math.isclose(v, C.feature_value(stored, meth, "continuous"), rel_tol=RTOL, abs_tol=1e-12), meth
    assert C.truth_record(o, "continuous")["truth_status"] == "interrupted"


def test_r02_normal_path_through_the_driver_is_unchanged(CP, monkeypatch, tmp_path):
    """No fault: cmd_run with checkpointing on writes the same record as the stored v3.0 run (to round-off; same frozen
    decisions), removes the checkpoint and ledgers the exact stored line hash."""
    out = _out(tmp_path, "D4_test", 1)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1)
    o = res[0]
    assert o["worker_status"] == "ok" and "feature_recovery" not in o
    diff, missing = compare_to_stored(o, FX.result_for("D4_test", o["run_id"]))
    assert diff == [] and missing == []
    same_decisions(o, FX.result_for("D4_test", o["run_id"]))
    assert "task_retry" not in o and "feature_checkpoint_error" not in o
    assert not os.listdir(tmp_path / "results" / "feature_checkpoints" / "D4_test")
    line = open(os.path.join(out, "results", "D4_test.jsonl"), "rb").readline()
    ev = [e for e in led if e.get("event") == "completed"][0]
    assert ev["record_sha256"] == SER.stored_line_sha256(line) and "feature_recovery" not in ev
    assert math.isfinite(o["A_H_peak_m"])
