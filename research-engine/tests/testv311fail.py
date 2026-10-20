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
HELDMETH = ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")
CEX = "Caf93a5f0b849d73a"


@pytest.fixture(scope="module")
def CP():
    import campaign
    return campaign


RUNS = {}


@pytest.fixture
def cachedbatch(monkeypatch):
    import cwvalid3.sampled as SA
    real = SA.HeldBatch

    class Cached(real):
        post = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.ckey = hashlib.sha256(pickle.dumps((a, sorted(k.items())), protocol=4)).hexdigest()

        def run(self):
            if self.ckey not in RUNS:
                real.run(self)
                RUNS[self.ckey] = copy.deepcopy(self.__dict__)
            else:
                self.__dict__.update(copy.deepcopy(RUNS[self.ckey]))
            if type(self).post is not None:
                type(self).post(self)
            return self
    monkeypatch.setattr(SA, "HeldBatch", Cached)
    return Cached


def cex():
    m = FX.manifest("D4test", 1)[0]
    assert m["case_id"] == CEX
    return m


def decisions(o):
    out = {}
    for key, s in FX.freeze()["selection"].items():
        cell, meth, crit = key.split("|")
        if cell != o["cell"] or s.get("threshold") is None:
            continue
        v, st, _ = C.featavail(o, meth, "held")
        out[key] = bool(st == "ok" and v <= s["threshold"])
    return out


RTOL = 1e-9


def _close(x, y):
    if isinstance(x, bool) or isinstance(y, bool) or not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(_close(p, q) for p, q in zip(x, y))
        return x == y
    return math.isclose(x, y, rel_tol=RTOL, abs_tol=1e-12)


def cmpstored(o, stored):
    a, b = SER.normalize(o), SER.normalize(stored)
    keys = {k for k in a if not SER.volat(k) and k not in SER.V31ONLYFIELDS} & {k for k in b if not SER.volat(k)}
    return sorted(k for k in keys if not _close(a[k], b[k])), sorted({k for k in b if not SER.volat(k)} - set(a))


def samedecs(o, ref, kind="held"):
    sel = FX.freeze()["selection"]
    for key, s in sel.items():
        cell, meth, crit = key.split("|")
        if cell != o["cell"] or s.get("threshold") is None:
            continue
        va, sa, _ = C.featavail(o, meth, kind)
        vb, sb, _ = C.featavail(ref, meth, kind)
        assert (sa == "ok" and va <= s["threshold"]) == (sb == "ok" and vb <= s["threshold"]), key


@pytest.fixture
def normal(CP, cachedbatch):
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "feature_errors" not in o
    return o


def checksurv(o, normal, affected, status=None):
    for meth in HELDMETH:
        v, st, why = C.featavail(o, meth, "held")
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


def testr01normpath(normal):
    stored = FX.resultfor("D4test", normal["run_id"])
    diff, missing = cmpstored(normal, stored)
    assert diff == [] and missing == []
    assert math.isclose(normal["A_H_peak_m"], stored["A_H_peak_m"], rel_tol=RTOL)
    samedecs(normal, stored)
    assert decisions(normal)["D4|A_H|1m"] is True
    assert C.truth_record(normal, "held")["truth_status"] == "completed_valid"


def testr01cachedprop(CP, normal, cachedbatch, monkeypatch):
    import cwvalid3.sampled as SA
    monkeypatch.setattr(SA, "HeldBatch", cachedbatch.__mro__[1])
    fresh = CP.workheld([cex()])[0]
    assert SER.physics_sha256(fresh) == SER.physics_sha256(normal)


def testr01naivepass(CP, normal, cachedbatch, monkeypatch):
    cachedrun = cachedbatch.run

    def run(self):
        if "A_naive" in self.ests:
            raise FloatingPointError("injected naive-pass failure")
        return cachedrun(self)
    monkeypatch.setattr(cachedbatch, "run", run)
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "injected naive-pass failure" in o["naive_pass_error"]
    assert "A_naive" in o["feature_errors"]
    checksurv(o, normal, {"A_naive"}, {"A_naive": "naive_pass_failed"})


def testr01naivecase(CP, normal, monkeypatch):
    import cwvalid3.heldmet as HM
    real = HM.CaseView

    def view(batch, i, *a, **k):
        if "A_naive" in batch.ests:
            raise IndexError("injected A_naive case-view failure")
        return real(batch, i, *a, **k)
    monkeypatch.setattr(HM, "CaseView", view)
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "failed_stage" not in o and "error" not in o
    assert "injected A_naive case-view failure" in o["feature_errors"]["A_naive"]
    checksurv(o, normal, {"A_naive"}, {"A_naive": "naive_feature_failed"})


def testr01naiveest(CP, normal, monkeypatch):
    import cwvalid3.heldmet as HM
    real = HM.estimfeat

    def feat(cv, est):
        if est == "A_naive":
            raise RuntimeError("injected A_naive extraction failure")
        return real(cv, est)
    monkeypatch.setattr(HM, "estimfeat", feat)
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "failed_stage" not in o
    assert "injected A_naive extraction failure" in o["feature_errors"]["A_naive"]
    checksurv(o, normal, {"A_naive"}, {"A_naive": "naive_feature_failed"})
    assert decisions(o)["D4|A_H|1m"] is True


def testr01naivediag(CP, normal, monkeypatch):
    def boom(*a, **k):
        raise ValueError("injected naive-pass diagnostic failure")
    monkeypatch.setattr(CP, "naivepassdiag", boom)
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok"
    assert o["cw_naive_pass_max_state_diff_m"] != o["cw_naive_pass_max_state_diff_m"]
    assert "injected naive-pass diagnostic failure" in o["feature_errors"]["cw_naive_pass_diagnostic"]
    checksurv(o, normal, set(), {"A_naive": "ok"})
    assert o["A_naive_peak_m"] == normal["A_naive_peak_m"]


def testr01satfail(CP, normal, monkeypatch):
    import cwvalid3.heldmet as HM

    def boom(cv):
        raise RuntimeError("injected saturation-summary failure")
    monkeypatch.setattr(HM, "satsumm", boom)
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "saturation" in o["feature_errors"]
    checksurv(o, normal, {"sat"})


def testr01primest(CP, normal, monkeypatch):
    import cwvalid3.heldmet as HM
    real = HM.estimfeat

    def feat(cv, est):
        if est == "A_H":
            raise RuntimeError("injected A_H extraction failure")
        return real(cv, est)
    monkeypatch.setattr(HM, "estimfeat", feat)
    o = CP.workheld([cex()])[0]
    checksurv(o, normal, {"A_H"}, {"A_H": "feature_extraction_failed"})
    assert decisions(normal)["D4|A_H|1m"] is True and decisions(o)["D4|A_H|1m"] is False


def testr01nomcw(CP, cachedbatch, monkeypatch):
    def mark(self):
        if self.haseci:
            self.status["cw"][0] = "numerical_failure"
            self.t_end["cw"][0] = 1000.0
    monkeypatch.setattr(cachedbatch, "post", staticmethod(mark))
    o = CP.workheld([cex()])[0]
    for meth in HELDMETH:
        assert C.featavail(o, meth, "held")[1] == "unavailable", meth
    assert C.truth_record(o, "held")["truth_status"] == "numerical_failure"
    assert not any(decisions(o).values())


def testr01ckptwrite(CP, normal, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(CP.CKPTENV, str(blocker / "ck"))
    o = CP.workheld([cex()])[0]
    assert o["worker_status"] == "ok" and "error" not in o and o["feature_checkpoint_error"]
    checksurv(o, normal, set(), {"A_H": "ok", "A_naive": "ok"})
    assert SER.physics_sha256(o) == SER.physics_sha256(normal)


def testr01contckpt(CP, monkeypatch, tmp_path):
    m = FX.manifest("D0test", 1)[0]
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(CP.CKPTENV, str(blocker / "ck"))
    o = CP.workcont(m)[0]
    assert o["worker_status"] == "ok" and o["feature_checkpoint_error"]
    for meth in ("A", "B", "C", "sep", "sat"):
        assert C.featavail(o, meth, "continuous")[1] == "ok", meth
    diff, missing = cmpstored(o, FX.resultfor("D0test", m["run_id"]))
    assert diff == [] and missing == []


def testr01featerrs(normal):
    r = {**normal, "feature_errors": {"A_naive": "x"}, "A_naive_status": "naive_feature_failed"}
    assert C.truth_record(r, "held")["truth_status"] == "completed_valid"
    assert C.featavail(r, "A_naive", "held")[1] == "unavailable"
    assert C.truth_record({**normal, "error": "x"}, "held")["truth_status"] == "unresolved"


def _out(tmp_path, cs, n):
    (tmp_path / "manifests").mkdir()
    lines = FX.manlines(cs)[:n]
    assert [json.loads(b)["index"] for b in lines] == [m["index"] for m in FX.manifest(cs, n)]
    (tmp_path / "manifests" / f"{cs}.jsonl").write_bytes(b"".join(b + b"\n" for b in lines))
    return str(tmp_path)


def _run(CP, monkeypatch, out, cell, split, batch, fault=None, workers=2):
    for k in (CP.CKPTENV, CP.SESSENV, CP.FAULTENV):
        monkeypatch.setenv(k, "")
    monkeypatch.setenv(CP.FAULTENV, fault or "")
    CP.cmdrun(cell, split, workers=workers, batch=batch, out=out)
    res = [json.loads(x) for x in open(os.path.join(out, "results", f"{cell}{split}.jsonl"), encoding="utf-8")]
    led = [json.loads(x) for x in open(os.path.join(out, "ledger.jsonl"), encoding="utf-8")]
    return {r["index"]: r for r in res}, led


def testr02processloss(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4test", 2)
    m0, m1 = FX.manifest("D4test", 2)
    ckdir = tmp_path / "results" / "checkpoints" / "D4test"
    ckdir.mkdir(parents=True)
    stale = {**{k: m1[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")},
             "feature_stage": "complete", "A_H_peak_m": 123.0, "A_H_status": "ok", "checkpoint_session": "an-earlier-run",
             "checkpoint_engine_sha": CP.enginever()}
    (ckdir / f"{m1['run_id']}.json").write_text(json.dumps(stale), encoding="utf-8")
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault="loss_held_after_combined")
    assert sorted(res) == [0, 1]
    stored = {i: FX.resultfor("D4test", m["run_id"]) for i, m in ((0, m0), (1, m1))}
    for i, o in res.items():
        assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "driver_nominal_recovery"
        assert o["task_retry"] == "lost again when re-executed alone in a fresh process"
        assert "feature_checkpoint_recovered" not in o and "eci_status" not in o and "cw_vs_eci_peak_m" not in o
        for meth in HELDMETH:
            v, st, why = C.featavail(o, meth, "held")
            assert st == "ok", (meth, why)
            ref = C.feature_value(stored[i], meth, "held")
            assert abs(v - ref) <= 1e-9 * max(abs(ref), 1e-12), (meth, v, ref)
        tr = C.truth_record(o, "held")
        assert tr["truth_status"] == "interrupted" and C.critlabel(tr, 1.0) == "interrupted"
    assert res[1]["stale_checkpoint_ignored"] == "session" and res[1]["A_H_peak_m"] != 123.0
    thr = FX.freeze()["selection"]["D4|A_H|1m"]["threshold"]
    ev = C.evalrule([C.feature_value(res[0], "A_H", "held")], ["interrupted"], thr)
    assert ev["n_accepted"] == 1 and ev["acc_interrupted"] == 1 and ev["conservative_adverse_accepted"] == 1
    ev = [e for e in led if e.get("event") == "completed"]
    assert len(ev) == 2 and all(e["feature_recovery"] == "driver_nominal_recovery" for e in ev)


def testr02tasksbroken(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4test", 2)
    m0, m1 = FX.manifest("D4test", 2)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault=f"loss_held_after_combined@{m0['run_id']}", workers=1)
    assert res[0]["worker_status"] == "worker_crash" and res[0]["feature_recovery"] == "driver_nominal_recovery"
    o = res[1]
    assert o["worker_status"] == "ok" and "feature_recovery" not in o and o["task_retry"].startswith("re-executed alone")
    diff, missing = cmpstored(o, FX.resultfor("D4test", m1["run_id"]))
    assert diff == [] and missing == []
    assert C.truth_record(o, "held")["truth_status"] == C.truth_record(FX.resultfor("D4test", m1["run_id"]), "held")["truth_status"]
    assert sum(e.get("event") == "completed" for e in led) == 2


def testr02procloss(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4test", 2)
    res, _ = _run(CP, monkeypatch, out, "D4", "test", batch=2, fault="loss_held_after_features")
    for o in res.values():
        assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "checkpoint"
        assert o["feature_checkpoint_recovered"] is True and "checkpoint_session" not in o
        for meth in ("A_H", "B_H", "C_H", "sep", "sat"):
            assert C.featavail(o, meth, "held")[1] == "ok", meth
        assert C.featavail(o, "A_naive", "held")[1] == "unavailable"
        assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def testr02genrecov(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4test", 1)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1, fault="loss_held_after_combined,fail_recovery")
    o = res[0]
    assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "failed"
    assert "injected fault at fail_recovery" in o["feature_recovery_error"]
    for meth in HELDMETH:
        assert C.featavail(o, meth, "held")[1] == "unavailable"
    assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def testr02recovnom(CP, cachedbatch, monkeypatch):
    m = cex()

    def mark(self):
        self.status["cw"][0] = "numerical_failure"
    monkeypatch.setattr(cachedbatch, "post", staticmethod(mark))
    got = CP.recfeats([m])[m["run_id"]]
    o = CP.mergefeats(CP.crashrec(m, RuntimeError("lost"), None), got, "driver_nominal_recovery")
    assert o["A_H_status"] == "nominal_failed"
    for meth in HELDMETH:
        assert C.featavail(o, meth, "held")[1] == "unavailable", meth
    assert C.truth_record(o, "held")["truth_status"] == "interrupted"


def testr02stalewrong(CP, tmp_path):
    m0, m1 = FX.manifest("D4test", 2)
    good = {**{k: m0[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")}, "feature_stage": "complete",
            "A_H_peak_m": 0.5, "A_H_status": "ok", "checkpoint_session": "S1", "checkpoint_engine_sha": CP.enginever()}
    cas = {"run_id": {**good, "run_id": m1["run_id"]}, "incomplete": {**good, "feature_stage": "partial"},
           "engine": {**good, "checkpoint_engine_sha": "0" * 16}, "session": {**good, "checkpoint_session": "S0"}}
    for why, ck in cas.items():
        (tmp_path / f"{m0['run_id']}.json").write_text(json.dumps(ck), encoding="utf-8")
        o = CP.crashrec(m0, RuntimeError("lost"), str(tmp_path), session="S1")
        assert o["stale_checkpoint_ignored"] == why and "A_H_peak_m" not in o
        assert C.featavail(o, "A_H", "held")[1] == "unavailable"
    (tmp_path / f"{m0['run_id']}.json").write_text("{not json", encoding="utf-8")
    assert CP.crashrec(m0, RuntimeError("lost"), str(tmp_path), session="S1")["stale_checkpoint_ignored"].startswith("unreadable")
    (tmp_path / f"{m0['run_id']}.json").write_text(json.dumps(good), encoding="utf-8")
    o = CP.crashrec(m0, RuntimeError("lost"), str(tmp_path), session="S1")
    assert o["feature_recovery"] == "checkpoint" and o["A_H_peak_m"] == 0.5 and "checkpoint_session" not in o


def testr02contloss(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D0test", 1)
    res, _ = _run(CP, monkeypatch, out, "D0", "test", batch=1, fault="loss_continuous_before_features")
    o = res[0]
    stored = FX.resultfor("D0test", o["run_id"])
    assert o["worker_status"] == "worker_crash" and o["feature_recovery"] == "driver_nominal_recovery"
    for meth in ("A", "B", "C", "sep", "sat"):
        v, st, _ = C.featavail(o, meth, "continuous")
        assert st == "ok" and math.isclose(v, C.feature_value(stored, meth, "continuous"), rel_tol=RTOL, abs_tol=1e-12), meth
    assert C.truth_record(o, "continuous")["truth_status"] == "interrupted"


def testr02normpath(CP, monkeypatch, tmp_path):
    out = _out(tmp_path, "D4test", 1)
    res, led = _run(CP, monkeypatch, out, "D4", "test", batch=1)
    o = res[0]
    assert o["worker_status"] == "ok" and "feature_recovery" not in o
    diff, missing = cmpstored(o, FX.resultfor("D4test", o["run_id"]))
    assert diff == [] and missing == []
    samedecs(o, FX.resultfor("D4test", o["run_id"]))
    assert "task_retry" not in o and "feature_checkpoint_error" not in o
    assert not os.listdir(tmp_path / "results" / "checkpoints" / "D4test")
    line = open(os.path.join(out, "results", "D4test.jsonl"), "rb").readline()
    ev = [e for e in led if e.get("event") == "completed"][0]
    assert ev["record_sha256"] == SER.stored_line_sha256(line) and "feature_recovery" not in ev
    assert math.isfinite(o["A_H_peak_m"])
