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
NRAD = 2.0 * math.pi / T


def contrec(**kw):
    r = {"cell": "D0", "worker_status": "ok", "horizon_s": T, "n_radps": NRAD}
    for b in ("cw", "eci", "nerm"):
        r.update({f"{b}_status": "ok", f"{b}_completed_horizon": True, f"{b}_t_final_s": T,
                  f"{b}_terminal_arrived": True, f"{b}_terminal_dwell_margin_s": 900.0})
    r.update({"cw_vs_eci_max_pos_disc_m": 0.4, "cw_vs_nerm_max_pos_disc_m": 0.4 + 1e-6,
              "cw_vs_eci_common_horizon_s": T, "cw_vs_nerm_common_horizon_s": T,
              "d1_peak_m": 0.3, "d1_status": "ok", "d1u_peak_m": 0.2, "d1u_status": "ok", "cres_peak_m": 0.1,
              "cres_status": "ok", "sep_m": 900.0, "feature_sat_fraction": 0.25})
    r.update(kw)
    return r


def heldrec(**kw):
    r = {"cell": "D3", "worker_status": "ok", "horizon_s": T, "n_radps": NRAD}
    for b in ("cw", "eci", "nerm"):
        r.update({f"{b}_status": "completed", f"{b}_t_end_s": T, f"{b}_arrived": False, f"{b}_dwell_margin_s": None})
    r.update({"cw_vs_eci_peak_m": 3.0, "cw_vs_nerm_peak_m": 3.0 + 1e-7, "cw_vs_eci_common_end_s": T,
              "cw_vs_nerm_common_end_s": T, "A_H_peak_m": 2.0, "A_H_status": "ok", "B_H_peak_m": 1.0, "B_H_status": "ok",
              "A_naive_peak_m": 2.0, "A_naive_status": "ok", "C_H_peak_m": 1.5, "C_H_status": "ok", "sep_m": 4000.0,
              "feature_sat_fraction": 1.0})
    r.update(kw)
    return r


def testvalidrecslabels():
    tr = C.truth_record(contrec())
    assert tr["truth_status"] == "completed_valid"
    assert C.critlabel(tr, 1.0) == "ok" and C.critlabel(tr, 0.1) == "violation"
    assert C.arrlabel(contrec(), tr)["arrival_status"] == "arrived"
    th = C.truth_record(heldrec())
    assert th["truth_status"] == "completed_valid" and C.critlabel(th, 1.0) == "violation"
    assert C.arrlabel(heldrec(), th)["arrival_status"] == "completed_non_arrival"


def testmissarrbool():
    r = contrec()
    del r["eci_terminal_arrived"], r["cw_terminal_arrived"], r["nerm_terminal_arrived"]
    tr = C.truth_record(r)
    assert C.arrlabel(r, tr)["arrival_status"] == "unresolved"
    assert CAL.arrlabel(r, CAL.truth_record(r))["arrival_status"] == "completed_non_arrival"
    r = contrec(eci_terminal_arrived=1)
    assert C.arrlabel(r, C.truth_record(r))["arrival_status"] == "unresolved"
    r = contrec(cw_terminal_dwell_margin_s=None)
    assert C.arrlabel(r, C.truth_record(r))["arrival_status"] == "unresolved"


@pytest.mark.parametrize("field,value", [("eci_status", "okay"), ("eci_status", None), ("nerm_status", 3),
                                         ("cw_status", "completed")])
def testunkstatusunr(field, value):
    tr = C.truth_record(contrec(**{field: value}))
    assert tr["truth_status"] == "unresolved"
    assert C.critlabel(tr, 1.0) == "unresolved" and C.isadverse("unresolved")


def testshortrefpair():
    assert C.truth_record(contrec(eci_t_final_s=T - 10.0))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(eci_completed_horizon=None))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(cw_vs_eci_common_horizon_s=T - 1.0))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(cw_vs_nerm_common_horizon_s=None))["truth_status"] == "unresolved"
    assert C.truth_record(heldrec(eci_t_end_s=T - 2.0))["truth_status"] == "unresolved"
    assert C.truth_record(heldrec(cw_vs_eci_common_end_s=T - 0.5))["truth_status"] == "unresolved"
    assert C.truth_record(heldrec(horizon_s=None))["truth_status"] == "unresolved"


def testnegmissdisc():
    for k, v in (("cw_vs_eci_max_pos_disc_m", -0.1), ("cw_vs_eci_max_pos_disc_m", None),
                 ("cw_vs_nerm_max_pos_disc_m", float("nan")), ("cw_vs_eci_max_pos_disc_m", True)):
        assert C.truth_record(contrec(**{k: v}))["truth_status"] == "unresolved"


def testexitfaildis():
    tr = C.truth_record(contrec(eci_status="terminated", nerm_status="terminated", eci_completed_horizon=False,
                                nerm_completed_horizon=False, eci_t_final_s=3000.0, nerm_t_final_s=3000.0))
    assert tr["truth_status"] == "physical_domain_exit" and C.critlabel(tr, 1.0) == "exit"
    assert C.arrlabel(contrec(), tr)["arrival_status"] == "not_applicable"
    tr = C.truth_record(heldrec(eci_status="domain_exit", eci_t_end_s=3000.0))
    assert tr["truth_status"] == "unresolved"
    tr = C.truth_record(heldrec(nerm_status="numerical_failure", nerm_t_end_s=100.0))
    assert tr["truth_status"] == "numerical_failure" and C.critlabel(tr, 1.0) == "failure"


@pytest.mark.parametrize("ws", ["worker_crash", "missing"])
def testopsintown(ws):
    r = {"cell": "D1", "worker_status": ws, "run_id": "R0"}
    tr = C.truth_record(r)
    assert tr["truth_status"] == "interrupted" and C.critlabel(tr, 1.0) == "interrupted"
    assert C.isadverse("interrupted")
    assert C.featavail(r, "A")[1] == "unavailable"
    assert CAL.truth_record(r)["truth_status"] == "numerical_failure"


def testexcunkworker():
    assert C.truth_record({"cell": "D0", "worker_status": "exception"})["truth_status"] == "numerical_failure"
    assert C.truth_record({"cell": "D0", "worker_status": "weird"})["truth_status"] == "unresolved"
    assert C.truth_record({"cell": "D0"})["truth_status"] == "unresolved"


def testfeatavailindep():
    r = contrec(worker_status="reference_failed", eci_status=None, nerm_status=None, reference_error="boom")
    v, st, _ = C.featavail(r, "A")
    assert st == "ok" and v == 0.3
    assert C.truth_record(r)["truth_status"] == "unresolved"
    r = heldrec(eci_status="numerical_failure", eci_t_end_s=10.0)
    assert C.featavail(r, "A_H")[1] == "ok"


def testfeatavailrules():
    assert C.featavail(contrec(cw_t_final_s=T - 5.0), "A")[1] == "unavailable"
    assert C.featavail(contrec(cw_completed_horizon=False), "A")[1] == "unavailable"
    assert C.featavail(contrec(d1_status="failed"), "A")[1] == "unavailable"
    assert C.featavail(contrec(d1_peak_m=-1.0), "A")[1] == "unavailable"
    assert C.featavail(contrec(d1_peak_m=float("inf")), "A")[1] == "unavailable"
    assert C.featavail(contrec(worker_status="exception"), "A")[1] == "unavailable"
    assert C.featavail(heldrec(feature_sat_fraction=1.0), "sat") == (1.0, "ok", "")
    assert C.featavail(heldrec(feature_sat_fraction=0.0), "sat")[1] == "ok"
    assert C.featavail(heldrec(feature_sat_fraction=1.0 + 1e-12), "sat")[1] == "unavailable"
    assert C.featavail(heldrec(cw_t_end_s=T - 1.0), "A_H")[1] == "unavailable"
    with pytest.raises(ValueError):
        C.featavail(heldrec(), "A")


def testexplkindmism():
    assert C.truth_record(heldrec(), kind="continuous")["truth_status"] == "unresolved"
    assert C.truth_record(heldrec(), kind="held")["truth_status"] == "completed_valid"
    with pytest.raises(ValueError):
        C.truth_record(heldrec(), kind="sampled")


def testlabelpartident():
    labels = ["ok", "violation", "exit", "failure", "unresolved", "interrupted", "ok"]
    acc = [True, True, False, True, True, True, False]
    cnt = C.labelcounts(acc, labels)
    assert sum(cnt[f"acc_{x}"] for x in C.LABELS) == sum(acc)
    assert sum(cnt.values()) == len(labels)
    with pytest.raises(ValueError):
        C.labelcounts([True], ["completed"])
    with pytest.raises(ValueError):
        C.checklabelpart({"acc_ok": 2}, 3)
    with pytest.raises(ValueError):
        C.isadverse("bogus")
    assert set(C.ADVLABELS) == set(C.LABELS) - {"ok"}


@pytest.mark.parametrize("cell", ["D0", "D1", "D3", "D4", "D5"])
def testcontrmatch(cell):
    kind = C.kindof({"cell": cell})
    recs = FX.results(f"{cell}cal")
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
            assert CAL.critlabel(t0, tol) == C.critlabel(t1, tol)
        meths = ("A", "B", "C", "sep", "sat") if kind == "continuous" else ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")
        for m in meths:
            a, b = CAL.feature_value(r, m, kind), C.feature_value(r, m, kind)
            assert (math.isnan(a) and math.isnan(b)) or a == b


def manif(cell, split, n):
    return FX.manifest(f"{cell}{split}", n)


@pytest.fixture(scope="module")
def CP():
    import campaign
    return campaign


def testcontreffail(CP, monkeypatch):
    m = manif("D0", "test", 1)[0]
    real = CP.propag

    def boom(scn, cfg, law, models):
        if "eci" in models:
            raise RuntimeError("injected reference failure")
        return real(scn, cfg, law, models)
    monkeypatch.setattr(CP, "propag", boom)
    o = CP.workcont(m)[0]
    assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "reference"
    assert "injected reference failure" in o["reference_error"] and o["feature_stage"] == "complete"
    for meth in ("A", "B", "C", "sep", "sat"):
        assert C.featavail(o, meth, "continuous")[1] == "ok", meth
    tr = C.truth_record(o, "continuous")
    assert tr["truth_status"] == "unresolved" and C.critlabel(tr, 1.0) == "unresolved"
    assert math.isnan(CAL.feature_value({**o, "worker_status": "exception"}, "A", "continuous"))


def testcontnomfail(CP, monkeypatch):
    m = manif("D0", "test", 1)[0]

    def boom(scn, cfg, law, models):
        raise RuntimeError("injected nominal failure")
    monkeypatch.setattr(CP, "propag", boom)
    o = CP.workcont(m)[0]
    assert o["worker_status"] == "exception" and o["failed_stage"] == "nominal"
    assert C.featavail(o, "A", "continuous")[1] == "unavailable"
    assert C.truth_record(o, "continuous")["truth_status"] == "numerical_failure"


def testcontmetricfail(CP, monkeypatch):
    import cwvalid3.continuous as CONT
    m = manif("D0", "test", 1)[0]

    def boom(*a, **k):
        raise ValueError("injected metric failure")
    monkeypatch.setattr(CONT, "evalgen", boom)
    o = CP.workcont(m)[0]
    assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "metrics"
    assert C.featavail(o, "A", "continuous")[1] == "ok"


def testheldrefmetric(CP, monkeypatch):
    import cwvalid3.heldmet as HM
    recs = manif("D4", "test", 2)

    def boom(*a, **k):
        raise RuntimeError("injected held metric failure")
    monkeypatch.setattr(HM, "peakdisc", boom)
    outs = CP.workheld(recs)
    for o in outs:
        assert o["worker_status"] == "reference_failed" and o["failed_stage"] == "reference"
        for meth in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"):
            assert C.featavail(o, meth, "held")[1] == "ok", meth
        assert C.truth_record(o, "held")["truth_status"] == "unresolved"


def testheldbatchexc(CP, monkeypatch):
    import cwvalid3.sampled as SA
    recs = manif("D4", "test", 2)
    real = SA.HeldBatch.run
    badid = recs[1]["scenario"]["rel0"]

    def run(self):
        if self.N > 1:
            raise FloatingPointError("injected batch failure")
        if np.allclose(self.rel0[0], badid):
            raise FloatingPointError("injected single-case failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    outs = CP.workheld(recs)
    assert [o["case_id"] for o in outs] == [r["case_id"] for r in recs]
    assert outs[0]["worker_status"] == "ok" and "case by case" in outs[0]["batch_retry"]
    assert outs[1]["worker_status"] == "exception" and outs[1]["failed_stage"] == "features"
    assert C.truth_record(outs[1], "held")["truth_status"] == "numerical_failure"


def testcamprefwrite(CP):
    for sub in ("raw/results", "raw/manifests", "", "reports"):
        with pytest.raises(SystemExit):
            CP.outpaths(os.path.join(CP.V3, sub))
    assert CP.outpaths(os.path.join(CP.V3, "v31", "raw", "campaign"))["res"].endswith("results")


def testheldreffail(CP, monkeypatch):
    import cwvalid3.sampled as SA
    recs = manif("D4", "test", 1)
    real = SA.HeldBatch.run

    def run(self):
        if self.haseci:
            raise FloatingPointError("injected reference-branch failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    o = CP.workheld(recs)[0]
    assert o["worker_status"] == "reference_failed" and o["feature_source"] == "cw_estimator_only_fallback"
    for meth in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"):
        assert C.featavail(o, meth, "held")[1] == "ok", meth
    assert C.truth_record(o, "held")["truth_status"] == "unresolved"


def testheldnaivepass(CP, monkeypatch):
    import cwvalid3.sampled as SA
    recs = manif("D4", "test", 1)
    real = SA.HeldBatch.run

    def run(self):
        if "A_naive" in self.ests:
            raise FloatingPointError("injected naive-pass failure")
        return real(self)
    monkeypatch.setattr(SA.HeldBatch, "run", run)
    o = CP.workheld(recs)[0]
    assert o["worker_status"] == "ok" and o["A_naive_status"] == "naive_pass_failed"
    assert C.featavail(o, "A_naive", "held")[1] == "unavailable"
    assert C.featavail(o, "A_H", "held")[1] == "ok"
    assert C.truth_record(o, "held")["truth_status"] in ("completed_valid", "physical_domain_exit")


def testfeatckptwritt(CP, monkeypatch, tmp_path):
    from cwvalid3 import serialization as SER
    m = manif("D0", "test", 1)[0]
    monkeypatch.setenv(CP.CKPTENV, str(tmp_path))
    o = CP.workcont(m)[0]
    ck = tmp_path / f"{m['run_id']}.json"
    assert ck.exists()
    stored = json.loads(ck.read_text(encoding="utf-8"))
    assert stored["feature_stage"] == "complete" and stored["d1_peak_m"] == o["d1_peak_m"]
    assert "eci_status" not in stored
    crash = CP.crashrec(m, RuntimeError("BrokenProcessPool"), str(tmp_path))
    assert crash["worker_status"] == "worker_crash" and crash["feature_checkpoint_recovered"] is True
    assert C.truth_record(crash, "continuous")["truth_status"] == "interrupted"
    assert C.feature_value(crash, "A", "continuous") == o["d1_peak_m"]
    nock = CP.crashrec({**m, "run_id": "Rnone"}, RuntimeError("x"), str(tmp_path))
    assert C.featavail(nock, "A", "continuous")[1] == "unavailable"
    assert SER.volat("batch_members_sha256")


def testhorincok():
    for kw in ({"horizon_s": 0.0}, {"horizon_s": -5.0}, {"horizon_s": 10.0}, {"n_radps": None}):
        r = contrec(**kw)
        assert C.truth_record(r)["truth_status"] == "unresolved", kw
        assert C.featavail(r, "A")[1] == "unavailable", kw
    assert C.truth_record(heldrec(horizon_s=10.0))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(failed_stage="reference"))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(reference_error="boom"))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(cw_numerical_failure=True))["truth_status"] == "unresolved"
    assert C.truth_record(contrec(cw_numerical_failure=False))["truth_status"] == "completed_valid"


def testv31evalrule():
    out = C.evalrule([0.1, 0.1, 0.1, 0.1, 5.0, float("nan")], ["ok", "interrupted", "unresolved", "ok", "violation", "ok"], 1.0)
    assert out["n_accepted"] == 4 and out["conservative_adverse_accepted"] == 2
    assert sum(out[f"acc_{x}"] for x in C.LABELS) == out["n_accepted"]
    assert sum(out[f"ref_{x}"] for x in C.LABELS) == out["n_referred"] == 2
    with pytest.raises(ValueError):
        C.evalrule([0.1], ["bogus"], 1.0)
    assert CAL.evalrule([0.1, 0.1], ["ok", "interrupted"], 1.0)["conservative_adverse_accepted"] == 0


def testv31loaderkeeps(tmp_path):
    sys.path.insert(0, os.path.join(REPO, "experiments", "v31"))
    import v31data
    from cwvalid3 import serialization as SER
    man = manif("D0", "test", 3)
    stored = {r["run_id"]: r for r in FX.results("D0test") if r["run_id"] in {m["run_id"] for m in man}}
    assert len(stored) == 3
    (tmp_path / "manifests").mkdir()
    (tmp_path / "results").mkdir()
    with open(tmp_path / "manifests" / "D0t1.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for m in man:
            SER.write_record(fh, m, schema=None)
    a, b = stored[man[0]["run_id"]], stored[man[1]["run_id"]]
    reffailed = {k: v for k, v in a.items() if not k.startswith(("eci_", "nerm_", "cw_vs_", "nerm_vs_"))}
    reffailed.update(worker_status="reference_failed", failed_stage="reference", reference_error="injected")
    with open(tmp_path / "results" / "D0t1.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        SER.write_record(fh, reffailed)
        SER.write_record(fh, b)
    df, rc = v31data.loadsplit("D0", "t1", root=str(tmp_path))
    assert rc["n_missing"] == 1
    r0, r1, r2 = (df[df.run_id == m["run_id"]].iloc[0] for m in man)
    assert r0.truth_status == "unresolved" and r0.f_A_status == "ok" and r0.f_A == a["d1_peak_m"]
    assert r2.worker_status == "missing" and r2.truth_status == "interrupted" and r2.label_1m == "interrupted"
    assert r1.truth_status == C.truth_record(b, "continuous")["truth_status"]
    ev = v31data.evaluate(df, "A", "1m", 1e9)
    assert ev["n_accepted"] == 2 and ev["acc_unresolved"] == 1


@pytest.mark.parametrize("variant", ["lower", "upper", "mixed"])
def testguardscaseins(CP, variant):
    if os.name != "nt":
        pytest.skip("case-insensitive filesystem check")
    sys.path.insert(0, os.path.join(REPO, "experiments", "v31"))
    import v31paths
    target = os.path.join(CP.V3, "raw", "results")
    t = {"lower": target.lower(), "upper": target.upper(), "mixed": os.path.join(os.path.dirname(CP.V3), "V3", "raw", "Results")}[variant]
    with pytest.raises(SystemExit):
        CP.v30treeguard(t)
    with pytest.raises(PermissionError):
        v31paths.guardv30(os.path.join(t, "x.csv"))
    ok = os.path.join(CP.V3, "v31", "raw").upper()
    assert CP.v30treeguard(ok)
