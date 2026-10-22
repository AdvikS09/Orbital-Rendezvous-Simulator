import json
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "v2"))

from cwvalid.estimate import firstorderdisc
from cwvalid.estimate2 import estvar, residonlydiag, unsatjacest
from cwvalid.experiment import RunCfg, makelaw, runbranch
from cwvalid.scenario import axisscen

RES = os.path.join(os.path.dirname(REPO), "results")


def _cw(axis, sep, wn=6.0, am=0.05):
    scn = axisscen("t", axis, sep, wn, am)
    sims, _ = runbranch(scn, RunCfg(models=("cw",), reference="cw", crosscheck=None))
    return scn, sims["cw"]


def testvarbequals():
    scn, sim = _cw("+R", 300.0)
    assert not sim.satswitchtimes
    law = makelaw(scn)
    a = firstorderdisc(sim, law, scn.n, scn.R, scn.horizon)["d1_peak_m"]
    b = unsatjacest(sim, law, scn.n, scn.R, scn.horizon)["peak_m"]
    assert b == pytest.approx(a, rel=1e-12)


def testvarbdiff():
    scn, sim = _cw("+R", 3000.0)
    assert sim.satswitchtimes
    law = makelaw(scn)
    a = firstorderdisc(sim, law, scn.n, scn.R, scn.horizon)["d1_peak_m"]
    b = unsatjacest(sim, law, scn.n, scn.R, scn.horizon)["peak_m"]
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(b / a - 1.0) > 1e-3


def testvarcmatch():
    from cwvalid.dynamics import acccw, accnerm
    scn, sim = _cw("+V", 2000.0)
    c = residonlydiag(sim, scn.n, scn.R, scn.horizon)
    t = np.linspace(0.0, scn.horizon, 40001)
    X = sim.states(t)
    r = np.array([np.subtract(accnerm(scn.n, scn.R, *x), acccw(scn.n, scn.R, *x)) for x in X])
    v = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t)[:, None], axis=0)])
    e = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t)[:, None], axis=0)])
    peak = np.linalg.norm(e, axis=1).max()
    assert c["peak_m"] == pytest.approx(peak, rel=1e-4)


def testvarcscales():
    s1 = _cw("+H", 100.0)
    s2 = _cw("+H", 200.0)
    c1 = residonlydiag(s1[1], s1[0].n, s1[0].R, s1[0].horizon)["peak_m"]
    c2 = residonlydiag(s2[1], s2[0].n, s2[0].R, s2[0].horizon)["peak_m"]
    assert c2 / c1 == pytest.approx(4.0, rel=0.02)


def testnonfintrajgives(monkeypatch):
    scn, sim = _cw("+R", 300.0)
    import cwvalid.estimate2 as E

    def bad(*a, **k):
        return (float("nan"),) * 3
    monkeypatch.setattr(E, "accnerm", bad)
    r = E.residonlydiag(sim, scn.n, scn.R, scn.horizon)
    assert r["status"] == "failed" and math.isnan(r["peak_m"])
    out = estvar(sim, makelaw(scn), scn.n, scn.R, scn.horizon)
    assert out["cres_status"] == "failed" and math.isnan(out["cres_peak_m"])


def testfitedgecases():
    import estfit as F
    t, st = F.upper(np.array([1.0, 2.0, np.nan]), np.array([False, False, True]))
    assert t is None and st.startswith("not_calibratable")
    t, st = F.upper(np.array([1.0, 2.0, 5.0]), np.array([False, True, True]))
    assert st == "ok" and t == pytest.approx(0.8 * 2.0)
    vals = np.array([3.0, 1.0, np.nan, 2.0])
    t, st = F.accquant(vals, 0.5)
    assert st == "ok" and t == 2.0 and np.mean(np.isfinite(vals) & (vals <= t)) >= 0.5
    t, st = F.accquant(vals, 1.0)
    assert t is None and st.startswith("not_calibratable")


def testnanfeatsalways():
    import estfit as F
    import pandas as pd
    D = pd.DataFrame({"d1": [0.1, np.nan, 5.0], "cw_arrived": [True, True, True], "cw_margin_s": [1000.0, 1000.0, np.nan]})
    acc = F.ruleaccept(D, {"feature": "d1", "threshold": 1.0, "status": "ok", "combined": False})
    assert acc.tolist() == [True, False, False]
    acc = F.ruleaccept(D, {"feature": "d1", "threshold": 10.0, "status": "ok", "combined": True, "m_min_s": 380.0})
    assert acc.tolist() == [True, False, False]
    assert F.ruleaccept(D, {"feature": "d1", "threshold": None, "status": "not_calibratable"}) is None


def testfailawaremetrics():
    import esteval as EV
    import pandas as pd
    L = pd.DataFrame({"traj_ok_1": [True, False, True, True], "adequate": [True, False, True, True],
                      "false_fav": [False, False, False, False], "false_unfav": [False, False, True, False],
                      "label_valid": [True, True, True, False], "ref_termination": ["horizon", "altitude_floor", "horizon", "horizon"],
                      "ref_status": ["ok", "terminated", "ok", "failed"], "disc_m": [0.1, 5.0, 0.2, np.nan]})
    acc = np.array([True, True, False, True])
    m = EV.metrics(acc, L, "traj_ok_1", featfinite=np.array([True, True, True, True]))
    assert m["n_total"] == 4 and m["n_label_invalid"] == 1 and m["n_evaluated"] == 3
    assert m["accepted"] == 2 and m["incorrect_accepted"] == 1
    assert m["referred"] == 1 and m["unnecessary_referrals"] == 1
    assert m["ref_domain_exits"] == 1 and m["ref_domain_exits_accepted"] == 1
    z = EV.metrics(np.zeros(4, bool), L, "traj_ok_1", featfinite=np.ones(4, bool))
    assert z["accepted"] == 0 and math.isnan(z["incorrect_rate"])


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "frozenv2", "freezerecord.json")), reason="v2 not frozen yet")
def testv2freezehashes():
    sys.path.insert(0, os.path.join(REPO, "experiments", "v2"))
    import v2common as C
    rec = json.load(open(os.path.join(C.FROZENV2DIR, "freezerecord.json")))
    assert C.sha256(C.FROZENV2) == rec["estimator_v2_frozen_sha256"]
    fr = json.load(open(C.FROZENV2))
    assert fr["v1_frozen_sha256"] == C.sha256(C.FROZENV1)
    assert fr["A_threshold_equals_v1_frozen"] is True


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "freshtest", "setdefinition.json")), reason="fresh test not generated yet")
def testfreshtestiso():
    import v2common as C
    sd = json.load(open(os.path.join(C.RAW, "freshtest", "setdefinition.json")))
    assert sd["overlap_with_archive"] == []
    assert C.sha256(os.path.join(C.RAW, "freshtest", "scenarios.jsonl")) == sd["scenarios_sha256"]
    assert sd["estimator_v2_frozen_sha256"] == C.sha256(C.FROZENV2)
    fr = json.load(open(C.FROZENV2))
    assert fr["frozen_at"] <= sd["written"]
    fresh = {json.loads(l)["sid"] for l in open(os.path.join(C.RAW, "freshtest", "scenarios.jsonl")) if l.strip()}
    for c in ("maingrid/grid", "maingrid/refine", "maingrid/refinebands", "screening/dev"):
        dev = {json.loads(l)["sid"] for l in open(os.path.join(C.RESULTS, c, "scenarios.jsonl")) if l.strip()}
        assert not (fresh & dev)


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "families", "familiesrecord.json")), reason="families not declared")
def testboundfamsdecl():
    import v2common as C
    rec = json.load(open(os.path.join(C.RAW, "families", "familiesrecord.json")))
    assert C.sha256(os.path.join(C.RAW, "families", "families.json")) == rec["families_sha256"]


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "v2", "raw", "verifier", "selection.json")), reason="verifier cases not selected")
def testverselsame():
    import v2common as C
    rec = json.load(open(os.path.join(C.RAW, "verifier", "selection.json")))
    assert C.sha256(os.path.join(C.RAW, "verifier", "cases.json")) == rec["cases_sha256"]
