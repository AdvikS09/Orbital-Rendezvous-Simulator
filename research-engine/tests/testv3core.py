import math

import numpy as np
import pytest

from cwvalid.control import PDLaw
from cwvalid3 import calibration as CAL
from cwvalid3 import ledger as L
from cwvalid3.controllers import lqrlaw, pdlaw, unsatstab
from cwvalid3.domains import CELLS, makecases, meanmotionalt, stream, checkunused

N400 = meanmotionalt(400e3)


def testpdmatcmds():
    rng = np.random.default_rng(1)
    for _ in range(200):
        wn = rng.uniform(3, 10) * N400
        am = float(np.exp(rng.uniform(np.log(0.005), np.log(0.2))))
        s = rng.normal(size=6) * np.r_[[5000.0] * 3, [5.0] * 3]
        p, m = PDLaw.bandw(wn, am), pdlaw(wn, am)
        assert p.command(*s) == m.command(*s)
        assert p.raw(*s) == m.raw(*s)


def testmatlawhas():
    m = lqrlaw(N400, 6 * N400, 0.05)
    assert not hasattr(m, "kp") and not hasattr(m, "kd")
    assert m.saturdist is None
    assert not m.ispd()
    assert pdlaw(6 * N400, 0.05).ispd()


def testjacmatchfinite():
    rng = np.random.default_rng(2)
    law = lqrlaw(N400, 5 * N400, 0.02)
    for _ in range(50):
        s = rng.normal(size=6) * np.r_[[3000.0] * 3, [3.0] * 3]
        J, sat = law.jacobian(s)
        eps = 1e-6 * np.maximum(np.abs(s), 1.0)
        Jfd = np.empty((3, 6))
        for k in range(6):
            sp, sm = s.copy(), s.copy()
            sp[k] += eps[k]
            sm[k] -= eps[k]
            Jfd[:, k] = (np.array(law.command(*sp)) - np.array(law.command(*sm))) / (2 * eps[k])
        assert np.allclose(J, Jfd, rtol=1e-5, atol=1e-12)


def testlqrstableover():
    for alt in (400e3,):
        n = meanmotionalt(alt)
        for r in np.linspace(3, 10, 15):
            assert unsatstab(lqrlaw(n, r * n, 0.05), n) < 0


def teststrictjsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"run_id": "a", "v": 1}\n{"run_id": "b", "v": 2}\n')
    assert len(L.readjsonlstrict(str(p), key="run_id")) == 2
    p.write_text('{"run_id": "a"}\n{"run_id": "a"}\n')
    with pytest.raises(L.StrictJSONLErr, match="duplicate"):
        L.readjsonlstrict(str(p), key="run_id")
    p.write_text('{"run_id": "a"}\n{"run_id": "b"')
    with pytest.raises(L.StrictJSONLErr, match="truncated"):
        L.readjsonlstrict(str(p), key="run_id")
    p.write_text('{"run_id": "a"}\nnot json\n')
    with pytest.raises(L.StrictJSONLErr, match="malformed"):
        L.readjsonlstrict(str(p), key="run_id")
    p.write_text('{"x": 1}\n')
    with pytest.raises(L.StrictJSONLErr, match="missing key"):
        L.readjsonlstrict(str(p), key="run_id")


def testrunidsep():
    pid = L.physical_id([1, 2, 3, 0, 0, 0], 400e3, 1.0, np.eye(3, 6), 0.05)
    a = L.run_id(pid, {"cmd_mode": "continuous", "Ts": None})
    b = L.run_id(pid, {"cmd_mode": "sampled", "Ts": 10.0})
    assert a != b and a == L.run_id(pid, {"Ts": None, "cmd_mode": "continuous"})


def testucbclosedforms():
    assert CAL.ucb(0, 0, 0.05) == 1.0
    assert CAL.ucb(5, 5, 0.05) == 1.0
    n, lev = 300, 0.05
    assert math.isclose(CAL.ucb(0, n, lev), 1 - lev ** (1 / n), rel_tol=1e-12)
    from scipy.stats import binom
    for b, n in ((1, 200), (3, 1000), (10, 2500)):
        u = CAL.ucb(b, n, 1e-4)
        assert math.isclose(binom.cdf(b, n, u), 1e-4, rel_tol=1e-6)
    M = 1000
    nmin = CAL.nadv(0.01, 0.05, M)
    assert CAL.ucb(0, nmin, 0.05 / M) <= 0.01 < CAL.ucb(0, nmin - 1, 0.05 / M)
    assert CAL.nminforb(0, 0.01, 0.05, M) == nmin


def testselruleties():
    v = np.array([0.1, 0.2, 0.3, np.nan, 0.5])
    adv = np.array([False, False, True, False, False])
    cands = [{"cid": 0, "threshold": 0.25, "source": "a"}, {"cid": 1, "threshold": 0.4, "source": "b"}]
    rows, best = CAL.calfam(v, adv, cands, alpha=0.9, level=0.05)
    assert rows[0]["n_accepted"] == 2 and rows[1]["n_accepted"] == 3 and rows[1]["n_adverse"] == 1
    rows, best = CAL.calfam(v, adv, cands, alpha=1e-9, level=0.05)
    assert best is None


def testcandsdeduporder():
    vals = np.r_[np.zeros(50), np.arange(1, 51, dtype=float)]
    c = CAL.buildcands(vals, histor=0.0)
    th = [x["threshold"] for x in c]
    assert len(th) == len(set(th)) and c[0]["source"] == "historical_v2_frozen"


def teststrdistseed():
    a, _ = stream("D0", "cal")
    b, _ = stream("D0", "test")
    c, _ = stream("D3", "cal")
    va, vb, vc = a.random(5), b.random(5), c.random(5)
    assert not np.allclose(va, vb) and not np.allclose(va, vc)
    u = checkunused()
    assert not u["root_entropy_in_archive_seeds"] and not u["first_draw_collisions"]


def testvelusescell():
    for cell in ("D0", "D2"):
        recs, _ = makecases(cell, "audit", 60)
        n = meanmotionalt(CELLS[cell]["altitude"])
        for r in recs:
            d = r["draw"]
            v = np.linalg.norm(d["rel0"][3:])
            assert math.isclose(v, d["nu"] * n * d["sep_m"], rel_tol=1e-9, abs_tol=1e-12)
            if d["velocity"] == "em1":
                assert math.isclose(d["rel0"][4], -2 * n * d["rel0"][0], rel_tol=1e-12, abs_tol=1e-12)


def testlabelsstatustax():
    base = {"worker_status": "ok", "cell": "D0", "cw_status": "ok", "cw_completed_horizon": True, "eci_status": "ok",
            "nerm_status": "ok", "cw_vs_eci_max_pos_disc_m": 0.5, "cw_vs_nerm_max_pos_disc_m": 0.5 + 1e-9}
    tr = CAL.truth_record(base)
    assert tr["truth_status"] == "completed_valid" and CAL.critlabel(tr, 1.0) == "ok"
    assert CAL.critlabel(tr, 0.1) == "violation"
    near = dict(base, cw_vs_eci_max_pos_disc_m=1.0 + 1e-7)
    assert CAL.critlabel(CAL.truth_record(near), 1.0) == "unresolved"
    ex = dict(base, eci_status="terminated", nerm_status="terminated")
    assert CAL.critlabel(CAL.truth_record(ex), 1.0) == "exit"
    mix = dict(base, eci_status="terminated")
    assert CAL.critlabel(CAL.truth_record(mix), 1.0) == "unresolved"
    fail = dict(base, eci_status="failed")
    assert CAL.critlabel(CAL.truth_record(fail), 1.0) == "failure"
    assert math.isnan(CAL.feature_value(dict(base, d1_peak_m=0.3, d1_status="failed: x"), "A", "continuous"))
    assert CAL.feature_value(dict(base, d1_peak_m=0.3, d1_status="ok"), "A", "continuous") == 0.3
