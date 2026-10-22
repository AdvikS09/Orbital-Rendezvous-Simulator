import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cwvalid.constants import MUEARTH
from cwvalid3.heldmet import CaseView, casesumm, peakdisc
import cwvalid3.sampled as SA
from cwvalid3.sampled import EXIT, HeldBatch, pdmatrix, reltoeci

from . import v31fx as FX

K0 = np.zeros((3, 6))
INTERIOR = [-290e3, 0.0, 0.0, -30.0, 0.0, 0.0]
FIRSTSTEP = [-299.95e3, 0.0, 0.0, -100.0, 0.0, 0.0]
BELOW = [-300.5e3, 0.0, 0.0, -10.0, 0.0, 0.0]
COMP = [100.0, -50.0, 20.0, 0.01, 0.0, -0.02]
FLOORTOLM = 1e-6


def batch(rel0s, h=2.0, Ts=2.0, lambdas=(3.0,), horizon=0.05, run=True, K=None, amax=0.05):
    N = len(rel0s)
    K = np.broadcast_to(K0 if K is None else K, (N, 3, 6))
    hb = HeldBatch(rel0s, K, [amax] * N, 400e3, Ts, horizon_orbits=horizon, h=h, models=("cw", "nerm", "eci"),
                   ests=("A_H", "B_H"), lambdas=lambdas)
    return hb.run() if run else hb


def floorgap(hb, i, b, t):
    y = hb.dense(i, [t])
    return float(hb.radius(y, b)[0] - hb.rfloor)


@pytest.fixture(scope="module")
def interior():
    return batch([INTERIOR, COMP])


def testintexitdense(interior):
    hb = interior
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit"
        j = hb.exit_step[b][0]
        te = hb.t_end[b][0]
        assert hb.t[j] < te < hb.t[j + 1]
        assert not hb.exit_fallback[b][0]
        assert abs(floorgap(hb, 0, b, te)) < FLOORTOLM
        cols = hb.cols(b)
        assert not np.array_equal(hb.Y[j + 1, 0, cols], hb.Y[j, 0, cols])
        assert hb.radius(hb.Y[j + 1, 0], b) < hb.rfloor
        assert np.array_equal(hb.Y[j + 2, 0, cols], hb.Y[j + 1, 0, cols])


def testdensemaskedevent(interior):
    hb = interior
    for b in ("nerm", "eci", "hom:3"):
        te = hb.t_end[b][0]
        cols = hb.cols(b)
        y = hb.dense(0, [te - 0.25, te, te + 1e-6, te + 3.0, hb.T])
        assert np.all(np.isfinite(y[:2][:, cols]))
        assert np.all(np.isnan(y[2:][:, cols]))
    y = hb.dense(0, np.linspace(0.0, hb.T, 101))
    assert np.all(np.isfinite(y[:, hb.cols("cw")]))
    assert np.all(np.isfinite(y[:, hb.cols("A_H")]))
    assert np.all(np.isfinite(hb.dense(0, [hb.T], mask=False)))


def testexittimematch(interior):
    hb = interior
    t0 = hb.Y[0, 0, hb.ceci:hb.ceci + 6]
    rc, vc = reltoeci(t0[None, :3], t0[None, 3:], np.asarray(INTERIOR)[None, :])

    def f(t, y):
        r = y[:3]
        return np.r_[y[3:], -MUEARTH * r / np.linalg.norm(r) ** 3]

    def ev(t, y):
        return np.linalg.norm(y[:3]) - hb.rfloor
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(f, (0.0, hb.T), np.r_[rc[0], vc[0]], method="DOP853", rtol=1e-13, atol=1e-9, events=ev)
    tref = float(sol.t_events[0][0])
    assert abs(hb.t_end["eci"][0] - tref) < 1e-7


def testexitinfirst():
    hb = batch([FIRSTSTEP])
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit"
        assert hb.exit_step[b][0] == 0 and 0.0 < hb.t_end[b][0] < hb.t[1]
        assert abs(floorgap(hb, 0, b, hb.t_end[b][0])) < FLOORTOLM


def testinitstatebelow():
    hb = batch([BELOW, COMP])
    for b in ("nerm", "eci", "hom:3"):
        assert hb.status[b][0] == "domain_exit" and hb.t_end[b][0] == 0.0
        assert hb.exit_step[b][0] == EXIT and not hb.exit_fallback[b][0]
        assert np.all(np.isnan(hb.dense(0, [0.5, hb.T])[:, hb.cols(b)]))
        assert hb.status[b][1] == "completed"
    pk = peakdisc(CaseView(hb, 0), "cw", "eci")
    assert pk["common_end_s"] == 0.0 and math.isfinite(pk["peak_m"])


def testexitstepnode():
    ref = batch([INTERIOR], lambdas=())
    js = 40
    assert ref.exit_step["nerm"][0] > js
    rn = float(ref.radius(ref.Y[js, 0], "nerm"))
    hb = batch([INTERIOR], lambdas=(), run=False)
    hb.rfloor = rn
    hb.run()
    assert hb.status["nerm"][0] == "domain_exit"
    assert hb.t_end["nerm"][0] == hb.t[js] and hb.exit_step["nerm"][0] == js
    assert not hb.exit_fallback["nerm"][0]
    assert np.array_equal(hb.Y[js, 0], ref.Y[js, 0])
    assert abs(floorgap(hb, 0, "nerm", hb.t[js])) <= 1e-9


def testbranchexitin(interior):
    hb = interior
    jh, jn = hb.exit_step["hom:3"][0], hb.exit_step["nerm"][0]
    assert jh < jn
    th, tn = hb.t_end["hom:3"][0], hb.t_end["nerm"][0]
    y = hb.dense(0, [0.5 * (th + tn)])
    assert np.all(np.isnan(y[:, hb.cols("hom:3")])) and np.all(np.isfinite(y[:, hb.cols("nerm")]))
    cv = CaseView(hb, 0)
    pk = peakdisc(cv, "hom:3", "nerm")
    assert pk["common_end_s"] == th and pk["t_peak_s"] <= th + 1e-9 and math.isfinite(pk["peak_m"])


def testcomplcompunaff(interior):
    other = batch([[-250e3, 0.0, 0.0, -10.0, 0.0, 0.0], COMP])
    hb = interior
    assert all(other.status[b][0] == "completed" for b in other.branchnames())
    for b in hb.branchnames():
        assert hb.status[b][1] == "completed" and hb.t_end[b][1] == hb.T and hb.exit_step[b][1] == -1
    assert np.array_equal(hb.Y[:, 1, :], other.Y[:, 1, :])
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    sb, _ = casesumm(hb, 1, pairs=pairs)
    so, _ = casesumm(other, 1, pairs=pairs)
    assert json.dumps(sb, sort_keys=True) == json.dumps(so, sort_keys=True)
    alone = batch([COMP])
    assert np.allclose(hb.Y[:, 1, :], alone.Y[:, 0, :], rtol=1e-14, atol=1e-18)


def teststephalvagrees():
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    s1, _ = casesumm(batch([INTERIOR], h=2.0), 0, pairs=pairs)
    s2, _ = casesumm(batch([INTERIOR], h=1.0), 0, pairs=pairs)
    for b in ("nerm", "eci"):
        assert abs(s1[f"{b}_t_end_s"] - s2[f"{b}_t_end_s"]) < 1e-7
    for p in ("cw_vs_eci", "cw_vs_nerm"):
        assert abs(s1[f"{p}_peak_m"] - s2[f"{p}_peak_m"]) < 1e-5
        assert s1[f"{p}_t_peak_s"] == s1[f"{p}_common_end_s"]
    assert s1["nerm_vs_eci_peak_m"] < 1e-4 and s2["nerm_vs_eci_peak_m"] < 1e-4


def testcommonivaldisc(interior):
    hb = interior
    cv = CaseView(hb, 0)
    pk = peakdisc(cv, "cw", "eci")
    tc = min(hb.t_end["cw"][0], hb.t_end["eci"][0])
    assert pk["common_end_s"] == tc and pk["t_peak_s"] <= tc + 1e-9
    tt = np.linspace(0.0, tc, 20001)
    d = np.linalg.norm(cv.relrows("cw", hb.dense(0, tt))[:, :3] - cv.relrows("eci", hb.dense(0, tt))[:, :3],
                       axis=1)
    assert np.all(np.isfinite(d))
    assert pk["peak_m"] >= d.max() - 1e-6 and pk["peak_m"] <= d.max() + 1e-3
    g = cv.grid("eci")
    assert np.all(np.isnan(g[cv.tm > tc + 1e-9])) and np.all(np.isfinite(g[cv.tm <= tc]))


def camprec(cell, split, index):
    man = FX.planned(f"{cell}{split}", index)
    return man, FX.resultfor(f"{cell}{split}", man["run_id"])


def testrepr():
    man, res = camprec("D4", "dev", 56)
    assert man["case_id"] == "Cbfad11507250fd59"
    hb = HeldBatch([man["scenario"]["rel0"]], np.array([man["K"]]), [man["scenario"]["a_max"]], 400e3, 10.0,
                   horizon_orbits=1.0, h=2.0, models=("cw", "nerm", "eci"), ests=("A_H", "B_H", "C_H")).run()
    for b in ("nerm", "eci"):
        assert hb.status[b][0] == res[f"{b}_status"] == "domain_exit"
        assert abs(hb.t_end[b][0] - res[f"{b}_t_end_s"]) < 1e-8
        assert hb.t[hb.exit_step[b][0]] == 3916.0
        assert abs(floorgap(hb, 0, b, hb.t_end[b][0])) < FLOORTOLM
    sm, _ = casesumm(hb, 0, pairs=(("cw", "eci"), ("cw", "nerm"), ("nerm", "eci")))
    assert abs(res["cw_vs_eci_peak_m"] - 55164.7034834116) < 1e-6
    assert abs(sm["cw_vs_eci_peak_m"] - 55166.86759277691) < 1e-3
    assert sm["nerm_vs_eci_peak_m"] < 1e-5
    for k in ("A_H_peak_m", "B_H_peak_m", "C_H_peak_m", "cw_final_range_m"):
        assert abs(sm[k] - res[k]) <= 1e-9 * max(1.0, abs(res[k]))


def testdensestateevent(interior):
    hb = interior
    t0 = hb.Y[0, 0, hb.ceci:hb.ceci + 6]
    rc, vc = reltoeci(t0[None, :3], t0[None, 3:], np.asarray(INTERIOR)[None, :])

    def f(t, y):
        r = y[:3]
        return np.r_[y[3:], -MUEARTH * r / np.linalg.norm(r) ** 3]
    te = hb.t_end["eci"][0]
    sol = solve_ivp(f, (0.0, te), np.r_[rc[0], vc[0]], method="DOP853", rtol=1e-13, atol=1e-9, dense_output=True)
    yref = sol.sol(te)
    y = hb.dense(0, [te])[0]
    c = hb.ceci
    assert np.linalg.norm(y[c + 6:c + 9] - yref[:3]) < 1e-4
    assert np.linalg.norm(y[c + 9:c + 12] - yref[3:]) < 1e-6


def testnumfailfrozen():
    hb = HeldBatch([[1000.0, 0, 0, 0, 0, 0]], pdmatrix(0.5)[None], [None], 400e3, 10.0, h=2.0, models=("cw", "nerm"),
                   ests=("A_H",)).run()
    assert hb.status["nerm"][0] in ("domain_exit", "numerical_failure")
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


def testfailedrootbrack(monkeypatch):
    def nobracket(*a, **k):
        raise ValueError("f(a) and f(b) must have different signs")
    monkeypatch.setattr(SA, "brentq", nobracket)
    hb = batch([INTERIOR], lambdas=())
    for b in ("nerm", "eci"):
        j = hb.exit_step[b][0]
        assert hb.status[b][0] == "domain_exit" and hb.exit_fallback[b][0]
        assert hb.t_end[b][0] == hb.t[j + 1]


def testsatgainexit():
    K = pdmatrix(6 * 0.00113)
    pairs = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
    s1, _ = casesumm(batch([INTERIOR], h=2.0, lambdas=(), K=K), 0, pairs=pairs)
    hb2 = batch([INTERIOR], h=1.0, lambdas=(), K=K)
    s2, _ = casesumm(hb2, 0, pairs=pairs)
    assert hb2.status["eci"][0] == "domain_exit" and hb2.sat_at_update[0].any()
    for b in ("nerm", "eci"):
        assert abs(floorgap(hb2, 0, b, hb2.t_end[b][0])) < FLOORTOLM
        assert abs(s1[f"{b}_t_end_s"] - s2[f"{b}_t_end_s"]) < 1e-7
    assert abs(s1["cw_vs_eci_peak_m"] - s2["cw_vs_eci_peak_m"]) < 1e-5
