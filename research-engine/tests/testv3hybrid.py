import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwvalid3 import hybmap as hm

ALT = 400e3
R, N, T = hm.orbit(ALT)
CCONST = np.array([2.0e-6, -1.0e-6, 5.0e-7])


def pd(wn):
    K = np.zeros((3, 6))
    K[:, :3] = wn * wn * np.eye(3)
    K[:, 3:] = 2.0 * wn * np.eye(3)
    return K


def lqrlike(wn):
    K = pd(wn)
    K[0, 4] = 0.3 * wn
    K[1, 3] = -0.3 * wn
    return K


RELA, KA, AMAXA = np.array([3000.0, 0, 0, 0, 0, 0]), pd(6 * N), 0.05


def testresidmatchdirect():
    pos = np.array([[3000.0, -200.0, 50.0], [-12000.0, 0, 0], [0, 0, 20000.0], [1e5, 2e5, -3e4]])
    x, y, z = pos.T
    rr = np.sqrt((R + x) ** 2 + y ** 2 + z ** 2)
    f = 1.0 - (R / rr) ** 3
    direct = np.stack([N * N * ((R + x) * f - 3 * x), N * N * y * f, N * N * z * f], -1)
    got = hm.residexact(pos, N, R)
    assert np.allclose(got, direct, rtol=1e-8, atol=0)
    small = np.array([[3.0, -2.0, 1.0]])
    so = (3 * N * N / R) * np.array([[-(2 * 9 - 4 - 1) / 2, 3 * -2.0, 3 * 1.0]])
    assert np.allclose(hm.residexact(small, N, R), so, rtol=1e-5)


@pytest.mark.parametrize("Ts", [10.0, 7.0])
@pytest.mark.parametrize("kind", hm.KINDS)
def testlincheckconst(kind, Ts):
    out = hm.lincheck(lqrlike(6 * N), ALT, Ts, CCONST, kind=kind, rel0=RELA)
    assert out["rel_err_updates"] < 1e-12
    assert out["rel_err_grid"] < 1e-12
    assert out["rel_err_closed_form"] < 1e-12


def testnosatident():
    for amax in (math.inf, None, 1e3):
        a = hm.heldsens(RELA, KA, amax, ALT, 10.0, "A_H")
        b = hm.heldsens(RELA, KA, amax, ALT, 10.0, "B_H")
        assert not a["sat_at_update"].any()
        assert np.array_equal(a["e"], b["e"])
        assert a["feature_m"] == b["feature_m"]
    a = hm.heldsens(RELA, KA, AMAXA, ALT, 10.0, "A_H")
    b = hm.heldsens(RELA, KA, AMAXA, ALT, 10.0, "B_H")
    assert a["sat_at_update"].any() and not a["sat_at_update"].all()
    assert abs(a["feature_m"] - b["feature_m"]) > 1e-3 * a["feature_m"]


@pytest.mark.parametrize("kind", hm.KINDS)
def testaugmatchvar(kind):
    rel, a, b = hm.checkaugconsist(RELA, KA, AMAXA, ALT, 10.0, kind, rtol=1e-10)
    assert rel < 1e-10
    assert np.array_equal(a["sat_at_update"], b["sat_at_update"])


def testaugmatchlqr():
    rel, _, _ = hm.checkaugconsist([-5000, 4000, 1000, 0, 0, 0], lqrlike(6 * N),
                                   0.03, ALT, 7.0, "A_H", rtol=1e-10)
    assert rel < 1e-10


def testquadconv():
    for kind in hm.KINDS:
        f8 = hm.heldsens(RELA, KA, AMAXA, ALT, 30.0, kind, nodes=8)["feature_m"]
        f16 = hm.heldsens(RELA, KA, AMAXA, ALT, 30.0, kind, nodes=16)["feature_m"]
        fsub = hm.heldsens(RELA, KA, AMAXA, ALT, 30.0, kind, sub=0.25)["feature_m"]
        assert abs(f8 - f16) < 1e-9 * f16
        assert abs(f8 - fsub) < 1e-9 * f16


def testpartlastival():
    Ts = 7.0
    assert abs(T / Ts - round(T / Ts)) > 1e-3
    out = hm.heldsens(RELA, KA, AMAXA, ALT, Ts, "A_H")
    tu = out["t_update"]
    assert tu.size == math.ceil(T / Ts)
    assert tu[0] == 0.0 and np.all(np.diff(tu) == Ts) and tu[-1] < T
    assert 0.0 < T - tu[-1] < Ts
    assert out["t"][-1] == T
    assert np.allclose(out["e"][-1], out["e_update"][-1], rtol=1e-12, atol=1e-15)
    tq = np.unique(np.concatenate((out["t"], tu[-1] + np.linspace(0.1, T - tu[-1] - 0.1, 7))))
    out2 = hm.heldsens(RELA, KA, AMAXA, ALT, Ts, "A_H", tq=tq)
    idx = np.searchsorted(tq, out["t"])
    assert np.allclose(out2["e"][idx], out["e"], rtol=0, atol=1e-12 * out["feature_m"])


def testnommatchdense():
    Ts, horizon = 10.0, 0.1
    _, _, Th = hm.orbit(ALT, horizon)
    A, B = hm.cwmats(N)
    tq = hm.mongrid(Th)
    nom = hm.heldcwnom(RELA, KA, AMAXA, N, Ts, Th, tq)
    s = RELA.copy()
    for k, t0 in enumerate(nom["t_update"]):
        u, _, _ = hm.law(KA, AMAXA, s)
        t1 = min(t0 + Ts, Th)
        sol = solve_ivp(lambda t, x: A @ x + B @ u, (t0, t1), s, method="DOP853",
                        rtol=1e-13, atol=1e-10)
        s = sol.y[:, -1]
        assert np.allclose(s, nom["s_update"][k + 1], rtol=1e-9, atol=1e-7)


def testhderivheld():
    Ts, horizon = 10.0, 0.15
    _, _, Th = hm.orbit(ALT, horizon)
    A, B = hm.cwmats(N)
    tq = np.append(hm.updatetimes(Ts, Th), Th)
    e = hm.heldsens(RELA, KA, AMAXA, ALT, Ts, "A_H", tq=tq, horizon_orbits=horizon)

    def branch(lam):
        s, out = RELA.copy(), [RELA.copy()]
        for t0 in tq[:-1]:
            u, _, _ = hm.law(KA, AMAXA, s)
            t1 = min(t0 + Ts, Th)
            sol = solve_ivp(lambda t, x: A @ x + B @ (u + lam * hm.residexact(x[:3], N, R)),
                            (t0, t1), s, method="DOP853", rtol=1e-13, atol=1e-12)
            s = sol.y[:, -1]
            out.append(s)
        return np.array(out)

    lam = 1e-2
    fd = (branch(lam) - branch(-lam)) / (2 * lam)
    assert e["sat_at_update"].any() and not e["sat_at_update"].all()
    scale = np.abs(e["e"][:, :3]).max()
    assert np.abs(fd[:, :3] - e["e"][:, :3]).max() < 1e-7 * scale
    b = hm.heldsens(RELA, KA, AMAXA, ALT, Ts, "B_H", tq=tq, horizon_orbits=horizon)
    assert np.abs(fd[:, :3] - b["e"][:, :3]).max() > 1e-2 * scale
