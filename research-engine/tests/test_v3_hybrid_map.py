"""Tests of the independent discrete-map held-command estimators (cwvalid3.hybrid_map)."""

import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwvalid3 import hybrid_map as hm  # noqa: E402

ALT = 400e3
R, N, T = hm.orbit(ALT)
C_CONST = np.array([2.0e-6, -1.0e-6, 5.0e-7])


def pd(wn):
    K = np.zeros((3, 6))
    K[:, :3] = wn * wn * np.eye(3)
    K[:, 3:] = 2.0 * wn * np.eye(3)
    return K


def lqr_like(wn):
    K = pd(wn)
    K[0, 4] = 0.3 * wn
    K[1, 3] = -0.3 * wn
    return K


# +R 3000 m, wn = 6n, a_max = 0.05: saturated at the start, then unsaturated (switching)
REL_A, K_A, AMAX_A = np.array([3000.0, 0, 0, 0, 0, 0]), pd(6 * N), 0.05


def test_residual_matches_direct_formula_and_second_order_limit():
    pos = np.array([[3000.0, -200.0, 50.0], [-12000.0, 0, 0], [0, 0, 20000.0], [1e5, 2e5, -3e4]])
    x, y, z = pos.T
    rr = np.sqrt((R + x) ** 2 + y ** 2 + z ** 2)
    f = 1.0 - (R / rr) ** 3
    direct = np.stack([N * N * ((R + x) * f - 3 * x), N * N * y * f, N * N * z * f], -1)
    got = hm.residual_exact(pos, N, R)
    assert np.allclose(got, direct, rtol=1e-8, atol=0)
    small = np.array([[3.0, -2.0, 1.0]])
    so = (3 * N * N / R) * np.array([[-(2 * 9 - 4 - 1) / 2, 3 * -2.0, 3 * 1.0]])
    assert np.allclose(hm.residual_exact(small, N, R), so, rtol=1e-5)


@pytest.mark.parametrize("Ts", [10.0, 7.0])
@pytest.mark.parametrize("kind", hm.KINDS)
def test_linear_check_constant_forcing(kind, Ts):
    out = hm.linear_check(lqr_like(6 * N), ALT, Ts, C_CONST, kind=kind, rel0=REL_A)
    assert out["rel_err_updates"] < 1e-12
    assert out["rel_err_grid"] < 1e-12
    assert out["rel_err_closed_form"] < 1e-12


def test_no_saturation_identity_A_H_equals_B_H():
    for a_max in (math.inf, None, 1e3):
        a = hm.held_sensitivity(REL_A, K_A, a_max, ALT, 10.0, "A_H")
        b = hm.held_sensitivity(REL_A, K_A, a_max, ALT, 10.0, "B_H")
        assert not a["sat_at_update"].any()
        assert np.array_equal(a["e"], b["e"])
        assert a["feature_m"] == b["feature_m"]
    # with saturation they must differ
    a = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, 10.0, "A_H")
    b = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, 10.0, "B_H")
    assert a["sat_at_update"].any() and not a["sat_at_update"].all()
    assert abs(a["feature_m"] - b["feature_m"]) > 1e-3 * a["feature_m"]


@pytest.mark.parametrize("kind", hm.KINDS)
def test_augmented_matches_variation_of_constants(kind):
    rel, a, b = hm.check_augmented_consistency(REL_A, K_A, AMAX_A, ALT, 10.0, kind, rtol=1e-10)
    assert rel < 1e-10
    assert np.array_equal(a["sat_at_update"], b["sat_at_update"])


def test_augmented_matches_lqr_like_partial_hold():
    rel, _, _ = hm.check_augmented_consistency([-5000, 4000, 1000, 0, 0, 0], lqr_like(6 * N),
                                               0.03, ALT, 7.0, "A_H", rtol=1e-10)
    assert rel < 1e-10


def test_quadrature_convergence():
    for kind in hm.KINDS:
        f8 = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, 30.0, kind, nodes=8)["feature_m"]
        f16 = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, 30.0, kind, nodes=16)["feature_m"]
        f_sub = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, 30.0, kind, sub=0.25)["feature_m"]
        assert abs(f8 - f16) < 1e-9 * f16
        assert abs(f8 - f_sub) < 1e-9 * f16


def test_partial_last_interval():
    Ts = 7.0
    assert abs(T / Ts - round(T / Ts)) > 1e-3          # Ts does not divide T
    out = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, Ts, "A_H")
    tu = out["t_update"]
    assert tu.size == math.ceil(T / Ts)
    assert tu[0] == 0.0 and np.all(np.diff(tu) == Ts) and tu[-1] < T
    assert 0.0 < T - tu[-1] < Ts
    assert out["t"][-1] == T
    assert np.allclose(out["e"][-1], out["e_update"][-1], rtol=1e-12, atol=1e-15)
    # requesting extra points inside the partial hold changes nothing at the common points
    tq = np.unique(np.concatenate((out["t"], tu[-1] + np.linspace(0.1, T - tu[-1] - 0.1, 7))))
    out2 = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, Ts, "A_H", tq=tq)
    idx = np.searchsorted(tq, out["t"])
    assert np.allclose(out2["e"][idx], out["e"], rtol=0, atol=1e-12 * out["feature_m"])


def test_nominal_matches_dense_ode_solution():
    Ts, horizon = 10.0, 0.1
    _, _, Th = hm.orbit(ALT, horizon)
    A, B = hm.cw_matrices(N)
    tq = hm.monitoring_grid(Th)
    nom = hm.held_cw_nominal(REL_A, K_A, AMAX_A, N, Ts, Th, tq)
    s = REL_A.copy()
    for k, t0 in enumerate(nom["t_update"]):
        u, _, _ = hm.law(K_A, AMAX_A, s)
        t1 = min(t0 + Ts, Th)
        sol = solve_ivp(lambda t, x: A @ x + B @ u, (t0, t1), s, method="DOP853",
                        rtol=1e-13, atol=1e-10)
        s = sol.y[:, -1]
        assert np.allclose(s, nom["s_update"][k + 1], rtol=1e-9, atol=1e-7)


def test_A_H_is_derivative_of_held_homotopy():
    """Central difference of the held lambda-branch (each branch closes its own held loop)."""
    Ts, horizon = 10.0, 0.15
    _, _, Th = hm.orbit(ALT, horizon)
    A, B = hm.cw_matrices(N)
    tq = np.append(hm.update_times(Ts, Th), Th)
    e = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, Ts, "A_H", tq=tq, horizon_orbits=horizon)

    def branch(lam):
        s, out = REL_A.copy(), [REL_A.copy()]
        for t0 in tq[:-1]:
            u, _, _ = hm.law(K_A, AMAX_A, s)
            t1 = min(t0 + Ts, Th)
            sol = solve_ivp(lambda t, x: A @ x + B @ (u + lam * hm.residual_exact(x[:3], N, R)),
                            (t0, t1), s, method="DOP853", rtol=1e-13, atol=1e-12)
            s = sol.y[:, -1]
            out.append(s)
        return np.array(out)

    lam = 1e-2
    fd = (branch(lam) - branch(-lam)) / (2 * lam)
    assert e["sat_at_update"].any() and not e["sat_at_update"].all()
    scale = np.abs(e["e"][:, :3]).max()
    assert np.abs(fd[:, :3] - e["e"][:, :3]).max() < 1e-7 * scale      # observed ~1e-9
    b = hm.held_sensitivity(REL_A, K_A, AMAX_A, ALT, Ts, "B_H", tq=tq, horizon_orbits=horizon)
    assert np.abs(fd[:, :3] - b["e"][:, :3]).max() > 1e-2 * scale       # B_H is not the derivative
