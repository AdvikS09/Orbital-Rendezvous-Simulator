"""Independent discrete-map reference implementation of the held-command (zero-order-hold)
first-variation estimators A_H, B_H and C_H (conventions: v3_extension/methods/V3_CONVENTIONS.md).

Purpose
-------
Verification only. This module is written deliberately independently of ``cwvalid3.sampled``:
it imports nothing from it, and it uses a different numerical method (exact matrix exponentials
plus composite Gauss-Legendre quadrature of the variation-of-constants integral, instead of a
step-by-step Runge-Kutta propagation). Agreement between the two therefore tests both.

Frames, units, constants
------------------------
LVLH of a circular chief (x radial, y along-track, z orbit normal), SI units, state
s = (x, y, z, xdot, ydot, zdot). R = R_EARTH + altitude, n = sqrt(MU_EARTH / R^3),
horizon T = horizon_orbits * 2 pi / n. Constants come from ``cwvalid.constants``.

Equations implemented
---------------------
CW: s' = A s + B u, A = [[0, I], [M, W]], M = diag(3n^2, 0, -n^2),
W = [[0, 2n, 0], [-2n, 0, 0], [0, 0, 0]], B = [0; I].

Residual (exact minus CW acceleration, position only), with rho^2 = x^2 + y^2 + z^2,
q = (2 R x + rho^2) / R^2, f = 1 - (1 + q)^(-3/2) = -expm1(-1.5 log1p(q)):
    r = n^2 ((R + x) f - 3x,  y f,  z f).
The x component is evaluated cancellation-free as n^2 (R h2(q) + 1.5 rho^2 / R + x f) with
h2(q) = f - 1.5 q (power series for |q| < 0.1).

Law: U(s) = normsat(-K s), K 3x6. Jacobian J_U = -K if |K s| <= a_max, else
(a_max/|a|)(I - a_hat a_hat^T)(-K) with a = -K s.

Hold: update times t_k = k Ts for t_k < T, holds [t_k, t_{k+1}), the last one [t_K, T]
(partial when Ts does not divide T). u_k = U(s0(t_k)) is constant (LVLH components) on a hold.

Nominal (exact): with E(tau) = expm(tau [[A, B], [0, 0]]) = [[Phi(tau), Gamma(tau)], [0, I]],
    s0(t_k + tau) = Phi(tau) s0(t_k) + Gamma(tau) u_k.

Sensitivities (e(0) = 0, e continuous at updates), on hold k:
    A_H: e' = A e + B m_k + B r(s0(t)),  m_k = J_U(s0(t_k)) e(t_k)
    B_H: the same with m_k = -K e(t_k)
    C_H: e' = A_C e + B r(s0(t)), A_C = [[0, I], [0, 0]], m_k = 0  (e_pos'' = r)
Variation of constants (Phi_e, Gamma_e built from A, or from A_C for C_H):
    e(t_k + tau) = Phi_e(tau) e_k + Gamma_e(tau) m_k + F_k(tau),
    F_k(tau) = int_0^tau Phi_e(tau - sigma) B r(s0(t_k + sigma)) d sigma.
F_k is evaluated by composite Gauss-Legendre quadrature (``nodes`` nodes per sub-interval,
sub-intervals of length <= ``sub`` s). In ``held_sensitivity`` each hold is cut at the requested
output times and every piece is subdivided; F_k at the piece ends is accumulated with the
semigroup identity F_k(b') = Phi_e(b' - b) F_k(b) + int_b^b' Phi_e(b' - sigma) B r d sigma, and
s0 at a quadrature node is evaluated exactly from the piece start with the expm formula.

Augmented-state formulation (``held_sensitivity_augmented``, a separate computational path):
z = (e, m), z' = [[A_e, B], [0, 0]] z + [B r; 0] inside a hold, fixed-time reset at t_k
    z+ = [[I, 0], [J_k, 0]] z-   (J_k = J_U(s0(t_k)), -K, or 0 for A_H, B_H, C_H),
solved as z(t_k + tau) = expm(tau M_e) z_k+ + int_0^tau expm((tau - sigma) M_e) [B r; 0] d sigma
with one composite Gauss-Legendre rule over the whole [0, tau] (no piece recursion), and the
nominal at the nodes from [s0; u](t_k + sigma) = expm(sigma [[A, B], [0, 0]]) [s0_k; u_k].
``check_augmented_consistency`` asserts that both give the same e(t).

Checks: ``linear_check`` (r replaced by a constant c, a_max = inf: exact discrete recursion
e_{k+1} = (Phi(Ts) - Gamma(Ts) K) e_k + Gamma(Ts) c from one expm of a 12x12 augmented matrix)
and the no-saturation identity A_H == B_H when a_max = inf (see tests/test_v3_hybrid_map.py).

Output: e(t) on a requested grid (default: the 0.5 s monitoring grid t_j = 0.5 j plus T), the
feature max_j |e_pos(t_j)| and the per-update saturation flags.
"""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from cwvalid.constants import MU_EARTH, R_EARTH

KINDS = ("A_H", "B_H", "C_H")
DT_MONITOR = 0.5
_TTOL = 1e-9  # [s] time tolerance for update-time / horizon-end bookkeeping


# ----------------------------------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------------------------------

def orbit(altitude, horizon_orbits=1.0):
    """Return (R, n, T) for a circular chief at ``altitude`` [m]."""
    R = R_EARTH + float(altitude)
    n = math.sqrt(MU_EARTH / R ** 3)
    return R, n, float(horizon_orbits) * 2.0 * math.pi / n


def cw_matrices(n):
    """CW system matrix A (6x6) and input matrix B = [0; I] (6x3)."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 0] = 3.0 * n * n
    A[5, 2] = -n * n
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return A, B


def double_integrator_matrix():
    """A_C = [[0, I], [0, 0]] (dynamics of the residual-only estimator C_H)."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    return A


def augmented(A, B):
    """[[A, B], [0, 0]] (9x9)."""
    M = np.zeros((9, 9))
    M[:6, :6] = A
    M[:6, 6:] = B
    return M


def _binom_series_coeffs(nterms=40):
    """Coefficients b_j of (1+q)^(-3/2) = sum_j b_j q^j."""
    b = [1.0]
    for j in range(nterms):
        b.append(b[-1] * (-1.5 - j) / (j + 1))
    return np.array(b)


_B32 = _binom_series_coeffs(40)


def _h2(q):
    """h2(q) = f(q) - 1.5 q with f = 1 - (1+q)^(-3/2); cancellation-free for small |q|."""
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    small = np.abs(q) < 0.1
    if np.any(small):
        qs = q[small]
        acc = np.zeros_like(qs)
        for c in _B32[:1:-1]:            # Horner over b_40 ... b_2
            acc = acc * qs + c
        out[small] = -qs * qs * acc
    if np.any(~small):
        ql = q[~small]
        out[~small] = -np.expm1(-1.5 * np.log1p(ql)) - 1.5 * ql
    return out


def residual_exact(pos, n, R):
    """r(rho) = a_exact - a_CW for a circular two-body chief; pos (..., 3) -> (..., 3)."""
    pos = np.asarray(pos, dtype=float)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    rho2 = x * x + y * y + z * z
    q = (2.0 * R * x + rho2) / (R * R)
    f = -np.expm1(-1.5 * np.log1p(q))
    n2 = n * n
    out = np.empty(pos.shape)
    out[..., 0] = n2 * (R * _h2(q) + 1.5 * rho2 / R + x * f)
    out[..., 1] = n2 * y * f
    out[..., 2] = n2 * z * f
    return out


def law(K, a_max, s):
    """(u, saturated, J_U) for U(s) = normsat(-K s) at a single state s (6,)."""
    a = -(K @ s)
    na = float(np.linalg.norm(a))
    if na > a_max:
        ah = a / na
        J = (a_max / na) * (np.eye(3) - np.outer(ah, ah)) @ (-K)
        return a * (a_max / na), True, J
    return a, False, -K


def update_times(Ts, T):
    """t_k = k Ts for t_k < T (t_0 = 0)."""
    kmax = int(math.ceil(T / Ts)) + 1
    t = Ts * np.arange(kmax + 1, dtype=float)
    return t[t < T - _TTOL]


def monitoring_grid(T, dt=DT_MONITOR):
    """t_j = dt j for t_j <= T, plus T itself."""
    j = int(math.floor(T / dt + 1e-9))
    t = dt * np.arange(j + 1, dtype=float)
    if T - t[-1] > _TTOL:
        t = np.append(t, T)
    else:
        t[-1] = T
    return t


class ExpmCache:
    """expm(tau * M) for many tau, computed once per distinct float value (stacked expm)."""

    def __init__(self, M):
        self.M = np.asarray(M, dtype=float)
        self._d = {}

    def __call__(self, taus):
        taus = np.asarray(taus, dtype=float)
        flat = taus.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        missing = [float(v) for v in uniq if float(v) not in self._d]
        if missing:
            E = expm(np.asarray(missing)[:, None, None] * self.M)
            for v, Ev in zip(missing, E):
                self._d[v] = Ev
        Eu = np.stack([self._d[float(v)] for v in uniq])
        return Eu[inv].reshape(taus.shape + self.M.shape)


def _gl_rule(L, sub, nodes):
    """Composite Gauss-Legendre nodes/weights on [0, L], sub-interval length <= sub.

    Returns (sigma, w, sigma_rev) with sigma_rev = L - sigma evaluated as the mirrored node
    (so that equal offsets produce identical floats across intervals and are cached once)."""
    x, w = leggauss(int(nodes))
    M = max(1, int(math.ceil(L / sub - 1e-12)))
    h = L / M
    p = np.arange(M, dtype=float)[:, None]
    sig = (p * h + 0.5 * h * (1.0 + x[None, :])).ravel()
    sig_rev = ((M - 1 - p) * h + 0.5 * h * (1.0 - x[None, :])).ravel()
    ww = np.tile(0.5 * h * w, M)
    return sig, ww, sig_rev


def _check_kind(kind):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _amax(a_max):
    return math.inf if a_max is None else float(a_max)


def _check_tq(tq, T):
    tq = np.asarray(tq, dtype=float)
    if tq.ndim != 1 or tq.size == 0:
        raise ValueError("tq must be a non-empty 1-D array")
    if np.any(tq < -_TTOL) or np.any(tq > T + _TTOL):
        raise ValueError("tq must lie in [0, T]")
    return np.clip(tq, 0.0, T)


# ----------------------------------------------------------------------------------------------
# Held CW nominal (exact)
# ----------------------------------------------------------------------------------------------

def _nominal_updates(rel0, K, a_max, A, B, Ts, T, En):
    tu = update_times(Ts, T)
    Ku = tu.size
    tau = np.append(tu[1:], T) - tu
    E = En(tau)
    s = np.empty((Ku + 1, 6))
    u = np.empty((Ku, 3))
    sat = np.zeros(Ku, dtype=bool)
    J = np.empty((Ku, 3, 6))
    margin = np.full(Ku, np.inf)
    s[0] = np.asarray(rel0, dtype=float)
    for k in range(Ku):
        u[k], sat[k], J[k] = law(K, a_max, s[k])
        if np.isfinite(a_max):
            margin[k] = abs(np.linalg.norm(K @ s[k]) / a_max - 1.0)
        s[k + 1] = E[k, :6, :6] @ s[k] + E[k, :6, 6:] @ u[k]
    return dict(t_update=tu, tau_hold=tau, s_update=s, u_update=u, sat_at_update=sat, J_update=J,
                switch_margin=margin)


def held_cw_nominal(rel0, K, a_max, n, Ts, T, tq, _cache=None):
    """Exact held-command CW nominal at times tq (via expm of [[A, B], [0, 0]]).

    Returns dict(t, s0 (len(tq), 6), t_update, tau_hold, s_update (K+1 rows incl. T),
    u_update, sat_at_update, J_update, switch_margin = | |K s_k|/a_max - 1 |)."""
    a_max = _amax(a_max)
    K = np.asarray(K, dtype=float)
    A, B = cw_matrices(n)
    En = _cache if _cache is not None else ExpmCache(augmented(A, B))
    nom = _nominal_updates(rel0, K, a_max, A, B, Ts, T, En)
    tq = _check_tq(tq, T)
    tu = nom["t_update"]
    hk = np.clip(np.searchsorted(tu, tq, side="right") - 1, 0, tu.size - 1)
    a = tq - tu[hk]
    E = En(a)
    s0 = (np.einsum("gij,gj->gi", E[:, :6, :6], nom["s_update"][hk])
          + np.einsum("gij,gj->gi", E[:, :6, 6:], nom["u_update"][hk]))
    nom.update(t=tq, s0=s0)
    return nom


# ----------------------------------------------------------------------------------------------
# Sensitivity, variation-of-constants form (primary)
# ----------------------------------------------------------------------------------------------

def _setup(rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing):
    _check_kind(kind)
    R, n, T = orbit(altitude, horizon_orbits)
    K = np.asarray(K, dtype=float)
    a_max = _amax(a_max)
    A, B = cw_matrices(n)
    En = ExpmCache(augmented(A, B))
    Ae = double_integrator_matrix() if kind == "C_H" else A
    Ee = ExpmCache(augmented(Ae, B)) if kind == "C_H" else En
    tq = monitoring_grid(T) if tq is None else _check_tq(tq, T)
    if forcing is None:
        def forcing(pos, _n=n, _R=R):
            return residual_exact(pos, _n, _R)
    nom = _nominal_updates(rel0, K, a_max, A, B, Ts, T, En)
    return R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom


def _feedback(kind, k, nom, K):
    if kind == "A_H":
        return nom["J_update"][k]
    if kind == "B_H":
        return -K
    return None


def _finish(out, tq, e, s0):
    p = np.linalg.norm(e[:, :3], axis=1)
    j = int(np.argmax(p))
    out.update(t=tq, e=e, s0=s0, feature_m=float(p[j]), t_peak_s=float(tq[j]))
    return out


def held_sensitivity(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
                     horizon_orbits=1.0, forcing=None):
    """Held-command first variation e(t) by variation of constants + composite Gauss-Legendre.

    Parameters: rel0 (6,), K (3, 6), a_max [m/s^2] (None or inf: unsaturated), altitude [m],
    Ts hold period [s], kind in {"A_H", "B_H", "C_H"}, tq output times (default: 0.5 s monitoring
    grid plus T), sub max quadrature sub-interval length [s], nodes Gauss-Legendre nodes per
    sub-interval, horizon_orbits, forcing optional callable pos (..., 3) -> (..., 3) replacing the
    exact residual r (used by ``linear_check``).

    Returns dict(t, e (len(t), 6), s0 (len(t), 6), feature_m = max_j |e_pos(t_j)|, t_peak_s,
    sat_at_update, plus t_update, e_update (K+1 rows incl. T), m_update, s_update, u_update,
    switch_margin, n_updates, T, n, R, kind, sub, nodes)."""
    R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom = _setup(
        rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing)
    tu, s_u, u_u = nom["t_update"], nom["s_update"], nom["u_update"]
    Ku = tu.size

    # breakpoints: update times, horizon end, output times
    P = np.unique(np.concatenate((tu, [T], tq)))
    hk = np.clip(np.searchsorted(tu, P, side="right") - 1, 0, Ku - 1)
    a = P - tu[hk]                       # offset inside the hold (0 exactly at update times)
    EnP = En(a)
    s0P = (np.einsum("gij,gj->gi", EnP[:, :6, :6], s_u[hk])
           + np.einsum("gij,gj->gi", EnP[:, :6, 6:], u_u[hk]))

    # pieces [P_i, P_{i+1}] never straddle an update time
    npc = P.size - 1
    L = np.diff(P)
    hk_pc = hk[:-1]
    G = np.empty((npc, 6))
    for Lv in np.unique(L):
        idx = np.nonzero(L == Lv)[0]
        rho, w, rho_rev = _gl_rule(float(Lv), sub, nodes)
        Er = En(rho)                                   # (Q, 9, 9) nominal from piece start
        W = w[:, None, None] * (Ee(rho_rev)[:, :6, :6] @ B)   # (Q, 6, 3)
        S = s0P[idx]
        U = u_u[hk_pc[idx]]
        pos = (np.einsum("qij,gj->gqi", Er[:, :3, :6], S)
               + np.einsum("qij,gj->gqi", Er[:, :3, 6:], U))
        r = forcing(pos)
        G[idx] = np.einsum("qij,gqj->gi", W, r)

    # F_k at piece ends: F(b') = Phi_e(b' - b) F(b) + G
    PhiL = Ee(L)[:, :6, :6]
    F_end = np.empty((npc, 6))
    Fc = np.zeros(6)
    for i in range(npc):
        if a[i] == 0.0:
            Fc = np.zeros(6)
        Fc = PhiL[i] @ Fc + G[i]
        F_end[i] = Fc
    FP = np.zeros((P.size, 6))
    nz = a[1:] != 0.0
    FP[1:][nz] = F_end[nz]

    # e at update times
    tau = nom["tau_hold"]
    Eh = Ee(tau)
    end_idx = np.searchsorted(P, np.append(tu[1:], T))   # P index of each hold's end
    F_hold = F_end[end_idx - 1]
    e_u = np.zeros((Ku + 1, 6))
    m_u = np.zeros((Ku, 3))
    for k in range(Ku):
        Jk = _feedback(kind, k, nom, K)
        if Jk is not None:
            m_u[k] = Jk @ e_u[k]
        e_u[k + 1] = Eh[k, :6, :6] @ e_u[k] + Eh[k, :6, 6:] @ m_u[k] + F_hold[k]

    # e at every breakpoint, then at the requested times
    EeP = Ee(a)
    eP = (np.einsum("gij,gj->gi", EeP[:, :6, :6], e_u[hk])
          + np.einsum("gij,gj->gi", EeP[:, :6, 6:], m_u[hk]) + FP)
    iq = np.searchsorted(P, tq)
    out = dict(t_update=tu, e_update=e_u, m_update=m_u, s_update=s_u, u_update=u_u,
               sat_at_update=nom["sat_at_update"].copy(), switch_margin=nom["switch_margin"],
               n_updates=int(Ku), T=T, n=n, R=R, kind=kind, sub=sub, nodes=nodes)
    return _finish(out, tq, eP[iq], s0P[iq])


# ----------------------------------------------------------------------------------------------
# Sensitivity, augmented-state + reset-map form (independent path for self-consistency)
# ----------------------------------------------------------------------------------------------

def held_sensitivity_augmented(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
                               horizon_orbits=1.0, forcing=None):
    """Same quantity as ``held_sensitivity`` via z = (e, m), z' = M_e z + [B r; 0] inside a
    hold and the reset z+ = [[I, 0], [J_k, 0]] z- at t_k; one composite Gauss-Legendre rule over
    the whole [0, tau] for every (hold, offset) pair and the nominal at the nodes taken straight
    from the update state. Returns the same keys as ``held_sensitivity`` (plus z_update_minus)."""
    R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom = _setup(
        rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing)
    tu = nom["t_update"]
    Ku = tu.size
    w_u = np.hstack((nom["s_update"][:Ku], nom["u_update"]))      # [s0_k; u_k]
    BR = np.zeros((9, 3))
    BR[3:6, :] = np.eye(3)                                       # [B; 0]

    def integrals(tau_v, holds):
        """int_0^tau expm((tau - sigma) M_e) [B r(s0(t_k + sigma)); 0] d sigma for each hold."""
        sig, w, sig_rev = _gl_rule(tau_v, sub, nodes)
        En_s = En(sig)                                           # (Q, 9, 9)
        pos = np.einsum("qij,gj->gqi", En_s[:, :3, :], w_u[holds])
        r = forcing(pos) * w[None, :, None]
        Wm = Ee(sig_rev) @ BR                                    # (Q, 9, 3)
        return np.einsum("qij,gqj->gi", Wm, r)

    tau = nom["tau_hold"]
    I_hold = np.empty((Ku, 9))
    for tv in np.unique(tau):
        idx = np.nonzero(tau == tv)[0]
        I_hold[idx] = integrals(float(tv), idx)

    Eh = Ee(tau)
    zm = np.zeros((Ku + 1, 9))            # z just before each update (row Ku: at T)
    zp = np.zeros((Ku, 9))                # z just after each update
    for k in range(Ku):
        Rk = np.zeros((9, 9))
        Rk[:6, :6] = np.eye(6)
        Jk = _feedback(kind, k, nom, K)
        if Jk is not None:
            Rk[6:, :6] = Jk
        zp[k] = Rk @ zm[k]
        zm[k + 1] = Eh[k] @ zp[k] + I_hold[k]

    hk = np.clip(np.searchsorted(tu, tq, side="right") - 1, 0, Ku - 1)
    off = tq - tu[hk]
    e = np.empty((tq.size, 6))
    for tv in np.unique(off):
        idx = np.nonzero(off == tv)[0]
        z = Ee(np.array([tv]))[0] @ zp[hk[idx]].T
        e[idx] = (z.T + integrals(float(tv), hk[idx]))[:, :6]
    Eq = En(off)
    s0 = np.einsum("gij,gj->gi", Eq[:, :6, :], w_u[hk])
    out = dict(t_update=tu, e_update=zm[:, :6], m_update=zp[:, 6:], z_update_minus=zm,
               s_update=nom["s_update"], u_update=nom["u_update"],
               sat_at_update=nom["sat_at_update"].copy(), switch_margin=nom["switch_margin"],
               n_updates=int(Ku), T=T, n=n, R=R, kind=kind, sub=sub, nodes=nodes)
    return _finish(out, tq, e, s0)


def check_augmented_consistency(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
                                horizon_orbits=1.0, rtol=1e-10):
    """Run both formulations and assert max_t |e_vc - e_aug| <= rtol * max_t |e_vc|.
    Returns (relative difference, result_vc, result_aug)."""
    a = held_sensitivity(rel0, K, a_max, altitude, Ts, kind, tq, sub, nodes, horizon_orbits)
    b = held_sensitivity_augmented(rel0, K, a_max, altitude, Ts, kind, tq, sub, nodes,
                                   horizon_orbits)
    scale = max(np.abs(a["e"]).max(), np.finfo(float).tiny)
    d = float(max(np.abs(a["e"] - b["e"]).max(), np.abs(a["e_update"] - b["e_update"]).max()))
    rel = d / scale
    assert rel <= rtol, f"augmented vs variation-of-constants mismatch: rel {rel:.3e} > {rtol:.1e}"
    return rel, a, b


# ----------------------------------------------------------------------------------------------
# Analytically soluble linear check
# ----------------------------------------------------------------------------------------------

def linear_check(K, altitude, Ts, c, kind="A_H", tq=None, sub=0.5, nodes=8, horizon_orbits=1.0,
                 rel0=None):
    """Replace r(s0(t)) by a constant vector c and take a_max = inf. Then exactly
        e_{k+1} = (Phi(tau_k) - Gamma(tau_k) K) e_k + Gamma(tau_k) c        (A_H, B_H)
        e_{k+1} = Phi_C(tau_k) e_k + Gamma_C(tau_k) c                       (C_H)
    with Phi, Gamma (feedback) and Gamma (constant input) read from ONE expm of the 12x12 matrix
    tau [[A_e, B, B], [0, 0, 0], [0, 0, 0]] per distinct hold length; inside a hold
    e(t_k + tau) = Phi(tau) e_k + Gamma(tau) (m_k + c). The k-step value is also checked against
    the closed form [[Ad, g], [0, 1]]^k [0; 1] (geometric matrix series).

    Compares with ``held_sensitivity(..., forcing=c)`` and returns dict(rel_err_updates,
    rel_err_grid, rel_err_closed_form, quad, exact_update, exact_grid)."""
    _check_kind(kind)
    R, n, T = orbit(altitude, horizon_orbits)
    K = np.asarray(K, dtype=float)
    c = np.asarray(c, dtype=float).reshape(3)
    rel0 = np.zeros(6) if rel0 is None else rel0
    quad = held_sensitivity(rel0, K, math.inf, altitude, Ts, kind, tq, sub, nodes,
                            horizon_orbits, forcing=lambda pos: np.broadcast_to(c, pos.shape))
    A, B = cw_matrices(n)
    Ae = double_integrator_matrix() if kind == "C_H" else A
    Kf = np.zeros((3, 6)) if kind == "C_H" else K
    M12 = np.zeros((12, 12))
    M12[:6, :6] = Ae
    M12[:6, 6:9] = B
    M12[:6, 9:12] = B

    def blocks(tv):
        E = expm(tv * M12)
        return E[:6, :6], E[:6, 6:9], E[:6, 9:12]

    tu = quad["t_update"]
    tau = np.append(tu[1:], T) - tu
    cache = {}
    e = np.zeros((tu.size + 1, 6))
    for k, tv in enumerate(tau):
        if tv not in cache:
            cache[tv] = blocks(tv)
        Phi, Gm, Gc = cache[tv]
        e[k + 1] = (Phi - Gm @ Kf) @ e[k] + Gc @ c

    # closed form at the last full hold (geometric matrix series via a block power)
    Phi, Gm, Gc = blocks(float(tau[0]))
    kfull = int(np.sum(tau == tau[0]))
    Mb = np.zeros((7, 7))
    Mb[:6, :6] = Phi - Gm @ Kf
    Mb[:6, 6] = Gc @ c
    Mb[6, 6] = 1.0
    e_cf = np.linalg.matrix_power(Mb, kfull)[:6, 6]
    scale_u = max(np.abs(e).max(), np.finfo(float).tiny)
    rel_cf = float(np.abs(e_cf - e[kfull]).max() / scale_u)

    tq_ = quad["t"]
    hk = np.clip(np.searchsorted(tu, tq_, side="right") - 1, 0, tu.size - 1)
    off = tq_ - tu[hk]
    eg = np.empty((tq_.size, 6))
    for tv in np.unique(off):
        idx = np.nonzero(off == tv)[0]
        Ph, Gm_, Gc_ = blocks(float(tv))
        ek = e[hk[idx]]
        eg[idx] = ek @ (Ph - Gm_ @ Kf).T + Gc_ @ c
    rel_u = float(np.abs(quad["e_update"] - e).max() / scale_u)
    rel_g = float(np.abs(quad["e"] - eg).max() / max(np.abs(eg).max(), np.finfo(float).tiny))
    return dict(rel_err_updates=rel_u, rel_err_grid=rel_g, rel_err_closed_form=rel_cf,
                quad=quad, exact_update=e, exact_grid=eg)


__all__ = ["KINDS", "orbit", "cw_matrices", "residual_exact", "law", "update_times",
           "monitoring_grid", "held_cw_nominal", "held_sensitivity", "held_sensitivity_augmented",
           "check_augmented_consistency", "linear_check"]
