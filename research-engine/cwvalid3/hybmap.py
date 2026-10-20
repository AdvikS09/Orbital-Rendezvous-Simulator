# Reference implementation of the held-command estimators A_H, B_H and C_H. It imports nothing
# from sampled.py on purpose, so the two implementations can be checked against each other.
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from cwvalid.constants import MUEARTH, REARTH

KINDS = ("A_H", "B_H", "C_H")
DTMONITOR = 0.5
TTOL = 1e-9


def orbit(altitude, horizon_orbits=1.0):
    R = REARTH + float(altitude)
    n = math.sqrt(MUEARTH / R ** 3)
    return R, n, float(horizon_orbits) * 2.0 * math.pi / n


def cwmats(n):
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 0] = 3.0 * n * n
    A[5, 2] = -n * n
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return A, B


def dblintegmat():
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    return A


def aug(A, B):
    M = np.zeros((9, 9))
    M[:6, :6] = A
    M[:6, 6:] = B
    return M


# Coefficients b_j of (1 + q)^(-3/2) = sum_j b_j q^j
def binomseriescoef(nterms=40):
    b = [1.0]
    for j in range(nterms):
        b.append(b[-1] * (-1.5 - j) / (j + 1))
    return np.array(b)


_B32 = binomseriescoef(40)


# h2(q) = f(q) - 1.5 q with f = 1 - (1 + q)^(-3/2). The direct form loses most of its digits to
# cancellation for small |q|, so the series is used there.
def _h2(q):
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    small = np.abs(q) < 0.1
    if np.any(small):
        qs = q[small]
        acc = np.zeros_like(qs)
        for c in _B32[:1:-1]:
            acc = acc * qs + c
        out[small] = -qs * qs * acc
    if np.any(~small):
        ql = q[~small]
        out[~small] = -np.expm1(-1.5 * np.log1p(ql)) - 1.5 * ql
    return out


# Exact two-body acceleration minus the CW acceleration, for a circular chief
def residexact(pos, n, R):
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
    a = -(K @ s)
    na = float(np.linalg.norm(a))
    if na > a_max:
        ah = a / na
        J = (a_max / na) * (np.eye(3) - np.outer(ah, ah)) @ (-K)
        return a * (a_max / na), True, J
    return a, False, -K


def updatetimes(Ts, T):
    kmax = int(math.ceil(T / Ts)) + 1
    t = Ts * np.arange(kmax + 1, dtype=float)
    return t[t < T - TTOL]


def mongrid(T, dt=DTMONITOR):
    j = int(math.floor(T / dt + 1e-9))
    t = dt * np.arange(j + 1, dtype=float)
    if T - t[-1] > TTOL:
        t = np.append(t, T)
    else:
        t[-1] = T
    return t


# expm(tau M) is computed once per distinct tau; the same offsets recur in every hold
class ExpmCache:

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


# Composite Gauss-Legendre on [0, L]. The mirrored nodes are computed as L - sigma so equal offsets
# give identical floats in every interval and hit the expm cache.
def glrule(L, sub, nodes):
    x, w = leggauss(int(nodes))
    M = max(1, int(math.ceil(L / sub - 1e-12)))
    h = L / M
    p = np.arange(M, dtype=float)[:, None]
    sig = (p * h + 0.5 * h * (1.0 + x[None, :])).ravel()
    sigrev = ((M - 1 - p) * h + 0.5 * h * (1.0 - x[None, :])).ravel()
    ww = np.tile(0.5 * h * w, M)
    return sig, ww, sigrev


def checkkind(kind):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def amax(a_max):
    return math.inf if a_max is None else float(a_max)


def checktq(tq, T):
    tq = np.asarray(tq, dtype=float)
    if tq.ndim != 1 or tq.size == 0:
        raise ValueError("tq must be a non-empty 1-D array")
    if np.any(tq < -TTOL) or np.any(tq > T + TTOL):
        raise ValueError("tq must lie in [0, T]")
    return np.clip(tq, 0.0, T)


def nomupd(rel0, K, a_max, A, B, Ts, T, En):
    tu = updatetimes(Ts, T)
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


def heldcwnom(rel0, K, a_max, n, Ts, T, tq, _cache=None):
    a_max = amax(a_max)
    K = np.asarray(K, dtype=float)
    A, B = cwmats(n)
    En = _cache if _cache is not None else ExpmCache(aug(A, B))
    nom = nomupd(rel0, K, a_max, A, B, Ts, T, En)
    tq = checktq(tq, T)
    tu = nom["t_update"]
    hk = np.clip(np.searchsorted(tu, tq, side="right") - 1, 0, tu.size - 1)
    a = tq - tu[hk]
    E = En(a)
    s0 = (np.einsum("gij,gj->gi", E[:, :6, :6], nom["s_update"][hk])
          + np.einsum("gij,gj->gi", E[:, :6, 6:], nom["u_update"][hk]))
    nom.update(t=tq, s0=s0)
    return nom


def setup(rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing):
    checkkind(kind)
    R, n, T = orbit(altitude, horizon_orbits)
    K = np.asarray(K, dtype=float)
    a_max = amax(a_max)
    A, B = cwmats(n)
    En = ExpmCache(aug(A, B))
    Ae = dblintegmat() if kind == "C_H" else A
    Ee = ExpmCache(aug(Ae, B)) if kind == "C_H" else En
    tq = mongrid(T) if tq is None else checktq(tq, T)
    if forcing is None:
        def forcing(pos, _n=n, _R=R):
            return residexact(pos, _n, _R)
    nom = nomupd(rel0, K, a_max, A, B, Ts, T, En)
    return R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom


def fb(kind, k, nom, K):
    if kind == "A_H":
        return nom["J_update"][k]
    if kind == "B_H":
        return -K
    return None


def finish(out, tq, e, s0):
    p = np.linalg.norm(e[:, :3], axis=1)
    j = int(np.argmax(p))
    out.update(t=tq, e=e, s0=s0, feature_m=float(p[j]), t_peak_s=float(tq[j]))
    return out


# First variation e(t) by variation of constants, with composite Gauss-Legendre quadrature in each hold
def heldsens(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
             horizon_orbits=1.0, forcing=None):
    R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom = setup(
        rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing)
    tu, su, uu = nom["t_update"], nom["s_update"], nom["u_update"]
    Ku = tu.size

    P = np.unique(np.concatenate((tu, [T], tq)))
    hk = np.clip(np.searchsorted(tu, P, side="right") - 1, 0, Ku - 1)
    a = P - tu[hk]
    EnP = En(a)
    s0P = (np.einsum("gij,gj->gi", EnP[:, :6, :6], su[hk])
           + np.einsum("gij,gj->gi", EnP[:, :6, 6:], uu[hk]))

    npc = P.size - 1
    L = np.diff(P)
    hkpc = hk[:-1]
    G = np.empty((npc, 6))
    for Lv in np.unique(L):
        idx = np.nonzero(L == Lv)[0]
        rho, w, rhorev = glrule(float(Lv), sub, nodes)
        Er = En(rho)
        W = w[:, None, None] * (Ee(rhorev)[:, :6, :6] @ B)
        S = s0P[idx]
        U = uu[hkpc[idx]]
        pos = (np.einsum("qij,gj->gqi", Er[:, :3, :6], S)
               + np.einsum("qij,gj->gqi", Er[:, :3, 6:], U))
        r = forcing(pos)
        G[idx] = np.einsum("qij,gqj->gi", W, r)

    PhiL = Ee(L)[:, :6, :6]
    FEnd = np.empty((npc, 6))
    Fc = np.zeros(6)
    for i in range(npc):
        if a[i] == 0.0:
            Fc = np.zeros(6)
        Fc = PhiL[i] @ Fc + G[i]
        FEnd[i] = Fc
    FP = np.zeros((P.size, 6))
    nz = a[1:] != 0.0
    FP[1:][nz] = FEnd[nz]

    tau = nom["tau_hold"]
    Eh = Ee(tau)
    endidx = np.searchsorted(P, np.append(tu[1:], T))
    FHold = FEnd[endidx - 1]
    eu = np.zeros((Ku + 1, 6))
    mu = np.zeros((Ku, 3))
    for k in range(Ku):
        Jk = fb(kind, k, nom, K)
        if Jk is not None:
            mu[k] = Jk @ eu[k]
        eu[k + 1] = Eh[k, :6, :6] @ eu[k] + Eh[k, :6, 6:] @ mu[k] + FHold[k]

    EeP = Ee(a)
    eP = (np.einsum("gij,gj->gi", EeP[:, :6, :6], eu[hk])
          + np.einsum("gij,gj->gi", EeP[:, :6, 6:], mu[hk]) + FP)
    iq = np.searchsorted(P, tq)
    out = dict(t_update=tu, e_update=eu, m_update=mu, s_update=su, u_update=uu,
               sat_at_update=nom["sat_at_update"].copy(), switch_margin=nom["switch_margin"],
               n_updates=int(Ku), T=T, n=n, R=R, kind=kind, sub=sub, nodes=nodes)
    return finish(out, tq, eP[iq], s0P[iq])


# Same quantity via an augmented state z = (e, m) and a reset map at each update. Only used to
# cross-check heldsens.
def heldsensaug(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
                horizon_orbits=1.0, forcing=None):
    R, n, T, K, a_max, A, B, En, Ee, tq, forcing, nom = setup(
        rel0, K, a_max, altitude, Ts, kind, tq, horizon_orbits, forcing)
    tu = nom["t_update"]
    Ku = tu.size
    wu = np.hstack((nom["s_update"][:Ku], nom["u_update"]))
    BR = np.zeros((9, 3))
    BR[3:6, :] = np.eye(3)

    def int2(tauv, holds):
        sig, w, sigrev = glrule(tauv, sub, nodes)
        EnS = En(sig)
        pos = np.einsum("qij,gj->gqi", EnS[:, :3, :], wu[holds])
        r = forcing(pos) * w[None, :, None]
        Wm = Ee(sigrev) @ BR
        return np.einsum("qij,gqj->gi", Wm, r)

    tau = nom["tau_hold"]
    IHold = np.empty((Ku, 9))
    for tv in np.unique(tau):
        idx = np.nonzero(tau == tv)[0]
        IHold[idx] = int2(float(tv), idx)

    Eh = Ee(tau)
    zm = np.zeros((Ku + 1, 9))
    zp = np.zeros((Ku, 9))
    for k in range(Ku):
        Rk = np.zeros((9, 9))
        Rk[:6, :6] = np.eye(6)
        Jk = fb(kind, k, nom, K)
        if Jk is not None:
            Rk[6:, :6] = Jk
        zp[k] = Rk @ zm[k]
        zm[k + 1] = Eh[k] @ zp[k] + IHold[k]

    hk = np.clip(np.searchsorted(tu, tq, side="right") - 1, 0, Ku - 1)
    off = tq - tu[hk]
    e = np.empty((tq.size, 6))
    for tv in np.unique(off):
        idx = np.nonzero(off == tv)[0]
        z = Ee(np.array([tv]))[0] @ zp[hk[idx]].T
        e[idx] = (z.T + int2(float(tv), hk[idx]))[:, :6]
    Eq = En(off)
    s0 = np.einsum("gij,gj->gi", Eq[:, :6, :], wu[hk])
    out = dict(t_update=tu, e_update=zm[:, :6], m_update=zp[:, 6:], z_update_minus=zm,
               s_update=nom["s_update"], u_update=nom["u_update"],
               sat_at_update=nom["sat_at_update"].copy(), switch_margin=nom["switch_margin"],
               n_updates=int(Ku), T=T, n=n, R=R, kind=kind, sub=sub, nodes=nodes)
    return finish(out, tq, e, s0)


def checkaugconsist(rel0, K, a_max, altitude, Ts, kind, tq=None, sub=0.5, nodes=8,
                    horizon_orbits=1.0, rtol=1e-10):
    a = heldsens(rel0, K, a_max, altitude, Ts, kind, tq, sub, nodes, horizon_orbits)
    b = heldsensaug(rel0, K, a_max, altitude, Ts, kind, tq, sub, nodes,
                    horizon_orbits)
    scale = max(np.abs(a["e"]).max(), np.finfo(float).tiny)
    d = float(max(np.abs(a["e"] - b["e"]).max(), np.abs(a["e_update"] - b["e_update"]).max()))
    rel = d / scale
    assert rel <= rtol, f"augmented vs variation-of-constants mismatch: rel {rel:.3e} > {rtol:.1e}"
    return rel, a, b


# With r(s0) replaced by a constant c and no saturation the recursion has a closed form, which gives
# an exact answer to test the quadrature against.
def lincheck(K, altitude, Ts, c, kind="A_H", tq=None, sub=0.5, nodes=8, horizon_orbits=1.0,
             rel0=None):
    checkkind(kind)
    R, n, T = orbit(altitude, horizon_orbits)
    K = np.asarray(K, dtype=float)
    c = np.asarray(c, dtype=float).reshape(3)
    rel0 = np.zeros(6) if rel0 is None else rel0
    quad = heldsens(rel0, K, math.inf, altitude, Ts, kind, tq, sub, nodes,
                    horizon_orbits, forcing=lambda pos: np.broadcast_to(c, pos.shape))
    A, B = cwmats(n)
    Ae = dblintegmat() if kind == "C_H" else A
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

    Phi, Gm, Gc = blocks(float(tau[0]))
    kfull = int(np.sum(tau == tau[0]))
    Mb = np.zeros((7, 7))
    Mb[:6, :6] = Phi - Gm @ Kf
    Mb[:6, 6] = Gc @ c
    Mb[6, 6] = 1.0
    ecf = np.linalg.matrix_power(Mb, kfull)[:6, 6]
    scaleu = max(np.abs(e).max(), np.finfo(float).tiny)
    relcf = float(np.abs(ecf - e[kfull]).max() / scaleu)

    tq_ = quad["t"]
    hk = np.clip(np.searchsorted(tu, tq_, side="right") - 1, 0, tu.size - 1)
    off = tq_ - tu[hk]
    eg = np.empty((tq_.size, 6))
    for tv in np.unique(off):
        idx = np.nonzero(off == tv)[0]
        Ph, Gm_, Gc_ = blocks(float(tv))
        ek = e[hk[idx]]
        eg[idx] = ek @ (Ph - Gm_ @ Kf).T + Gc_ @ c
    relu = float(np.abs(quad["e_update"] - e).max() / scaleu)
    relg = float(np.abs(quad["e"] - eg).max() / max(np.abs(eg).max(), np.finfo(float).tiny))
    return dict(rel_err_updates=relu, rel_err_grid=relg, rel_err_closed_form=relcf,
                quad=quad, exact_update=e, exact_grid=eg)
