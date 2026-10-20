# Fixed-step batch propagation of held-command (zero-order-hold) branches. The command is constant
# in LVLH inside each hold, so each hold is a smooth ODE and steps aligned with the updates stay accurate.
import math

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dc
from scipy.optimize import brentq

from cwvalid.constants import ALTFLOOR, DEFINC, MUEARTH, REARTH

NS = _dc.N_STAGES
TA = np.array(_dc.A[:NS, :NS], dtype=float)
TB = np.array(_dc.B, dtype=float)
RELMOD = ("cw", "so", "nerm")
ESTIM = ("A_H", "B_H", "A_naive", "C_H")
BLOWUPREL = 1e15
ENDTOLS = 1e-9
EXIT = -2


def cwmatrix(n):
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4], A[4, 3], A[5, 2] = 3.0 * n * n, 2.0 * n, -2.0 * n, -n * n
    return A


# Exact minus CW acceleration for a circular two-body chief (the Coriolis terms cancel)
def resexact(P, n, R):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    q = (2.0 * R * x + x * x + y * y + z * z) / (R * R)
    f = -np.expm1(-1.5 * np.log1p(q))
    n2 = n * n
    out = np.empty(P.shape)
    out[..., 0] = n2 * (R + x) * f - 3.0 * n2 * x
    out[..., 1] = n2 * y * f
    out[..., 2] = n2 * z * f
    return out


def residquad(P, n, R):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    c = 3.0 * n * n / R
    out = np.empty(P.shape)
    out[..., 0] = -0.5 * c * (2.0 * x * x - y * y - z * z)
    out[..., 1] = c * x * y
    out[..., 2] = c * x * z
    return out


def command(K, amax, S):
    raw = -np.einsum("nij,nj->ni", K, S)
    mag = np.sqrt(np.einsum("ni,ni->n", raw, raw))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(mag > amax, amax / mag, 1.0)
    return raw * scale[:, None], raw, mag


# Jacobian of normsat(-K s), plus which cases are saturated
def jacu(K, amax, S):
    raw = -np.einsum("nij,nj->ni", K, S)
    mag = np.sqrt(np.einsum("ni,ni->n", raw, raw))
    sat = mag > amax
    J = -K.copy()
    if np.any(sat):
        a = raw[sat] / mag[sat, None]
        P = np.eye(3)[None, :, :] - a[:, :, None] * a[:, None, :]
        J[sat] = (amax[sat] / mag[sat])[:, None, None] * np.einsum("nij,njk->nik", P, -K[sat])
    return J, sat, mag


# Scalar PD law as a gain matrix, K = [wn^2 I, 2 wn I] (critically damped)
def pdmatrix(wn):
    return np.hstack((wn * wn * np.eye(3), 2.0 * wn * np.eye(3)))


# LVLH axes and the in-plane frame rate |h| / r^2
def lvlhframe(rt, vt):
    r = np.sqrt(np.einsum("...i,...i->...", rt, rt))
    xh = rt / r[..., None]
    h = np.cross(rt, vt)
    hn = np.sqrt(np.einsum("...i,...i->...", h, h))
    zh = h / hn[..., None]
    yh = np.cross(zh, xh)
    return np.stack((xh, yh, zh), axis=-2), hn / (r * r)


def ecitorel(rt, vt, rc, vc):
    C, wz = lvlhframe(rt, vt)
    rho = np.einsum("...ij,...j->...i", C, rc - rt)
    rhod = np.einsum("...ij,...j->...i", C, vc - vt)
    rhod[..., 0] += wz * rho[..., 1]
    rhod[..., 1] -= wz * rho[..., 0]
    return np.concatenate((rho, rhod), axis=-1)


def reltoeci(rt, vt, rel):
    C, wz = lvlhframe(rt, vt)
    rho, rhod = rel[..., :3], rel[..., 3:]
    wxrho = np.stack((-wz * rho[..., 1], wz * rho[..., 0], np.zeros_like(rho[..., 0])), axis=-1)
    Ct = np.swapaxes(C, -1, -2)
    return (rt + np.einsum("...ij,...j->...i", Ct, rho), vt + np.einsum("...ij,...j->...i", Ct, rhod + wxrho))


class HeldBatch:

    def __init__(self, rel0, K, amax, altitude, Ts, horizon_orbits=1.0, h=None,
                 models=("cw", "so", "nerm", "eci"), ests=("A_H", "B_H", "C_H"), lambdas=(),
                 inclination=DEFINC, altitude_floor=ALTFLOOR, nearswitchtol=1e-9):
        self.rel0 = np.atleast_2d(np.asarray(rel0, float))
        self.N = self.rel0.shape[0]
        self.K = np.asarray(K, float).reshape(self.N, 3, 6)
        am = np.array([np.inf if a is None else float(a) for a in np.atleast_1d(amax)], float)
        self.amax = np.broadcast_to(am, (self.N,)).copy()
        self.altitude = float(altitude)
        self.R = REARTH + self.altitude
        self.n = math.sqrt(MUEARTH / self.R ** 3)
        self.T = horizon_orbits * 2.0 * math.pi / self.n
        self.Ts = float(Ts)
        self.h = float(Ts if h is None else h)
        m = self.Ts / self.h
        if abs(m - round(m)) > 1e-9:
            raise ValueError("the step h must divide the hold period Ts")
        self.models = tuple(models)
        self.ests = tuple(ests)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.inclination = inclination
        self.rfloor = REARTH + altitude_floor
        self.nearswitchtol = nearswitchtol
        self.A6 = cwmatrix(self.n)
        if self.ests and "cw" not in self.models:
            raise ValueError("estimators need the cw branch")
        nfull = int(math.floor(self.T / self.h + 1e-9))
        t = np.arange(nfull + 1) * self.h
        if t[-1] < self.T - 1e-9:
            t = np.append(t, self.T)
        else:
            t[-1] = self.T
        self.t = t
        self.S = t.size - 1
        kmax = int(math.floor(self.T / self.Ts + 1e-9))
        self.updatetimes = np.array([k * self.Ts for k in range(kmax + 1) if k * self.Ts < self.T - 1e-9])
        self.updatestep = np.rint(self.updatetimes / self.h).astype(int)
        self.relnames = [b for b in RELMOD if b in self.models] + [f"hom:{lam:.12g}" for lam in self.lambdas]
        self.nr = len(self.relnames)
        self.ne = len(self.ests)
        self.ceci = 6 * self.nr
        self.haseci = "eci" in self.models
        self.cest = self.ceci + (12 if self.haseci else 0)
        self.D = self.cest + 6 * self.ne
        self.hasresid = np.array([b == "nerm" or b.startswith("hom:") for b in self.relnames])
        self.residcoef = np.array([1.0 if b == "nerm" else (float(b[4:]) if b.startswith("hom:") else 0.0)
                                   for b in self.relnames])
        self.isso = np.array([b == "so" for b in self.relnames])
        self.cmdnames = list(self.relnames) + (["eci"] if self.haseci else [])

    def relslice(self, b):
        k = self.relnames.index(b)
        return slice(6 * k, 6 * k + 6)

    def estslice(self, e):
        k = self.ests.index(e)
        return slice(self.cest + 6 * k, self.cest + 6 * k + 6)

    def _y0(self):
        Y = np.zeros((self.N, self.D))
        for k in range(self.nr):
            Y[:, 6 * k:6 * k + 6] = self.rel0
        if self.haseci:
            ci, si = math.cos(self.inclination), math.sin(self.inclination)
            v = math.sqrt(MUEARTH / self.R)
            t0 = np.array([self.R, 0.0, 0.0, 0.0, v * ci, v * si])
            Y[:, self.ceci:self.ceci + 6] = t0
            rc, vc = reltoeci(np.broadcast_to(t0[:3], (self.N, 3)), np.broadcast_to(t0[3:], (self.N, 3)), self.rel0)
            Y[:, self.ceci + 6:self.ceci + 9] = rc
            Y[:, self.ceci + 9:self.ceci + 12] = vc
        return Y

    def rhs(self, Y, Urel, Ueci, Mest, K=None, amax=None):
        n, R = self.n, self.R
        out = np.empty_like(Y)
        N = Y.shape[0]
        if self.nr:
            S = Y[:, :6 * self.nr].reshape(N, self.nr, 6)
            dS = S @ self.A6.T
            dS[:, :, 3:] += Urel
            P = S[:, :, :3]
            if np.any(self.hasresid):
                sel = self.hasresid
                dS[:, sel, 3:] += self.residcoef[sel][None, :, None] * resexact(P[:, sel], n, R)
            if np.any(self.isso):
                dS[:, self.isso, 3:] += residquad(P[:, self.isso], n, R)
            out[:, :6 * self.nr] = dS.reshape(N, 6 * self.nr)
        if self.haseci:
            c = self.ceci
            rt, vt, rc, vc = Y[:, c:c + 3], Y[:, c + 3:c + 6], Y[:, c + 6:c + 9], Y[:, c + 9:c + 12]
            C, _ = lvlhframe(rt, vt)
            rtn = np.sqrt(np.einsum("ni,ni->n", rt, rt))
            rcn = np.sqrt(np.einsum("ni,ni->n", rc, rc))
            out[:, c:c + 3] = vt
            out[:, c + 3:c + 6] = -MUEARTH * rt / (rtn ** 3)[:, None]
            out[:, c + 6:c + 9] = vc
            out[:, c + 9:c + 12] = -MUEARTH * rc / (rcn ** 3)[:, None] + np.einsum("nji,nj->ni", C, Ueci)
        if self.ne:
            s0 = Y[:, self.relslice("cw")]
            r0 = resexact(s0[:, :3], n, R)
            E = Y[:, self.cest:].reshape(N, self.ne, 6)
            dE = E @ self.A6.T
            dE[:, :, 3:] += r0[:, None, :]
            for k, e in enumerate(self.ests):
                if e in ("A_H", "B_H"):
                    dE[:, k, 3:] += Mest[:, k]
                elif e == "C_H":
                    dE[:, k, :3] = E[:, k, 3:]
                    dE[:, k, 3:] = r0
                elif e == "A_naive":
                    J, _, _ = jacu(self.K if K is None else K, self.amax if amax is None else amax, s0)
                    dE[:, k, 3:] += np.einsum("nij,nj->ni", J, E[:, k])
            out[:, self.cest:] = dE.reshape(N, 6 * self.ne)
        return out

    def _step(self, Y, Urel, Ueci, Mest, h):
        k = np.empty((NS,) + Y.shape)
        k[0] = self.rhs(Y, Urel, Ueci, Mest)
        for i in range(1, NS):
            k[i] = self.rhs(Y + h * np.tensordot(TA[i, :i], k[:i], axes=(0, 0)), Urel, Ueci, Mest)
        return Y + h * np.tensordot(TB, k, axes=(0, 0))

    def releci(self, Y):
        c = self.ceci
        return ecitorel(Y[..., c:c + 3], Y[..., c + 3:c + 6], Y[..., c + 6:c + 9], Y[..., c + 9:c + 12])

    def _update(self, Y, Urel, Ueci, Mest, ku):
        for k, b in enumerate(self.relnames):
            Urel[:, k] = command(self.K, self.amax, Y[:, 6 * k:6 * k + 6])[0]
        if self.haseci:
            Ueci[:] = command(self.K, self.amax, self.releci(Y))[0]
        s0 = Y[:, self.relslice("cw")] if "cw" in self.relnames else None
        if s0 is not None:
            J, sat, mag = jacu(self.K, self.amax, s0)
            self.sat_at_update[:, ku] = sat
            with np.errstate(invalid="ignore"):
                self.nearswitch[:, ku] = np.abs(mag - self.amax) <= self.nearswitchtol * self.amax
            for k, e in enumerate(self.ests):
                Ek = Y[:, self.estslice(e)]
                if e == "A_H":
                    Mest[:, k] = np.einsum("nij,nj->ni", J, Ek)
                elif e == "B_H":
                    Mest[:, k] = np.einsum("nij,nj->ni", -self.K, Ek)

    def run(self):
        N, S = self.N, self.S
        Y = self._y0()
        Urel = np.zeros((N, self.nr, 3))
        Ueci = np.zeros((N, 3))
        Mest = np.zeros((N, max(self.ne, 1), 3))
        self.sat_at_update = np.zeros((N, self.updatetimes.size), bool)
        self.nearswitch = np.zeros((N, self.updatetimes.size), bool)
        self.Y = np.empty((S + 1, N, self.D))
        self.Urel = np.empty((S, N, self.nr, 3))
        self.Ueci = np.empty((S, N, 3))
        self.Mest = np.empty((S, N, max(self.ne, 1), 3))
        self.Y[0] = Y
        names = self.branchnames()
        self.status = {b: np.array(["completed"] * N, dtype=object) for b in names}
        self.t_end = {b: np.full(N, self.T) for b in names}
        self.exit_step = {b: np.full(N, -1) for b in names}
        self.exit_fallback = {b: np.zeros(N, bool) for b in names}
        frozen = {b: np.zeros(N, bool) for b in names}
        for b in names:
            if self.exits(b):
                ex0 = self.radius(Y, b) < self.rfloor
                self.status[b][ex0] = "domain_exit"
                self.t_end[b][ex0] = self.t[0]
                self.exit_step[b][ex0] = EXIT
                frozen[b] |= ex0
        ku = 0
        for j in range(S):
            if ku < self.updatestep.size and self.updatestep[ku] == j:
                self._update(Y, Urel, Ueci, Mest, ku)
                ku += 1
            self.Urel[j], self.Ueci[j], self.Mest[j] = Urel, Ueci, Mest
            h = self.t[j + 1] - self.t[j]
            Yn = self._step(Y, Urel, Ueci, Mest, h)
            for b in names:
                cols = self.cols(b)
                was = frozen[b].copy()
                Ynb = Yn[:, cols]
                lim = 1e12 if b == "eci" else BLOWUPREL
                bad = (~np.all(np.isfinite(Ynb), axis=1)) | (np.max(np.abs(Ynb), axis=1) > lim)
                new = bad & ~frozen[b]
                if np.any(new):
                    self.status[b][new] = "numerical_failure"
                    self.t_end[b][new] = self.t[j]
                    frozen[b] |= new
                if self.exits(b):
                    ex = (self.radius(Yn, b) < self.rfloor) & ~frozen[b]
                    for i in np.where(ex)[0]:
                        self.t_end[b][i], self.exit_fallback[b][i] = self._locate_exit(Y, Yn, Urel, Ueci, Mest, i, b, j)
                        self.status[b][i] = "domain_exit"
                        self.exit_step[b][i] = j
                    frozen[b] |= ex
                fz = np.where(was | new)[0]
                if fz.size:
                    Yn[np.ix_(fz, cols)] = Y[np.ix_(fz, cols)]
            Y = Yn
            self.Y[j + 1] = Y
        return self

    def branchnames(self):
        return list(self.relnames) + (["eci"] if self.haseci else []) + list(self.ests)

    def cols(self, b):
        if b == "eci":
            return list(range(self.ceci, self.ceci + 12))
        sl = self.relslice(b) if b in self.relnames else self.estslice(b)
        return list(range(sl.start, sl.stop))

    @staticmethod
    def exits(b):
        return b in ("nerm", "eci") or b.startswith("hom:")

    def radius(self, Y, b):
        if b == "eci":
            c = self.ceci + 6
            return np.sqrt(np.einsum("...i,...i->...", Y[..., c:c + 3], Y[..., c:c + 3]))
        Sb = Y[..., self.relslice(b)]
        return np.sqrt((self.R + Sb[..., 0]) ** 2 + Sb[..., 1] ** 2 + Sb[..., 2] ** 2)

    def _locate_exit(self, Y, Yn, Urel, Ueci, Mest, i, b, j):
        t0, t1 = self.t[j], self.t[j + 1]
        sl = slice(i, i + 1)
        F0 = self.rhs(Y[sl], Urel[sl], Ueci[sl], Mest[sl], self.K[sl], self.amax[sl])[0]
        F1 = self.rhs(Yn[sl], Urel[sl], Ueci[sl], Mest[sl], self.K[sl], self.amax[sl])[0]

        def g(t):
            y = hermite(t0, t1, Y[i], Yn[i], F0, F1, t)
            return float(self.radius(y[None, :], b)[0] - self.rfloor)
        try:
            return float(brentq(g, t0, t1, xtol=1e-9)), False
        except ValueError:
            return float(t1), True

    def casederivs(self, i):
        Yi = self.Y[:, i, :]
        Ki = np.broadcast_to(self.K[i], (self.S, 3, 6))
        ai = np.full(self.S, self.amax[i])
        Fr = self.rhs(Yi[:-1], self.Urel[:, i], self.Ueci[:, i], self.Mest[:, i], Ki, ai)
        Fl = self.rhs(Yi[1:], self.Urel[:, i], self.Ueci[:, i], self.Mest[:, i], Ki, ai)
        return Fr, Fl

    def dense(self, i, tq, Fr=None, Fl=None, mask=True):
        if Fr is None:
            Fr, Fl = self.casederivs(i)
        tq = np.atleast_1d(np.asarray(tq, float))
        j = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, self.S - 1)
        t0, t1 = self.t[j], self.t[j + 1]
        out = hermite(t0[:, None], t1[:, None], self.Y[j, i], self.Y[j + 1, i], Fr[j], Fl[j], tq[:, None])
        if mask:
            for b in self.branchnames():
                if self.status[b][i] != "completed":
                    after = tq > self.t_end[b][i] + ENDTOLS
                    if np.any(after):
                        out[np.ix_(after, self.cols(b))] = np.nan
        return out


def hermite(t0, t1, y0, y1, f0, f1, t):
    h = t1 - t0
    s = (t - t0) / h
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * f0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * f1
