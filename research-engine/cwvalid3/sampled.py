"""Batch fixed-step propagation of held-command (zero-order-hold) closed-loop branches,
with held-command first-variation estimators propagated in the same steps.

Why fixed steps are accurate here. With a held command every branch is a smooth ODE inside each
hold interval: the command is constant in LVLH components, so the motion inside a hold is the
uncontrolled relative (or two-body) motion plus a constant LVLH acceleration, on the orbital time
scale 1/n, independent of the controller bandwidth. Steps of the 8th-order Dormand-Prince tableau
(DOP853 coefficients from SciPy) are taken with a constant step h that divides the hold period, so
update instants are step boundaries; the default h equals the hold period. The 0.5 s monitoring
grid is evaluated afterwards by cubic Hermite interpolation between steps (see held_metrics).

Branches (vectorised over a batch of cases sharing altitude, horizon and hold period; see
v3_extension/methods/V3_CONVENTIONS.md):
  cw, so, nerm    relative models with their own held commands
  hom:<lam>       homotopy branch s' = A s + B U(s(t_k)) + lam B r(s) (finite-difference checks)
  eci             inertial two-spacecraft reference; the held LVLH command is rotated with the
                  current target frame; feedback uses the reconstructed relative state
Estimators along the CW branch s0:
  A_H      e' = A e + B J_U(s0(t_k)) e(t_k) + B r(s0(t))
  B_H      e' = A e + B (-K) e(t_k) + B r(s0(t))
  A_naive  e' = (A + B J_U(s0(t))) e + B r(s0(t))       (continuous formula; diagnostic)
  C_H      e_pos'' = r(s0(t))                              (residual only; diagnostic)
A_naive can change saturation status inside a hold; use a finer step (e.g. 0.5 s) for it.

Physical-domain exit (chaser altitude below 100 km) is detected for nerm, eci and hom at t = 0 and
at step ends, and located inside the step by cubic Hermite interpolation; the branch is then frozen.
A numerical failure (non-finite state or |state| > 1e15 m) freezes the branch at the last good step.

v3.1 correction (C01). In v3.0 a branch that left the domain inside step j had its stored step end
Y[j+1] overwritten with the start-of-step state Y[j], so the dense output on [t_j, t_{j+1}] was not
the interpolant on which the exit had been located (the exit time and status were right; the dense
state at t_end was up to 565 m above the floor and peak discrepancies of exit cases were biased).
Now the exiting rows keep the actual trial-step endpoint, which is the endpoint _locate_exit used,
and are frozen from step j+1 on; rows frozen earlier and rows that fail numerically in step j are
still frozen at the last good state. dense() returns NaN in a branch's columns after its t_end,
so no reader can use the frozen placeholder. exit_step[b][i] and exit_fallback[b][i] record the
step that contained the event and whether the root bracket failed (t_end then defaults to t_{j+1});
exit_step is -1 when there is no exit and EXIT_AT_T0 (-2) when the initial state is already below the
floor. The status field is authoritative.
"""

import math

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dc
from scipy.optimize import brentq

from cwvalid.constants import ALTITUDE_FLOOR, DEFAULT_INCLINATION, MU_EARTH, R_EARTH

_NS = _dc.N_STAGES
_TA = np.array(_dc.A[:_NS, :_NS], dtype=float)
_TB = np.array(_dc.B, dtype=float)
REL_MODELS = ("cw", "so", "nerm")
ESTIMATORS = ("A_H", "B_H", "A_naive", "C_H")
BLOWUP_REL = 1e15
END_TOL_S = 1e-9
EXIT_AT_T0 = -2           # exit_step sentinel: initial state below the floor (no in-step event)


def cw_matrix(n):
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4], A[4, 3], A[5, 2] = 3.0 * n * n, 2.0 * n, -2.0 * n, -n * n
    return A


def resid_exact(P, n, R):
    """r(rho) = a_exact - a_CW for the circular two-body chief (Coriolis terms cancel)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    q = (2.0 * R * x + x * x + y * y + z * z) / (R * R)
    f = -np.expm1(-1.5 * np.log1p(q))
    n2 = n * n
    out = np.empty(P.shape)
    out[..., 0] = n2 * (R + x) * f - 3.0 * n2 * x
    out[..., 1] = n2 * y * f
    out[..., 2] = n2 * z * f
    return out


def resid_quadratic(P, n, R):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    c = 3.0 * n * n / R
    out = np.empty(P.shape)
    out[..., 0] = -0.5 * c * (2.0 * x * x - y * y - z * z)
    out[..., 1] = c * x * y
    out[..., 2] = c * x * z
    return out


def command(K, amax, S):
    """normsat(-K s): K (N,3,6), amax (N,) (np.inf = no cap), S (N,6) -> (u, raw, |raw|)."""
    raw = -np.einsum("nij,nj->ni", K, S)
    mag = np.sqrt(np.einsum("ni,ni->n", raw, raw))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(mag > amax, amax / mag, 1.0)
    return raw * scale[:, None], raw, mag


def jac_u(K, amax, S):
    """Jacobian of normsat(-K s) at S -> (J (N,3,6), saturated (N,), |raw| (N,))."""
    raw = -np.einsum("nij,nj->ni", K, S)
    mag = np.sqrt(np.einsum("ni,ni->n", raw, raw))
    sat = mag > amax
    J = -K.copy()
    if np.any(sat):
        a = raw[sat] / mag[sat, None]
        P = np.eye(3)[None, :, :] - a[:, :, None] * a[:, None, :]
        J[sat] = (amax[sat] / mag[sat])[:, None, None] * np.einsum("nij,njk->nik", P, -K[sat])
    return J, sat, mag


def pd_matrix(wn):
    """Scalar PD law as a 3x6 gain matrix K = [wn^2 I, 2 wn I]."""
    return np.hstack((wn * wn * np.eye(3), 2.0 * wn * np.eye(3)))


def lvlh_frame(rt, vt):
    """LVLH rows (x, y, z) as (...,3,3) and the in-plane frame rate |h|/r^2."""
    r = np.sqrt(np.einsum("...i,...i->...", rt, rt))
    xh = rt / r[..., None]
    h = np.cross(rt, vt)
    hn = np.sqrt(np.einsum("...i,...i->...", h, h))
    zh = h / hn[..., None]
    yh = np.cross(zh, xh)
    return np.stack((xh, yh, zh), axis=-2), hn / (r * r)


def eci_to_rel(rt, vt, rc, vc):
    C, wz = lvlh_frame(rt, vt)
    rho = np.einsum("...ij,...j->...i", C, rc - rt)
    rhod = np.einsum("...ij,...j->...i", C, vc - vt)
    rhod[..., 0] += wz * rho[..., 1]
    rhod[..., 1] -= wz * rho[..., 0]
    return np.concatenate((rho, rhod), axis=-1)


def rel_to_eci(rt, vt, rel):
    C, wz = lvlh_frame(rt, vt)
    rho, rhod = rel[..., :3], rel[..., 3:]
    w_x_rho = np.stack((-wz * rho[..., 1], wz * rho[..., 0], np.zeros_like(rho[..., 0])), axis=-1)
    Ct = np.swapaxes(C, -1, -2)
    return (rt + np.einsum("...ij,...j->...i", Ct, rho), vt + np.einsum("...ij,...j->...i", Ct, rhod + w_x_rho))


class HeldBatch:
    """Cases sharing altitude, horizon (orbits) and hold period Ts; one gain matrix and cap per case."""

    def __init__(self, rel0, K, amax, altitude, Ts, horizon_orbits=1.0, h=None,
                 models=("cw", "so", "nerm", "eci"), estimators=("A_H", "B_H", "C_H"), lambdas=(),
                 inclination=DEFAULT_INCLINATION, altitude_floor=ALTITUDE_FLOOR, near_switch_tol=1e-9):
        self.rel0 = np.atleast_2d(np.asarray(rel0, float))
        self.N = self.rel0.shape[0]
        self.K = np.asarray(K, float).reshape(self.N, 3, 6)
        am = np.array([np.inf if a is None else float(a) for a in np.atleast_1d(amax)], float)
        self.amax = np.broadcast_to(am, (self.N,)).copy()
        self.altitude = float(altitude)
        self.R = R_EARTH + self.altitude
        self.n = math.sqrt(MU_EARTH / self.R ** 3)
        self.T = horizon_orbits * 2.0 * math.pi / self.n
        self.Ts = float(Ts)
        self.h = float(Ts if h is None else h)
        m = self.Ts / self.h
        if abs(m - round(m)) > 1e-9:
            raise ValueError("the step h must divide the hold period Ts")
        self.models = tuple(models)
        self.estimators = tuple(estimators)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.inclination = inclination
        self.r_floor = R_EARTH + altitude_floor
        self.near_switch_tol = near_switch_tol
        self.A6 = cw_matrix(self.n)
        if self.estimators and "cw" not in self.models:
            raise ValueError("estimators need the cw branch")
        nfull = int(math.floor(self.T / self.h + 1e-9))
        t = np.arange(nfull + 1) * self.h
        if t[-1] < self.T - 1e-9:
            t = np.append(t, self.T)
        else:
            t[-1] = self.T
        self.t = t                       # step boundaries
        self.S = t.size - 1              # number of steps
        kmax = int(math.floor(self.T / self.Ts + 1e-9))
        self.update_times = np.array([k * self.Ts for k in range(kmax + 1) if k * self.Ts < self.T - 1e-9])
        self.update_step = np.rint(self.update_times / self.h).astype(int)
        # combined state: relative branches (contiguous), eci target+chaser, estimators (contiguous)
        self.rel_names = [b for b in REL_MODELS if b in self.models] + [f"hom:{lam:.12g}" for lam in self.lambdas]
        self.nr = len(self.rel_names)
        self.ne = len(self.estimators)
        self.c_eci = 6 * self.nr
        self.has_eci = "eci" in self.models
        self.c_est = self.c_eci + (12 if self.has_eci else 0)
        self.D = self.c_est + 6 * self.ne
        # coefficient of the exact residual: 1 for nerm, lambda (any sign) for a homotopy branch
        self.has_resid = np.array([b == "nerm" or b.startswith("hom:") for b in self.rel_names])
        self.resid_coef = np.array([1.0 if b == "nerm" else (float(b[4:]) if b.startswith("hom:") else 0.0)
                                    for b in self.rel_names])
        self.is_so = np.array([b == "so" for b in self.rel_names])
        self.cmd_names = list(self.rel_names) + (["eci"] if self.has_eci else [])

    # -- layout helpers ------------------------------------------------------------
    def rel_slice(self, b):
        k = self.rel_names.index(b)
        return slice(6 * k, 6 * k + 6)

    def est_slice(self, e):
        k = self.estimators.index(e)
        return slice(self.c_est + 6 * k, self.c_est + 6 * k + 6)

    def _y0(self):
        Y = np.zeros((self.N, self.D))
        for k in range(self.nr):
            Y[:, 6 * k:6 * k + 6] = self.rel0
        if self.has_eci:
            ci, si = math.cos(self.inclination), math.sin(self.inclination)
            v = math.sqrt(MU_EARTH / self.R)
            t0 = np.array([self.R, 0.0, 0.0, 0.0, v * ci, v * si])
            Y[:, self.c_eci:self.c_eci + 6] = t0
            rc, vc = rel_to_eci(np.broadcast_to(t0[:3], (self.N, 3)), np.broadcast_to(t0[3:], (self.N, 3)), self.rel0)
            Y[:, self.c_eci + 6:self.c_eci + 9] = rc
            Y[:, self.c_eci + 9:self.c_eci + 12] = vc
        return Y

    # -- right-hand side (vectorised over the leading axis; K/amax broadcast per row) ----------
    def rhs(self, Y, Urel, Ueci, Mest, K=None, amax=None):
        """Y (N,D); Urel (N,nr,3); Ueci (N,3); Mest (N,ne,3) held estimator memory terms."""
        n, R = self.n, self.R
        out = np.empty_like(Y)
        N = Y.shape[0]
        if self.nr:
            S = Y[:, :6 * self.nr].reshape(N, self.nr, 6)
            dS = S @ self.A6.T
            dS[:, :, 3:] += Urel
            P = S[:, :, :3]
            if np.any(self.has_resid):
                sel = self.has_resid
                dS[:, sel, 3:] += self.resid_coef[sel][None, :, None] * resid_exact(P[:, sel], n, R)
            if np.any(self.is_so):
                dS[:, self.is_so, 3:] += resid_quadratic(P[:, self.is_so], n, R)
            out[:, :6 * self.nr] = dS.reshape(N, 6 * self.nr)
        if self.has_eci:
            c = self.c_eci
            rt, vt, rc, vc = Y[:, c:c + 3], Y[:, c + 3:c + 6], Y[:, c + 6:c + 9], Y[:, c + 9:c + 12]
            C, _ = lvlh_frame(rt, vt)
            rtn = np.sqrt(np.einsum("ni,ni->n", rt, rt))
            rcn = np.sqrt(np.einsum("ni,ni->n", rc, rc))
            out[:, c:c + 3] = vt
            out[:, c + 3:c + 6] = -MU_EARTH * rt / (rtn ** 3)[:, None]
            out[:, c + 6:c + 9] = vc
            out[:, c + 9:c + 12] = -MU_EARTH * rc / (rcn ** 3)[:, None] + np.einsum("nji,nj->ni", C, Ueci)
        if self.ne:
            s0 = Y[:, self.rel_slice("cw")]
            r0 = resid_exact(s0[:, :3], n, R)
            E = Y[:, self.c_est:].reshape(N, self.ne, 6)
            dE = E @ self.A6.T
            dE[:, :, 3:] += r0[:, None, :]
            for k, e in enumerate(self.estimators):
                if e in ("A_H", "B_H"):
                    dE[:, k, 3:] += Mest[:, k]
                elif e == "C_H":
                    dE[:, k, :3] = E[:, k, 3:]
                    dE[:, k, 3:] = r0
                elif e == "A_naive":
                    J, _, _ = jac_u(self.K if K is None else K, self.amax if amax is None else amax, s0)
                    dE[:, k, 3:] += np.einsum("nij,nj->ni", J, E[:, k])
            out[:, self.c_est:] = dE.reshape(N, 6 * self.ne)
        return out

    def _step(self, Y, Urel, Ueci, Mest, h):
        k = np.empty((_NS,) + Y.shape)
        k[0] = self.rhs(Y, Urel, Ueci, Mest)
        for i in range(1, _NS):
            k[i] = self.rhs(Y + h * np.tensordot(_TA[i, :i], k[:i], axes=(0, 0)), Urel, Ueci, Mest)
        return Y + h * np.tensordot(_TB, k, axes=(0, 0))

    def rel_eci(self, Y):
        c = self.c_eci
        return eci_to_rel(Y[..., c:c + 3], Y[..., c + 3:c + 6], Y[..., c + 6:c + 9], Y[..., c + 9:c + 12])

    def _update(self, Y, Urel, Ueci, Mest, ku):
        for k, b in enumerate(self.rel_names):
            Urel[:, k] = command(self.K, self.amax, Y[:, 6 * k:6 * k + 6])[0]
        if self.has_eci:
            Ueci[:] = command(self.K, self.amax, self.rel_eci(Y))[0]
        s0 = Y[:, self.rel_slice("cw")] if "cw" in self.rel_names else None
        if s0 is not None:
            J, sat, mag = jac_u(self.K, self.amax, s0)
            self.sat_at_update[:, ku] = sat
            with np.errstate(invalid="ignore"):
                self.near_switch[:, ku] = np.abs(mag - self.amax) <= self.near_switch_tol * self.amax
            for k, e in enumerate(self.estimators):
                Ek = Y[:, self.est_slice(e)]
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
        self.sat_at_update = np.zeros((N, self.update_times.size), bool)
        self.near_switch = np.zeros((N, self.update_times.size), bool)
        self.Y = np.empty((S + 1, N, self.D))
        self.Urel = np.empty((S, N, self.nr, 3))
        self.Ueci = np.empty((S, N, 3))
        self.Mest = np.empty((S, N, max(self.ne, 1), 3))
        self.Y[0] = Y
        names = self.branch_names()
        self.status = {b: np.array(["completed"] * N, dtype=object) for b in names}
        self.t_end = {b: np.full(N, self.T) for b in names}
        self.exit_step = {b: np.full(N, -1) for b in names}
        self.exit_fallback = {b: np.zeros(N, bool) for b in names}
        frozen = {b: np.zeros(N, bool) for b in names}
        for b in names:
            if self.exits(b):
                ex0 = self.radius(Y, b) < self.r_floor
                self.status[b][ex0] = "domain_exit"
                self.t_end[b][ex0] = self.t[0]
                self.exit_step[b][ex0] = EXIT_AT_T0
                frozen[b] |= ex0
        ku = 0
        for j in range(S):
            if ku < self.update_step.size and self.update_step[ku] == j:
                self._update(Y, Urel, Ueci, Mest, ku)
                ku += 1
            self.Urel[j], self.Ueci[j], self.Mest[j] = Urel, Ueci, Mest
            h = self.t[j + 1] - self.t[j]
            Yn = self._step(Y, Urel, Ueci, Mest, h)
            for b in names:
                cols = self.cols(b)
                was = frozen[b].copy()
                Ynb = Yn[:, cols]
                lim = 1e12 if b == "eci" else BLOWUP_REL
                bad = (~np.all(np.isfinite(Ynb), axis=1)) | (np.max(np.abs(Ynb), axis=1) > lim)
                new = bad & ~frozen[b]
                if np.any(new):
                    self.status[b][new] = "numerical_failure"
                    self.t_end[b][new] = self.t[j]
                    frozen[b] |= new
                if self.exits(b):
                    ex = (self.radius(Yn, b) < self.r_floor) & ~frozen[b]
                    for i in np.where(ex)[0]:
                        self.t_end[b][i], self.exit_fallback[b][i] = self._locate_exit(Y, Yn, Urel, Ueci, Mest, i, b, j)
                        self.status[b][i] = "domain_exit"
                        self.exit_step[b][i] = j
                    frozen[b] |= ex
                # rows that exit in this step keep the trial endpoint the exit was located on
                fz = np.where(was | new)[0]
                if fz.size:
                    Yn[np.ix_(fz, cols)] = Y[np.ix_(fz, cols)]
            Y = Yn
            self.Y[j + 1] = Y
        return self

    def branch_names(self):
        return list(self.rel_names) + (["eci"] if self.has_eci else []) + list(self.estimators)

    def cols(self, b):
        if b == "eci":
            return list(range(self.c_eci, self.c_eci + 12))
        sl = self.rel_slice(b) if b in self.rel_names else self.est_slice(b)
        return list(range(sl.start, sl.stop))

    @staticmethod
    def exits(b):
        """Branches checked against the altitude floor."""
        return b in ("nerm", "eci") or b.startswith("hom:")

    def radius(self, Y, b):
        if b == "eci":
            c = self.c_eci + 6
            return np.sqrt(np.einsum("...i,...i->...", Y[..., c:c + 3], Y[..., c:c + 3]))
        Sb = Y[..., self.rel_slice(b)]
        return np.sqrt((self.R + Sb[..., 0]) ** 2 + Sb[..., 1] ** 2 + Sb[..., 2] ** 2)

    def _locate_exit(self, Y, Yn, Urel, Ueci, Mest, i, b, j):
        t0, t1 = self.t[j], self.t[j + 1]
        sl = slice(i, i + 1)
        F0 = self.rhs(Y[sl], Urel[sl], Ueci[sl], Mest[sl], self.K[sl], self.amax[sl])[0]
        F1 = self.rhs(Yn[sl], Urel[sl], Ueci[sl], Mest[sl], self.K[sl], self.amax[sl])[0]

        def g(t):
            y = hermite(t0, t1, Y[i], Yn[i], F0, F1, t)
            return float(self.radius(y[None, :], b)[0] - self.r_floor)
        try:
            return float(brentq(g, t0, t1, xtol=1e-9)), False
        except ValueError:
            return float(t1), True

    # -- dense output for one case --------------------------------------------------------
    def case_derivatives(self, i):
        """Right derivatives at steps 0..S-1 and left derivatives at steps 1..S for case i
        (held command of each step), vectorised over steps."""
        Yi = self.Y[:, i, :]
        Ki = np.broadcast_to(self.K[i], (self.S, 3, 6))
        ai = np.full(self.S, self.amax[i])
        Fr = self.rhs(Yi[:-1], self.Urel[:, i], self.Ueci[:, i], self.Mest[:, i], Ki, ai)
        Fl = self.rhs(Yi[1:], self.Urel[:, i], self.Ueci[:, i], self.Mest[:, i], Ki, ai)
        return Fr, Fl

    def dense(self, i, tq, Fr=None, Fl=None, mask=True):
        """Combined state of case i at times tq (vectorised cubic Hermite between steps). With mask,
        the columns of a branch that ended before the horizon are NaN for tq > t_end + END_TOL_S."""
        if Fr is None:
            Fr, Fl = self.case_derivatives(i)
        tq = np.atleast_1d(np.asarray(tq, float))
        j = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, self.S - 1)
        t0, t1 = self.t[j], self.t[j + 1]
        out = hermite(t0[:, None], t1[:, None], self.Y[j, i], self.Y[j + 1, i], Fr[j], Fl[j], tq[:, None])
        if mask:
            for b in self.branch_names():
                if self.status[b][i] != "completed":
                    after = tq > self.t_end[b][i] + END_TOL_S
                    if np.any(after):
                        out[np.ix_(after, self.cols(b))] = np.nan
        return out


def hermite(t0, t1, y0, y1, f0, f1, t):
    h = t1 - t0
    s = (t - t0) / h
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * f0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * f1


__all__ = ["HeldBatch", "hermite", "cw_matrix", "resid_exact", "resid_quadratic", "command", "jac_u", "pd_matrix",
           "lvlh_frame", "eci_to_rel", "rel_to_eci", "ESTIMATORS", "REL_MODELS"]
