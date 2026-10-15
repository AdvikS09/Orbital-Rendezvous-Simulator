"""Closed-loop / open-loop propagation of one branch (one model, one command source).

Integration
-----------
* ``DOP853`` (default): scipy's adaptive explicit Runge-Kutta 8(5,3) with dense
  output (Hairer, Norsett & Wanner). Tolerances are per-component (scaled
  absolute + relative) and are set by the convergence study.
* ``RK4``: fixed-step classical Runge-Kutta, the integrator of the original
  engine, with cubic-Hermite interpolation between nodes for metric extraction.

Effort accounting
-----------------
The applied-acceleration magnitude is integrated as an extra state component
(d(dv)/dt = |a_applied|), so Delta-v is computed by the solver with the same
error control as the trajectory; it is never formed by summing right-hand-side
evaluations (which include rejected and repeated stages).

Sampled control
---------------
For ``sampled`` commands the integration is restarted at every update instant
t_k = k*Ts with the command held (zero-order hold, in LVLH axes) on
[t_k, t_{k+1}); update instants are independent of solver steps.
"""

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from . import dynamics as dyn
from .constants import ALTITUDE_FLOOR, DEFAULT_INCLINATION, MU_EARTH, R_EARTH
from .control import CommandSpec


@dataclass
class SolverConfig:
    method: str = "DOP853"          # 'DOP853' | 'RK45' | 'RK4'
    rtol: float = 1e-13             # production setting chosen by the convergence study
    atol_rel_pos: float = 1e-10     # [m]   relative models
    atol_rel_vel: float = 1e-13     # [m/s] relative models
    atol_eci_pos: float = 1e-7      # [m]   inertial states
    atol_eci_vel: float = 1e-10     # [m/s] inertial states
    atol_dv: float = 1e-13          # [m/s] Delta-v accumulator
    max_step: float = 20.0          # [s] DOP853 step cap (np.inf to disable)
    rk4_dt: float | None = None     # [s] fixed step for RK4
    sat_events: bool = True         # locate saturation switches and restart there
    max_sat_restarts: int = 2000    # safety cap on restarts per branch

    def as_dict(self):
        d = asdict(self)
        if not math.isfinite(d["max_step"]):
            d["max_step"] = "inf"
        return d


@dataclass
class EciConfig:
    """Inertial-branch settings (target on a circular orbit at the ascending node)."""
    inclination: float = DEFAULT_INCLINATION
    force_target: dyn.InertialForce = field(default_factory=dyn.InertialForce)
    force_chaser: dyn.InertialForce = field(default_factory=dyn.InertialForce)
    exact_frame_rate: bool = True   # include wx = r a_h / h (matters only if perturbed)

    def as_dict(self):
        return {"inclination_rad": self.inclination,
                "force_target": self.force_target.as_dict(),
                "force_chaser": self.force_chaser.as_dict(),
                "exact_frame_rate": self.exact_frame_rate}


class _RK4Solution:
    """Cubic-Hermite dense output over fixed RK4 nodes (callable like OdeSolution)."""

    def __init__(self, t, y, f):
        self.t = t
        self.y = y  # (N, m)
        self.f = f  # (N, m)

    def __call__(self, tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        t = self.t
        i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, t.size - 2)
        h = t[i + 1] - t[i]
        s = (tq - t[i]) / h
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        y = (h00[:, None] * self.y[i] + (h10 * h)[:, None] * self.f[i]
             + h01[:, None] * self.y[i + 1] + (h11 * h)[:, None] * self.f[i + 1])
        return y.T  # (m, Nq) like OdeSolution


@dataclass
class Segment:
    t0: float
    t1: float
    sol: object                 # OdeSolution or _RK4Solution
    u_hold: tuple | None = None  # held LVLH command for sampled mode


class SimResult:
    """Result of one branch propagation; evaluates states/commands on any time grid."""

    def __init__(self, model, cmd, n, R, t_end, solver, eci, rel0):
        self.model = model
        self.cmd = cmd
        self.n = n
        self.R = R
        self.t_end_requested = float(t_end)
        self.solver = solver
        self.eci = eci
        self.rel0 = np.asarray(rel0, dtype=float)
        self.y0 = np.concatenate((self.rel0, [0.0]))   # replaced by the full initial state in simulate()
        self.segments: list[Segment] = []
        self.t_final = 0.0
        self.status = "ok"           # 'ok' | 'terminated' | 'failed'
        self.termination = "horizon"  # 'horizon' | 'altitude_floor' | 'solver_failure'
        self.message = ""
        self.nfev = 0
        self.nsteps = 0
        self.wall_time = 0.0
        self.dv_final = float("nan")
        self.sat_switch_times = []
        self.sat_events_disabled_at = None

    # ------------------------------------------------------------------
    def _raw_eval(self, t):
        """Evaluate the augmented solution vector at times t -> (m, N)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.segments:   # failed before any accepted step: hold the initial state
            return np.repeat(np.asarray(self.y0, dtype=float)[:, None], t.size, axis=1)
        if len(self.segments) == 1:
            return self.segments[0].sol(t)
        breaks = np.array([s.t0 for s in self.segments])
        idx = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(self.segments) - 1)
        out = None
        for k in np.unique(idx):
            sel = idx == k
            v = self.segments[k].sol(t[sel])
            if out is None:
                out = np.empty((v.shape[0], t.size))
            out[:, sel] = v
        return out

    def eci_states(self, t):
        """(target (N,6), chaser (N,6)) for the inertial branch."""
        if self.model != "eci":
            raise ValueError("eci_states only defined for the eci model")
        Y = self._raw_eval(t)
        return Y[0:6].T, Y[6:12].T

    def target_pert_acc(self, target):
        ft = self.eci.force_target
        if not (ft.use_j2 or ft.use_drag) or not self.eci.exact_frame_rate:
            return None
        return np.array([ft.perturbation(*row) for row in target])

    def states(self, t):
        """LVLH relative states (N, 6) at times t."""
        if self.model == "eci":
            tgt, chs = self.eci_states(t)
            rel, _ = dyn.eci_to_lvlh_vectorized(tgt, chs, self.target_pert_acc(tgt))
            return rel
        return self._raw_eval(t)[0:6].T

    def dv_cumulative(self, t):
        Y = self._raw_eval(t)
        return Y[-1]

    def held_commands(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        breaks = np.array([s.t0 for s in self.segments])
        idx = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(self.segments) - 1)
        U = np.array([s.u_hold for s in self.segments], dtype=float)
        return U[idx]

    def commands(self, t, states=None):
        """(raw (N,3), applied (N,3)) LVLH commands acting at times t.

        continuous: law of current state; sampled: law of state at the last
        update instant (held); prescribed: the replayed history (raw = applied);
        none: zeros."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        mode = self.cmd.mode
        if mode == "none":
            z = np.zeros((t.size, 3))
            return z, z
        if mode == "prescribed":
            u = np.asarray(self.cmd.prescribed_vec(t), dtype=float)
            return u, u
        law = self.cmd.law
        if mode == "continuous":
            S = self.states(t) if states is None else states
            return law.raw_vec(S), law.applied_vec(S)
        if mode == "sampled":
            t_upd = np.array([s.t0 for s in self.segments])
            S_upd = self.states(t_upd)
            raw_u = law.raw_vec(S_upd)
            app_u = law.applied_vec(S_upd)
            idx = np.clip(np.searchsorted(t_upd, t, side="right") - 1, 0, t_upd.size - 1)
            return raw_u[idx], app_u[idx]
        raise ValueError(mode)

    def summary(self):
        return {
            "model": self.model, "cmd_mode": self.cmd.mode, "status": self.status,
            "termination": self.termination, "message": self.message,
            "t_final_s": self.t_final, "t_end_requested_s": self.t_end_requested,
            "nfev": self.nfev, "nsteps": self.nsteps, "wall_time_s": self.wall_time,
            "dv_aug_mps": self.dv_final, "n_segments": len(self.segments),
            "n_sat_switch_events": len(self.sat_switch_times),
            "sat_events_disabled_at_s": self.sat_events_disabled_at,
        }


# ---------------------------------------------------------------------------
# Right-hand sides
# ---------------------------------------------------------------------------


def _rel_rhs(model, n, R, cmd, hold):
    acc = dyn.RELATIVE_MODELS[model]
    mode = cmd.mode
    sqrt = math.sqrt
    if mode == "continuous":
        command = cmd.law.command

        def rhs(t, s):
            x, y, z, vx, vy, vz, _ = s.tolist()
            ax, ay, az = acc(n, R, x, y, z, vx, vy, vz)
            ux, uy, uz = command(x, y, z, vx, vy, vz)
            return [vx, vy, vz, ax + ux, ay + uy, az + uz, sqrt(ux * ux + uy * uy + uz * uz)]
        return rhs
    if mode in ("sampled", "none"):
        def rhs(t, s):
            x, y, z, vx, vy, vz, _ = s.tolist()
            ax, ay, az = acc(n, R, x, y, z, vx, vy, vz)
            ux, uy, uz = hold[0]
            return [vx, vy, vz, ax + ux, ay + uy, az + uz, sqrt(ux * ux + uy * uy + uz * uz)]
        return rhs
    if mode == "prescribed":
        pres = cmd.prescribed

        def rhs(t, s):
            x, y, z, vx, vy, vz, _ = s.tolist()
            ax, ay, az = acc(n, R, x, y, z, vx, vy, vz)
            ux, uy, uz = pres(t)
            return [vx, vy, vz, ax + ux, ay + uy, az + uz, sqrt(ux * ux + uy * uy + uz * uz)]
        return rhs
    raise ValueError(mode)


def _eci_rhs(eci, cmd, hold):
    ft, fc = eci.force_target, eci.force_chaser
    exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)
    mode = cmd.mode
    command = cmd.law.command if cmd.law is not None else None
    pres = cmd.prescribed
    sqrt = math.sqrt
    to_lvlh = dyn.eci_to_lvlh_scalar

    def rhs(t, s):
        tx, ty, tz, tvx, tvy, tvz, cx, cy, cz, cvx, cvy, cvz, _ = s.tolist()
        atx, aty, atz = ft.acceleration(tx, ty, tz, tvx, tvy, tvz)
        acx, acy, acz = fc.acceleration(cx, cy, cz, cvx, cvy, cvz)
        ap = ft.perturbation(tx, ty, tz, tvx, tvy, tvz) if exact else None
        rel, (xh, yh, zh) = to_lvlh((tx, ty, tz, tvx, tvy, tvz), (cx, cy, cz, cvx, cvy, cvz), ap)
        if mode == "continuous":
            ux, uy, uz = command(*rel)
        elif mode == "prescribed":
            ux, uy, uz = pres(t)
        else:
            ux, uy, uz = hold[0]
        uix = ux * xh[0] + uy * yh[0] + uz * zh[0]
        uiy = ux * xh[1] + uy * yh[1] + uz * zh[1]
        uiz = ux * xh[2] + uy * yh[2] + uz * zh[2]
        return [tvx, tvy, tvz, atx, aty, atz, cvx, cvy, cvz,
                acx + uix, acy + uiy, acz + uiz, sqrt(ux * ux + uy * uy + uz * uz)]
    return rhs


def _rk4_integrate(rhs, t0, t1, y0, dt, floor=None):
    """Fixed-step RK4; stops after the first step that ends below the altitude
    floor (``floor(t, y) < 0``). Returns (solution, nfev, nsteps, hit_floor)."""
    nstep = max(1, int(round((t1 - t0) / dt)))
    t = np.linspace(t0, t1, nstep + 1)
    y = np.empty((t.size, y0.size))
    f = np.empty_like(y)
    y[0] = y0
    hit = False
    for i in range(nstep):
        h = t[i + 1] - t[i]
        yi = y[i]
        k1 = np.asarray(rhs(t[i], yi))
        k2 = np.asarray(rhs(t[i] + 0.5 * h, yi + 0.5 * h * k1))
        k3 = np.asarray(rhs(t[i] + 0.5 * h, yi + 0.5 * h * k2))
        k4 = np.asarray(rhs(t[i] + h, yi + h * k3))
        f[i] = k1
        y[i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if floor is not None and floor(t[i + 1], y[i + 1]) < 0:
            t, y, f = t[:i + 2], y[:i + 2], f[:i + 2]
            nstep, hit = i + 1, True
            break
    f[-1] = np.asarray(rhs(t[-1], y[-1]))
    return _RK4Solution(t, y, f), 4 * nstep + 1, nstep, hit


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def simulate(model, rel0, cmd: CommandSpec, t_end, *, n, R, solver=None, eci=None,
             altitude_floor=ALTITUDE_FLOOR):
    """Propagate one branch from LVLH relative state ``rel0`` over [0, t_end].

    model: 'cw' | 'so' | 'nerm' | 'eci'.
    Returns :class:`SimResult` (never raises on solver failure; status set).
    """
    solver = solver or SolverConfig()
    rel0 = np.asarray(rel0, dtype=float)
    res = SimResult(model, cmd, n, R, t_end, solver, eci, rel0)
    hold = [(0.0, 0.0, 0.0)]

    if model == "eci":
        eci = eci or EciConfig()
        res.eci = eci
        from src.orbit import circular_orbit_state
        target0 = circular_orbit_state(R, MU_EARTH, eci.inclination)
        ap0 = None
        if eci.exact_frame_rate and (eci.force_target.use_j2 or eci.force_target.use_drag):
            ap0 = np.array(eci.force_target.perturbation(*target0))
        chaser0 = dyn.lvlh_to_eci(target0, rel0, ap0)
        y0 = np.concatenate((target0, chaser0, [0.0]))
        rhs = _eci_rhs(eci, cmd, hold)
        atol = np.array([solver.atol_eci_pos] * 3 + [solver.atol_eci_vel] * 3
                        + [solver.atol_eci_pos] * 3 + [solver.atol_eci_vel] * 3 + [solver.atol_dv])
        r_floor = R_EARTH + altitude_floor

        def ev_floor(t, s):
            return math.sqrt(s[6] * s[6] + s[7] * s[7] + s[8] * s[8]) - r_floor
    else:
        if model not in dyn.RELATIVE_MODELS:
            raise ValueError(f"unknown model {model!r}")
        y0 = np.concatenate((rel0, [0.0]))
        rhs = _rel_rhs(model, n, R, cmd, hold)
        atol = np.array([solver.atol_rel_pos] * 3 + [solver.atol_rel_vel] * 3 + [solver.atol_dv])
        r_floor = R_EARTH + altitude_floor

        def ev_floor(t, s):
            return math.sqrt((R + s[0]) ** 2 + s[1] ** 2 + s[2] ** 2) - r_floor
    ev_floor.terminal = True
    ev_floor.direction = -1
    events = [ev_floor] if model in ("nerm", "eci") else None

    # saturation-switch event (continuous feedback with a cap): g = |a_raw| - a_max
    ev_sat = None
    sat_dir = [1]
    n_restarts = 0
    law = cmd.law
    if (solver.sat_events and cmd.mode == "continuous" and law is not None and law.a_max is not None
            and solver.method != "RK4"):
        kp, kd, am = law.kp, law.kd, law.a_max
        if model == "eci":
            ft = eci.force_target
            exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)

            def ev_sat(t, s):
                tg = (s[0], s[1], s[2], s[3], s[4], s[5])
                ap = ft.perturbation(*tg) if exact else None
                r, _ = dyn.eci_to_lvlh_scalar(tg, (s[6], s[7], s[8], s[9], s[10], s[11]), ap)
                ax, ay, az = -kp * r[0] - kd * r[3], -kp * r[1] - kd * r[4], -kp * r[2] - kd * r[5]
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        else:
            def ev_sat(t, s):
                ax, ay, az = -kp * s[0] - kd * s[3], -kp * s[1] - kd * s[4], -kp * s[2] - kd * s[5]
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        ev_sat.terminal = True
        sat_dir[0] = -1 if ev_sat(0.0, y0) > 0 else 1

    # update instants
    if cmd.mode == "sampled":
        Ts = float(cmd.sample_period)
        k = int(math.floor(t_end / Ts + 1e-9))
        breaks = [i * Ts for i in range(k + 1)]
        if breaks[-1] < t_end - 1e-9:
            breaks.append(t_end)
        else:
            breaks[-1] = t_end
    else:
        breaks = [0.0, float(t_end)]
    if getattr(cmd, "breakpoints", None) and cmd.mode != "sampled":
        extra = [float(x) for x in cmd.breakpoints if 0.0 < float(x) < t_end]
        breaks = sorted(set(breaks) | set(extra))

    res.y0 = y0
    tic = time.perf_counter()
    y = y0
    try:
        for a, b in zip(breaks[:-1], breaks[1:]):
            if cmd.mode == "sampled":
                if model == "eci":
                    tgt, chs = y[0:6], y[6:12]
                    ap = None
                    ft = eci.force_target
                    if eci.exact_frame_rate and (ft.use_j2 or ft.use_drag):
                        ap = ft.perturbation(*tgt)
                    rel, _ = dyn.eci_to_lvlh_scalar(tuple(tgt), tuple(chs), ap)
                else:
                    rel = tuple(y[0:6])
                hold[0] = cmd.law.command(*rel)
            u_seg = hold[0] if cmd.mode in ("sampled", "none") else None
            if solver.method == "RK4":
                if solver.rk4_dt is None:
                    raise ValueError("RK4 requires rk4_dt")
                sol, nfev, nst, hit = _rk4_integrate(rhs, a, b, y, solver.rk4_dt,
                                                     ev_floor if model in ("nerm", "eci") else None)
                res.segments.append(Segment(a, float(sol.t[-1]), sol, u_seg))
                res.nfev += nfev
                res.nsteps += nst
                y = sol.y[-1]
                res.t_final = float(sol.t[-1])
                if hit:
                    res.status, res.termination = "terminated", "altitude_floor"
                    res.message = "RK4: first step ending below the altitude floor (resolution = step)"
                    break
                continue
            # Inner loop: with saturation-switch events the integration restarts at
            # every located switch, so no step straddles the kink of the saturated law.
            t_cur, stop = a, False
            while t_cur < b - 1e-12:
                evs = list(events) if events else []
                i_sat = None
                if ev_sat is not None and n_restarts < solver.max_sat_restarts:
                    ev_sat.direction = sat_dir[0]
                    i_sat = len(evs)
                    evs.append(ev_sat)
                out = solve_ivp(rhs, (t_cur, b), y, method=solver.method, rtol=solver.rtol, atol=atol,
                                max_step=solver.max_step, dense_output=True, events=evs or None)
                res.nfev += out.nfev
                res.nsteps += max(0, out.t.size - 1)
                if out.status == -1:
                    res.status, res.termination, res.message = "failed", "solver_failure", out.message
                    if out.sol is not None:
                        res.segments.append(Segment(t_cur, float(out.t[-1]), out.sol, u_seg))
                    res.t_final = float(out.t[-1])
                    stop = True
                    break
                t_new = float(out.t[-1])
                if t_new > t_cur:
                    res.segments.append(Segment(t_cur, t_new, out.sol, u_seg))
                res.t_final = t_new
                y = out.y[:, -1]
                if out.status == 1:
                    floor_hit = events and out.t_events[0].size > 0
                    if floor_hit:
                        res.status, res.termination = "terminated", "altitude_floor"
                        res.message = out.message
                        stop = True
                        break
                    # saturation switch located: flip the sought crossing direction and restart
                    sat_dir[0] = -sat_dir[0]
                    n_restarts += 1
                    res.sat_switch_times.append(t_new)
                    if n_restarts >= solver.max_sat_restarts and res.sat_events_disabled_at is None:
                        res.sat_events_disabled_at = t_new
                    if t_new <= t_cur + 1e-12:   # no progress: disable switch events for the rest of the branch
                        n_restarts = solver.max_sat_restarts
                        res.sat_events_disabled_at = t_new
                    t_cur = t_new
                    continue
                t_cur = t_new
            if stop:
                break
    except Exception as exc:  # numerical failure is recorded, never silently dropped
        res.status, res.termination, res.message = "failed", "solver_failure", repr(exc)
    res.wall_time = time.perf_counter() - tic
    if res.segments:
        res.dv_final = float(res._raw_eval([res.t_final])[-1, 0])
    return res


__all__ = ["SolverConfig", "EciConfig", "SimResult", "Segment", "simulate"]
