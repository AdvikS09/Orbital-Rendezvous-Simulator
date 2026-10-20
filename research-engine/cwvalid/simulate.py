import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from . import dynamics as dyn
from .constants import ALTFLOOR, DEFINC, MUEARTH, REARTH


@dataclass
class SolverCfg:
    method: str = "DOP853"
    rtol: float = 1e-13
    atol_rel_pos: float = 1e-10
    atol_rel_vel: float = 1e-13
    atol_eci_pos: float = 1e-7
    atol_eci_vel: float = 1e-10
    atol_dv: float = 1e-13
    max_step: float = 20.0
    rk4_dt: float | None = None
    sat_events: bool = True
    max_sat_restarts: int = 2000

    def asdict2(self):
        d = asdict(self)
        if not math.isfinite(d["max_step"]):
            d["max_step"] = "inf"
        return d


@dataclass
class EciCfg:
    inclination: float = DEFINC
    force_target: dyn.InertForce = field(default_factory=dyn.InertForce)
    force_chaser: dyn.InertForce = field(default_factory=dyn.InertForce)
    exact_frame_rate: bool = True

    def asdict2(self):
        return {"inclination_rad": self.inclination,
                "force_target": self.force_target.asdict2(),
                "force_chaser": self.force_chaser.asdict2(),
                "exact_frame_rate": self.exact_frame_rate}


class Rk4Sol:

    def __init__(self, t, y, f):
        self.t = t
        self.y = y
        self.f = f

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
        return y.T


@dataclass
class Segment:
    t0: float
    t1: float
    sol: object
    u_hold: tuple | None = None


class SimRes:

    def __init__(self, model, cmd, n, R, t_end, solver, eci, rel0):
        self.model = model
        self.cmd = cmd
        self.n = n
        self.R = R
        self.tendreq = float(t_end)
        self.solver = solver
        self.eci = eci
        self.rel0 = np.asarray(rel0, dtype=float)
        self.y0 = np.concatenate((self.rel0, [0.0]))
        self.segments = []
        self.tfinal = 0.0
        self.status = "ok"
        self.termination = "horizon"
        self.message = ""
        self.nfev = 0
        self.nsteps = 0
        self.walltime = 0.0
        self.dvfinal = float("nan")
        self.satswitchtimes = []
        self.sateventsoff = None

    def raweval(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.segments:
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

    def ecistates(self, t):
        if self.model != "eci":
            raise ValueError("eci_states only defined for the eci model")
        Y = self.raweval(t)
        return Y[0:6].T, Y[6:12].T

    def tgtpertacc(self, target):
        ft = self.eci.force_target
        if not (ft.use_j2 or ft.use_drag) or not self.eci.exact_frame_rate:
            return None
        return np.array([ft.pert2(*row) for row in target])

    def states(self, t):
        if self.model == "eci":
            tgt, chs = self.ecistates(t)
            rel, _ = dyn.ecilvlhvec(tgt, chs, self.tgtpertacc(tgt))
            return rel
        return self.raweval(t)[0:6].T

    def dvcum(self, t):
        Y = self.raweval(t)
        return Y[-1]

    def heldcmds(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        breaks = np.array([s.t0 for s in self.segments])
        idx = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(self.segments) - 1)
        U = np.array([s.u_hold for s in self.segments], dtype=float)
        return U[idx]

    def commands(self, t, states=None):
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
            return law.rawvec(S), law.applvec(S)
        if mode == "sampled":
            tupd = np.array([s.t0 for s in self.segments])
            SUpd = self.states(tupd)
            rawu = law.rawvec(SUpd)
            appu = law.applvec(SUpd)
            idx = np.clip(np.searchsorted(tupd, t, side="right") - 1, 0, tupd.size - 1)
            return rawu[idx], appu[idx]
        raise ValueError(mode)

    def summary(self):
        return {
            "model": self.model, "cmd_mode": self.cmd.mode, "status": self.status,
            "termination": self.termination, "message": self.message,
            "t_final_s": self.tfinal, "t_end_requested_s": self.tendreq,
            "nfev": self.nfev, "nsteps": self.nsteps, "wall_time_s": self.walltime,
            "dv_aug_mps": self.dvfinal, "n_segments": len(self.segments),
            "n_sat_switch_events": len(self.satswitchtimes),
            "sat_events_disabled_at_s": self.sateventsoff,
        }


def relrhs(model, n, R, cmd, hold):
    acc = dyn.RELMODELS[model]
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


def ecirhs(eci, cmd, hold):
    ft, fc = eci.force_target, eci.force_chaser
    exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)
    mode = cmd.mode
    command = cmd.law.command if cmd.law is not None else None
    pres = cmd.prescribed
    sqrt = math.sqrt
    tolvlh = dyn.ecilvlhscalar

    def rhs(t, s):
        tx, ty, tz, tvx, tvy, tvz, cx, cy, cz, cvx, cvy, cvz, _ = s.tolist()
        atx, aty, atz = ft.acc3(tx, ty, tz, tvx, tvy, tvz)
        acx, acy, acz = fc.acc3(cx, cy, cz, cvx, cvy, cvz)
        ap = ft.pert2(tx, ty, tz, tvx, tvy, tvz) if exact else None
        rel, (xh, yh, zh) = tolvlh((tx, ty, tz, tvx, tvy, tvz), (cx, cy, cz, cvx, cvy, cvz), ap)
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


def rk4integ(rhs, t0, t1, y0, dt, floor=None):
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
    return Rk4Sol(t, y, f), 4 * nstep + 1, nstep, hit


def simulate(model, rel0, cmd, t_end, *, n, R, solver=None, eci=None,
             altitude_floor=ALTFLOOR):
    solver = solver or SolverCfg()
    rel0 = np.asarray(rel0, dtype=float)
    res = SimRes(model, cmd, n, R, t_end, solver, eci, rel0)
    hold = [(0.0, 0.0, 0.0)]

    if model == "eci":
        eci = eci or EciCfg()
        res.eci = eci
        from src.orbit import circorbitstate
        target0 = circorbitstate(R, MUEARTH, eci.inclination)
        ap0 = None
        if eci.exact_frame_rate and (eci.force_target.use_j2 or eci.force_target.use_drag):
            ap0 = np.array(eci.force_target.pert2(*target0))
        chaser0 = dyn.lvlhtoeci(target0, rel0, ap0)
        y0 = np.concatenate((target0, chaser0, [0.0]))
        rhs = ecirhs(eci, cmd, hold)
        atol = np.array([solver.atol_eci_pos] * 3 + [solver.atol_eci_vel] * 3
                        + [solver.atol_eci_pos] * 3 + [solver.atol_eci_vel] * 3 + [solver.atol_dv])
        rfloor = REARTH + altitude_floor

        def evfloor(t, s):
            return math.sqrt(s[6] * s[6] + s[7] * s[7] + s[8] * s[8]) - rfloor
    else:
        if model not in dyn.RELMODELS:
            raise ValueError(f"unknown model {model!r}")
        y0 = np.concatenate((rel0, [0.0]))
        rhs = relrhs(model, n, R, cmd, hold)
        atol = np.array([solver.atol_rel_pos] * 3 + [solver.atol_rel_vel] * 3 + [solver.atol_dv])
        rfloor = REARTH + altitude_floor

        def evfloor(t, s):
            return math.sqrt((R + s[0]) ** 2 + s[1] ** 2 + s[2] ** 2) - rfloor
    evfloor.terminal = True
    evfloor.direction = -1
    events = [evfloor] if model in ("nerm", "eci") else None

    evsat = None
    satdir = [1]
    nrestarts = 0
    law = cmd.law
    if (solver.sat_events and cmd.mode == "continuous" and law is not None and law.a_max is not None
            and solver.method != "RK4"):
        kp, kd, am = law.kp, law.kd, law.a_max
        if model == "eci":
            ft = eci.force_target
            exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)

            def evsat(t, s):
                tg = (s[0], s[1], s[2], s[3], s[4], s[5])
                ap = ft.pert2(*tg) if exact else None
                r, _ = dyn.ecilvlhscalar(tg, (s[6], s[7], s[8], s[9], s[10], s[11]), ap)
                ax, ay, az = -kp * r[0] - kd * r[3], -kp * r[1] - kd * r[4], -kp * r[2] - kd * r[5]
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        else:
            def evsat(t, s):
                ax, ay, az = -kp * s[0] - kd * s[3], -kp * s[1] - kd * s[4], -kp * s[2] - kd * s[5]
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        evsat.terminal = True
        satdir[0] = -1 if evsat(0.0, y0) > 0 else 1

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
                        ap = ft.pert2(*tgt)
                    rel, _ = dyn.ecilvlhscalar(tuple(tgt), tuple(chs), ap)
                else:
                    rel = tuple(y[0:6])
                hold[0] = cmd.law.command(*rel)
            useg = hold[0] if cmd.mode in ("sampled", "none") else None
            if solver.method == "RK4":
                if solver.rk4_dt is None:
                    raise ValueError("RK4 requires rk4_dt")
                sol, nfev, nst, hit = rk4integ(rhs, a, b, y, solver.rk4_dt,
                                               evfloor if model in ("nerm", "eci") else None)
                res.segments.append(Segment(a, float(sol.t[-1]), sol, useg))
                res.nfev += nfev
                res.nsteps += nst
                y = sol.y[-1]
                res.tfinal = float(sol.t[-1])
                if hit:
                    res.status, res.termination = "terminated", "altitude_floor"
                    res.message = "RK4: first step ending below the altitude floor (resolution = step)"
                    break
                continue
            tcur, stop = a, False
            while tcur < b - 1e-12:
                evs = list(events) if events else []
                if evsat is not None and nrestarts < solver.max_sat_restarts:
                    evsat.direction = satdir[0]
                    evs.append(evsat)
                out = solve_ivp(rhs, (tcur, b), y, method=solver.method, rtol=solver.rtol, atol=atol,
                                max_step=solver.max_step, dense_output=True, events=evs or None)
                res.nfev += out.nfev
                res.nsteps += max(0, out.t.size - 1)
                if out.status == -1:
                    res.status, res.termination, res.message = "failed", "solver_failure", out.message
                    if out.sol is not None:
                        res.segments.append(Segment(tcur, float(out.t[-1]), out.sol, useg))
                    res.tfinal = float(out.t[-1])
                    stop = True
                    break
                tnew = float(out.t[-1])
                if tnew > tcur:
                    res.segments.append(Segment(tcur, tnew, out.sol, useg))
                res.tfinal = tnew
                y = out.y[:, -1]
                if out.status == 1:
                    floorhit = events and out.t_events[0].size > 0
                    if floorhit:
                        res.status, res.termination = "terminated", "altitude_floor"
                        res.message = out.message
                        stop = True
                        break
                    satdir[0] = -satdir[0]
                    nrestarts += 1
                    res.satswitchtimes.append(tnew)
                    if nrestarts >= solver.max_sat_restarts and res.sateventsoff is None:
                        res.sateventsoff = tnew
                    if tnew <= tcur + 1e-12:
                        nrestarts = solver.max_sat_restarts
                        res.sateventsoff = tnew
                    tcur = tnew
                    continue
                tcur = tnew
            if stop:
                break
    except Exception as exc:
        res.status, res.termination, res.message = "failed", "solver_failure", repr(exc)
    res.walltime = time.perf_counter() - tic
    if res.segments:
        res.dvfinal = float(res.raweval([res.tfinal])[-1, 0])
    return res
