import math
import time
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from cwvalid import dynamics as dyn
from cwvalid.constants import ALTFLOOR, MUEARTH, REARTH
from cwvalid.control import CmdSpec, PDLaw
from cwvalid.dynamics import acccw, accnerm
from cwvalid.estimate import valcwcov, valestseg
from cwvalid.simulate import EciCfg, Segment, SimRes, SolverCfg, ecirhs, relrhs, rk4integ

RTOL = 1e-9
ATOL = 1e-9
DTOUT = 2.0
MAXSTEP = 20.0


# Same as cwvalid.simulate.simulate, but the saturation-switch event is built from the law itself,
# so it works for any gain matrix and not only the scalar PD law.
def simgen(model, rel0, cmd, t_end, *, n, R, solver=None, eci=None,
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
        raw, am = law.raw, law.a_max
        if model == "eci":
            ft = eci.force_target
            exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)

            def evsat(t, s):
                tg = (s[0], s[1], s[2], s[3], s[4], s[5])
                ap = ft.pert2(*tg) if exact else None
                r, _ = dyn.ecilvlhscalar(tg, (s[6], s[7], s[8], s[9], s[10], s[11]), ap)
                ax, ay, az = raw(r[0], r[1], r[2], r[3], r[4], r[5])
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        else:
            def evsat(t, s):
                ax, ay, az = raw(s[0], s[1], s[2], s[3], s[4], s[5])
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


def runbranchgen(scn, cfg, law):
    if cfg.cmd_mode == "sampled":
        cmd = CmdSpec("sampled", law=law, sample_period=cfg.sample_period)
    elif cfg.cmd_mode == "continuous":
        cmd = CmdSpec("continuous", law=law)
    else:
        raise ValueError(cfg.cmd_mode)
    sims = {}
    for m in cfg.models:
        sims[m] = simgen(m, scn.rel0, cmd, scn.horizon, n=scn.n, R=scn.R, solver=cfg.solver,
                         eci=cfg.eci if m == "eci" else None)
    return sims, cmd


def evalgen(scn, cfg, sims, law, keep_histories=False, hist_stride=1):
    from cwvalid.experiment import evaluate
    row, hist = evaluate(scn, cfg, sims, keep_histories, hist_stride)
    if isinstance(law, PDLaw):
        kp, kd = law.kp, law.kd
    elif getattr(law, "ispd", None) and law.ispd():
        kp, kd = float(law.K[0, 0]), float(law.K[0, 3])
    else:
        kp = kd = float("nan")
    row["kp"], row["kd"] = kp, kd
    row["sat_distance_m"] = (law.a_max / kp) if (law.a_max and np.isfinite(kp)) else float("nan")
    row["controller"] = law.asdict2() if hasattr(law, "K") else {"rule": "scalar_pd", **law.asdict2()}
    return row, hist


def segeval(cwsim):
    segs = cwsim.segments
    t0s = [s.t0 for s in segs]

    def xat(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)
    return xat


def integr(rhs, cwsim, t_end, dt_out=DTOUT, rtol=RTOL, atol=ATOL):
    valcwcov(cwsim, t_end)
    breaks = [0.0] + sorted(set(t for t in cwsim.satswitchtimes if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    ts, es = [], []
    nfev = 0

    def failed(reason):
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "final_m": float("nan"),
                "peak_vel_mps": float("nan"), "nfev": nfev, "status": "failed", "failure_reason": reason}
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=rtol, atol=atol, max_step=MAXSTEP, dense_output=True)
        nfev += out.nfev
        try:
            valestseg(out, a, b)
        except ValueError as exc:
            return failed(f"segment [{a}, {b}]: {exc}")
        k = max(2, int(math.ceil((b - a) / dt_out)) + 1)
        tt = np.linspace(a, b, k)
        ts.append(tt)
        es.append(out.sol(tt).T)
        e = out.y[:, -1]
    if not ts:
        return failed("no completed estimator segments")
    T = np.concatenate(ts)
    E = np.vstack(es)
    if not np.all(np.isfinite(E)):
        return failed("non-finite estimator dense-output samples")
    dp = np.linalg.norm(E[:, :3], axis=1)
    i = int(np.argmax(dp))
    return {"peak_m": float(dp[i]), "t_peak_s": float(T[i]), "final_m": float(dp[-1]),
            "peak_vel_mps": float(np.linalg.norm(E[:, 3:], axis=1).max()), "nfev": nfev, "status": "ok",
            "t": T, "E": E}


def makerhs(cwsim, law, n, R, kind):
    xat = segeval(cwsim)
    n2 = n * n
    am = law.a_max
    sqrt = math.sqrt
    (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5), (c0, c1, c2, c3, c4, c5) = law._k

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = xat(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = accnerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acccw(n, R, x0, x1, x2, x3, x4, x5)
        if kind == "C":
            return [e3, e4, e5, an[0] - ac[0], an[1] - ac[1], an[2] - ac[2]]
        w0 = -(a0 * e0 + a1 * e1 + a2 * e2 + a3 * e3 + a4 * e4 + a5 * e5)
        w1 = -(b0 * e0 + b1 * e1 + b2 * e2 + b3 * e3 + b4 * e4 + b5 * e5)
        w2 = -(c0 * e0 + c1 * e1 + c2 * e2 + c3 * e3 + c4 * e4 + c5 * e5)
        if kind == "A" and am is not None:
            rx, ry, rz = law.raw(x0, x1, x2, x3, x4, x5)
            m = sqrt(rx * rx + ry * ry + rz * rz)
            if m > am:
                ax, ay, az = rx / m, ry / m, rz / m
                dot = ax * w0 + ay * w1 + az * w2
                s = am / m
                w0, w1, w2 = s * (w0 - ax * dot), s * (w1 - ay * dot), s * (w2 - az * dot)
        return [e3, e4, e5,
                3.0 * n2 * e0 + 2.0 * n * e4 + w0 + (an[0] - ac[0]),
                -2.0 * n * e3 + w1 + (an[1] - ac[1]),
                -n2 * e2 + w2 + (an[2] - ac[2])]
    return rhs


# A uses the saturated Jacobian, B uses -K, C keeps only the residual term
def estmat(cwsim, law, n, R, t_end, kind, dt_out=DTOUT, rtol=RTOL, atol=ATOL, keep=False):
    out = integr(makerhs(cwsim, law, n, R, kind), cwsim, t_end, dt_out, rtol, atol)
    if not keep:
        out.pop("t", None)
        out.pop("E", None)
    return out


def estvargen(cwsim, law, n, R, t_end):
    if isinstance(law, PDLaw):
        from cwvalid.estimate2 import estvar
        return estvar(cwsim, law, n, R, t_end)
    out = {}
    for key, kind in (("d1", "A"), ("d1u", "B"), ("cres", "C")):
        try:
            r = estmat(cwsim, law, n, R, t_end, kind)
            out.update({f"{key}_peak_m": r["peak_m"], f"{key}_t_peak_s": r["t_peak_s"], f"{key}_final_m": r["final_m"],
                        f"{key}_status": r["status"], f"{key}_nfev": r["nfev"],
                        f"{key}_failure_reason": r.get("failure_reason", "")})
        except Exception as exc:
            out.update({f"{key}_peak_m": float("nan"), f"{key}_t_peak_s": float("nan"), f"{key}_final_m": float("nan"),
                        f"{key}_status": f"failed: {exc!r}", f"{key}_nfev": 0, f"{key}_failure_reason": repr(exc)})
    return out
