"""Continuous-feedback propagation and first-order estimators for a general matrix law (P07).

Why this module exists: cwvalid.simulate builds its saturation-switch event from law.kp/law.kd,
cwvalid.estimate*/estimate_v2 hard-code the scalar PD product -kp e - kd e_dot, and
cwvalid.experiment.evaluate rebuilds a scalar PDLaw for its kp/kd/sat_distance fields. For a
general 3x6 gain matrix each of these would silently use the wrong controller. This module

* ``simulate_general``: a line-by-line copy of cwvalid.simulate.simulate whose only change is
  that the saturation-switch function is g = |law.raw(s)| - a_max (any law with ``raw``). For
  cwvalid.control.PDLaw, ``raw`` is the same arithmetic as the production inline expression,
  and for a MatrixLaw holding a PD matrix it is bit-identical as well (tests/test_v3_controllers).
* ``estimator_variants_general``: A (saturated Jacobian), B (-K) and C (residual only) with the
  production settings (DOP853, rtol = atol = 1e-9, max step 20 s, restarts at the CW branch's
  located saturation switches, 2 s output grid, the archive's completion/coverage guards).
* ``evaluate_general``: production cwvalid.experiment.evaluate (whose metrics read the ACTUAL law
  from each branch's CommandSpec) followed by replacement of the law-reconstructed scalar fields.
For PDLaw inputs the production functions are called directly, so D0-D2 keep the historical
feature identity exactly.
"""

import math
import time
from bisect import bisect_right

import numpy as np
from scipy.integrate import solve_ivp

from cwvalid import dynamics as dyn
from cwvalid.constants import ALTITUDE_FLOOR, MU_EARTH, R_EARTH
from cwvalid.control import CommandSpec, PDLaw
from cwvalid.dynamics import acc_cw, acc_nerm
from cwvalid.estimate import _validate_cw_coverage, _validate_estimator_segment
from cwvalid.simulate import EciConfig, Segment, SimResult, SolverConfig, _eci_rhs, _rel_rhs, _rk4_integrate

RTOL = 1e-9
ATOL = 1e-9
DT_OUT = 2.0
MAX_STEP = 20.0


def simulate_general(model, rel0, cmd: CommandSpec, t_end, *, n, R, solver=None, eci=None,
                     altitude_floor=ALTITUDE_FLOOR):
    """cwvalid.simulate.simulate with a law-agnostic saturation-switch event (see module doc)."""
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

    ev_sat = None
    sat_dir = [1]
    n_restarts = 0
    law = cmd.law
    if (solver.sat_events and cmd.mode == "continuous" and law is not None and law.a_max is not None
            and solver.method != "RK4"):
        raw, am = law.raw, law.a_max
        if model == "eci":
            ft = eci.force_target
            exact = eci.exact_frame_rate and (ft.use_j2 or ft.use_drag)

            def ev_sat(t, s):
                tg = (s[0], s[1], s[2], s[3], s[4], s[5])
                ap = ft.perturbation(*tg) if exact else None
                r, _ = dyn.eci_to_lvlh_scalar(tg, (s[6], s[7], s[8], s[9], s[10], s[11]), ap)
                ax, ay, az = raw(r[0], r[1], r[2], r[3], r[4], r[5])
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        else:
            def ev_sat(t, s):
                ax, ay, az = raw(s[0], s[1], s[2], s[3], s[4], s[5])
                return math.sqrt(ax * ax + ay * ay + az * az) - am
        ev_sat.terminal = True
        sat_dir[0] = -1 if ev_sat(0.0, y0) > 0 else 1

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
            t_cur, stop = a, False
            while t_cur < b - 1e-12:
                evs = list(events) if events else []
                if ev_sat is not None and n_restarts < solver.max_sat_restarts:
                    ev_sat.direction = sat_dir[0]
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
                    sat_dir[0] = -sat_dir[0]
                    n_restarts += 1
                    res.sat_switch_times.append(t_new)
                    if n_restarts >= solver.max_sat_restarts and res.sat_events_disabled_at is None:
                        res.sat_events_disabled_at = t_new
                    if t_new <= t_cur + 1e-12:
                        n_restarts = solver.max_sat_restarts
                        res.sat_events_disabled_at = t_new
                    t_cur = t_new
                    continue
                t_cur = t_new
            if stop:
                break
    except Exception as exc:
        res.status, res.termination, res.message = "failed", "solver_failure", repr(exc)
    res.wall_time = time.perf_counter() - tic
    if res.segments:
        res.dv_final = float(res._raw_eval([res.t_final])[-1, 0])
    return res


def run_branches_general(scn, cfg, law):
    """Propagate cfg.models with the given law (continuous or sampled per cfg.cmd_mode)."""
    if cfg.cmd_mode == "sampled":
        cmd = CommandSpec("sampled", law=law, sample_period=cfg.sample_period)
    elif cfg.cmd_mode == "continuous":
        cmd = CommandSpec("continuous", law=law)
    else:
        raise ValueError(cfg.cmd_mode)
    sims = {}
    for m in cfg.models:
        sims[m] = simulate_general(m, scn.rel0, cmd, scn.horizon, n=scn.n, R=scn.R, solver=cfg.solver,
                                   eci=cfg.eci if m == "eci" else None)
    return sims, cmd


def evaluate_general(scn, cfg, sims, law, keep_histories=False, hist_stride=1):
    """Production evaluate() with the law-reconstructed scalar fields replaced by the actual law."""
    from cwvalid.experiment import evaluate
    row, hist = evaluate(scn, cfg, sims, keep_histories, hist_stride)
    if isinstance(law, PDLaw):
        kp, kd = law.kp, law.kd
    elif getattr(law, "is_pd", None) and law.is_pd():
        kp, kd = float(law.K[0, 0]), float(law.K[0, 3])
    else:
        kp = kd = float("nan")
    row["kp"], row["kd"] = kp, kd
    row["sat_distance_m"] = (law.a_max / kp) if (law.a_max and np.isfinite(kp)) else float("nan")
    row["controller"] = law.as_dict() if hasattr(law, "K") else {"rule": "scalar_pd", **law.as_dict()}
    return row, hist


# ---------------------------------------------------------------------------------------------
# continuous first-order estimators for a general K
# ---------------------------------------------------------------------------------------------


def _segment_eval(cw_sim):
    segs = cw_sim.segments
    t0s = [s.t0 for s in segs]

    def x_at(t):
        k = bisect_right(t0s, t) - 1
        k = 0 if k < 0 else (len(segs) - 1 if k >= len(segs) else k)
        return segs[k].sol(t)
    return x_at


def _integrate(rhs, cw_sim, t_end, dt_out=DT_OUT, rtol=RTOL, atol=ATOL):
    _validate_cw_coverage(cw_sim, t_end)
    breaks = [0.0] + sorted(set(t for t in cw_sim.sat_switch_times if 0.0 < t < t_end)) + [t_end]
    e = np.zeros(6)
    ts, es = [], []
    nfev = 0

    def failed(reason):
        return {"peak_m": float("nan"), "t_peak_s": float("nan"), "final_m": float("nan"),
                "peak_vel_mps": float("nan"), "nfev": nfev, "status": "failed", "failure_reason": reason}
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        out = solve_ivp(rhs, (a, b), e, method="DOP853", rtol=rtol, atol=atol, max_step=MAX_STEP, dense_output=True)
        nfev += out.nfev
        try:
            _validate_estimator_segment(out, a, b)
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


def _make_rhs(cw_sim, law, n, R, kind):
    x_at = _segment_eval(cw_sim)
    n2 = n * n
    am = law.a_max
    sqrt = math.sqrt
    (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5), (c0, c1, c2, c3, c4, c5) = law._k

    def rhs(t, e):
        x0, x1, x2, x3, x4, x5 = x_at(t)[:6].tolist()
        e0, e1, e2, e3, e4, e5 = e.tolist()
        an = acc_nerm(n, R, x0, x1, x2, x3, x4, x5)
        ac = acc_cw(n, R, x0, x1, x2, x3, x4, x5)
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


def estimator_matrix(cw_sim, law, n, R, t_end, kind, dt_out=DT_OUT, rtol=RTOL, atol=ATOL, keep=False):
    """kind 'A' (saturated Jacobian), 'B' (-K) or 'C' (residual only) for a MatrixLaw."""
    out = _integrate(_make_rhs(cw_sim, law, n, R, kind), cw_sim, t_end, dt_out, rtol, atol)
    if not keep:
        out.pop("t", None)
        out.pop("E", None)
    return out


def estimator_variants_general(cw_sim, law, n, R, t_end):
    """Same keys as cwvalid.estimate_v2.estimator_variants; PDLaw -> the production function."""
    if isinstance(law, PDLaw):
        from cwvalid.estimate_v2 import estimator_variants
        return estimator_variants(cw_sim, law, n, R, t_end)
    out = {}
    for key, kind in (("d1", "A"), ("d1u", "B"), ("cres", "C")):
        try:
            r = estimator_matrix(cw_sim, law, n, R, t_end, kind)
            out.update({f"{key}_peak_m": r["peak_m"], f"{key}_t_peak_s": r["t_peak_s"], f"{key}_final_m": r["final_m"],
                        f"{key}_status": r["status"], f"{key}_nfev": r["nfev"],
                        f"{key}_failure_reason": r.get("failure_reason", "")})
        except Exception as exc:
            out.update({f"{key}_peak_m": float("nan"), f"{key}_t_peak_s": float("nan"), f"{key}_final_m": float("nan"),
                        f"{key}_status": f"failed: {exc!r}", f"{key}_nfev": 0, f"{key}_failure_reason": repr(exc)})
    return out


__all__ = ["simulate_general", "run_branches_general", "evaluate_general", "estimator_matrix",
           "estimator_variants_general"]
