"""Common-command propagation (diagnostic).

The CW closed loop is run first; its *applied* acceleration history u_CW(t)
(saturated PD of the CW state) is then replayed open-loop through each model.
This is distinct from independently evaluated feedback: the same command acts
on every model, so differences come from the dynamics alone. It is a
diagnostic, not an additive decomposition of the closed-loop difference.
"""

import numpy as np

from . import metrics as M
from .control import CommandSpec
from .experiment import make_law, run_branches
from .simulate import simulate


def _aligned(sim_a, tr_a, sim_b, tr_b):
    """Positions/commands of two traces on the shorter trace's times."""
    short, other, sim_o = (tr_a, tr_b, sim_b) if tr_a.t[-1] <= tr_b.t[-1] else (tr_b, tr_a, sim_a)
    t = short.t
    n = t.size
    if other.t.size >= n and np.array_equal(other.t[:n], t):
        So, Uo = other.S[:n], other.app[:n]
    else:
        So = sim_o.states(t)
        Uo = sim_o.commands(t, So)[1]
    if short is tr_a:
        return t, short.S, So, short.app, Uo
    return t, So, short.S, Uo, short.app


def common_command(scn, cfg, models=("cw", "nerm", "eci", "so")):
    if cfg.cmd_mode != "continuous":
        raise ValueError("common-command replay is defined for the continuous-feedback CW loop only")
    sims_cl, _ = run_branches(scn, cfg)
    cw = sims_cl["cw"]
    law = make_law(scn, cfg.unrestricted)

    def pres(t):
        return law.command(*cw.states([t])[0])

    def pres_vec(tt):
        return law.applied_vec(cw.states(tt))

    # restart integration at the CW loop's saturation switches (kinks of u_CW(t))
    cmd = CommandSpec("prescribed", prescribed=pres, prescribed_vec=pres_vec, label="CW applied command replay",
                      breakpoints=tuple(cw.sat_switch_times))
    sims_cc = {}
    for m in models:
        sims_cc[m] = simulate(m, scn.rel0, cmd, scn.horizon, n=scn.n, R=scn.R,
                              solver=cfg.solver, eci=cfg.eci if m == "eci" else None)
    T = scn.horizon
    grid = M.common_grid(T, cfg.dt_eval)
    tr_cl = {m: M.make_trace(s, grid) for m, s in sims_cl.items()}
    tr_cc = {m: M.make_trace(s, grid) for m, s in sims_cc.items()}
    row = {"sid": scn.sid, "sep_m": scn.separation, **{f"meta_{k}": v for k, v in scn.meta.items()},
           "wn_ratio": scn.wn_ratio, "a_max": scn.a_max}
    ref = cfg.reference
    for m in models:
        if m == "cw":
            _, Sa, Sb, _, _ = _aligned(sims_cc["cw"], tr_cc["cw"], cw, tr_cl["cw"])
            row["cc_cw_replay_vs_cw_closed_max_m"] = float(np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1).max())
            continue
        t1, Sa, Sb, _, _ = _aligned(sims_cc[m], tr_cc[m], cw, tr_cl["cw"])
        d_cc = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
        row[f"cc_{m}_vs_cw_max_m"] = float(d_cc.max())
        row[f"cc_{m}_vs_cw_t_max_s"] = float(t1[int(np.argmax(d_cc))])
        row[f"cc_{m}_final_pos_m"] = float(np.linalg.norm(tr_cc[m].S[-1, :3]))
        row[f"cc_{m}_final_speed_mps"] = float(np.linalg.norm(tr_cc[m].S[-1, 3:]))
        row[f"cc_{m}_end_s"] = float(tr_cc[m].t[-1])
        t2, Sa, Sb, Ua, Ub = _aligned(sims_cl[m], tr_cl[m], cw, tr_cl["cw"])
        d_cl = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
        row[f"cl_{m}_vs_cw_max_m"] = float(d_cl.max())
        du = np.linalg.norm(Ua - Ub, axis=1)
        row[f"cl_{m}_cmd_diff_max_mps2"] = float(du.max())
        row[f"masking_ratio_{m}"] = float(d_cc.max() / d_cl.max()) if d_cl.max() > 0 else float("nan")
    hist = {"t": grid}
    for m, tr in tr_cl.items():
        hist[f"cl_{m}_t"], hist[f"cl_{m}_S"], hist[f"cl_{m}_app"], hist[f"cl_{m}_raw"] = tr.t, tr.S, tr.app, tr.raw
    for m, tr in tr_cc.items():
        hist[f"cc_{m}_t"], hist[f"cc_{m}_S"], hist[f"cc_{m}_app"] = tr.t, tr.S, tr.app
    return row, hist


__all__ = ["common_command"]
