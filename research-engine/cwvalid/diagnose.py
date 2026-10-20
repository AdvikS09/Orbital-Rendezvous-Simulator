import numpy as np

from . import metrics as M
from .control import CmdSpec
from .experiment import makelaw, runbranch
from .simulate import simulate


def align(sima, tra, simb, trb):
    short, other, simo = (tra, trb, simb) if tra.t[-1] <= trb.t[-1] else (trb, tra, sima)
    t = short.t
    n = t.size
    if other.t.size >= n and np.array_equal(other.t[:n], t):
        So, Uo = other.S[:n], other.app[:n]
    else:
        So = simo.states(t)
        Uo = simo.commands(t, So)[1]
    if short is tra:
        return t, short.S, So, short.app, Uo
    return t, So, short.S, Uo, short.app


def common_command(scn, cfg, models=("cw", "nerm", "eci", "so")):
    if cfg.cmd_mode != "continuous":
        raise ValueError("common-command replay is defined for the continuous-feedback CW loop only")
    simscl, _ = runbranch(scn, cfg)
    c = simscl["cw"]
    law = makelaw(scn, cfg.unrestricted)

    def pres(t):
        return law.command(*c.states([t])[0])

    def presvec(tt):
        return law.applvec(c.states(tt))

    cmd = CmdSpec("prescribed", prescribed=pres, prescribed_vec=presvec, label="CW applied command replay",
                  breakpoints=tuple(c.satswitchtimes))
    simscc = {}
    for m in models:
        simscc[m] = simulate(m, scn.rel0, cmd, scn.horizon, n=scn.n, R=scn.R,
                             solver=cfg.solver, eci=cfg.eci if m == "eci" else None)
    T = scn.horizon
    grid = M.commongrid(T, cfg.dt_eval)
    trcl = {m: M.maketrace(s, grid) for m, s in simscl.items()}
    trcc = {m: M.maketrace(s, grid) for m, s in simscc.items()}
    row = {"sid": scn.sid, "sep_m": scn.separation, **{f"meta_{k}": v for k, v in scn.meta.items()},
           "wn_ratio": scn.wn_ratio, "a_max": scn.a_max}
    for m in models:
        if m == "cw":
            _, Sa, Sb, _, _ = align(simscc["cw"], trcc["cw"], c, trcl["cw"])
            row["cc_cw_replay_vs_cw_closed_max_m"] = float(np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1).max())
            continue
        t1, Sa, Sb, _, _ = align(simscc[m], trcc[m], c, trcl["cw"])
        dcc = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
        row[f"cc_{m}_vs_cw_max_m"] = float(dcc.max())
        row[f"cc_{m}_vs_cw_t_max_s"] = float(t1[int(np.argmax(dcc))])
        row[f"cc_{m}_final_pos_m"] = float(np.linalg.norm(trcc[m].S[-1, :3]))
        row[f"cc_{m}_final_speed_mps"] = float(np.linalg.norm(trcc[m].S[-1, 3:]))
        row[f"cc_{m}_end_s"] = float(trcc[m].t[-1])
        t2, Sa, Sb, Ua, Ub = align(simscl[m], trcl[m], c, trcl["cw"])
        dcl = np.linalg.norm(Sa[:, :3] - Sb[:, :3], axis=1)
        row[f"cl_{m}_vs_cw_max_m"] = float(dcl.max())
        du = np.linalg.norm(Ua - Ub, axis=1)
        row[f"cl_{m}_cmd_diff_max_mps2"] = float(du.max())
        row[f"masking_ratio_{m}"] = float(dcc.max() / dcl.max()) if dcl.max() > 0 else float("nan")
    hist = {"t": grid}
    for m, tr in trcl.items():
        hist[f"cl_{m}_t"], hist[f"cl_{m}_S"], hist[f"cl_{m}_app"], hist[f"cl_{m}_raw"] = tr.t, tr.S, tr.app, tr.raw
    for m, tr in trcc.items():
        hist[f"cc_{m}_t"], hist[f"cc_{m}_S"], hist[f"cc_{m}_app"] = tr.t, tr.S, tr.app
    return row, hist
