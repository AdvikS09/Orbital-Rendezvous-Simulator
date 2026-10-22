from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .control import CmdSpec, PDLaw
from .simulate import EciCfg, SolverCfg, simulate


@dataclass
class RunCfg:
    models: tuple = ("cw", "nerm", "eci", "so")
    reference: str = "eci"
    crosscheck: str | None = "nerm"
    solver: SolverCfg = field(default_factory=SolverCfg)
    crit: M.OutcCrit = field(default_factory=M.OutcCrit)
    dt_eval: float = 0.5
    cmd_mode: str = "continuous"
    sample_period: float | None = None
    unrestricted: bool = False
    eci: EciCfg = field(default_factory=EciCfg)
    dwell_amb_s: float = 0.5
    label: str = ""

    def asdict2(self):
        return {"models": list(self.models), "reference": self.reference,
                "crosscheck": self.crosscheck, "solver": self.solver.asdict2(),
                "crit": self.crit.asdict2(), "dt_eval": self.dt_eval,
                "cmd_mode": self.cmd_mode, "sample_period": self.sample_period,
                "unrestricted": self.unrestricted, "eci": self.eci.asdict2(),
                "dwell_amb_s": self.dwell_amb_s, "label": self.label}


def makelaw(scn, unrestricted=False):
    wn = scn.wn_ratio * scn.n
    return PDLaw.bandw(wn, None if unrestricted else scn.a_max)


def makecmd(scn, cfg):
    law = makelaw(scn, cfg.unrestricted)
    if cfg.cmd_mode == "none":
        return CmdSpec("none", law=None)
    if cfg.cmd_mode == "sampled":
        return CmdSpec("sampled", law=law, sample_period=cfg.sample_period)
    return CmdSpec("continuous", law=law)


def runbranch(scn, cfg, cmdoverr=None):
    cmd = cmdoverr or makecmd(scn, cfg)
    sims = {}
    for m in cfg.models:
        sims[m] = simulate(m, scn.rel0, cmd, scn.horizon, n=scn.n, R=scn.R,
                           solver=cfg.solver, eci=cfg.eci if m == "eci" else None)
    return sims, cmd


def evaluate(scn, cfg, sims, keep_histories=False, hist_stride=1):
    if cfg.reference not in sims:
        raise ValueError(f"reference model {cfg.reference!r} is not among the propagated models {list(sims)}")
    T = scn.horizon
    grid = M.commongrid(T, cfg.dt_eval)
    traces = {m: M.maketrace(s, grid) for m, s in sims.items()}
    row = {"sid": scn.sid, "family": scn.family}
    for k, v in scn.meta.items():
        row[f"meta_{k}"] = v
    for i, k in enumerate(("x0", "y0", "z0", "vx0", "vy0", "vz0")):
        row[k] = scn.rel0[i]
    law = makelaw(scn, cfg.unrestricted)
    row.update({"sep_m": scn.separation, "wn_ratio": scn.wn_ratio, "a_max": scn.a_max,
                "unrestricted": cfg.unrestricted, "horizon_orbits": scn.horizon_orbits,
                "horizon_s": T, "n_radps": scn.n, "R_m": scn.R, "kp": law.kp, "kd": law.kd,
                "sat_distance_m": (scn.a_max / law.kp) if scn.a_max else float("nan"),
                "cmd_mode": cfg.cmd_mode, "sample_period_s": cfg.sample_period,
                "reference": cfg.reference, "crosscheck": cfg.crosscheck})
    for m, s in sims.items():
        for k, v in s.summary().items():
            if k != "model":
                row[f"{m}_{k}"] = v
        for k, v in M.legacyarr(s, T, cfg.crit).items():
            row[f"{m}_{k}"] = v
        for k, v in M.branchdiags(s, traces[m], T, cfg.crit).items():
            row[f"{m}_{k}"] = v
    ref = cfg.reference
    pairs = [(m, ref) for m in sims if m != ref]
    if cfg.crosscheck and cfg.crosscheck in sims and (cfg.crosscheck, ref) not in pairs:
        pairs.append((cfg.crosscheck, ref))
    if cfg.crosscheck and cfg.crosscheck in sims and "cw" in sims and cfg.crosscheck != "cw":
        pairs.append(("cw", cfg.crosscheck))
    for a, b in pairs:
        for k, v in M.pairdiags(sims[a], traces[a], sims[b], traces[b], T, cfg.crit).items():
            row[f"{a}_vs_{b}_{k}"] = v
    for m in sims:
        if m == ref:
            continue
        row[f"outcome_terminal_{m}"] = M.classify(row[f"{m}_terminal_arrived"], row[f"{ref}_terminal_arrived"])
        row[f"outcome_legacy_{m}"] = M.classify(row[f"{m}_legacy_arrived"], row[f"{ref}_legacy_arrived"])
        row[f"outcome_posonly_{m}"] = M.classify(row[f"{m}_pos_only_arrived"], row[f"{ref}_pos_only_arrived"])
    amb = []
    cc = cfg.crosscheck
    if cc and cc in sims:
        if row[f"{cc}_terminal_arrived"] != row[f"{ref}_terminal_arrived"]:
            amb.append("terminal_ref_vs_crosscheck")
        if row[f"{cc}_legacy_arrived"] != row[f"{ref}_legacy_arrived"]:
            amb.append("legacy_ref_vs_crosscheck")
    for m in sims:
        dm = row.get(f"{m}_terminal_dwell_margin_s")
        if dm is not None and np.isfinite(dm) and abs(dm) < cfg.dwell_amb_s:
            amb.append(f"{m}_dwell_margin")
        if sims[m].status == "failed":
            amb.append(f"{m}_numerical_failure")
    row["ambiguity_flags"] = ";".join(amb)
    row["ambiguous"] = bool(amb)
    hist = None
    if keep_histories:
        stride = max(1, int(hist_stride))

        def dec(a):
            if stride == 1 or a.shape[0] < 3:
                return a
            idx = np.unique(np.append(np.arange(0, a.shape[0], stride), a.shape[0] - 1))
            return a[idx]
        hist = {"t": dec(grid), "hist_stride": np.array(stride), "dt_eval": np.array(cfg.dt_eval)}
        for m, tr in traces.items():
            hist[f"{m}_t"] = dec(tr.t)
            hist[f"{m}_S"] = dec(tr.S)
            hist[f"{m}_raw"] = dec(tr.raw)
            hist[f"{m}_app"] = dec(tr.app)
            hist[f"{m}_dv"] = dec(tr.dv)
    return row, hist


def runscen(scn, cfg, keep_histories=False, hist_stride=1):
    sims, _ = runbranch(scn, cfg)
    return evaluate(scn, cfg, sims, keep_histories, hist_stride)
