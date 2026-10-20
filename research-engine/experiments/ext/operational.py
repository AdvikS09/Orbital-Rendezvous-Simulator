import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RES = os.path.join(os.path.dirname(ROOT), "results", "operational")

from cwvalid import metrics as M
from cwvalid.control import CmdSpec
from cwvalid.dynamics import InertForce
from cwvalid.experiment import makelaw
from cwvalid.scenario import Scenario
from cwvalid.simulate import EciCfg, SolverCfg, simulate

CRIT = M.OutcCrit()
SOLVER = SolverCfg()
BAREA = 1.0


def branches():
    two = InertForce()
    j2 = InertForce(use_j2=True)
    j2d = InertForce(use_j2=True, use_drag=True, area=BAREA)
    j2dhi = InertForce(use_j2=True, use_drag=True, area=2.0 * BAREA)
    j2dlo = InertForce(use_j2=True, use_drag=True, area=0.5 * BAREA)
    E = lambda ft, fc, exact=True: EciCfg(force_target=ft, force_chaser=fc, exact_frame_rate=exact)
    return {
        "cw_cont": ("cw", "continuous", None, None),
        "cw_s1": ("cw", "sampled", 1.0, None),
        "cw_s10": ("cw", "sampled", 10.0, None),
        "nl_cont": ("eci", "continuous", None, E(two, two)),
        "nerm_cont": ("nerm", "continuous", None, None),
        "nl_s1": ("eci", "sampled", 1.0, E(two, two)),
        "nl_s10": ("eci", "sampled", 10.0, E(two, two)),
        "j2_cont": ("eci", "continuous", None, E(j2, j2)),
        "j2_cont_base": ("eci", "continuous", None, E(j2, j2, exact=False)),
        "j2d_cont": ("eci", "continuous", None, E(j2d, j2d)),
        "j2d_hi": ("eci", "continuous", None, E(j2d, j2dhi)),
        "j2d_lo": ("eci", "continuous", None, E(j2d, j2dlo)),
        "j2d_s10": ("eci", "sampled", 10.0, E(j2d, j2d)),
    }


STAGES = [("model", "cw_cont", "nl_cont"), ("formulation", "nerm_cont", "nl_cont"),
          ("pert_J2", "nl_cont", "j2_cont"), ("pert_drag", "j2_cont", "j2d_cont"),
          ("diff_drag_hi", "j2d_cont", "j2d_hi"), ("diff_drag_lo", "j2d_cont", "j2d_lo"),
          ("sampling_1s", "nl_cont", "nl_s1"), ("sampling_10s", "nl_cont", "nl_s10"),
          ("cw_sampling_10s", "cw_cont", "cw_s10"), ("combined", "cw_cont", "j2d_s10"),
          ("frame_fix", "j2_cont", "j2_cont_base"), ("model_s10", "cw_s10", "nl_s10")]


def runcase(d):
    scn = Scenario.from_dict(d)
    law = makelaw(scn)
    T = scn.horizon
    grid = M.commongrid(T, 0.5)
    sims, traces, diag = {}, {}, {}
    for key, (model, mode, Ts, eci) in branches().items():
        cmd = CmdSpec(mode, law=law, sample_period=Ts)
        t0 = time.perf_counter()
        s = simulate(model, scn.rel0, cmd, T, n=scn.n, R=scn.R, solver=SOLVER, eci=eci)
        sims[key] = s
        traces[key] = M.maketrace(s, grid)
        dg = M.branchdiags(s, traces[key], T, CRIT)
        dg.update(M.legacyarr(s, T, CRIT))
        dg.update({"status": s.status, "termination": s.termination, "nfev": s.nfev,
                   "wall_s": time.perf_counter() - t0})
        diag[key] = dg
    rows = []
    base = {"sid": scn.sid, "sep_m": scn.separation, "wn_ratio": scn.wn_ratio, "a_max": scn.a_max,
            **{f"meta_{k}": v for k, v in scn.meta.items()}}
    for key, dg in diag.items():
        rows.append({**base, "kind": "branch", "branch": key, **dg})
    for stage, a, b in STAGES:
        pdg = M.pairdiags(sims[a], traces[a], sims[b], traces[b], T, CRIT)
        rows.append({**base, "kind": "stage", "stage": stage, "a": a, "b": b,
                     "a_terminal": diag[a]["terminal_arrived"], "b_terminal": diag[b]["terminal_arrived"],
                     "outcome": M.classify(diag[a]["terminal_arrived"], diag[b]["terminal_arrived"]),
                     "a_arrival_s": diag[a]["terminal_arrival_time_s"], "b_arrival_s": diag[b]["terminal_arrival_time_s"],
                     "a_dv": diag[a]["dv_aug_mps"], "b_dv": diag[b]["dv_aug_mps"], **pdg})
    hist = {"t": grid[::4]}
    for key, tr in traces.items():
        idx = np.unique(np.append(np.arange(0, tr.t.size, 4), tr.t.size - 1))
        hist[f"{key}_t"], hist[f"{key}_S"], hist[f"{key}_app"] = tr.t[idx], tr.S[idx], tr.app[idx]
    np.savez_compressed(os.path.join(RES, "histories", f"{scn.sid}.npz"), **hist)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "histories"), exist_ok=True)
    cas = json.load(open(os.path.join(RES, "cases.json")))
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(runcase, cas)):
            rows.extend(r)
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(cas)} cases ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df[df.kind == "branch"].dropna(axis=1, how="all").to_csv(os.path.join(RES, "branches.csv"), index=False)
    df[df.kind == "stage"].dropna(axis=1, how="all").to_csv(os.path.join(RES, "stages.csv"), index=False)
    print(f"done: {len(cas)} cases in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
