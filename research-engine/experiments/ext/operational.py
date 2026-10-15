"""Step 8 - sampled control and perturbation sensitivity with a staged comparison.

For each case in results/operational/cases.json (representative and boundary
cases chosen from the main-grid analysis) the following branches are run, each
closing its own loop on its own state:

  key            model  command          force model (target / chaser)
  cw_cont        CW     continuous       -
  cw_s1, cw_s10  CW     sampled 1 s/10 s -
  nl_cont        ECI    continuous       two-body
  nerm_cont      NERM   continuous       two-body (formulation cross-check)
  nl_s1, nl_s10  ECI    sampled 1 s/10 s two-body
  j2_cont        ECI    continuous       J2 (exact LVLH frame rate)
  j2_cont_base   ECI    continuous       J2 (baseline frame rate, wx = 0)
  j2d_cont       ECI    continuous       J2 + drag, equal Cd*A/m = 0.022 m^2/kg
  j2d_lo, j2d_hi ECI    continuous       J2 + drag, chaser A doubled / halved
                                         (exploratory differential drag)
  j2d_s10        ECI    sampled 10 s     J2 + drag, equal ballistic coefficients

Spacecraft: Cd = 2.2, A = 1 m^2, m = 100 kg (manuscript Table 3); exponential
density rho0 = 3.9e-12 kg/m^3 at 400 km, H = 60 km. The 0.5x/2x ballistic
ratios are an exploratory sensitivity range, not a specific vehicle pair.
Inclination 51.6 deg; target released at the ascending node.

Staged comparison (peak position difference and terminal outcome):
  model      : cw_cont vs nl_cont        (two-body model disagreement)
  pert_J2    : nl_cont vs j2_cont        (effect of J2 omitted from CW)
  pert_drag  : j2_cont vs j2d_cont       (effect of common-mode drag)
  diff_drag  : j2d_cont vs j2d_lo/j2d_hi (exploratory differential drag)
  sampling   : nl_cont vs nl_s1 / nl_s10 (controller sampling, nonlinear)
  combined   : cw_cont vs j2d_s10        (CW prediction vs most operational branch)
  frame_fix  : j2_cont vs j2_cont_base   (effect of the baseline wx omission)

Usage: python experiments/ext/operational.py [--workers 7]
"""

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

from cwvalid import metrics as M  # noqa: E402
from cwvalid.control import CommandSpec  # noqa: E402
from cwvalid.dynamics import InertialForce  # noqa: E402
from cwvalid.experiment import make_law  # noqa: E402
from cwvalid.scenario import Scenario  # noqa: E402
from cwvalid.simulate import EciConfig, SolverConfig, simulate  # noqa: E402

CRIT = M.OutcomeCriteria()
SOLVER = SolverConfig()
B_AREA = 1.0


def branches():
    two = InertialForce()
    j2 = InertialForce(use_j2=True)
    j2d = InertialForce(use_j2=True, use_drag=True, area=B_AREA)
    j2d_hi = InertialForce(use_j2=True, use_drag=True, area=2.0 * B_AREA)
    j2d_lo = InertialForce(use_j2=True, use_drag=True, area=0.5 * B_AREA)
    E = lambda ft, fc, exact=True: EciConfig(force_target=ft, force_chaser=fc, exact_frame_rate=exact)
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
        "j2d_hi": ("eci", "continuous", None, E(j2d, j2d_hi)),
        "j2d_lo": ("eci", "continuous", None, E(j2d, j2d_lo)),
        "j2d_s10": ("eci", "sampled", 10.0, E(j2d, j2d)),
    }


STAGES = [("model", "cw_cont", "nl_cont"), ("formulation", "nerm_cont", "nl_cont"),
          ("pert_J2", "nl_cont", "j2_cont"), ("pert_drag", "j2_cont", "j2d_cont"),
          ("diff_drag_hi", "j2d_cont", "j2d_hi"), ("diff_drag_lo", "j2d_cont", "j2d_lo"),
          ("sampling_1s", "nl_cont", "nl_s1"), ("sampling_10s", "nl_cont", "nl_s10"),
          ("cw_sampling_10s", "cw_cont", "cw_s10"), ("combined", "cw_cont", "j2d_s10"),
          ("frame_fix", "j2_cont", "j2_cont_base"), ("model_s10", "cw_s10", "nl_s10")]


def run_case(d):
    scn = Scenario.from_dict(d)
    law = make_law(scn)
    T = scn.horizon
    grid = M.common_grid(T, 0.5)
    sims, traces, diag = {}, {}, {}
    for key, (model, mode, Ts, eci) in branches().items():
        cmd = CommandSpec(mode, law=law, sample_period=Ts)
        t0 = time.perf_counter()
        s = simulate(model, scn.rel0, cmd, T, n=scn.n, R=scn.R, solver=SOLVER, eci=eci)
        sims[key] = s
        traces[key] = M.make_trace(s, grid)
        dg = M.branch_diagnostics(s, traces[key], T, CRIT)
        dg.update(M.legacy_arrival(s, T, CRIT))
        dg.update({"status": s.status, "termination": s.termination, "nfev": s.nfev,
                   "wall_s": time.perf_counter() - t0})
        diag[key] = dg
    rows = []
    base = {"sid": scn.sid, "sep_m": scn.separation, "wn_ratio": scn.wn_ratio, "a_max": scn.a_max,
            **{f"meta_{k}": v for k, v in scn.meta.items()}}
    for key, dg in diag.items():
        rows.append({**base, "kind": "branch", "branch": key, **dg})
    for stage, a, b in STAGES:
        pdg = M.paired_diagnostics(sims[a], traces[a], sims[b], traces[b], T, CRIT)
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
    cases = json.load(open(os.path.join(RES, "cases.json")))
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(run_case, cases)):
            rows.extend(r)
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(cases)} cases ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df[df.kind == "branch"].dropna(axis=1, how="all").to_csv(os.path.join(RES, "operational_branches.csv"), index=False)
    df[df.kind == "stage"].dropna(axis=1, how="all").to_csv(os.path.join(RES, "operational_stages.csv"), index=False)
    print(f"done: {len(cases)} cases in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
