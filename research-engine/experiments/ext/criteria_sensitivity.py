"""Sensitivity of the terminal-arrival outcome definition (study assumptions).

On the representative/boundary cases (results/operational/cases.json) plus the
disagreement-band cases of verification pass 2, each scenario is propagated once
(CW and inertial reference, production settings) and the arrival labels are
re-evaluated for every combination of
  position tolerance {0.5, 1, 2} m, speed tolerance {0.005, 0.01, 0.02} m/s,
  minimum remaining dwell {0, 150, 300, 600} s
and for the legacy position-only test. Reported per combination: number of
reference arrivals, CW arrivals, CW/reference label disagreements, and label
changes relative to the primary definition (1 m, 0.01 m/s, 300 s).

Output: results/criteria_sensitivity/criteria_sensitivity.csv, tables/tab_criteria_sensitivity
Usage:  python experiments/ext/criteria_sensitivity.py [--workers 7]
"""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid import metrics as M  # noqa: E402
from cwvalid.experiment import RunConfig, run_branches  # noqa: E402
from cwvalid.scenario import Scenario  # noqa: E402

POS = (0.5, 1.0, 2.0)
VEL = (0.005, 0.01, 0.02)
DWELL = (0.0, 150.0, 300.0, 600.0)
OUT = os.path.join(RU.RESULTS, "criteria_sensitivity")


def one(d):
    scn = Scenario.from_dict(d)
    cfg = RunConfig(models=("cw", "eci"), reference="eci", crosscheck=None)
    sims, _ = run_branches(scn, cfg)
    grid = M.common_grid(scn.horizon, 0.5)
    tr = {m: M.make_trace(s, grid) for m, s in sims.items()}
    rows = []
    for p, v, w in itertools.product(POS, VEL, DWELL):
        crit = M.OutcomeCriteria(pos_tol=p, vel_tol=v, dwell=w)
        lab = {m: M.branch_diagnostics(sims[m], tr[m], scn.horizon, crit)["terminal_arrived"] for m in sims}
        rows.append({"sid": scn.sid, "pos_tol": p, "vel_tol": v, "dwell": w, "cw": lab["cw"], "ref": lab["eci"]})
    crit = M.OutcomeCriteria()
    lab = {m: M.legacy_arrival(sims[m], scn.horizon, crit)["legacy_arrived"] for m in sims}
    rows.append({"sid": scn.sid, "pos_tol": 1.0, "vel_tol": float("nan"), "dwell": float("nan"), "cw": lab["cw"],
                 "ref": lab["eci"], "legacy": True})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cases = json.load(open(os.path.join(RU.RESULTS, "operational", "cases.json")))
    p2 = os.path.join(RU.RESULTS, "verification", "pass2_cases.json")
    if os.path.exists(p2):
        cases += json.load(open(p2))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for rr in ex.map(one, cases) for r in rr]
    df = pd.DataFrame(rows)
    df["legacy"] = df.get("legacy", False).fillna(False)
    df.to_csv(os.path.join(OUT, "criteria_sensitivity.csv"), index=False)
    base = df[(df.pos_tol == 1.0) & (df.vel_tol == 0.01) & (df.dwell == 300.0)].set_index("sid")
    out = []
    for key, g in df.groupby(["legacy", "pos_tol", "vel_tol", "dwell"], dropna=False):
        g = g.set_index("sid")
        out.append({"definition": "legacy position-only" if key[0] else f"p<{key[1]:g} m, v<{key[2]:g} m/s, dwell>={key[3]:g} s",
                    "N": len(g), "reference arrivals": int(g.ref.sum()), "CW arrivals": int(g.cw.sum()),
                    "CW/reference disagreements": int((g.cw != g.ref).sum()),
                    "reference labels changed vs primary": int((g.ref != base.ref.loc[g.index]).sum()),
                    "CW labels changed vs primary": int((g.cw != base.cw.loc[g.index]).sum())})
    t = pd.DataFrame(out)
    RU.write_table(t, "tab_criteria_sensitivity", "Sensitivity of the terminal-arrival definition on the "
                   f"{len(base)} representative, boundary and disagreement-band cases: arrivals, CW/reference label "
                   "disagreements and label changes relative to the primary definition (1 m, 0.01 m/s, 300 s).",
                   "tab:criteria")
    print(t.to_string())


if __name__ == "__main__":
    main()
