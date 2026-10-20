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
import reportutil as RU

from cwvalid import metrics as M
from cwvalid.experiment import RunCfg, runbranch
from cwvalid.scenario import Scenario

POS = (0.5, 1.0, 2.0)
VEL = (0.005, 0.01, 0.02)
DWELL = (0.0, 150.0, 300.0, 600.0)
OUT = os.path.join(RU.RESULTS, "criteria")


def one(d):
    scn = Scenario.from_dict(d)
    cfg = RunCfg(models=("cw", "eci"), reference="eci", crosscheck=None)
    sims, _ = runbranch(scn, cfg)
    grid = M.commongrid(scn.horizon, 0.5)
    tr = {m: M.maketrace(s, grid) for m, s in sims.items()}
    rows = []
    for p, v, w in itertools.product(POS, VEL, DWELL):
        crit = M.OutcCrit(pos_tol=p, vel_tol=v, dwell=w)
        lab = {m: M.branchdiags(sims[m], tr[m], scn.horizon, crit)["terminal_arrived"] for m in sims}
        rows.append({"sid": scn.sid, "pos_tol": p, "vel_tol": v, "dwell": w, "cw": lab["cw"], "ref": lab["eci"]})
    crit = M.OutcCrit()
    lab = {m: M.legacyarr(sims[m], scn.horizon, crit)["legacy_arrived"] for m in sims}
    rows.append({"sid": scn.sid, "pos_tol": 1.0, "vel_tol": float("nan"), "dwell": float("nan"), "cw": lab["cw"],
                 "ref": lab["eci"], "legacy": True})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cas = json.load(open(os.path.join(RU.RESULTS, "operational", "cases.json")))
    p2 = os.path.join(RU.RESULTS, "verification", "pass2cases.json")
    if os.path.exists(p2):
        cas += json.load(open(p2))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for rr in ex.map(one, cas) for r in rr]
    df = pd.DataFrame(rows)
    df["legacy"] = df.get("legacy", False).fillna(False)
    df.to_csv(os.path.join(OUT, "criteria.csv"), index=False)
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
    RU.writetable(t, "criteria", "Sensitivity of the terminal-arrival definition on the "
                  f"{len(base)} representative, boundary and disagreement-band cases: arrivals, CW/reference label "
                  "disagreements and label changes relative to the primary definition (1 m, 0.01 m/s, 300 s).",
                  "tab:criteria")
    print(t.to_string())


if __name__ == "__main__":
    main()
