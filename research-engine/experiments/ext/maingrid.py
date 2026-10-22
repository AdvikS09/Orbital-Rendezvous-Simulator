import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RES = os.path.join(os.path.dirname(ROOT), "results", "maingrid")

from cwvalid.refine import famkeycols, findbrack, newpoints, predic
from cwvalid.runner import loadres, runcamp
from cwvalid.scenario import axisscen

AXISLIST = ["+R", "-R", "+V", "-V", "+H", "-H"]
SEPS = np.geomspace(50.0, 50_000.0, 40)
WN = [4.0, 6.0, 8.0]
AMAX = [0.01, 0.05, 0.10]
PILOTSEED = 20261010

CONFIG = {
    "models": ["cw", "eci", "nerm", "so"],
    "reference": "eci",
    "crosscheck": "nerm",
    "dt_eval": 0.5,
    "solver": {"rtol": 1e-13, "atol_rel_pos": 1e-10, "atol_rel_vel": 1e-13,
               "atol_eci_pos": 1e-7, "atol_eci_vel": 1e-10, "atol_dv": 1e-13, "max_step": 20.0,
               "sat_events": True},
    "crit": {"pos_tol": 1.0, "vel_tol": 0.01, "dwell": 300.0, "legacy_samples": 2001},
    "label": "main_grid",
}


def gridscen(family="grid"):
    out = []
    for ax in AXISLIST:
        for wn in WN:
            for am in AMAX:
                for k, s in enumerate(SEPS):
                    scn = axisscen(family, ax, float(s), wn, am, extra_meta={"grid_index": k})
                    out.append(scn)
    return out


def writescen(path, scns):
    with open(path, "w") as fh:
        for s in scns:
            fh.write(json.dumps(s.to_dict()) + "\n")


def cmdpilot(workers):
    scns = gridscen()
    rng = np.random.default_rng(PILOTSEED)
    idx = rng.choice(len(scns), size=80, replace=False)
    pilot = [scns[i] for i in sorted(idx)]
    d = os.path.join(RES, "pilot")
    os.makedirs(d, exist_ok=True)
    writescen(os.path.join(d, "scenarios.jsonl"), pilot)
    t0 = time.perf_counter()
    runcamp(d, pilot, CONFIG, workers=workers, keephist=False)
    wall = time.perf_counter() - t0
    df = loadres(d)
    prof = {"n": len(df), "wall_s": wall, "workers": workers,
            "mean_run_wall_s": float(df["run_wall_s"].mean()),
            "p95_run_wall_s": float(df["run_wall_s"].quantile(0.95)),
            "max_run_wall_s": float(df["run_wall_s"].max()),
            **{f"mean_{m}_solver_wall_s": float(df[f"{m}_wall_time_s"].mean()) for m in CONFIG["models"]},
            **{f"mean_{m}_nfev": float(df[f"{m}_nfev"].mean()) for m in CONFIG["models"]},
            "projected_full_grid_wall_s": float(df["run_wall_s"].mean() * 2160 / workers)}
    json.dump(prof, open(os.path.join(d, "profile.json"), "w"), indent=2)
    print(json.dumps(prof, indent=2))


def cmdgrid(workers):
    scns = gridscen()
    assert len(scns) == 2160
    d = os.path.join(RES, "grid")
    os.makedirs(d, exist_ok=True)
    writescen(os.path.join(d, "scenarios.jsonl"), scns)
    json.dump({"axes": AXISLIST, "separations_m": SEPS.tolist(), "wn_ratio": WN, "a_max": AMAX,
               "velocity": "zero", "horizon_orbits": 1.0, "config": CONFIG},
              open(os.path.join(d, "definition.json"), "w"), indent=2)
    runcamp(d, scns, CONFIG, workers=workers, keephist=False)


def cmdrefine(workers, rounds, tol):
    gdir = os.path.join(RES, "grid")
    rdir = os.path.join(RES, "refine")
    os.makedirs(rdir, exist_ok=True)
    preds = predic("eci", (0.1, 1.0))
    log = []
    allrefine = []
    if os.path.exists(os.path.join(rdir, "scenarios.jsonl")):
        from cwvalid.scenario import Scenario
        allrefine = [Scenario.from_dict(json.loads(l)) for l in open(os.path.join(rdir, "scenarios.jsonl")) if l.strip()]
    for r in range(rounds):
        df = loadres(gdir)
        if os.path.exists(os.path.join(rdir, "results.jsonl")):
            df = pd.concat([df, loadres(rdir)], ignore_index=True)
        br = findbrack(df, preds, tol)
        unr = int((~br["resolved"]).sum()) if len(br) else 0
        log.append({"round": r, "n_points": len(df), "n_brackets": len(br), "unresolved": unr})
        print(f"[refine] round {r}: {len(df)} points, {len(br)} brackets, {unr} unresolved", flush=True)
        if unr == 0:
            break
        pts = newpoints(br, r)
        existing = {s.sid for s in allrefine}
        new = []
        for fam, sep in pts:
            meta = dict(zip(famkeycols(), fam))
            scn = axisscen("refine", meta["meta_axis"], sep, float(meta["wn_ratio"]), float(meta["a_max"]),
                           extra_meta={"refine_round": r})
            if scn.sid not in existing:
                new.append(scn)
                existing.add(scn.sid)
        allrefine.extend(new)
        writescen(os.path.join(rdir, "scenarios.jsonl"), allrefine)
        runcamp(rdir, allrefine, CONFIG, workers=workers, keephist=False)
    df = loadres(gdir)
    if os.path.exists(os.path.join(rdir, "results.jsonl")):
        df = pd.concat([df, loadres(rdir)], ignore_index=True)
    br = findbrack(df, preds, tol)
    br.to_csv(os.path.join(rdir, "boundarybrackets.csv"), index=False)
    json.dump({"tol_rel_width": tol, "rounds": log}, open(os.path.join(rdir, "refinelog.json"), "w"), indent=2)
    print(f"[refine] final: {len(br)} brackets, {int((~br['resolved']).sum()) if len(br) else 0} unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "grid", "refine"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    if a.cmd == "pilot":
        cmdpilot(a.workers)
    elif a.cmd == "grid":
        cmdgrid(a.workers)
    else:
        cmdrefine(a.workers, a.rounds, a.tol)


if __name__ == "__main__":
    main()
