"""Step 5 - main grid (2,160 paired scenarios) with pilot, refinement and summaries.

Grid: circular 400 km chief; six signed axis directions (+/-R, +/-V, +/-H);
40 log-spaced separations 50 m - 50 km; omega_n/n in {4, 6, 8};
a_max in {0.01, 0.05, 0.10} m/s^2; zero initial LVLH velocity; one-orbit horizon;
Kp = omega_n^2, Kd = 2 omega_n. Each paired scenario runs CW, the inertial
nonlinear reference (ECI, primary), the direct-LVLH exact nonlinear model
(NERM, cross-check) and the second-order model (SO), each closing its own loop.

Usage (repository root):
  python experiments/ext/main_grid.py pilot   [--workers 7]   # 80 seeded scenarios
  python experiments/ext/main_grid.py grid    [--workers 7]   # full 2,160
  python experiments/ext/main_grid.py refine  [--workers 7] [--rounds 8] [--tol 0.005]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RES = os.path.join(os.path.dirname(ROOT), "results", "main_grid")

from cwvalid.refine import family_key_cols, find_brackets, new_points, predicates  # noqa: E402
from cwvalid.runner import load_results, run_campaign, write_manifest  # noqa: E402
from cwvalid.scenario import AXES, axis_scenario  # noqa: E402

AXIS_LIST = ["+R", "-R", "+V", "-V", "+H", "-H"]
SEPS = np.geomspace(50.0, 50_000.0, 40)
WN = [4.0, 6.0, 8.0]
AMAX = [0.01, 0.05, 0.10]
PILOT_SEED = 20261010

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


def grid_scenarios(family="grid"):
    out = []
    for ax in AXIS_LIST:
        for wn in WN:
            for am in AMAX:
                for k, s in enumerate(SEPS):
                    scn = axis_scenario(family, ax, float(s), wn, am, extra_meta={"grid_index": k})
                    out.append(scn)
    return out


def write_scenarios(path, scns):
    with open(path, "w") as fh:
        for s in scns:
            fh.write(json.dumps(s.to_dict()) + "\n")


def cmd_pilot(workers):
    scns = grid_scenarios()
    rng = np.random.default_rng(PILOT_SEED)
    idx = rng.choice(len(scns), size=80, replace=False)
    pilot = [scns[i] for i in sorted(idx)]
    d = os.path.join(RES, "pilot")
    os.makedirs(d, exist_ok=True)
    write_scenarios(os.path.join(d, "scenarios.jsonl"), pilot)
    t0 = time.perf_counter()
    run_campaign(d, pilot, CONFIG, workers=workers, keep_hist=False)
    wall = time.perf_counter() - t0
    df = load_results(d)
    prof = {"n": len(df), "wall_s": wall, "workers": workers,
            "mean_run_wall_s": float(df["run_wall_s"].mean()),
            "p95_run_wall_s": float(df["run_wall_s"].quantile(0.95)),
            "max_run_wall_s": float(df["run_wall_s"].max()),
            **{f"mean_{m}_solver_wall_s": float(df[f"{m}_wall_time_s"].mean()) for m in CONFIG["models"]},
            **{f"mean_{m}_nfev": float(df[f"{m}_nfev"].mean()) for m in CONFIG["models"]},
            "projected_full_grid_wall_s": float(df["run_wall_s"].mean() * 2160 / workers)}
    json.dump(prof, open(os.path.join(d, "pilot_profile.json"), "w"), indent=2)
    print(json.dumps(prof, indent=2))


def cmd_grid(workers):
    scns = grid_scenarios()
    assert len(scns) == 2160
    d = os.path.join(RES, "grid")
    os.makedirs(d, exist_ok=True)
    write_scenarios(os.path.join(d, "scenarios.jsonl"), scns)
    json.dump({"axes": AXIS_LIST, "separations_m": SEPS.tolist(), "wn_ratio": WN, "a_max": AMAX,
               "velocity": "zero", "horizon_orbits": 1.0, "config": CONFIG},
              open(os.path.join(d, "grid_definition.json"), "w"), indent=2)
    run_campaign(d, scns, CONFIG, workers=workers, keep_hist=False)


def cmd_refine(workers, rounds, tol):
    gdir = os.path.join(RES, "grid")
    rdir = os.path.join(RES, "refine")
    os.makedirs(rdir, exist_ok=True)
    preds = predicates("eci", (0.1, 1.0))
    log = []
    all_refine = []
    if os.path.exists(os.path.join(rdir, "scenarios.jsonl")):
        from cwvalid.scenario import Scenario
        all_refine = [Scenario.from_dict(json.loads(l)) for l in open(os.path.join(rdir, "scenarios.jsonl")) if l.strip()]
    for r in range(rounds):
        df = load_results(gdir)
        if os.path.exists(os.path.join(rdir, "results.jsonl")):
            df = pd.concat([df, load_results(rdir)], ignore_index=True)
        br = find_brackets(df, preds, tol)
        unresolved = int((~br["resolved"]).sum()) if len(br) else 0
        log.append({"round": r, "n_points": len(df), "n_brackets": len(br), "unresolved": unresolved})
        print(f"[refine] round {r}: {len(df)} points, {len(br)} brackets, {unresolved} unresolved", flush=True)
        if unresolved == 0:
            break
        pts = new_points(br, r)
        existing = {s.sid for s in all_refine}
        new = []
        for fam, sep in pts:
            meta = dict(zip(family_key_cols(), fam))
            scn = axis_scenario("refine", meta["meta_axis"], sep, float(meta["wn_ratio"]), float(meta["a_max"]),
                                extra_meta={"refine_round": r})
            if scn.sid not in existing:
                new.append(scn)
                existing.add(scn.sid)
        all_refine.extend(new)
        write_scenarios(os.path.join(rdir, "scenarios.jsonl"), all_refine)
        run_campaign(rdir, all_refine, CONFIG, workers=workers, keep_hist=False)
    df = load_results(gdir)
    if os.path.exists(os.path.join(rdir, "results.jsonl")):
        df = pd.concat([df, load_results(rdir)], ignore_index=True)
    br = find_brackets(df, preds, tol)
    br.to_csv(os.path.join(rdir, "boundary_brackets.csv"), index=False)
    json.dump({"tol_rel_width": tol, "rounds": log}, open(os.path.join(rdir, "refine_log.json"), "w"), indent=2)
    print(f"[refine] final: {len(br)} brackets, {int((~br['resolved']).sum()) if len(br) else 0} unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "grid", "refine"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    if a.cmd == "pilot":
        cmd_pilot(a.workers)
    elif a.cmd == "grid":
        cmd_grid(a.workers)
    else:
        cmd_refine(a.workers, a.rounds, a.tol)


if __name__ == "__main__":
    main()
