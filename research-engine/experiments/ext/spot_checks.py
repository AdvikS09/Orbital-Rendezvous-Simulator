"""Single-scenario numerical diagnostics quoted in VERIFICATION_REPORT.md section 2.

1. grid-925b9dfc512a (+cross-track 17.3 km, 8n, 0.01 m/s^2) at the superseded
   settings (rtol 1e-12, no switch events) and at production settings:
   formulation spread (NERM vs inertial) and the response of each formulation to
   initial-position perturbations of 1e-9, 1e-6 and 1e-3 m along the offset.
2. grid-db779c3b50e1 (-cross-track 12.1 km, 8n, 0.01 m/s^2): formulation spread at
   production settings and with a 5 s maximum step, and how much each formulation's
   trajectory changes between the two settings.

All differences are maximum position differences on the 0.5 s grid over the
common horizon of the two branches.

Output: results/verification/spot_checks.csv
Usage:  python experiments/ext/spot_checks.py
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid import metrics as M  # noqa: E402
from cwvalid.experiment import RunConfig, make_cmd  # noqa: E402
from cwvalid.scenario import Scenario  # noqa: E402
from cwvalid.simulate import SolverConfig, simulate  # noqa: E402

PROD = SolverConfig()
SUPERSEDED = SolverConfig(rtol=1e-12, atol_rel_pos=1e-9, atol_rel_vel=1e-12, atol_eci_pos=1e-6,
                          atol_eci_vel=1e-9, atol_dv=1e-12, max_step=20.0, sat_events=False)
STEP5 = dataclasses.replace(PROD, max_step=5.0)


def load(sid):
    for line in open(os.path.join(RU.RESULTS, "main_grid", "grid", "scenarios.jsonl")):
        d = json.loads(line)
        if d["sid"] == sid:
            return Scenario.from_dict(d)
    raise KeyError(sid)


def traj(s, model, solver, rel0=None):
    cmd = make_cmd(s, RunConfig())
    sim = simulate(model, np.asarray(s.rel0 if rel0 is None else rel0, float), cmd, s.horizon,
                   n=s.n, R=s.R, solver=solver)
    tr = M.make_trace(sim, M.common_grid(s.horizon, 0.5))
    return tr.t, tr.S[:, :3]


def maxdiff(a, b):
    ta, pa = a
    tb, pb = b
    k = min(len(ta), len(tb))
    return float(np.max(np.linalg.norm(pa[:k] - pb[:k], axis=1)))


def main():
    rows = []
    s = load("grid-925b9dfc512a")
    axis = np.asarray(s.rel0[:3], float) / np.linalg.norm(s.rel0[:3])
    for name, solver in (("superseded rtol 1e-12, no switch events", SUPERSEDED), ("production", PROD)):
        base = {m: traj(s, m, solver) for m in ("nerm", "eci")}
        rows.append({"sid": s.sid, "settings": name, "quantity": "formulation spread NERM vs inertial",
                     "perturbation_m": 0.0, "value_m": maxdiff(base["nerm"], base["eci"])})
        for d in (1e-9, 1e-6, 1e-3):
            r0 = np.array(s.rel0, float)
            r0[:3] += d * axis
            for m in ("nerm", "eci"):
                resp = maxdiff(base[m], traj(s, m, solver, r0))
                rows.append({"sid": s.sid, "settings": name, "quantity": f"{m} response to initial-position change",
                             "perturbation_m": d, "value_m": resp, "response_over_perturbation": resp / d})
        print(f"  {s.sid} {name} done", flush=True)
    s = load("grid-db779c3b50e1")
    out = {}
    for name, solver in (("production", PROD), ("production with 5 s max step", STEP5)):
        out[name] = {m: traj(s, m, solver) for m in ("nerm", "eci")}
        rows.append({"sid": s.sid, "settings": name, "quantity": "formulation spread NERM vs inertial",
                     "perturbation_m": 0.0, "value_m": maxdiff(out[name]["nerm"], out[name]["eci"])})
    for m in ("nerm", "eci"):
        rows.append({"sid": s.sid, "settings": "production vs 5 s max step", "quantity": f"{m} change between settings",
                     "perturbation_m": 0.0, "value_m": maxdiff(out["production"][m], out["production with 5 s max step"][m])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RU.RESULTS, "verification", "spot_checks.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.to_string())


if __name__ == "__main__":
    main()
