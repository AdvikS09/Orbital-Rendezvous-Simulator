"""Trajectory and control histories for the boundary-family disagreement cases.

Re-runs (production configuration) the diagnostic point at the middle of each resolved disagreement
band ('inside band 0.5') and every interior point of the exploratory supplementary set, keeps the
histories of all four branches at 1 s spacing (every second point of the 0.5 s grid), and checks that
the rerun reproduces the stored result row (statuses, labels, peak difference, arrival times).

Output: results/v2/raw/boundary_families/histories/<sid>.npz, histories_index.csv
Usage:  python experiments/v2/family_histories.py
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

B = os.path.join(C.RAW, "boundary_families")
OUT = os.path.join(B, "histories")
CHECK = ["cw_status", "eci_status", "so_status", "nerm_status", "eci_termination", "cw_terminal_arrived", "eci_terminal_arrived",
         "so_terminal_arrived", "cw_vs_eci_max_pos_disc_m", "eci_terminal_arrival_time_s", "cw_terminal_arrival_time_s"]


def one(args):
    scn_d, stored = args
    from cwvalid.experiment import RunConfig, run_scenario
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(scn_d)
    row, hist = run_scenario(scn, RunConfig(), keep_histories=True, hist_stride=2)
    hist["scenario_json"] = np.array(json.dumps(scn_d))
    np.savez_compressed(os.path.join(OUT, f"{scn.sid}.npz"), **hist)
    diffs = []
    for k in CHECK:
        a, b = stored.get(k), row.get(k)
        if isinstance(a, float) and isinstance(b, float):
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                diffs.append(k)
        elif a != b:
            diffs.append(k)
    return {"sid": scn.sid, "family_id": scn.meta.get("family_id"), "label": scn.meta.get("diag_label"),
            "sep_m": scn.separation, "outcome_terminal_cw": row.get("outcome_terminal_cw"),
            "peak_m": row.get("cw_vs_eci_max_pos_disc_m"), "fields_differing_from_stored": ";".join(diffs),
            "file": f"histories/{scn.sid}.npz"}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for camp, sel in (("diagnostic", lambda lab: lab == "inside band 0.5"),
                      ("diagnostic_supplementary", lambda lab: lab.startswith("inside"))):
        scns = [json.loads(l) for l in open(os.path.join(B, camp, "scenarios.jsonl")) if l.strip()]
        stored = {r["sid"]: r for r in C.read_jsonl(os.path.join(B, camp, "results.jsonl"))}
        jobs += [(s, stored[s["sid"]]) for s in scns if sel(s["meta"].get("diag_label", ""))]
    with ProcessPoolExecutor(7) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(B, "histories_index.csv"), index=False)
    C.log_run("boundary_families", "python experiments/v2/family_histories.py", "completed", 0.0,
              f"{len(df)} histories; {int((df.fields_differing_from_stored != '').sum())} reruns differing from stored rows")
    print(df.to_string())


if __name__ == "__main__":
    main()
