import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C

B = os.path.join(C.RAW, "families")
OUT = os.path.join(B, "histories")
CHECK = ["cw_status", "eci_status", "so_status", "nerm_status", "eci_termination", "cw_terminal_arrived", "eci_terminal_arrived",
         "so_terminal_arrived", "cw_vs_eci_max_pos_disc_m", "eci_terminal_arrival_time_s", "cw_terminal_arrival_time_s"]


def one(args):
    scnd, stored = args
    from cwvalid.experiment import RunCfg, runscen
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(scnd)
    row, hist = runscen(scn, RunCfg(), keep_histories=True, hist_stride=2)
    hist["scenario_json"] = np.array(json.dumps(scnd))
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
                      ("supplementary", lambda lab: lab.startswith("inside"))):
        scns = [json.loads(l) for l in open(os.path.join(B, camp, "scenarios.jsonl")) if l.strip()]
        stored = {r["sid"]: r for r in C.readjsonl(os.path.join(B, camp, "results.jsonl"))}
        jobs += [(s, stored[s["sid"]]) for s in scns if sel(s["meta"].get("diag_label", ""))]
    with ProcessPoolExecutor(7) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(B, "historyindex.csv"), index=False)
    C.logrun("boundary_families", "python experiments/v2/famhist.py", "completed", 0.0,
             f"{len(df)} histories; {int((df.fields_differing_from_stored != '').sum())} reruns differing from stored rows")
    print(df.to_string())


if __name__ == "__main__":
    main()
