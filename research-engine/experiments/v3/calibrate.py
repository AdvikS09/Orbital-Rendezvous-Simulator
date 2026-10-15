"""P04 step 2: select one candidate per family from calibration labels with the frozen rule.

Refuses to run unless PROSPECTIVE_PLAN.json exists, the analysis/engine sources still match the
hashes frozen in the plan, and every planned calibration run has a result. Writes
  v3_extension/raw/calibration_freeze.json      selected thresholds (refuses to overwrite)
  v3_extension/results/calibration_candidates.csv   n, b, U for every candidate (calibration data)
Test data are never read here.
Usage: python code/experiments/v3/calibrate.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD  # noqa: E402
from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402

PLAN = os.path.join(VD.V3, "PROSPECTIVE_PLAN.json")
FREEZE = os.path.join(VD.V3, "raw", "calibration_freeze.json")


def check_frozen_code(plan):
    bad = []
    for rel, h in plan["implementation"]["code_sha256"].items():
        p = os.path.join(VD.REPO, rel)
        if not os.path.exists(p) or L.text_sha256(p) != h:
            bad.append(rel)
    return bad


def main():
    if not os.path.exists(PLAN):
        sys.exit("no PROSPECTIVE_PLAN.json: freeze the protocol first")
    if os.path.exists(FREEZE):
        sys.exit("calibration_freeze.json exists; the selection is made once")
    plan = json.load(open(PLAN))
    bad = check_frozen_code(plan)
    if bad:
        sys.exit(f"frozen sources changed since the plan: {bad}")
    mult = plan["multiplicity"]
    level, alpha = mult["per_candidate_level"], mult["alpha"]
    data, recon = {}, {}
    for cell in plan["domains"]:
        data[cell], recon[cell] = VD.load_split(cell, "cal")
    rows, sel = [], {}
    for f in plan["families"]:
        df = data[f["cell"]]
        v = df[f"f_{f['method']}"].to_numpy(float)
        adv = VD.adverse(df[f"label_{f['criterion']}"])
        cand_rows, best = CAL.calibrate_family(v, adv, f["candidates"], alpha, level)
        for r in cand_rows:
            rows.append({"family_id": f["family_id"], "cell": f["cell"], "method": f["method"], "criterion": f["criterion"],
                         "n_cal": int(v.size), **r, "selected": best is not None and r["cid"] == best["cid"]})
        sel[f["family_id"]] = {"cell": f["cell"], "method": f["method"], "criterion": f["criterion"],
                               "status": "selected" if best else "abstain_no_admissible_candidate",
                               "threshold": None if best is None else best["threshold"],
                               "source": None if best is None else best["source"],
                               "cid": None if best is None else best["cid"],
                               "cal_accepted": None if best is None else best["n_accepted"],
                               "cal_adverse": None if best is None else best["n_adverse"],
                               "cal_ucb": None if best is None else best["ucb"], "n_cal": int(v.size),
                               "cal_feature_unavailable": int((~np.isfinite(v)).sum())}
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(VD.V3, "results", "calibration_candidates.csv"), index=False)
    census = {c: {"truth_status": d.truth_status.value_counts().to_dict(), "label_1m": d["label_1m"].value_counts().to_dict(),
                  "label_0.1m": d["label_0.1m"].value_counts().to_dict(), "worker_status": d.worker_status.value_counts().to_dict(),
                  "reconcile": {k: recon[c][k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")}}
              for c, d in data.items()}
    rec = {"selected_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "plan_sha256": L.file_sha256(PLAN),
           "alpha": alpha, "per_candidate_level": level, "M": mult["M_total_candidates"],
           "calibration_results_sha256": {c: L.file_sha256(os.path.join(VD.RES, f"{c}_cal.jsonl")) for c in data},
           "census": census, "selection": sel,
           "note": "Selected from calibration labels only, by the rule frozen in PROSPECTIVE_PLAN.json; test labels unread."}
    json.dump(L.jsonable(rec), open(FREEZE, "w"), indent=1)
    s = pd.DataFrame(sel).T
    print(s[["status", "threshold", "source", "cal_accepted", "cal_adverse", "cal_ucb"]].to_string())
    print("written", FREEZE, L.file_sha256(FREEZE))


if __name__ == "__main__":
    main()
