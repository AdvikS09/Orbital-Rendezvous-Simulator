import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD
from cwvalid3 import calibration as CAL
from cwvalid3 import ledger as L

PLAN = os.path.join(VD.V3, "plan.json")
FREEZE = os.path.join(VD.V3, "raw", "freeze.json")


def checkfrozencode(plan):
    bad = []
    for rel, h in plan["implementation"]["code_sha256"].items():
        p = os.path.join(VD.REPO, rel)
        if not os.path.exists(p) or L.text_sha256(p) != h:
            bad.append(rel)
    return bad


def main():
    if not os.path.exists(PLAN):
        sys.exit("no plan.json: freeze the protocol first")
    if os.path.exists(FREEZE):
        sys.exit("freeze.json exists; the selection is made once")
    plan = json.load(open(PLAN))
    bad = checkfrozencode(plan)
    if bad:
        sys.exit(f"frozen sources changed since the plan: {bad}")
    mult = plan["multiplicity"]
    level, alpha = mult["per_candidate_level"], mult["alpha"]
    data, recon = {}, {}
    for cell in plan["domains"]:
        data[cell], recon[cell] = VD.loadsplit(cell, "cal")
    rows, sel = [], {}
    for f in plan["families"]:
        df = data[f["cell"]]
        v = df[f"f_{f['method']}"].to_numpy(float)
        adv = VD.adverse(df[f"label_{f['criterion']}"])
        candrows, best = CAL.calfam(v, adv, f["candidates"], alpha, level)
        for r in candrows:
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
    out.to_csv(os.path.join(VD.V3, "results", "candidates.csv"), index=False)
    census = {c: {"truth_status": d.truth_status.value_counts().to_dict(), "label_1m": d["label_1m"].value_counts().to_dict(),
                  "label_0.1m": d["label_0.1m"].value_counts().to_dict(), "worker_status": d.worker_status.value_counts().to_dict(),
                  "reconcile": {k: recon[c][k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")}}
              for c, d in data.items()}
    rec = {"selected_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "plan_sha256": L.filesha256(PLAN),
           "alpha": alpha, "per_candidate_level": level, "M": mult["M_total_candidates"],
           "calibration_results_sha256": {c: L.filesha256(os.path.join(VD.RES, f"{c}cal.jsonl")) for c in data},
           "census": census, "selection": sel,
           "note": "Selected from calibration labels only, by the rule frozen in plan.json; test labels unread."}
    json.dump(L.jsonable(rec), open(FREEZE, "w"), indent=1)
    s = pd.DataFrame(sel).T
    print(s[["status", "threshold", "source", "cal_accepted", "cal_adverse", "cal_ucb"]].to_string())
    print("written", FREEZE, L.filesha256(FREEZE))


if __name__ == "__main__":
    main()
