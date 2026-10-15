"""Re-execute runs recorded as worker_crash because the campaign driver was stopped deliberately.

The original records stay in raw/results/<cell>_<split>.jsonl (they are what the frozen procedure saw).
Re-executions go to raw/results/<cell>_<split>_reruns.jsonl with a 'rerun_of' field, and the ledger
receives 'rerun' events. A sensitivity check then recomputes the affected family selections with the
re-executed records substituted and reports whether any selected threshold would change; the frozen
selection in raw/calibration_freeze.json is not modified.

Usage: python code/experiments/v3/rerun_interrupted.py --cell D1 --split cal [--workers 2]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign as CP  # noqa: E402
from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    man = {r["run_id"]: r for r in L.read_jsonl_strict(os.path.join(CP.MAN, f"{a.cell}_{a.split}.jsonl"), key="run_id")}
    res = L.read_jsonl_strict(os.path.join(CP.RES, f"{a.cell}_{a.split}.jsonl"), key="run_id")
    crashed = [r["run_id"] for r in res if r.get("worker_status") == "worker_crash"]
    out_path = os.path.join(CP.RES, f"{a.cell}_{a.split}_reruns.jsonl")
    done = {r["run_id"] for r in L.read_jsonl_strict(out_path, key="run_id", allow_missing=True)}
    todo = [man[k] for k in crashed if k not in done]
    print(f"{a.cell}/{a.split}: {len(crashed)} interrupted runs, {len(todo)} to re-execute", flush=True)
    if CP.CELLS[a.cell]["cmd_mode"] != "continuous":
        raise SystemExit("held-cell reruns are not implemented (none were interrupted)")
    sw = CP.software()
    with ProcessPoolExecutor(a.workers) as ex, open(out_path, "a", encoding="utf-8", newline="\n") as fh, \
            open(CP.LEDGER, "a", encoding="utf-8", newline="\n") as lf:
        for outs in ex.map(CP.work_continuous, todo):
            for o in outs:
                o["software"] = sw
                o["rerun_of"] = "worker_crash record caused by a deliberate stop of the campaign driver"
                o["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                L.append_jsonl(out_path, o, fh)
                L.append_jsonl(CP.LEDGER, {"ledger_id": f"rerun:{o['run_id']}", "event": "rerun", "run_id": o["run_id"],
                                           "case_id": o["case_id"], "cell": a.cell, "split": a.split,
                                           "worker_status": o.get("worker_status"), "cpu_s": o.get("cpu_s"),
                                           "wall_s": o.get("wall_s"), "engine_sha": sw["engine_sha"],
                                           "time": o["completed_at"]}, lf)
    # sensitivity check of the affected selections
    reruns = {r["run_id"]: r for r in L.read_jsonl_strict(out_path, key="run_id")}
    plan = json.load(open(os.path.join(CP.V3, "PROSPECTIVE_PLAN.json")))
    frz = json.load(open(os.path.join(CP.V3, "raw", "calibration_freeze.json")))
    kind = "continuous"
    full = [reruns.get(r["run_id"], r) for r in res]
    level, alpha = plan["multiplicity"]["per_candidate_level"], plan["multiplicity"]["alpha"]
    rows = []
    for f in plan["families"]:
        if f["cell"] != a.cell:
            continue
        v = [CAL.feature_value(r, f["method"], kind) for r in full]
        lab = [CAL.criterion_label(CAL.truth_record(r), f["tol_m"]) for r in full]
        adv = [x in ("violation", "exit", "failure", "unresolved") for x in lab]
        _, best = CAL.calibrate_family(v, adv, f["candidates"], alpha, level)
        s = frz["selection"][f["family_id"]]
        rows.append({"family_id": f["family_id"], "frozen_threshold": s["threshold"],
                     "threshold_with_reruns": None if best is None else best["threshold"],
                     "same_selection": (s["threshold"] == (None if best is None else best["threshold"])),
                     "frozen_cal_accepted": s["cal_accepted"], "accepted_with_reruns": None if best is None else best["n_accepted"],
                     "adverse_with_reruns": None if best is None else best["n_adverse"]})
    rep = {"cell": a.cell, "split": a.split, "n_interrupted": len(crashed),
           "rerun_worker_status": {k: sum(1 for r in reruns.values() if r.get("worker_status") == k) for k in ("ok", "exception")},
           "selection_sensitivity": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(L.jsonable(rep), open(os.path.join(CP.V3, "raw", f"rerun_check_{a.cell}_{a.split}.json"), "w"), indent=1)
    print(json.dumps(L.jsonable(rep), indent=1))


if __name__ == "__main__":
    main()
