"""v3.1.1 R02: quantify how the driver-level nominal-only recovery (campaign.recover_features) differs from the normal
path, and check every frozen accept/refer decision.

The recovery path is used only after a worker process is lost before its features were checkpointed; it never ran in
the archived campaign (no archived record has feature_recovery). For held cells it propagates a CW + estimator batch
(no reference branches), a different state-vector shape from the combined batch of the normal path, so last bits may
differ; continuous cells repeat the normal computation. This script runs recover_features in-process on the c11 sample
(same groups and batch membership) and compares with the v3.1.1 normal-path records in
results/normal_path/records_v3.1.1.jsonl: per feature the number of exact matches and the largest absolute and
relative differences, and per frozen selection (calibration_freeze.json, every cell/method/criterion with a threshold)
the number of decision flips between the normal record (worker_status ok) and the recovered record (worker_crash with
feature_recovery 'driver_nominal_recovery', as cmd_run writes it).

Writes v3_1_1_followup/results/recovery_path_differences.csv, recovery_path_decisions.csv, recovery_path_summary.json.
Usage: python c12_recovery_path_differences.py
"""

import csv
import json
import os
import sys
import time

import v311paths as P

sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
import campaign as CP  # noqa: E402
import c11_normal_path as C11  # noqa: E402
from cwvalid3 import contracts as C  # noqa: E402
from cwvalid3 import serialization as SER  # noqa: E402

FEATS = {"held": ("A_H_peak_m", "B_H_peak_m", "C_H_peak_m", "A_naive_peak_m", "feature_sat_fraction", "sep_m", "cw_t_end_s"),
         "continuous": ("d1_peak_m", "d1u_peak_m", "cres_peak_m", "feature_sat_fraction", "sep_m", "cw_t_final_s")}
STATUS = {"held": ("cw_status", "A_H_status", "B_H_status", "C_H_status", "A_naive_status"),
          "continuous": ("cw_status", "d1_status", "d1u_status", "cres_status")}
METHODS = {"held": ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"), "continuous": ("A", "B", "C", "sep", "sat")}


def main():
    os.environ.pop(CP.CHECKPOINT_ENV, None)
    normal = {(r["c11_group"], r["run_id"]): r for r in (json.loads(x) for x in open(os.path.join(P.V311_RES, "normal_path", "records_v3.1.1.jsonl"),
                                                                   encoding="utf-8"))}
    sel = json.load(open(os.path.join(P.V3, "raw", "calibration_freeze.json"), encoding="utf-8"))["selection"]
    rows, drows = [], []
    t0, c0 = time.perf_counter(), time.process_time()
    for group, recs in C11.sample():
        got = CP.recover_features(recs)
        for m in recs:
            nrm = normal[(group, m["run_id"])]
            kind = C.kind_of({"cell": m["cell"]})
            fr = got[m["run_id"]]
            assert "feature_recovery_error" not in fr, fr
            rec = CP._merge_features(CP._crash_record(m, RuntimeError("process lost (simulated for c12)"), None), fr,
                                     "driver_nominal_recovery")
            row = {"group": group, "cell": m["cell"], "split": m["split"], "index": m["index"], "case_id": m["case_id"],
                   "normal_worker_status": nrm["worker_status"]}
            for k in STATUS[kind]:
                row[f"{k}_equal"] = SER.normalize(rec.get(k)) == SER.normalize(nrm.get(k))
            for k in FEATS[kind]:
                a, b = SER.normalize(nrm.get(k)), SER.normalize(rec.get(k))       # canonical: NaN -> None
                fa, fb = (isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
                row[f"{k}_normal"], row[f"{k}_recovered"] = a, b
                row[f"{k}_exact"] = a == b
                row[f"{k}_abs_diff"] = abs(a - b) if (fa and fb) else None
                row[f"{k}_rel_diff"] = (abs(a - b) / abs(a) if a else float(a != b)) if (fa and fb) else None
            rows.append(row)
            for key, s in sel.items():
                cell, meth, crit = key.split("|")
                if cell != m["cell"] or s.get("threshold") is None:
                    continue
                vn, sn, _ = C.feature_availability(nrm, meth, kind)
                vr, sr, why = C.feature_availability(rec, meth, kind)
                dn = sn == "ok" and vn <= s["threshold"]
                dr = sr == "ok" and vr <= s["threshold"]
                drows.append({"group": group, "case_id": m["case_id"], "selection": key, "threshold": s["threshold"],
                              "normal_available": sn, "recovered_available": sr, "recovered_reason": why,
                              "normal_accept": dn, "recovered_accept": dr, "flip": dn != dr,
                              "margin_m": (s["threshold"] - vn) if sn == "ok" else None})
    os.makedirs(P.V311_RES, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("group", "cell", "split", "index", "case_id",
                                                                         "normal_worker_status"), k))
    with open(P.guard(os.path.join(P.V311_RES, "recovery_path_differences.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(P.guard(os.path.join(P.V311_RES, "recovery_path_decisions.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)
    per = {}
    for kind in ("held", "continuous"):
        cells = ("D3", "D4") if kind == "held" else ("D0", "D1", "D2", "D5")
        rk = [r for r in rows if r["cell"] in cells]
        for k in FEATS[kind]:
            ad = [r[f"{k}_abs_diff"] for r in rk if r[f"{k}_abs_diff"] is not None]
            rd = [r[f"{k}_rel_diff"] for r in rk if r[f"{k}_rel_diff"] is not None]
            per[f"{kind}:{k}"] = {"n": len(rk), "n_exact": sum(bool(r[f"{k}_exact"]) for r in rk),
                                  "max_abs_diff": max(ad) if ad else None, "max_rel_diff": max(rd) if rd else None}
        for k in STATUS[kind]:
            per[f"{kind}:{k}"] = {"n": len(rk), "n_equal": sum(bool(r[f"{k}_equal"]) for r in rk)}
    flips = [d for d in drows if d["flip"]]
    avail = [d for d in drows if d["normal_available"] != d["recovered_available"]]
    margins = [abs(d["margin_m"]) for d in drows if d["margin_m"] is not None]
    summ = {"n_cases": len(rows), "n_held": sum(r["cell"] in ("D3", "D4") for r in rows),
            "n_continuous": sum(r["cell"] not in ("D3", "D4") for r in rows), "n_decisions": len(drows),
            "n_decision_flips": len(flips), "n_availability_changes": len(avail),
            "smallest_abs_margin_to_a_frozen_cutoff": min(margins) if margins else None,
            "per_feature": per, "engine_sha": CP.engine_version(), "wall_s": time.perf_counter() - t0,
            "cpu_s": time.process_time() - c0,
            "note": "recover_features run in-process on the c11 sample (same groups and membership as the normal-path run); "
                    "in cmd_run it runs in a fresh process after a process loss. Truth of a recovered record is "
                    "'interrupted' by construction; only features and their availability are compared.",
            "flips": flips}
    with open(P.guard(os.path.join(P.V311_RES, "recovery_path_summary.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_feature", "flips")}, indent=1))
    for k, v in per.items():
        print(k, v)


if __name__ == "__main__":
    main()
