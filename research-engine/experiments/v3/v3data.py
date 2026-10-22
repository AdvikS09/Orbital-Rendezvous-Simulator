import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")

from cwvalid3 import calibration as CAL
from cwvalid3 import ledger as L
from cwvalid3.domains import CELLS


def meth2(cell):
    return ("A", "B", "C", "sep", "sat") if CELLS[cell]["cmd_mode"] == "continuous" else ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")


def loadsplit(cell, split, reqfull=True):
    man = L.readjsonlstrict(os.path.join(MAN, f"{cell}{split.replace('_', '')}.jsonl"), key="run_id")
    res = L.readjsonlstrict(os.path.join(RES, f"{cell}{split.replace('_', '')}.jsonl"), key="run_id", allow_missing=True)
    rc = L.reconcile([r["run_id"] for r in man], res, "run_id")
    if reqfull and (rc["n_missing"] or rc["n_extra"] or rc["n_duplicates"]):
        raise SystemExit(f"{cell}/{split}: results do not reconcile with the manifest: {rc}")
    by = {r["run_id"]: r for r in res}
    kind = "continuous" if CELLS[cell]["cmd_mode"] == "continuous" else "held"
    recs = []
    for m in man:
        r = by.get(m["run_id"], {"run_id": m["run_id"], "worker_status": "missing"})
        tr = CAL.truth_record(r)
        d = {"run_id": m["run_id"], "case_id": m["case_id"], "physical_id": m["physical_id"], "cell": cell,
             "split": split, "index": m["index"], "worker_status": r.get("worker_status"),
             "sep_m": m["draw"]["sep_m"], "nu": m["draw"]["nu"], "wn_ratio": m["draw"]["wn_ratio"],
             "a_max": m["draw"]["a_max"], "axis": m["draw"]["axis"], "velocity": m["draw"]["velocity"],
             "n_radps": m["n_radps"], "cpu_s": r.get("cpu_s"), **tr,
             "label_1m": CAL.critlabel(tr, 1.0), "label_0.1m": CAL.critlabel(tr, 0.1),
             **{f"arr_{k}": v for k, v in CAL.arrlabel(r, tr).items()}}
        if m.get("source_case_id"):
            d["source_case_id"] = m["source_case_id"]
        for meth in meth2(cell):
            d[f"f_{meth}"] = CAL.feature_value(r, meth, kind)
        if kind == "held":
            d["sat_fraction"] = r.get("cw_sat_fraction")
            d["n_near_switch_updates"] = r.get("cw_n_near_switch_updates")
            d["n_status_changes"] = r.get("cw_n_status_changes")
            d["cw_arrived"] = r.get("cw_arrived")
            d["ref_arrived"] = r.get("eci_arrived")
            d["cw_max_range_m"] = r.get("cw_final_range_m")
            d["t_peak_disc_s"] = r.get("cw_vs_eci_t_peak_s")
        else:
            d["sat_fraction"] = r.get("feature_sat_fraction")
            d["n_sat_switch"] = r.get("cw_n_sat_switch_events")
            d["cw_arrived"] = r.get("cw_terminal_arrived")
            d["ref_arrived"] = r.get("eci_terminal_arrived")
            d["cw_max_range_m"] = r.get("cw_max_range_m")
            d["t_peak_disc_s"] = r.get("cw_vs_eci_t_max_pos_disc_s")
        recs.append(d)
    return pd.DataFrame(recs), rc


def adverse(labels):
    return np.isin(np.asarray(labels, object), ("violation", "exit", "failure", "unresolved"))
