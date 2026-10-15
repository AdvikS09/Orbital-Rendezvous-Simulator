"""Development-only analysis of the pilot (and later the development split): status census, runtime,
numerical resolution, feature/truth relation and acceptance at candidate cutoffs. Never reads
calibration or test results.

Usage: python code/experiments/v3/analyze_dev.py --splits pilot [dev]
Output: v3_extension/results/dev/<tag>_summary.json, <tag>_cases.csv
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
RES = os.path.join(V3, "raw", "results")
OUT = os.path.join(V3, "results", "dev")

from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS  # noqa: E402

DEV_SPLITS = ("pilot", "dev")
HIST = json.load(open(os.path.join(ARCH, "results/v2/raw/frozen_v2/estimator_v2_frozen.json")))["rules"]


def methods_for(cell):
    return ("A", "B", "C", "sep", "sat") if CELLS[cell]["cmd_mode"] == "continuous" else ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")


def hist_cutoff(method, tol):
    pref = "traj1" if tol == 1.0 else "traj01"
    feat = {"A": "d1", "A_H": "d1", "A_naive": "d1", "B": "d1u", "B_H": "d1u", "C": "cres", "C_H": "cres",
            "sep": "sep", "sat": "sat"}[method]
    r = HIST.get(f"{pref}_{feat}")
    return r["threshold"] if r and r.get("status") == "ok" else None


def case_table(cell, split):
    if split not in DEV_SPLITS:
        raise SystemExit("development analysis only")
    rows = L.read_jsonl_strict(os.path.join(RES, f"{cell}_{split}.jsonl"), key="run_id")
    kind = "continuous" if CELLS[cell]["cmd_mode"] == "continuous" else "held"
    recs = []
    for r in rows:
        tr = CAL.truth_record(r)
        d = {"run_id": r["run_id"], "case_id": r["case_id"], "cell": cell, "split": split, "index": r["index"],
             "worker_status": r.get("worker_status"), "sep_m": r.get("sep_m"), "nu": r.get("nu"),
             "wn_ratio": r.get("wn_ratio"), "a_max": r.get("a_max"), "cpu_s": r.get("cpu_s"), **tr,
             "label_1m": CAL.criterion_label(tr, 1.0), "label_01m": CAL.criterion_label(tr, 0.1),
             **{f"arr_{k}": v for k, v in CAL.arrival_label(r, tr).items()}}
        for m in methods_for(cell):
            d[f"f_{m}"] = CAL.feature_value(r, m, kind)
        if kind == "held":
            d["n_near_switch_updates"] = r.get("cw_n_near_switch_updates")
            d["sat_fraction"] = r.get("cw_sat_fraction")
            d["cw_naive_pass_max_state_diff_m"] = r.get("cw_naive_pass_max_state_diff_m")
        else:
            d["sat_fraction"] = r.get("feature_sat_fraction")
            d["n_sat_switch"] = r.get("cw_n_sat_switch_events")
        recs.append(d)
    return pd.DataFrame(recs)


def summarize(df, cell):
    s = {"cell": cell, "n": int(len(df)), "worker_status": df.worker_status.value_counts().to_dict(),
         "truth_status": df.truth_status.value_counts().to_dict(),
         "label_1m": df.label_1m.value_counts().to_dict(), "label_01m": df.label_01m.value_counts().to_dict(),
         "arrival_status": df.arr_arrival_status.value_counts().to_dict(),
         "cpu_s_per_case": {"mean": float(df.cpu_s.mean()), "median": float(df.cpu_s.median()),
                            "p95": float(df.cpu_s.quantile(0.95)), "max": float(df.cpu_s.max())},
         "u_num_m": {"median": float(df.u_num_m.median()), "p95": float(df.u_num_m.quantile(0.95)),
                     "max": float(df.u_num_m.max())}}
    ok = df.truth_status == "completed_valid"
    feats = {}
    for m in methods_for(cell):
        f = df[f"f_{m}"].to_numpy(float)
        ent = {"n_unavailable": int((~np.isfinite(f)).sum())}
        for tol, lab in ((1.0, "label_1m"), (0.1, "label_01m")):
            hc = hist_cutoff(m, tol)
            if hc is not None:
                acc = CAL.accept_mask(f, hc)
                bad = df[lab].isin(["violation", "exit", "failure", "unresolved"]).to_numpy()
                ent[f"hist_cutoff_{tol:g}m"] = {"threshold": hc, "accepted": int(acc.sum()),
                                                 "accepted_adverse": int((acc & bad).sum())}
        if m not in ("sep", "sat"):
            sel = ok.to_numpy() & np.isfinite(f) & (df.disc_m.to_numpy(float) > 1e-6)
            if sel.any():
                ratio = f[sel] / df.disc_m.to_numpy(float)[sel]
                near = sel & (df.disc_m.to_numpy(float) > 0.01) & (df.disc_m.to_numpy(float) < 100)
                ent["ratio_feature_over_truth"] = {"n": int(sel.sum()), "median": float(np.median(ratio)),
                                                   "min": float(ratio.min()), "max": float(ratio.max()),
                                                   "n_0.01_100m": int(near.sum()),
                                                   "min_0.01_100m": float((f[near] / df.disc_m.to_numpy(float)[near]).min()) if near.any() else None}
        feats[m] = ent
    s["features"] = feats
    return s


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["pilot"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or "_".join(a.splits)
    os.makedirs(OUT, exist_ok=True)
    frames, summ = [], []
    for cell in CELLS:
        parts = [case_table(cell, sp) for sp in a.splits if os.path.exists(os.path.join(RES, f"{cell}_{sp}.jsonl"))]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        frames.append(df)
        summ.append(summarize(df, cell))
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(os.path.join(OUT, f"{tag}_cases.csv"), index=False)
    json.dump(L.jsonable(summ), open(os.path.join(OUT, f"{tag}_summary.json"), "w"), indent=1)
    for s in summ:
        print(json.dumps(L.jsonable(s)))


if __name__ == "__main__":
    main()
