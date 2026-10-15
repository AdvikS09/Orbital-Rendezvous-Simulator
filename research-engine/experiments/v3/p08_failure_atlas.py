"""P08 failure atlas, status census and counterexamples (results/failure_atlas.csv).

Population: every v3 case of every cell and split (pilot, dev, cal, test; the transfer cohort is
reported separately), classified under the FROZEN selected 1 m rule of the cell's A-type feature
(A in continuous cells, A_H in held cells) and, for contrast, the B-type rule. Classification is
descriptive; nothing here changes a rule. Categories:
  accepted_adequate        accepted, truth ok
  accepted_violation       accepted, truth violates 1 m (underestimation that reached acceptance)
  accepted_exit            accepted, reference left the physical domain
  accepted_unresolved      accepted, truth unresolved or failed
  referred_conservative    referred, truth ok (coverage lost); split by feature/truth ratio
  referred_violation       referred, truth violates or exits (correct referral)
  referred_unavailable     feature unavailable
Axes summarised per category: separation/R, nu, omega_n/n, c_bar = a_max/(n^2 R), saturated-time
fraction, n T_s (held), feature/truth ratio. 2-D interaction table: violation-or-exit rate by
saturated-time fraction band x separation band.
Illustrative cases are chosen by recorded rules (one each, per cell where it exists):
  R1 largest resolved underprediction among accepted cases (minimum feature/truth, truth > 1 mm)
  R2 accepted case with the largest truth (an accepted violation if any exists)
  R3 referred adequate case with the smallest feature (most marginal conservative referral)
  R4 (held cells) largest |A_H - A_naive| / A_H among test cases
  R5 physical exit with the smallest initial separation
  R6 A-only and B-only acceptances on the test set (A wins and losses): the B-only case with the largest truth
Their full histories are regenerated with replay_case.py into raw/counterexamples/.
Usage: python code/experiments/v3/p08_failure_atlas.py [--no-replay]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS  # noqa: E402

R_E = 6378137.0
SPLITS = ("pilot", "dev", "cal", "test")


def categorize(df, f, thr, lab="label_1m"):
    v = df[f].to_numpy(float)
    acc = np.isfinite(v) & (v <= thr) if thr is not None else np.zeros(len(df), bool)
    labels = df[lab].to_numpy(object)
    cat = np.empty(len(df), dtype=object)
    cat[acc & (labels == "ok")] = "accepted_adequate"
    cat[acc & (labels == "violation")] = "accepted_violation"
    cat[acc & (labels == "exit")] = "accepted_exit"
    cat[acc & np.isin(labels, ["unresolved", "failure"])] = "accepted_unresolved"
    cat[~acc & ~np.isfinite(v)] = "referred_unavailable"
    rest = ~acc & np.isfinite(v)
    cat[rest & (labels == "ok")] = "referred_conservative"
    cat[rest & np.isin(labels, ["violation", "exit"])] = "referred_violation"
    cat[rest & np.isin(labels, ["unresolved", "failure"])] = "referred_unresolved"
    return cat, acc


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    frz = json.load(open(os.path.join(VD.V3, "raw", "calibration_freeze.json")))["selection"]
    rows, census, examples = [], [], []
    allc = []
    for cell in CELLS:
        held = CELLS[cell]["cmd_mode"] == "sampled"
        am, bm = ("A_H", "B_H") if held else ("A", "B")
        R = R_E + CELLS[cell]["altitude"]
        parts = []
        for sp in SPLITS:
            d, _ = VD.load_split(cell, sp)
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        n = df.n_radps.to_numpy(float)
        df["sep_over_R"] = df.sep_m / R
        df["cbar"] = df.a_max / (n ** 2 * R)
        df["nTs"] = (CELLS[cell]["Ts"] or 0.0) * n
        df["ratio_A"] = df[f"f_{am}"] / df.disc_m.where(df.disc_m > 1e-3)
        tA, tB = frz[f"{cell}|{am}|1m"]["threshold"], frz[f"{cell}|{bm}|1m"]["threshold"]
        df["cat_A"], accA = categorize(df, f"f_{am}", tA)
        df["cat_B"], accB = categorize(df, f"f_{bm}", tB)
        allc.append(df.assign(cellname=cell))
        for sp in SPLITS:
            s = df[df.split == sp]
            census.append({"kind": "status_census", "cell": cell, "split": sp, "n": len(s),
                           **{f"truth_{k}": int((s.truth_status == k).sum()) for k in
                              ("completed_valid", "physical_domain_exit", "numerical_failure", "unresolved")},
                           **{f"label1m_{k}": int((s.label_1m == k).sum()) for k in ("ok", "violation", "exit", "failure", "unresolved")},
                           **{f"arrival_{k}": int((s.arr_arrival_status == k).sum()) for k in
                              ("arrived", "completed_non_arrival", "unresolved", "not_applicable")},
                           "feature_unavailable_A": int((~np.isfinite(s[f"f_{am}"])).sum()),
                           "worker_not_ok": int((s.worker_status != "ok").sum()),
                           "interrupted_operational": int((s.worker_status == "worker_crash").sum()),
                           "numerical_failure_excl_interrupted": int(((s.truth_status == "numerical_failure") &
                                                                      (s.worker_status != "worker_crash")).sum())})
        for c in sorted(df.cat_A.unique()):
            s = df[df.cat_A == c]
            r = {"kind": "category_summary", "cell": cell, "rule": f"{am} 1 m frozen (t = {tA})", "category": c, "n": len(s),
                 "n_test": int((s.split == "test").sum())}
            for col in ("sep_over_R", "nu", "wn_ratio", "cbar", "sat_fraction", "ratio_A"):
                x = pd.to_numeric(s[col], errors="coerce").dropna()
                if len(x):
                    r[f"{col}_q10"], r[f"{col}_q50"], r[f"{col}_q90"] = (float(x.quantile(q)) for q in (0.1, 0.5, 0.9))
            if c == "referred_conservative":
                rr = pd.to_numeric(s.ratio_A, errors="coerce")
                r["n_ratio_le_1.25"] = int((rr <= 1.25).sum())
                r["n_ratio_gt_1.25"] = int((rr > 1.25).sum())
                r["n_truth_le_1mm"] = int((s.disc_m <= 1e-3).sum())
            rows.append(r)
        # A/B wins and losses (test)
        t = df[df.split == "test"]
        aa, bb = t.cat_A.str.startswith("accepted").to_numpy(), t.cat_B.str.startswith("accepted").to_numpy()
        good = (t.label_1m == "ok").to_numpy()
        rows.append({"kind": "A_vs_B_test", "cell": cell, "rule": f"{am} t={tA} vs {bm} t={tB}", "n": len(t),
                     "A_only": int((aa & ~bb).sum()), "A_only_adequate": int((aa & ~bb & good).sum()),
                     "B_only": int((~aa & bb).sum()), "B_only_adequate": int((~aa & bb & good).sum()),
                     "both": int((aa & bb).sum()), "neither": int((~aa & ~bb).sum())})
        # interaction table
        sb = pd.cut(df.sat_fraction.astype(float), [-1e-9, 1e-9, 0.25, 0.75, 1.0], labels=["0", "(0,0.25]", "(0.25,0.75]", "(0.75,1]"])
        pb = pd.cut(df.sep_m, [0, 500, 2000, 10000, 60000], labels=["<0.5 km", "0.5-2 km", "2-10 km", ">10 km"])
        bad = df.label_1m.isin(["violation", "exit"])
        tab = pd.DataFrame({"sat": sb, "sep": pb, "bad": bad}).groupby(["sat", "sep"], observed=False).bad.agg(["mean", "size"])
        for (si, pi), v in tab.iterrows():
            rows.append({"kind": "interaction_violation_rate", "cell": cell, "category": f"sat {si} | sep {pi}",
                         "n": int(v["size"]), "violation_or_exit_rate": float(v["mean"]) if v["size"] else np.nan})
        # illustrative cases by recorded rules
        acc_ok = df.cat_A.str.startswith("accepted") & (df.disc_m > 1e-3) & (df.truth_status == "completed_valid")
        if acc_ok.any():
            i = df[acc_ok].ratio_A.astype(float).idxmin()
            examples.append(("R1_largest_accepted_underprediction", cell, df.loc[i]))
        if df.cat_A.str.startswith("accepted").any():
            i = df[df.cat_A.str.startswith("accepted")].disc_m.astype(float).idxmax()
            examples.append(("R2_accepted_largest_truth", cell, df.loc[i]))
        rc = df.cat_A == "referred_conservative"
        if rc.any():
            i = df[rc][f"f_{am}"].astype(float).idxmin()
            examples.append(("R3_most_marginal_conservative_referral", cell, df.loc[i]))
        if held:
            tt = df[df.split == "test"]
            rel = (tt.f_A_H - tt.f_A_naive).abs() / tt.f_A_H
            if rel.notna().any():
                examples.append(("R4_largest_AH_vs_Anaive_test", cell, tt.loc[rel.idxmax()]))
        ex = df.truth_status == "physical_domain_exit"
        if ex.any():
            examples.append(("R5_smallest_separation_exit", cell, df.loc[df[ex].sep_m.idxmin()]))
        bo = (df.split == "test") & ~df.cat_A.str.startswith("accepted") & df.cat_B.str.startswith("accepted")
        if bo.any():
            examples.append(("R6_B_only_largest_truth_test", cell, df.loc[df[bo].disc_m.astype(float).idxmax()]))
    ex_rows = []
    for rule, cell, r in examples:
        ex_rows.append({"kind": "illustrative_case", "rule": rule, "cell": cell, "case_id": r.case_id, "split": r.split,
                        "category": r.cat_A, "truth_disc_m": r.disc_m, "feature_A": r[[c for c in r.index if c in ("f_A", "f_A_H")][0]],
                        "sep_m": r.sep_m, "nu": r.nu, "wn_ratio": r.wn_ratio, "a_max": r.a_max, "sat_fraction": r.sat_fraction,
                        "history": f"raw/counterexamples/{r.case_id}.csv"})
    out = pd.DataFrame(census + rows + ex_rows)
    out.to_csv(os.path.join(VD.V3, "results", "failure_atlas.csv"), index=False)
    pd.concat(allc, ignore_index=True).to_csv(os.path.join(VD.V3, "raw", "atlas_cases.csv"), index=False)
    if not a.no_replay:
        import replay_case as RC
        for r in ex_rows:
            RC.replay(r["case_id"], os.path.join(VD.V3, "raw", "counterexamples"))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[out.kind == "category_summary"][["cell", "category", "n", "n_test", "sat_fraction_q50", "ratio_A_q10",
                                                    "ratio_A_q50"]].to_string())
        print(pd.DataFrame(ex_rows)[["rule", "cell", "case_id", "split", "category", "truth_disc_m", "feature_A"]].to_string())


if __name__ == "__main__":
    main()
