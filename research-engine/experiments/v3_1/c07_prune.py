"""C07: keep only genuinely changed files in v3_1_correction/rebuilt and annotate the changed-output manifest.

  identical           byte-identical to the v3.0 original: the rebuilt copy is deleted (hash kept in the manifest)
  metadata_only       PDF/EPS differing only in the embedded CreationDate (checked here): copy deleted
  harness_artefact    raw/test_evaluation_record.json: evaluate_test.py appends a 'regenerations' entry whenever it
                      runs; every other field is checked equal to the frozen record, the appended entry is kept in
                      the manifest reason and the rebuilt copy is deleted
  superseded          TabV3_Numerics / FigV3_S1_Numerics: rebuilt unchanged only because make_assets.py reads the
                      uncorrected v3.0 audit; replaced by TabV31_Numerics / FigV31_S1_Numerics (copy deleted)
  corrected_output    kept: the v3.1 corrected version of a v3.0 output (see 'reason')
Writes v3_1_correction/results/CHANGED_OUTPUTS_MANIFEST.csv (annotated in place) and prints a summary.
Usage: python code/experiments/v3_1/c07_prune.py
"""

import csv
import json
import os
import re

import v31paths as P

REBUILT = os.path.join(P.V31, "rebuilt")
REASON = {
    "raw/atlas_cases.csv": "C01: disc_m, u_num_m, ratio_A (327 iid exit rows) and t_peak_disc_s (292) from the corrected held exit step; f_C_H differs by 1 ulp in 3 rows (re-batching, identical in the v3.0-engine replay)",
    "raw/test_decisions.csv": "C01: disc_m and u_num_m of the 101 D3/D4 test exit rows; f_C_H 1 ulp in 1 row (re-batching); every decision column identical",
    "results/dev/pilot_dev_cases.csv": "C01: disc_m and u_num_m of the 12 D3/D4 development exit rows (records produced by engine 25643c68, source not archived)",
    "results/dev/pilot_dev_summary.json": "C01: D3/D4 u_num_m p95 and max (all development rows): the exit rows' cross-check discrepancy fell from 2.3e-3 / 2.9e-3 m to 8.2e-7 / 3.3e-6 m; medians identical",
    "results/failure_atlas.csv": "C01: truth_disc_m of the two R5 illustrative exit cases (D3 Cec9aa9e05ba7b685, D4 Cd8c8a1e89454a319); every category count and quantile identical",
    "results/transfer_cohort_diagnostics.csv": "C01: disc_target_m and disc_ratio of the 14 held-target exit rows; transfer_cohort_summary.csv identical",
}


SUPERSEDED = {
    "manuscript_assets/TabV3_Numerics": "rebuilt unchanged only because make_assets.py reads the uncorrected v3.0 audit "
                                        "(results/numerical_resolution.csv); superseded by v3_1_correction/manuscript_assets/TabV31_Numerics.tex",
    "manuscript_assets/FigV3_S1_Numerics": "rebuilt unchanged only because make_assets.py reads the uncorrected v3.0 audit; "
                                           "superseded by v3_1_correction/manuscript_assets/FigV31_S1_Numerics.{pdf,png}",
}


def strip_date(b):
    b = re.sub(rb"/CreationDate \(D:[^)]*\)", b"", b)
    return re.sub(rb"%%CreationDate:[^\r\n]*", b"", b)


def main():
    path = os.path.join(P.V31_RES, "CHANGED_OUTPUTS_MANIFEST.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        f = os.path.join(REBUILT, r["path"])
        o = os.path.join(P.V3, r["path"])
        if r.get("disposition"):          # already annotated (the rebuilt copies of non-corrected files are gone)
            continue
        sup = next((v for k, v in SUPERSEDED.items() if r["path"].startswith(k)), None)
        if sup:
            disp, why = "superseded", sup
        elif r["identical"] == "True":
            disp, why = "identical", "byte-identical to the v3.0 original"
        elif r["path"].endswith((".pdf", ".eps")) and os.path.exists(f) and \
                strip_date(open(f, "rb").read()) == strip_date(open(o, "rb").read()):
            disp, why = "metadata_only", "differs only in the embedded CreationDate; PNG and plot data are byte-identical"
        elif r["path"] == "raw/test_evaluation_record.json":
            new, orig = json.load(open(f, encoding="utf-8")), json.load(open(o, encoding="utf-8"))
            regen = new.get("regenerations", [])[len(orig.get("regenerations", [])):]
            same = {k: v for k, v in new.items() if k != "regenerations"} == {k: v for k, v in orig.items() if k != "regenerations"}
            if not same:
                raise SystemExit("the rebuilt test_evaluation_record differs from the frozen record outside 'regenerations'")
            disp, why = "harness_artefact", ("evaluate_test.py appended a regenerations entry " + json.dumps(regen) +
                                             " (false because test_decisions.csv differs in disc_m/u_num_m); every other field "
                                             "equals the frozen record (checked)")
        else:
            disp, why = "corrected_output", REASON.get(r["path"], "")
            if not why:
                raise SystemExit(f"unexplained change: {r['path']}")
        r["disposition"], r["reason"] = disp, why
        if disp != "corrected_output" and os.path.exists(f):
            os.remove(f)
    for dp, dn, fn in os.walk(REBUILT, topdown=False):
        if not os.listdir(dp):
            os.rmdir(dp)
    cols = list(rows[0].keys())
    with open(P.guard_v30(path), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r["disposition"] for r in rows))
    for r in rows:
        if r["disposition"] == "corrected_output":
            print(r["path"], "|", r["reason"])


if __name__ == "__main__":
    main()
