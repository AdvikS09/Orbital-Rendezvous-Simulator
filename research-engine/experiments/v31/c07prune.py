import csv
import json
import os
import re

import v31paths as P

REBUILT = os.path.join(P.V31, "rebuilt")
REASON = {
    "raw/atlascases.csv": "C01: disc_m, u_num_m, ratio_A (327 iid exit rows) and t_peak_disc_s (292) from the corrected held exit step; f_C_H differs by 1 ulp in 3 rows (re-batching, identical in the v3.0-engine replay)",
    "raw/testdecisions.csv": "C01: disc_m and u_num_m of the 101 D3/D4 test exit rows; f_C_H 1 ulp in 1 row (re-batching); every decision column identical",
    "results/dev/pilotdevcases.csv": "C01: disc_m and u_num_m of the 12 D3/D4 development exit rows (records produced by engine 25643c68, source not archived)",
    "results/dev/pilotdevsummary.json": "C01: D3/D4 u_num_m p95 and max (all development rows): the exit rows' cross-check discrepancy fell from 2.3e-3 / 2.9e-3 m to 8.2e-7 / 3.3e-6 m; medians identical",
    "results/failureatlas.csv": "C01: truth_disc_m of the two R5 illustrative exit cases (D3 Cec9aa9e05ba7b685, D4 Cd8c8a1e89454a319); every category count and quantile identical",
    "results/cohortdiagnostics.csv": "C01: disc_target_m and disc_ratio of the 14 held-target exit rows; cohortsummary.csv identical",
}


OLD = {
    "assets/TabV3Numerics": "rebuilt unchanged only because assets.py reads the uncorrected v3.0 audit "
                                 "(results/numerical.csv); superseded by v31/assets/TabV31Numerics.tex",
    "assets/FigV3S1Numerics": "rebuilt unchanged only because assets.py reads the uncorrected v3.0 audit; "
                                    "superseded by v31/assets/FigV31S1Numerics.{pdf,png}",
}


def stripdate(b):
    b = re.sub(rb"/CreationDate \(D:[^)]*\)", b"", b)
    return re.sub(rb"%%CreationDate:[^\r\n]*", b"", b)


def main():
    path = os.path.join(P.V31RES, "changedoutputs.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        f = os.path.join(REBUILT, r["path"])
        o = os.path.join(P.V3, r["path"])
        if r.get("disposition"):
            continue
        sup = next((v for k, v in OLD.items() if r["path"].startswith(k)), None)
        if sup:
            disp, why = "superseded", sup
        elif r["identical"] == "True":
            disp, why = "identical", "byte-identical to the v3.0 original"
        elif r["path"].endswith((".pdf", ".eps")) and os.path.exists(f) and \
                stripdate(open(f, "rb").read()) == stripdate(open(o, "rb").read()):
            disp, why = "metadata_only", "differs only in the embedded CreationDate; PNG and plot data are byte-identical"
        elif r["path"] == "raw/testevaluation.json":
            new, orig = json.load(open(f, encoding="utf-8")), json.load(open(o, encoding="utf-8"))
            regen = new.get("regenerations", [])[len(orig.get("regenerations", [])):]
            same = {k: v for k, v in new.items() if k != "regenerations"} == {k: v for k, v in orig.items() if k != "regenerations"}
            if not same:
                raise SystemExit("the rebuilt testevaluation differs from the frozen record outside 'regenerations'")
            disp, why = "harness_artefact", ("evaltest.py appended a regenerations entry " + json.dumps(regen) +
                                             " (false because testdecisions.csv differs in disc_m/u_num_m); every other field "
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
    with open(P.guardv30(path), "w", newline="", encoding="utf-8") as fh:
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
