import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    base, _ = VD.loadsplit("D0", "dev")
    base = base.set_index("case_id")
    rows, summ = [], []
    for tgt, am in (("D3", "A_H"), ("D4", "A_H"), ("D5", "A")):
        c, _ = VD.loadsplit(tgt, "transfer_cohort")
        for _, r in c.iterrows():
            b = base.loc[r.source_case_id]
            rows.append({"target": tgt, "source_case_id": r.source_case_id, "case_id": r.case_id,
                         "sep_m": b.sep_m, "wn_ratio": b.wn_ratio, "a_max": b.a_max,
                         "disc_D0_m": b.disc_m, "disc_target_m": r.disc_m,
                         "disc_ratio": r.disc_m / b.disc_m if b.disc_m > 0 else np.nan,
                         "A_D0_m": b.f_A, "A_target_m": r[f"f_{am}"], "A_naive_target_m": r.get("f_A_naive", np.nan),
                         "label_D0": b.label_1m, "label_target": r.label_1m,
                         "sat_D0": b.sat_fraction, "sat_target": r.sat_fraction,
                         "truth_status_D0": b.truth_status, "truth_status_target": r.truth_status})
        d = pd.DataFrame([x for x in rows if x["target"] == tgt])
        ok = (d.truth_status_D0 == "completed_valid") & (d.truth_status_target == "completed_valid") & (d.disc_D0_m > 1e-3)
        lr = np.log10(d.disc_ratio[ok])
        flips = pd.crosstab(d.label_D0, d.label_target)
        summ.append({"target": tgt, "n_pairs": len(d), "n_compared": int(ok.sum()),
                     "log10_disc_ratio_q05": float(lr.quantile(0.05)), "log10_disc_ratio_q50": float(lr.median()),
                     "log10_disc_ratio_q95": float(lr.quantile(0.95)),
                     "label_ok_to_violation": int(((d.label_D0 == "ok") & d.label_target.isin(["violation", "exit"])).sum()),
                     "label_violation_to_ok": int((d.label_D0.isin(["violation", "exit"]) & (d.label_target == "ok")).sum()),
                     "label_crosstab": flips.to_json()})
    out = pd.DataFrame(rows)
    out.insert(0, "population", "transfer-diagnostic cohort: 300 D0 development draws, paired; exploratory")
    out.to_csv(os.path.join(VD.V3, "results", "cohortdiagnostics.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(VD.V3, "results", "cohortsummary.csv"), index=False)
    print(s.to_string())


if __name__ == "__main__":
    main()
