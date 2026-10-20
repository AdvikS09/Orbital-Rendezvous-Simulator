import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anadev as AD
from cwvalid3.domains import CELLS

ADV = ["violation", "exit", "failure", "unresolved"]


def auc(f, bad, good):
    m = np.isfinite(f) & (bad | good)
    if not (bad[m].any() and good[m].any()):
        return np.nan
    u = mannwhitneyu(f[m & bad], f[m & good], alternative="two-sided").statistic
    return float(u / ((m & bad).sum() * (m & good).sum()))


def main():
    rows = []
    for cell in CELLS:
        df = pd.concat([AD.casetable(cell, sp) for sp in ("pilot", "dev")], ignore_index=True)
        d = df.disc_m.to_numpy(float)
        comp = (df.truth_status == "completed_valid").to_numpy()
        aname = "A" if CELLS[cell]["cmd_mode"] == "continuous" else "A_H"
        fa = df[f"f_{aname}"].to_numpy(float)
        hista = AD.histcut(aname, 1.0)
        accahist = int((np.isfinite(fa) & (fa <= hista)).sum())
        for m in AD.meth2(cell):
            f = df[f"f_{m}"].to_numpy(float)
            r = {"population": "v3 pilot+dev (iid, development only)", "cell": cell, "method": m, "n": len(df),
                 "n_feature_unavailable": int((~np.isfinite(f)).sum())}
            for tag, lab in (("1m", "label_1m"), ("0.1m", "label_01m")):
                bad = df[lab].isin(["violation", "exit"]).to_numpy()
                good = (df[lab] == "ok").to_numpy()
                r[f"auc_{tag}"] = auc(f, bad, good)
            ok = comp & np.isfinite(f)
            r["spearman_feature_truth"] = float(spearmanr(f[ok], d[ok]).correlation) if ok.sum() > 2 else np.nan
            if m not in ("sep", "sat"):
                sel = ok & (d > 1e-3)
                rr = f[sel] / d[sel]
                r.update({"n_ratio": int(sel.sum()), "ratio_q05": float(np.quantile(rr, 0.05)),
                          "ratio_q50": float(np.quantile(rr, 0.5)), "ratio_q95": float(np.quantile(rr, 0.95)),
                          "frac_ratio_below_0.9": float(np.mean(rr < 0.9)), "frac_ratio_above_1.1": float(np.mean(rr > 1.1))})
                band = ok & (d > 0.01) & (d < 100)
                if band.any():
                    rb = f[band] / d[band]
                    r.update({"n_0.01_100m": int(band.sum()), "ratio_min_0.01_100m": float(rb.min()),
                              "ratio_max_0.01_100m": float(rb.max())})
                small = ok & (d <= 1e-3)
                if small.any():
                    r["abs_err_max_truth_le_1mm_m"] = float(np.max(np.abs(f[small] - d[small])))
            fin = np.sort(f[np.isfinite(f)])
            for p in (0.25, 0.5, 0.75):
                need = int(np.ceil(p * len(df)))
                if need <= fin.size:
                    t = fin[need - 1]
                    acc = np.isfinite(f) & (f <= t)
                    r[f"q{int(p * 100)}_threshold"] = float(t)
                    r[f"q{int(p * 100)}_accepted"] = int(acc.sum())
                    r[f"q{int(p * 100)}_adverse_1m"] = int((acc & df.label_1m.isin(ADV).to_numpy()).sum())
                    r[f"q{int(p * 100)}_adverse_0.1m"] = int((acc & df.label_01m.isin(ADV).to_numpy()).sum())
            if m in ("B", "B_H", "C", "C_H", "A_naive"):
                if accahist <= fin.size and accahist > 0:
                    t = fin[accahist - 1]
                    acc = np.isfinite(f) & (f <= t)
                    r.update({"rescale_target_accepted": accahist, "rescale_threshold": float(t),
                              "rescale_factor_vs_A_cutoff": float(t / hista),
                              "rescale_adverse_1m": int((acc & df.label_1m.isin(ADV).to_numpy()).sum())})
                    accref = np.isfinite(fa) & (fa <= hista)
                    r["A_hist_cutoff_adverse_1m"] = int((accref & df.label_1m.isin(ADV).to_numpy()).sum())
                    r["rescale_disagreements_with_A"] = int((acc ^ accref).sum())
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(AD.V3, "results", "ablation.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out[["cell", "method", "auc_1m", "auc_0.1m", "spearman_feature_truth", "ratio_q05", "ratio_q50", "ratio_q95",
                   "ratio_min_0.01_100m", "q50_adverse_1m", "q75_adverse_1m", "rescale_factor_vs_A_cutoff",
                   "rescale_adverse_1m", "A_hist_cutoff_adverse_1m"]].to_string())


if __name__ == "__main__":
    main()
