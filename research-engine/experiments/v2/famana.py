import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C

from cwvalid.runner import loadres

B = os.path.join(C.RAW, "families")
OUT = os.path.join(C.DERIVED, "families")


def cp(k, n):
    if n == 0:
        return (np.nan, np.nan)
    ci = binomtest(int(k), int(n)).proportion_ci(method="exact")
    return (ci.low, ci.high)


def main():
    os.makedirs(OUT, exist_ok=True)
    fams = json.load(open(os.path.join(B, "families.json")))["families"]
    v1 = json.load(open(C.FROZENV1))
    p = v1["params"]
    props = []
    for f in fams:
        for r in C.readjsonl(os.path.join(B, "search", f"{f['id']}.jsonl")):
            props.append({"family_id": f["id"], "purpose": r["purpose"], "models": r["models"], "sep_m": r["sep_m"],
                          "run_ok": r.get("run_ok"), "cw_status": r.get("cw_status"), "eci_status": r.get("eci_status"),
                          "eci_termination": r.get("eci_termination"), "nerm_status": r.get("nerm_status"),
                          "cw_arrived": r.get("cw_terminal_arrived"), "ref_arrived": r.get("eci_terminal_arrived"),
                          "nerm_arrived": r.get("nerm_terminal_arrived"), "cw_margin_s": r.get("cw_terminal_dwell_margin_s"),
                          "ref_margin_s": r.get("eci_terminal_dwell_margin_s"), "peak_m": r.get("cw_vs_eci_max_pos_disc_m"),
                          "ambiguity_flags": r.get("ambiguity_flags")})
    P = pd.DataFrame(props)
    P.to_csv(os.path.join(OUT, "propagations.csv"), index=False)
    pc = P.groupby(["family_id", "purpose"]).size().unstack(fill_value=0)
    pc["total"] = pc.sum(axis=1)
    pc["run_errors"] = P.groupby("family_id").run_ok.apply(lambda s: int((s != True).sum()))
    pc["ref_domain_exits"] = P.groupby("family_id").eci_termination.apply(lambda s: int((s == "altitude_floor").sum()))
    pc["ref_numerical_failures"] = P.groupby("family_id").eci_status.apply(lambda s: int((s == "failed").sum()))
    pc["cw_numerical_failures"] = P.groupby("family_id").cw_status.apply(lambda s: int((s == "failed").sum()))
    nb = P[P.models.isin(["pair", "ref"])]
    pc["nerm_vs_ref_label_differences"] = nb.groupby("family_id").apply(
        lambda g: int(((g.nerm_arrived.astype("boolean") != g.ref_arrived.astype("boolean")) & g.nerm_arrived.notna()).sum()))
    pc.reset_index().to_csv(os.path.join(OUT, "propagationcounts.csv"), index=False)
    off = pd.read_csv(os.path.join(B, "offsets.csv"))
    for tag, camp in (("predeclared", "diagnostic"), ("exploratory_supplementary", "supplementary")):
        if os.path.exists(os.path.join(B, camp, "results.jsonl")):
            analyse(tag, camp, fams, p, off, P)
    C.logrun("family_analysis", "python experiments/v2/famana.py", "completed", 0.0, f"{len(P)} search propagations analysed")


def analyse(tag, camp, fams, p, off, P):
    D = loadres(os.path.join(B, camp))
    S = pd.DataFrame(C.readjsonl(os.path.join(C.RAW, "features", f"{camp}.jsonl")))
    D = D.merge(S[["sid", "d1_peak_m", "cw_sat_total_s", "cw_final_pos_m", "cw_terminal_dwell_margin_s",
                   "cw_terminal_arrived"]].rename(columns={"cw_terminal_dwell_margin_s": "f_margin", "cw_terminal_arrived": "f_arr",
                                                           "cw_sat_total_s": "f_sat", "cw_final_pos_m": "f_final"}), on="sid", how="left")
    c = D.cw_terminal_arrived.astype(bool)
    ref = D.eci_terminal_arrived.astype(bool)
    d1 = D.d1_peak_m.astype(float)
    m = D.f_margin.astype(float)
    sat = D.f_sat.astype(float) / D.horizon_s.astype(float)
    peak = D.cw_vs_eci_max_pos_disc_m.astype(float)
    D["false_fav"] = c & ~ref
    D["false_unfav"] = ~c & ref
    D["both_arrive"] = c & ref
    D["adequate"] = (peak <= 1.0) & (c == ref)
    D["acc_combined"] = c & (m >= p["combined"]["m_min_s"]) & (d1 <= p["combined"]["D_max_m"])
    D["acc_traj_1m"] = d1 <= p["traj_1m"]["D1_max_m"]
    D["acc_outcome"] = (c & (m >= p["outcome"]["m_out_s"])) | (~c & (D.f_final.astype(float) >= p["outcome"]["f_out_m"]))
    D["ff_flag"] = c & (m < p["ff_flag"]["m_ff_s"])
    D["acc_sep_only"] = D.sep_m.astype(float) <= p["sep_only"]["s_max_m"]
    D["acc_sat_only"] = sat <= p["sat_only"]["tau"]
    D["ambiguous"] = D.ambiguous.astype(bool)
    D["nerm_ref_label_diff"] = D.nerm_terminal_arrived.astype(bool) != ref
    keep = ["sid", "meta_family_id", "meta_diag_label", "meta_boundary_status", "meta_boundary_kind", "sep_m", "cw_terminal_arrived",
            "eci_terminal_arrived", "nerm_terminal_arrived", "so_terminal_arrived", "cw_terminal_dwell_margin_s",
            "eci_terminal_dwell_margin_s", "cw_vs_eci_max_pos_disc_m", "d1_peak_m", "false_fav", "false_unfav", "both_arrive",
            "adequate", "acc_combined", "acc_traj_1m", "acc_outcome", "ff_flag", "acc_sep_only", "acc_sat_only", "ambiguous",
            "ambiguity_flags", "nerm_ref_label_diff", "eci_status", "eci_termination", "cw_status", "so_status"]
    D[keep].to_csv(os.path.join(OUT, f"diag{tag.split('_')[0]}.csv"), index=False)
    rows = []
    for f in fams:
        fid = f["id"]
        q = D[D.meta_family_id == fid]
        if tag != "predeclared" and q.empty:
            continue
        o = off[off.family_id == fid]
        res = o[o.status == "resolved"]
        rows.append({"family_id": fid, "direction": f["direction_label"], "velocity_rule": f["velocity_rule"],
                     "wn_ratio": f["wn_ratio"], "a_max": f["a_max"],
                     "n_boundaries": int(len(o[o.status != "no boundary in search range"])),
                     "n_resolved": int(len(res)), "status": ";".join(sorted(set(o.status.astype(str)))),
                     "offset_pct": ";".join(f"{100 * v:+.4f}" for v in o.get("offset_rel", pd.Series(dtype=float)).dropna()),
                     "offset_unc_pct": ";".join(f"{100 * v:.4f}" for v in o.get("offset_unc_rel", pd.Series(dtype=float)).dropna()),
                     "n_points": len(q), "n_false_fav": int(q.false_fav.sum()), "n_false_unfav": int(q.false_unfav.sum()),
                     "ff_flagged": int((q.ff_flag & q.false_fav).sum()),
                     "correct_arrivals": int(q.both_arrive.sum()), "correct_arrivals_flagged": int((q.ff_flag & q.both_arrive).sum()),
                     "combined_accepted": int(q.acc_combined.sum()), "combined_accepted_inadequate": int((q.acc_combined & ~q.adequate).sum()),
                     "outcome_accepted": int(q.acc_outcome.sum()),
                     "outcome_accepted_mismatch": int((q.acc_outcome & (q.false_fav | q.false_unfav)).sum()),
                     "outcome_referred_correct": int((~q.acc_outcome & ~(q.false_fav | q.false_unfav)).sum()),
                     "traj1m_accepted": int(q.acc_traj_1m.sum()), "sep_only_accepted": int(q.acc_sep_only.sum()),
                     "sat_only_accepted": int(q.acc_sat_only.sum()),
                     "sep_only_accepted_inadequate": int((q.acc_sep_only & ~q.adequate).sum()),
                     "sat_only_accepted_inadequate": int((q.acc_sat_only & ~q.adequate).sum()),
                     "ambiguous_points": int(q.ambiguous.sum()), "nerm_ref_label_diff": int(q.nerm_ref_label_diff.sum()),
                     "min_peak_m": float(q.cw_vs_eci_max_pos_disc_m.min()) if len(q) else np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, f"results{tag.split('_')[0]}.csv"), index=False)
    withff = T[T.n_false_fav > 0]
    allc = int((withff.ff_flagged == withff.n_false_fav).sum())
    lo, hi = cp(allc, len(withff))
    summ = {"point_set": tag, "n_families": len(T), "families_with_boundary": int((T.n_boundaries > 0).sum()),
            "families_with_resolved_offset": int((T.n_resolved > 0).sum()),
            "n_diagnostic_points": int(T.n_points.sum()), "n_false_fav_points": int(T.n_false_fav.sum()),
            "n_false_unfav_points": int(T.n_false_unfav.sum()), "families_with_false_fav": len(withff),
            "families_all_false_fav_flagged": allc, "families_all_false_fav_flagged_ci95": [lo, hi],
            "combined_accepted_points": int(T.combined_accepted.sum()),
            "combined_accepted_inadequate_points": int(T.combined_accepted_inadequate.sum()),
            "outcome_accepted_mismatch_points": int(T.outcome_accepted_mismatch.sum()),
            "correct_arrivals_flagged_points": int(T.correct_arrivals_flagged.sum()), "correct_arrivals_points": int(T.correct_arrivals.sum()),
            "n_search_propagations": int(len(P)), "n_search_run_errors": int((P.run_ok != True).sum()),
            "ambiguous_points": int(T.ambiguous_points.sum()), "nerm_ref_label_diff_points": int(T.nerm_ref_label_diff.sum()),
            "note": "designed challenge; nearby points are dependent; the interval is over families and assumes "
                    "families are exchangeable, which they are not by design"}
    C.writejson(os.path.join(OUT, f"summary{tag.split('_')[0]}.json"), summ)
    pd.set_option("display.width", 250)
    print(T.to_string())
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
