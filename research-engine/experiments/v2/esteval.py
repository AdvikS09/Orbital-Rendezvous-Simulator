import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C
import estfit as EF

OUT = os.path.join(C.DERIVED, "estimator")
FLOOR = 1e-6
NEARS = 600.0
SETS = {
    "dev": ("dev", None, "in-sample (fitted)"),
    "test": ("screening/test", "screening/test", "retrospective"),
    "stress": ("screening/stress", "screening/stress", "retrospective"),
    "band_test": ("screening/bandtest", "screening/bandtest", "retrospective"),
    "fresh": ("v2/raw/freshtest", "v2/raw/freshtest", "confirmatory"),
}


def cp(k, n):
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def metrics(acc, L, target, featfinite):
    acc = np.asarray(acc, bool)
    valid = L["label_valid"].to_numpy(bool)
    ok = L[target].to_numpy(bool)
    a, o = acc[valid], ok[valid]
    disc = L["disc_m"].to_numpy(float)[valid]
    ff = L["false_fav"].to_numpy(bool)[valid]
    fu = L["false_unfav"].to_numpy(bool)[valid]
    term = L["ref_termination"].astype(str).to_numpy()
    rstat = L["ref_status"].astype(str).to_numpy()
    out = {"n_total": int(acc.size), "n_label_invalid": int((~valid).sum()), "n_evaluated": int(valid.sum()),
           "accepted": int(a.sum()), "referred": int((~a).sum()),
           "incorrect_accepted": int((a & ~o).sum()), "unnecessary_referrals": int((~a & o).sum()),
           "violating_total": int((~o).sum()), "satisfying_total": int(o.sum()),
           "traj_violations_accepted": int((a & (disc > 1.0)).sum()), "traj_violations_total": int((disc > 1.0).sum()),
           "ff_accepted": int((a & ff).sum()), "ff_total": int(ff.sum()),
           "fu_accepted": int((a & fu).sum()), "fu_total": int(fu.sum()),
           "feature_nonfinite": int((~np.asarray(featfinite, bool)).sum()),
           "ref_domain_exits": int((term == "altitude_floor").sum()),
           "ref_domain_exits_accepted": int(((term == "altitude_floor") & acc).sum()),
           "ref_numerical_failures": int((rstat == "failed").sum())}
    for name, k, n in (("accept_rate", out["accepted"], out["n_evaluated"]),
                       ("incorrect_rate", out["incorrect_accepted"], out["accepted"]),
                       ("unnecessary_rate", out["unnecessary_referrals"], out["referred"]),
                       ("violators_accepted_rate", out["incorrect_accepted"], out["violating_total"])):
        lo, hi = cp(k, n)
        out[name] = (k / n) if n else float("nan")
        out[name + "_lo"], out[name + "_hi"] = lo, hi
    return out


def load(setname):
    s, featkey, status = SETS[setname]
    if setname == "dev":
        parts = []
        for c in EF.DEV:
            F, L, D, _ = EF.loadset(c)
            parts.append((F, L, D.assign(dev_source=c)))
        F = pd.concat([p[0] for p in parts], ignore_index=True)
        L = pd.concat([p[1] for p in parts], ignore_index=True)
        D = pd.concat([p[2] for p in parts], ignore_index=True)
    else:
        F, L, D, _ = EF.loadset(s, featkey)
    return F, L, D, status


def regime(F, D, L):
    sat = pd.to_numeric(D["cw_sat_total_s"], errors="coerce").fillna(0).to_numpy() > 0
    cwm = pd.to_numeric(D["cw_terminal_dwell_margin_s"], errors="coerce").to_numpy()
    rfm = pd.to_numeric(D["eci_terminal_dwell_margin_s"], errors="coerce").to_numpy()
    near = (np.isfinite(cwm) & (np.abs(cwm) <= NEARS)) | (np.isfinite(rfm) & (np.abs(rfm) <= NEARS))
    near |= ~L["outcome_ok"].to_numpy(bool)
    return np.where(near, "near_boundary", np.where(sat, "saturated", "unsaturated"))


def ratiostats(F, L, reg, setname):
    rows = []
    meas = L["disc_m"].to_numpy(float)
    valid = L["label_valid"].to_numpy(bool)
    for est in EF.EST:
        e = F[est].to_numpy(float)
        for rg in ("unsaturated", "saturated", "near_boundary", "all"):
            m = valid & ((reg == rg) if rg != "all" else True)
            fin = m & np.isfinite(e)
            use = fin & (meas >= FLOOR)
            r = e[use] / meas[use]
            rows.append({"set": setname, "estimator": est, "regime": rg, "n": int(m.sum()),
                         "n_estimate_failed": int((m & ~np.isfinite(e)).sum()),
                         "n_below_floor": int((fin & (meas < FLOOR)).sum()), "n_ratio": int(use.sum()),
                         "median_ratio": float(np.median(r)) if r.size else np.nan,
                         "p01_ratio": float(np.quantile(r, 0.01)) if r.size else np.nan,
                         "p99_ratio": float(np.quantile(r, 0.99)) if r.size else np.nan,
                         "min_ratio": float(r.min()) if r.size else np.nan, "max_ratio": float(r.max()) if r.size else np.nan,
                         "within_factor_1.1": float(np.mean((r >= 1 / 1.1) & (r <= 1.1))) if r.size else np.nan,
                         "within_factor_2": float(np.mean((r >= 0.5) & (r <= 2))) if r.size else np.nan,
                         "n_over_10pct": int((r > 1.1).sum()), "n_under_10pct": int((r < 1 / 1.1).sum()),
                         "spearman": float(pd.Series(e[use]).corr(pd.Series(meas[use]), method="spearman")) if r.size > 2 else np.nan})
    return rows


def main():
    fr = json.load(open(C.FROZENV2))
    rules = {k: r for k, r in fr["rules"].items()}
    h = C.sha256(C.FROZENV2)
    os.makedirs(OUT, exist_ok=True)
    perf, ratios = [], []
    for setname in SETS:
        s = SETS[setname][0]
        if setname != "dev" and not os.path.exists(os.path.join(C.RESULTS, s, "results.jsonl")):
            print(f"[evaluate] {setname}: no results yet, skipped")
            continue
        F, L, D, status = load(setname)
        miss = int(F["feature_status"].isna().sum())
        if miss:
            raise SystemExit(f"{setname}: {miss} scenarios have no v2 features")
        reg = regime(F, D, L)
        dec = pd.DataFrame({"sid": F["sid"], "regime": reg})
        for c in ("disc_m", "traj_ok_1", "traj_ok_01", "outcome_ok", "false_fav", "false_unfav", "adequate",
                  "label_valid", "ref_status", "ref_termination"):
            dec[c] = L[c].values
        for c in ("d1", "d1u", "cres", "sep", "sat", "cw_arrived", "cw_margin_s", "cw_final_m", "cw_status",
                  "d1_status", "d1u_status", "cres_status"):
            dec[c] = F[c].values
        dec["ref_margin_s"] = pd.to_numeric(D["eci_terminal_dwell_margin_s"], errors="coerce").values
        dec["cw_sat_total_s"] = pd.to_numeric(D["cw_sat_total_s"], errors="coerce").values
        for col in ("meta_axis", "meta_velocity", "meta_in_dev_domain", "dev_source"):
            if col in D:
                dec[col] = D[col].values
        for name, r in rules.items():
            acc = EF.ruleaccept(F, r)
            if acc is None:
                perf.append({"set": setname, "set_status": status, "rule": name, "target": r["target"],
                             "rule_status": r["status"]})
                continue
            dec[f"accept__{name}"] = acc
            m = metrics(acc, L, r["target"], np.isfinite(F[r["feature"]].to_numpy(float)))
            perf.append({"set": setname, "set_status": status, "rule": name, "target": r["target"],
                         "feature": r["feature"], "threshold": r["threshold"], "combined": r.get("combined", False),
                         "descriptive": r.get("descriptive", False), "rule_status": "ok", **m})
            if "meta_in_dev_domain" in dec and setname in ("test", "fresh"):
                for flag in (True, False):
                    idx = dec["meta_in_dev_domain"].astype(bool).to_numpy() == flag
                    mm = metrics(acc[idx], L[idx], r["target"], np.isfinite(F[r["feature"]].to_numpy(float))[idx])
                    perf.append({"set": f"{setname}:{'in' if flag else 'outside'}_dev_domain", "set_status": status,
                                 "rule": name, "target": r["target"], "feature": r["feature"], "threshold": r["threshold"],
                                 "combined": r.get("combined", False), "descriptive": r.get("descriptive", False),
                                 "rule_status": "ok", **mm})
        dec.to_csv(os.path.join(OUT, f"decisions{setname.replace('_', '')}.csv"), index=False)
        ratios += ratiostats(F, L, reg, setname)
        print(f"[evaluate] {setname} ({status}): {len(F)} scenarios, {int((~L.label_valid).sum())} invalid labels")
    P = pd.DataFrame(perf)
    P.to_csv(os.path.join(OUT, "performancelong.csv"), index=False)
    pairs = [("traj1_d1", o) for o in ("traj1_d1u", "traj1_cres", "traj1_sep", "traj1_sat")]
    pairs += [("comb_d1", o) for o in ("comb_d1u", "comb_cres", "v1_sep_only", "v1_sat_only")]
    pr = []
    for setname in SETS:
        p = os.path.join(OUT, f"decisions{setname.replace('_', '')}.csv")
        if not os.path.exists(p):
            continue
        dd = pd.read_csv(p)
        dd = dd[dd.label_valid.astype(bool)]
        for a, b in pairs:
            ca, cb = f"accept__{a}", f"accept__{b}"
            if ca not in dd or cb not in dd:
                continue
            xa, xb = dd[ca].astype(bool), dd[cb].astype(bool)
            n10, n01 = int((xa & ~xb).sum()), int((~xa & xb).sum())
            pv = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else float("nan")
            pr.append({"set": setname, "rule_a": a, "rule_b": b, "accepted_a": int(xa.sum()), "accepted_b": int(xb.sum()),
                       "a_only": n10, "b_only": n01, "mcnemar_exact_p": pv, "n": int(len(dd))})
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "paired.csv"), index=False)
    ac = []
    for setname in SETS:
        p = os.path.join(OUT, f"decisions{setname.replace('_', '')}.csv")
        if not os.path.exists(p):
            continue
        dd = pd.read_csv(p)
        dd = dd[dd.label_valid.astype(bool)]
        t1, cb = dd["accept__traj1_d1"].astype(bool), dd["accept__comb_d1"].astype(bool)
        onlyt = dd[t1 & ~cb]
        mism = ~dd.outcome_ok.astype(bool)
        ac.append({"set": setname, "n": len(dd), "traj1_d1_accepted": int(t1.sum()), "comb_d1_accepted": int(cb.sum()),
                   "removed_by_arrival_conditions": len(onlyt),
                   "removed_that_were_adequate": int(onlyt.adequate.astype(bool).sum()),
                   "removed_that_were_outcome_mismatches": int((~onlyt.outcome_ok.astype(bool)).sum()),
                   "removed_cw_nonarrivals": int((~onlyt.cw_arrived.astype(bool)).sum()),
                   "removed_cw_margin_below_m_min": int((onlyt.cw_arrived.astype(bool)).sum()),
                   "outcome_mismatches_in_set": int(mism.sum()),
                   "outcome_mismatches_accepted_by_traj1_d1": int((t1 & mism).sum()),
                   "removed_sids": ";".join(onlyt.sid.astype(str))})
    pd.DataFrame(ac).to_csv(os.path.join(OUT, "arrival.csv"), index=False)
    pd.DataFrame(ratios).to_csv(os.path.join(OUT, "ratiostats.csv"), index=False)
    C.writejson(os.path.join(OUT, "evaluation.json"), {"estimator_v2_frozen_sha256": h, "code": C.coderev(),
                                                              "sets": {k: v[2] for k, v in SETS.items()}})
    C.logrun("estimator_evaluate", "python experiments/v2/esteval.py", "completed", 0.0, f"{len(P)} performance rows")
    show = P[(P.rule.isin(["traj1_d1", "traj1_d1u", "traj1_cres", "traj1_sep", "traj1_sat", "comb_d1", "comb_d1u",
                           "comb_cres", "v1_sep_only", "v1_sat_only"])) & (~P.set.str.contains(":"))]
    pd.set_option("display.width", 250)
    print(show[["set", "rule", "accepted", "n_evaluated", "incorrect_accepted", "unnecessary_referrals", "referred",
                "ff_accepted", "fu_accepted", "feature_nonfinite"]].to_string())


if __name__ == "__main__":
    main()
