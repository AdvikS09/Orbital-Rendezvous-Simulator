"""Fit and freeze the v2 estimator comparison on the ORIGINAL development scenarios only.

Development data: the same 3,839 scenarios used to freeze the v1 screen (main grid 2,160,
boundary refinement 1,154, band refinement 325, dev set 200), with v2 features from
experiments/v2/estimator_features.py and labels from the stored paired runs (inertial
reference). Nothing from the broad test, stress, band test, fresh test or boundary
families enters this script.

Features (all CW-only, metres unless stated):
  d1    A  full first-order estimate (unchanged v1 estimator)
  d1u   B  same with the unsaturated Jacobian -K throughout
  cres  C  residual-only diagnostic (double integral of the CW acceleration residual)
  sep   initial separation rho0 [m]                       (v1 baseline input)
  sat   CW saturated time / horizon [-]                   (v1 baseline input)

Rules (same objective and padding convention as v1, K = 0.8):
  traj1_<f>     target traj_ok_1  (peak CW-to-reference difference <= 1 m):  accept iff f <= t
  traj01_<f>    target traj_ok_01 (<= 0.1 m):                                accept iff f <= t
                t = K * (smallest finite f among development cases that violate the target)
  comb_<f>      target adequate (traj_ok_1 AND outcome agreement), f in {d1, d1u, cres}:
                accept iff CW arrives AND CW dwell margin >= m_min AND f <= t_traj1_<f>
                m_min = the FROZEN v1 value (380.56 s), reused, not refitted (same construction
                as the v1 combined rule, whose distance threshold is the traj1 threshold)
  v1 baselines  sep_only, sat_only (target adequate) and all v1 rules: FROZEN v1 values, unchanged
  acc<p>_<f>    descriptive acceptance-target rules, p in {25, 50, 75} %: accept iff f <= q_p, where
                q_p is the smallest development value such that at least p % of ALL development
                cases (non-finite values count as never accepted) satisfy f <= q_p

Edge cases (each recorded in the freeze file):
  * no violating development case: the threshold is NOT calibratable; the rule is stored with
    status "not_calibratable" and is not evaluated (no invented threshold)
  * non-finite feature values: excluded from the threshold minimum, counted, and always REFERRED
    by every rule (a NaN never satisfies f <= t)
  * numerical failures of the CW branch: same as non-finite (referred), counted
  * after fitting, each rule is checked to accept no violating development case

Usage:  python experiments/v2/estimator_fit.py            (writes the freeze file; refuses to overwrite)
        python experiments/v2/estimator_fit.py --check    (recompute and compare with the freeze file)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

from cwvalid.runner import load_results  # noqa: E402

DEV = ["main_grid/grid", "main_grid/refine", "main_grid/refine_bands", "screening/dev"]
K = 0.8
FEATS = {"d1": "A: full D1 (saturated Jacobian)", "d1u": "B: D1 with unsaturated Jacobian -K",
         "cres": "C: residual-only diagnostic", "sep": "initial separation", "sat": "CW saturated-time fraction"}
EST = ("d1", "d1u", "cres")
ACC_TARGETS = (0.25, 0.5, 0.75)
CODE_FILES = ["cwvalid/estimate.py", "cwvalid/estimate_v2.py", "experiments/v2/estimator_features.py",
              "experiments/v2/estimator_fit.py", "experiments/v2/v2_common.py"]


def load_features(campaign):
    p = os.path.join(C.RAW, "estimator_features", campaign.replace("/", "__") + ".jsonl")
    rows = C.read_jsonl(p)
    S = pd.DataFrame(rows).drop_duplicates("sid", keep="last")
    return S, p


def feature_frame(S):
    F = pd.DataFrame({"sid": S["sid"].values})
    for k in EST:
        F[k] = pd.to_numeric(S[f"{k}_peak_m"], errors="coerce").values
    F["sep"] = pd.to_numeric(S["sep_m"], errors="coerce").values
    F["sat"] = (pd.to_numeric(S["cw_sat_total_s"], errors="coerce") / pd.to_numeric(S["horizon_s"], errors="coerce")).values
    F["cw_arrived"] = S["cw_terminal_arrived"].fillna(False).astype(bool).values
    F["cw_margin_s"] = pd.to_numeric(S["cw_terminal_dwell_margin_s"], errors="coerce").values
    F["cw_final_m"] = pd.to_numeric(S["cw_final_pos_m"], errors="coerce").values
    F["cw_status"] = S["cw_status"].values
    F["feature_status"] = S["feature_status"].values
    for k in EST:
        F[f"{k}_status"] = S[f"{k}_status"].values
    return F


def label_frame(D):
    disc = pd.to_numeric(D["cw_vs_eci_max_pos_disc_m"], errors="coerce")
    cw = D["cw_terminal_arrived"].astype(bool)
    nl = D["eci_terminal_arrived"].astype(bool)
    L = pd.DataFrame({"sid": D["sid"].values, "disc_m": disc.values})
    L["traj_ok_1"] = (disc <= 1.0).values
    L["traj_ok_01"] = (disc <= 0.1).values
    L["outcome_ok"] = (cw == nl).values
    L["false_fav"] = (cw & ~nl).values
    L["false_unfav"] = (~cw & nl).values
    L["adequate"] = L["traj_ok_1"] & L["outcome_ok"]
    L["ref_status"] = D["eci_status"].values
    L["ref_termination"] = D["eci_termination"].values
    L["cw_status_paired"] = D["cw_status"].values
    L["label_valid"] = (D["eci_status"].isin(["ok", "terminated"]) & D["cw_status"].eq("ok") & np.isfinite(disc)).values
    return L


def load_set(campaign_rel, feature_key=None):
    D = load_results(os.path.join(C.RESULTS, campaign_rel))
    sids = [json.loads(l)["sid"] for l in open(os.path.join(C.RESULTS, campaign_rel, "scenarios.jsonl")) if l.strip()]
    D = D[D.sid.isin(set(sids))].drop_duplicates("sid", keep="last").set_index("sid").loc[sids].reset_index()
    S, fpath = load_features(feature_key or campaign_rel)
    S = S.set_index("sid").reindex(sids).reset_index()
    F = feature_frame(S)
    L = label_frame(D)
    return F, L, D, fpath


def upper(values, bad):
    v = values[bad & np.isfinite(values)]
    if v.size == 0:
        return None, "not_calibratable: no violating development case with a finite feature"
    return float(K * v.min()), "ok"


def acc_quantile(values, p):
    n = values.size
    fin = np.sort(values[np.isfinite(values)])
    need = int(np.ceil(p * n))
    if need == 0:
        return None, "not_calibratable: zero cases requested"
    if need > fin.size:
        return None, f"not_calibratable: only {fin.size} finite values for target {p:.0%} of {n}"
    return float(fin[need - 1]), "ok"


def rule_accept(F, rule):
    """Accept mask for a frozen rule dict; NaN features are never accepted."""
    if rule.get("status") != "ok":
        return None
    f = F[rule["feature"]].to_numpy(dtype=float)
    acc = np.isfinite(f) & (f <= rule["threshold"])
    if rule.get("combined"):
        m = F["cw_margin_s"].to_numpy(dtype=float)
        acc &= F["cw_arrived"].to_numpy(bool) & np.isfinite(m) & (m >= rule["m_min_s"])
    return acc


def build_rules(F, L, v1):
    rules = {}
    bad1 = ~L["traj_ok_1"].to_numpy(bool)
    bad01 = ~L["traj_ok_01"].to_numpy(bool)
    for f in FEATS:
        vals = F[f].to_numpy(dtype=float)
        for tgt, bad, pref in (("traj_ok_1", bad1, "traj1"), ("traj_ok_01", bad01, "traj01")):
            t, st = upper(vals, bad)
            rules[f"{pref}_{f}"] = {"feature": f, "target": tgt, "threshold": t, "status": st, "combined": False,
                                    "text": f"accept iff {f} <= threshold (target {tgt})",
                                    "n_dev_nonfinite": int((~np.isfinite(vals)).sum())}
    m_min = v1["params"]["combined"]["m_min_s"]
    for f in EST:
        t = rules[f"traj1_{f}"]
        rules[f"comb_{f}"] = {"feature": f, "target": "adequate", "threshold": t["threshold"], "status": t["status"],
                              "combined": True, "m_min_s": m_min,
                              "text": f"accept iff CW arrives AND CW dwell margin >= m_min_s AND {f} <= threshold (target adequate)"}
    for f in FEATS:
        vals = F[f].to_numpy(dtype=float)
        for p in ACC_TARGETS:
            t, st = acc_quantile(vals, p)
            rules[f"acc{int(p * 100)}_{f}"] = {"feature": f, "target": "traj_ok_1", "threshold": t, "status": st,
                                               "combined": False, "descriptive": True, "dev_acceptance_target": p,
                                               "text": f"accept iff {f} <= dev quantile for {p:.0%} acceptance (descriptive)"}
    # v1 baselines reproduced exactly from the v1 freeze file (not refitted)
    rules["v1_sep_only"] = {"feature": "sep", "target": "adequate", "threshold": v1["params"]["sep_only"]["s_max_m"],
                            "status": "ok", "combined": False, "frozen_v1": True, "text": "v1 frozen: accept iff rho0 <= s_max"}
    rules["v1_sat_only"] = {"feature": "sat", "target": "adequate", "threshold": v1["params"]["sat_only"]["tau"],
                            "status": "ok", "combined": False, "frozen_v1": True, "text": "v1 frozen: accept iff sat_frac <= tau"}
    # fit checks: no calibrated rule accepts a violating development case
    checks = {}
    for name, r in rules.items():
        acc = rule_accept(F, r)
        if acc is None:
            checks[name] = "not evaluated (not calibratable)"
            continue
        ok = L[r["target"]].to_numpy(bool)
        nb = int((acc & ~ok).sum())
        checks[name] = {"dev_accepted": int(acc.sum()), "dev_n": int(acc.size), "dev_violations_accepted": nb}
        if nb and r.get("frozen_v1"):
            checks[name]["note"] = "v1 frozen baseline evaluated on its own (adequate) target"
        elif nb and not r.get("descriptive"):
            # the convention cannot separate the violating cases (e.g. ties at a feature value of 0):
            # no calibrated threshold exists; record instead of inventing one
            r["attempted_threshold"] = r["threshold"]
            r["threshold"] = None
            r["status"] = (f"not_calibratable: convention threshold K*min(violating) = {r['attempted_threshold']:.6g} "
                           f"still accepts {nb} violating development cases (ties at the minimum violating value)")
            checks[name]["note"] = "rule not calibratable; not evaluated"
    # combined rules inherit the calibration status of their trajectory component
    for f in EST:
        if rules[f"traj1_{f}"]["status"] != "ok":
            rules[f"comb_{f}"].update(threshold=None, status=rules[f"traj1_{f}"]["status"])
    return rules, checks


def fit():
    v1 = json.load(open(C.FROZEN_V1))
    parts, info = [], {}
    for c in DEV:
        F, L, D, fpath = load_set(c)
        n_missing = int(F["feature_status"].isna().sum())
        if n_missing:
            raise SystemExit(f"{c}: {n_missing} scenarios have no v2 features; compute them first")
        F["source"] = c
        parts.append((F, L))
        info[c] = {"n": len(F), "features_file": os.path.relpath(fpath, C.PROJ), "features_sha256": C.sha256(fpath),
                   "n_label_invalid": int((~L.label_valid).sum()),
                   "n_feature_exceptions": int((F.feature_status != "ok").sum()),
                   "n_nonfinite": {k: int((~np.isfinite(F[k].to_numpy(dtype=float))).sum()) for k in EST}}
    F = pd.concat([p[0] for p in parts], ignore_index=True)
    L = pd.concat([p[1] for p in parts], ignore_index=True)
    rules, checks = build_rules(F, L, v1)
    a_eq = rules["traj1_d1"]["threshold"] == v1["params"]["traj_1m"]["D1_max_m"]
    return v1, F, L, rules, checks, info, a_eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    v1, F, L, rules, checks, info, a_eq = fit()
    if a.check:
        fr = json.load(open(C.FROZEN_V2))
        same = all(fr["rules"][k]["threshold"] == rules[k]["threshold"] for k in rules)
        print("recomputed thresholds identical to freeze file:", same)
        return
    if os.path.exists(C.FROZEN_V2):
        sys.exit("v2 estimator method already frozen; refusing to overwrite")
    rev = C.code_revision()
    if rev["dirty"]:
        sys.exit("commit the v2 code before freezing (working tree has uncommitted changes)")
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": "eci (inertial two-spacecraft)", "K": K,
        "features": FEATS, "rules": rules, "dev_fit_checks": checks, "development": info, "n_dev": int(len(F)),
        "n_dev_violating": {t: int((~L[t]).sum()) for t in ("traj_ok_1", "traj_ok_01", "adequate", "outcome_ok")},
        "A_threshold_equals_v1_frozen": bool(a_eq), "v1_frozen_sha256": C.sha256(C.FROZEN_V1),
        "edge_case_policy": {"no_violating_dev_case": "threshold not calibratable; rule not evaluated",
                             "nonfinite_feature": "excluded from fitting minimum; always referred",
                             "cw_numerical_failure": "features non-finite; always referred",
                             "zero_accepted": "conditional rates reported as undefined (n/a), no interval",
                             "reference_label_invalid": "excluded from denominators and listed separately"},
        "code": rev, "code_sha256": {f: C.sha256(os.path.join(C.REPO, f)) for f in CODE_FILES},
        "environment": C.environment(),
        "confirmatory_set": "results/v2/raw/fresh_test (300 scenarios, seed 12001, broad-test distribution), generated AFTER this freeze",
        "retrospective_sets": ["screening/test (400)", "screening/stress (150)", "screening/band_test (69)"],
        "primary_comparison": (
            "On the fresh confirmatory set: (i) trajectory-only screens traj1_d1 (A) vs traj1_d1u (B), traj1_cres (C), "
            "traj1_sep and traj1_sat (target: peak <= 1 m); (ii) combined rules comb_d1 vs comb_d1u, comb_cres and the "
            "frozen v1 baselines v1_sep_only, v1_sat_only (target: adequate). A method is said to improve on another "
            "only if it accepts more scenarios AND has no more incorrect acceptances. Paired acceptance differences are "
            "described with the exact McNemar test on discordant pairs (descriptive; no multiplicity correction). "
            "Zero observed incorrect acceptances are reported with their Clopper-Pearson upper bound, never as zero risk. "
            "Acceptance-target rules (acc25/50/75) are secondary and descriptive."),
    }
    C.write_json(C.FROZEN_V2, frozen)
    rec = {"estimator_v2_frozen_sha256": C.sha256(C.FROZEN_V2), "code_commit": rev["commit"], "written": frozen["frozen_at"]}
    C.write_json(os.path.join(C.FROZEN_V2_DIR, "freeze_record.json"), rec)
    pd.DataFrame([{"rule": k, **{kk: vv for kk, vv in r.items() if kk != "text"}, "text": r["text"]} for k, r in rules.items()]
                 ).to_csv(os.path.join(C.FROZEN_V2_DIR, "frozen_rules.csv"), index=False)
    C.log_run("estimator_fit", "python experiments/v2/estimator_fit.py", "completed", 0.0, rec["estimator_v2_frozen_sha256"])
    print(json.dumps({k: {"threshold": r["threshold"], "status": r["status"]} for k, r in rules.items()}, indent=1))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
