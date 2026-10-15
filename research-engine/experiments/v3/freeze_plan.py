"""Freeze the prospective protocol (P04/P05) from development data only.

  python code/experiments/v3/freeze_plan.py --preview                  sample-size plan only, no files
  python code/experiments/v3/freeze_plan.py --n-cal 2500 --n-test 2500 --write
      writes v3_extension/PROSPECTIVE_PLAN.json, plans the calibration and test manifests (no
      propagation, no labels) and records hashes of the engine, the analysis code and the manifests.

Candidates: for every (cell, method, criterion) the historical v2 cutoff (where it exists) plus the
16 development quantiles 0.05..0.80 of that cell's development feature values (pilot + development
split). M = total candidate count over all families. Sample-size plan: minimum accepted counts for
b = 0..10 accepted adverse cases at level delta/M, and, for each candidate, the accepted count and
adverse fraction the development data project at the planned calibration size.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
PLAN = os.path.join(V3, "PROSPECTIVE_PLAN.json")

import analyze_dev as AD  # noqa: E402
import campaign as CP  # noqa: E402
from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS, DIST, ROOT_ENTROPY, SEED_LABEL, SPLITS, verify_unused  # noqa: E402

ALPHA, DELTA = 0.01, 0.05
RATIONALE = (
    "Chosen from development data before any calibration label existed. With M ~ 1012 candidates, delta/M ~ 4.9e-5 "
    "and alpha = 0.01, certification needs >= 987 accepted calibration cases with zero adverse cases (1247 with one, "
    "2042 with five). Projecting each candidate's development acceptance and a Jeffreys-type adverse rate "
    "(b + 0.5)/(n_acc + 1) (which does not treat zero development violations as zero risk) gives a projected "
    "certification probability for the primary D0 pair at 1 m of about 0.5 (A) and < 0.5 (B) with 2,500 calibration "
    "cases, 0.86/0.63 with 4,000 and 0.98/0.80 with 5,000; 4 of 64 families are projected to certify with 2,500, "
    "23 with 4,000 and 42 with 5,000. Measured development cost (2.34-2.54 CPU-s per continuous case, 1.93 and 1.47 "
    "per held case) projects about 27 CPU-h for 5,000 calibration + 2,500 test cases per cell, about 31 CPU-h for the "
    "whole extension, within the 48 CPU-h ceiling. The test size stays at the 2,500 default. Sizes are fixed; no "
    "sequential additions.")
CRITERIA = {"1m": 1.0, "0.1m": 0.1}
ADVERSE = ("violation", "exit", "failure", "unresolved")


def dev_tables():
    out = {}
    for cell in CELLS:
        parts = [AD.case_table(cell, sp) for sp in ("pilot", "dev")]
        out[cell] = pd.concat(parts, ignore_index=True)
    return out


def families(dev):
    fams = []
    for cell, df in dev.items():
        for m in AD.methods_for(cell):
            for crit, tol in CRITERIA.items():
                hc = AD.hist_cutoff(m, tol)
                cands = CAL.build_candidates(df[f"f_{m}"].to_numpy(float), hc)
                fams.append({"family_id": f"{cell}|{m}|{crit}", "cell": cell, "method": m, "criterion": crit, "tol_m": tol,
                             "historical_cutoff": hc,
                             "historical_cutoff_note": ("heuristic transfer of the continuous-feature number"
                                                        if m in ("A_H", "A_naive", "B_H", "C_H") else "same feature"),
                             "candidates": cands})
    return fams


def project(fams, dev, n_cal, M):
    level = DELTA / M
    rows = []
    for f in fams:
        df = dev[f["cell"]]
        v = df[f"f_{f['method']}"].to_numpy(float)
        lab = df["label_1m" if f["criterion"] == "1m" else "label_01m"].to_numpy(object)
        adv = np.isin(lab, ADVERSE)
        for c in f["candidates"]:
            acc = CAL.accept_mask(v, c["threshold"])
            na, nb = int(acc.sum()), int((acc & adv).sum())
            p_acc = na / v.size
            n_proj = int(round(p_acc * n_cal))
            r = nb / na if na else float("nan")
            bmax = -1
            for b in range(0, 60):
                if n_proj > b and CAL.ucb(b, n_proj, level) <= ALPHA:
                    bmax = b
                else:
                    if b > bmax + 1:
                        break
            p_pass = float(binom.cdf(bmax, n_proj, r)) if (bmax >= 0 and np.isfinite(r)) else (0.0 if bmax < 0 else float("nan"))
            rj = (nb + 0.5) / (na + 1.0)   # Jeffreys-type rate: does not treat zero dev adverse cases as zero risk
            p_pass_j = float(binom.cdf(bmax, n_proj, rj)) if bmax >= 0 else 0.0
            rows.append({"family_id": f["family_id"], "cell": f["cell"], "method": f["method"], "criterion": f["criterion"],
                         "cid": c["cid"], "source": c["source"], "threshold": c["threshold"], "dev_n": int(v.size),
                         "dev_accepted": na, "dev_adverse_accepted": nb, "dev_acceptance": p_acc,
                         "projected_cal_accepted": n_proj, "max_adverse_for_certificate": bmax,
                         "projected_pass_prob_at_dev_rate": p_pass, "jeffreys_rate": rj,
                         "projected_pass_prob_at_jeffreys_rate": p_pass_j})
    return pd.DataFrame(rows)


def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        return {"commit": c, "dirty": bool(d), "dirty_files": d.splitlines()[:50]}
    except Exception as exc:
        return {"error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if os.path.exists(PLAN) and a.write:
        sys.exit("PROSPECTIVE_PLAN.json exists; record changes as protocol deviations instead of rewriting it")
    dev = dev_tables()
    fams = families(dev)
    M = sum(len(f["candidates"]) for f in fams)
    level = DELTA / M
    nmin = {b: CAL.n_min_for_b(b, ALPHA, DELTA, M) for b in range(0, 11)}
    proj = project(fams, dev, a.n_cal, M)
    best = (proj[proj.projected_pass_prob_at_jeffreys_rate >= 0.5].sort_values("projected_cal_accepted", ascending=False)
            .groupby("family_id").head(1).set_index("family_id"))
    summary = []
    for f in fams:
        b = best.loc[f["family_id"]] if f["family_id"] in best.index else None
        summary.append({"family_id": f["family_id"], "n_candidates": len(f["candidates"]),
                        "projected_selected_threshold": None if b is None else float(b.threshold),
                        "projected_cal_accepted": None if b is None else int(b.projected_cal_accepted),
                        "projected_coverage": None if b is None else float(b.dev_acceptance)})
    print(f"families {len(fams)}, M = {M}, delta/M = {level:.3e}; n_min(b) = {nmin}")
    pp = proj.groupby("family_id").projected_pass_prob_at_jeffreys_rate.max()
    print("families with a candidate projected to certify with prob >= 0.5 at the Jeffreys rate:", int((pp >= 0.5).sum()),
          "of", len(pp), "; >= 0.8:", int((pp >= 0.8).sum()))
    print(pd.DataFrame(summary).to_string())
    os.makedirs(os.path.join(V3, "results", "plan"), exist_ok=True)
    proj.to_csv(os.path.join(V3, "results", "plan", f"sample_size_projection_ncal{a.n_cal}.csv"), index=False)
    if a.preview or not a.write:
        return
    # ---- freeze ------------------------------------------------------------------------------
    unused = verify_unused()
    if unused["root_entropy_in_archive_seeds"] or unused["first_draw_collisions"]:
        sys.exit(f"seed check failed: {unused}")
    man = {}
    for cell in CELLS:
        for split, n in (("cal", a.n_cal), ("test", a.n_test)):
            CP.cmd_plan(cell, split, n)
            sm = json.load(open(os.path.join(CP.MAN, f"{cell}_{split}_summary.json")))
            man[f"{cell}_{split}"] = {"n": n, "sha256": sm["manifest_sha256"], "rng": sm["rng"]}
    code_files = CP.ENGINE_FILES + [os.path.join(REPO, p) for p in (
        "cwvalid3/calibration.py", "experiments/v3/freeze_plan.py", "experiments/v3/analyze_dev.py",
        "experiments/v3/v3data.py", "experiments/v3/calibrate.py", "experiments/v3/evaluate_test.py")]
    plan = {
        "title": "CEAS v3 extension: prospective calibration and transfer protocol",
        "frozen_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_note": "Local process record. It is not proof of public preregistration or third-party independence.",
        "central_question": ("Can saturation-aware first-order trajectory-error screening offer reliably calibrated acceptance, "
                             "and does its advantage survive realistic changes in command timing and controller structure "
                             "within clearly specified rendezvous domains?"),
        "hypotheses": {
            "H1_primary": ("D0, criterion 1 m: the calibrated A rule (saturated Jacobian) and the calibrated B rule "
                           "(unsaturated Jacobian, same path/residual/integration/protocol) are compared at matched "
                           "calibrated risk. Primary contrast: paired test coverage difference A - B and net useful "
                           "acceptance, with paired bootstrap 95% intervals and exact McNemar. No direction is assumed."),
            "H2_sampled": ("D3 (1 s) and D4 (10 s): the held-command first variation A_H is the correct derivative "
                           "(verified in P06); compare A_H with A_naive, B_H and C_H at matched calibrated risk "
                           "and in prediction error; secondary."),
            "H3_transfer": ("D1, D2 (altitude), D3, D4 (command timing), D5 (controller structure): realised test "
                            "violation rates and coverage of T1 (D0 cutoff unchanged), T2 (similarity image, D1/D2 only) "
                            "and T3 (target-recalibrated). Descriptive for T1/T2 (no certificate); T3 inherits the "
                            "calibration statement of its own family."),
        },
        "primary_claim": "D0 | A vs B | 1 m | paired coverage and net useful acceptance on the D0 test set",
        "secondary_claims": ["0.1 m criterion (same protocol)", "arrival-label agreement among accepted cases (descriptive)",
                             "held-command and controller cells", "transfer modes T1/T2/T3"],
        "domains": {c: {**v, "distribution": DIST} for c, v in CELLS.items()},
        "distribution_definition": "cwvalid3/domains.py (velocity generated with the cell's own n; template SETS['test'])",
        "features": {"continuous": {"A": "d1_peak_m", "B": "d1u_peak_m", "C": "cres_peak_m", "sep": "sep_m",
                                    "sat": "CW saturated time / horizon"},
                     "held": {"A_H": "A_H_peak_m (0.5 s grid)", "B_H": "B_H_peak_m", "A_naive": "A_naive_peak_m",
                              "C_H": "C_H_peak_m", "sep": "sep_m", "sat": "held saturated time / horizon"}},
        "candidate_rule": "historical v2 cutoff (if any) + 16 development quantiles 0.05..0.80 (pilot + dev), dedup, accept iff feature <= t",
        "families": fams,
        "multiplicity": {"M_total_candidates": M, "delta": DELTA, "alpha": ALPHA, "per_candidate_level": level,
                         "families": len(fams), "allocation": "single family-wise budget over all cells, methods and criteria"},
        "selection_rule": "max calibration acceptance among U <= alpha; ties -> smaller threshold, then candidate order; none -> abstain",
        "sample_sizes": {"pilot_per_cell": 40, "dev_per_cell": 300, "cal_per_cell": a.n_cal, "test_per_cell": a.n_test,
                         "rationale": RATIONALE,
                         "n_min_accepted_by_b": nmin, "stopping_rule": "fixed sizes; no sequential additions"},
        "sample_size_projection_file": f"results/plan/sample_size_projection_ncal{a.n_cal}.csv",
        "projected_selection": summary,
        "transfer_modes": {"T1": "D0 selected cutoff unchanged (metres)", "T2": "D0 cutoff x R_target/R_D0 for D1, D2 only",
                           "T3": "target-calibrated (own family)", "T1_heuristic": "D0 A/B cutoff applied to A_H/A_naive/B_H"},
        "matched_workload": "development acceptance targets 25/50/75 % (dev quantiles 0.25/0.50/0.75) per method, compared on test",
        "rng": {"label": SEED_LABEL, "root_entropy": str(ROOT_ENTROPY), "bit_generator": "PCG64",
                "spawn_key": "(cell index, split index)", "splits": SPLITS, "unused_check": unused},
        "manifests": man,
        "labeling": {"band_factor": CAL.BAND_FACTOR, "band_floor_m": CAL.BAND_FLOOR_M, "adverse": list(ADVERSE),
                     "dwell_ambiguity_s": CAL.DWELL_AMBIGUITY_S, "reference": "eci", "crosscheck": "nerm",
                     "rule": "methods/calibration_spec.md section 2"},
        "exclusions": "none: every planned case resolves to a status; unavailable features are referrals; post-acceptance "
                      "reference failures stay accepted with unresolved truth",
        "compute_budget": {"ceiling_cpu_h": 48.0, "measured_cpu_s_per_case_dev": {
            c: float(dev[c].cpu_s.mean()) for c in CELLS},
            "projected_cpu_h_cal_test": float(sum(dev[c].cpu_s.mean() * (a.n_cal + a.n_test) for c in CELLS) / 3600.0)},
        "implementation": {"git": git_commit(), "code_sha256": {os.path.relpath(p, REPO).replace("\\", "/"): L.text_sha256(p)
                                                                 for p in code_files},
                           "engine_sha": CP.engine_version()},
        "stress_and_diagnostics": ("transfer_cohort: the 300 D0 development draws re-run as D3, D4 and D5 configurations "
                                   "(paired mechanism diagnostics, exploratory); failure atlas uses all splits "
                                   "descriptively with recorded selection rules; no stress data enter risk tables"),
        "protocol_deviations": [],
    }
    json.dump(L.jsonable(plan), open(PLAN, "w"), indent=1)
    print("written", PLAN, L.file_sha256(PLAN))


if __name__ == "__main__":
    main()
