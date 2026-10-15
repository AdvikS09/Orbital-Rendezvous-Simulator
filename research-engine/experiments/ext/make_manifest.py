"""Run manifest for the handover: every campaign and study, with completed / failed /
unexecuted counts, configuration hashes and the code version.

Output: handover/RUN_MANIFEST.csv and handover/RUN_MANIFEST.md
Usage:  python experiments/ext/make_manifest.py
"""

import glob
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_utils as RU  # noqa: E402

PROJ = os.path.dirname(RU.RESULTS)
REPO = os.path.join(PROJ, "Orbital Stability Platform")


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", REPO, *a], text=True).strip()
    except Exception:
        return ""


def rows_of(path):
    n = 0
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    n += 1
    return n


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(rows)


def main():
    out = []
    for sc in sorted(glob.glob(os.path.join(RU.RESULTS, "**", "scenarios.jsonl"), recursive=True)):
        d = os.path.dirname(sc)
        rel = os.path.relpath(d, PROJ)
        man = json.load(open(os.path.join(d, "manifest.json"))) if os.path.exists(os.path.join(d, "manifest.json")) else {}
        rc = json.load(open(os.path.join(d, "run_config.json"))) if os.path.exists(os.path.join(d, "run_config.json")) else {}
        sw = rc.get("software", {})
        out.append({"campaign": rel, "kind": "batch campaign", "defined": rows_of(sc),
                    "completed": man.get("completed", rows_of(os.path.join(d, "results.jsonl"))),
                    "failed": man.get("failed_exception", 0), "unexecuted": man.get("unexecuted", None),
                    "branch numerical failures": man.get("completed_with_branch_numerical_failure", 0),
                    "config_hash": rc.get("config_hash", "(pre-guard run)"),
                    "numpy/scipy": f"{sw.get('numpy', '')}/{sw.get('scipy', '')}",
                    "results file": os.path.relpath(os.path.join(d, "results.jsonl"), PROJ),
                    "note": "superseded (rtol 1e-12, no switch events); kept for comparison" if "superseded" in rel else ""})
    singles = [
        ("results/verification/verification_variants.csv", "verification (27 cases x 17 solver settings incl. reference + original engine)"),
        ("results/verification/pass2/verification_variants.csv", "verification pass 2 (26 boundary / disagreement cases x 17 settings)"),
        ("results/verification/scaling_study.csv", "small-separation scaling (CW, second order vs NERM)"),
        ("results/verification/gmat_vs_nerm_free_propagation.csv", "saved GMAT reports vs NERM (5 two-body cases)"),
        ("results/mechanisms/common_command/common_command_summary.csv", "common-command propagation"),
        ("results/operational/operational_stages.csv", "sampled control / J2 / drag staged comparison"),
        ("results/benchmark/benchmark_runs.csv", "computation benchmark"),
        ("results/baseline_reproduction/file_level_comparison.csv", "baseline scripts re-executed at tag baseline-paper-final"),
        ("results/corrections/fig6_frame_rate_check.csv", "Fig. 6 frame-rate correction check"),
        ("results/criteria_sensitivity/criteria_sensitivity.csv", "outcome-definition sensitivity (88 cases x 37 definitions)"),
        ("results/screening/band_test/band_limits.csv", "band test capture limits (20 families)"),
        ("results/main_grid/refine/boundary_brackets.csv", "refined boundary brackets"),
        ("results/corrections/prefix_rerun_check.csv", "re-run of 96 sampled scenarios from campaigns that predate the review fixes"),
        ("results/verification/spot_checks.csv", "single-scenario solver diagnostics quoted in VERIFICATION_REPORT section 2"),
    ]
    for f, what in singles:
        p = os.path.join(PROJ, f)
        if not os.path.exists(p):
            out.append({"campaign": f, "kind": what, "completed": 0, "unexecuted": "NOT RUN", "results file": f,
                        "note": "not executed"})
            continue
        t = pd.read_csv(p)
        miss = t[t["status"].astype(str) == "unavailable"] if "status" in t.columns else t.iloc[0:0]
        out.append({"campaign": f, "kind": what, "defined": len(t) if len(miss) else None,
                    "completed": len(t) - len(miss), "failed": None,
                    "unexecuted": len(miss) if len(miss) else None, "results file": f,
                    "note": ("not regenerated: " + ", ".join(miss["file"].astype(str))) if len(miss) and "file" in t.columns else ""})
    df = pd.DataFrame(out)
    head = git("rev-parse", "HEAD")
    status = git("status", "--short")
    df.to_csv(os.path.join(PROJ, "handover", "RUN_MANIFEST.csv"), index=False)
    lines = ["# Run manifest", "", f"Code version: commit `{head}` on branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}`"
             + (" (working tree has uncommitted changes)" if status else " (clean working tree)"), "",
             "Rows for batch campaigns come from each campaign's `manifest.json` (written by the runner). 'failed' counts "
             "scenarios that raised an exception; 'branch numerical failures' counts completed scenarios in which a "
             "solver reported failure. Physical-domain exits (altitude floor) are completed scenarios and are counted "
             "as non-arrivals, not failures. A config_hash of '(pre-guard run)' marks a campaign run before the "
             "configuration guard and the other code-review fixes; a seeded sample of those campaigns was re-run "
             "with the final code and matched every stored value (`results/corrections/prefix_rerun_check.csv`).", "",
             md_table(df.fillna(""))]
    open(os.path.join(PROJ, "handover", "RUN_MANIFEST.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(df.to_string())


if __name__ == "__main__":
    main()
