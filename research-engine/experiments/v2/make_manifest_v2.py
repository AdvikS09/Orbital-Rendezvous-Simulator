"""Input/output manifest with SHA-256 hashes and the run log for the v2 extension.

Writes
  handover_v2/IO_MANIFEST.csv   every v2 output file (raw and derived), the v1 inputs the v2 analyses read,
                                every tracked code file, and the v2 reports: path, role, bytes, SHA-256
  handover_v2/RUN_LOG_v2.md     every recorded v2 command (results/v2/raw/run_log.jsonl) with time, status,
                                wall time and code commit, plus environment and test-log summary
Usage: python experiments/v2/make_manifest_v2.py
"""

import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

HV2 = os.path.join(C.PROJ, "handover_v2")
V1_INPUTS = ["main_grid/grid", "main_grid/refine", "main_grid/refine_bands", "screening/dev", "screening/test",
             "screening/stress", "screening/band_test", "mechanisms/horizon2"]


def rows_for(path, role):
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        for f in sorted(files):
            p = os.path.join(root, f)
            out.append({"path": os.path.relpath(p, C.PROJ).replace("\\", "/"), "role": role,
                        "bytes": os.path.getsize(p), "sha256": C.sha256(p)})
    return out


def main():
    rows = []
    rows += rows_for(os.path.join(C.RAW), "v2 raw output")
    rows += rows_for(os.path.join(C.DERIVED), "v2 derived (tables, figures, analysis)")
    # v1 files read by v2: every v1 campaign's scenario and result files (read by the fresh-test overlap
    # check and the status census), the development/retrospective campaign records, the v1 frozen rules,
    # the superseded v1 outcome table and the v1 equation check
    import glob
    seen = set()
    v1_files = []
    for p in sorted(glob.glob(os.path.join(C.RESULTS, "**", "*.jsonl"), recursive=True)
                    + sorted(glob.glob(os.path.join(C.RESULTS, "**", "*.json"), recursive=True))):
        rel = os.path.relpath(p, C.RESULTS).replace("\\", "/")
        if rel.startswith("v2/") or "/histories" in rel or rel.startswith("baseline_reproduction/engine_baseline"):
            continue
        if os.path.basename(p) in ("scenarios.jsonl", "results.jsonl", "run_config.json", "manifest.json",
                                   "screen_features.jsonl", "set_definition.json", "screening_frozen.json"):
            v1_files.append(p)
    v1_files += [os.path.join(C.RESULTS, "tables", "tab_model_outcomes_refined.csv"),
                 os.path.join(C.RESULTS, "reference_equation_check.md")]
    for p in v1_files:
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        rows.append({"path": os.path.relpath(p, C.PROJ).replace("\\", "/"), "role": "v1 input read by v2",
                     "bytes": os.path.getsize(p), "sha256": C.sha256(p)})
    tracked = subprocess.check_output(["git", "-C", C.REPO, "ls-files"], text=True).split()
    for f in tracked:
        p = os.path.join(C.REPO, f)
        if os.path.isfile(p):
            rows.append({"path": "Orbital Stability Platform/" + f, "role": "code (git-tracked)", "bytes": os.path.getsize(p),
                         "sha256": C.sha256(p)})
    # v2 reports, logs and provenance records (RUN_LOG_v2.md is rewritten below, so it is hashed after writing)
    for root, _, files in os.walk(HV2):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, C.PROJ).replace("\\", "/")
            if f in ("IO_MANIFEST.csv", "RUN_LOG_v2.md"):
                continue
            rows.append({"path": rel, "role": "v2 report / log / provenance", "bytes": os.path.getsize(p), "sha256": C.sha256(p)})
    pd.DataFrame(rows).to_csv(os.path.join(HV2, "IO_MANIFEST.csv"), index=False)
    # run log
    log = [json.loads(l) for l in open(os.path.join(C.RAW, "run_log.jsonl"), encoding="utf-8") if l.strip()]
    env = C.environment()
    lines = ["# Run log (v2 extension)", "",
             f"Code at manifest time: commit `{C.code_revision()['commit']}` on branch `{C.code_revision()['branch']}`"
             + (" (tracked files have uncommitted changes)" if C.code_revision()["dirty"] else " (no uncommitted changes to tracked files)") + ".", "",
             f"Environment: Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, pandas {env['pandas']}; "
             f"{env['platform']}; {env['processor']}; {env['cpu_count']} logical CPUs. Parallel runs used 6 or 7 worker processes.", "",
             "Every command below was run from `Orbital Stability Platform/` with the project's virtual-environment Python. "
             "Wall times are elapsed seconds of the recorded step (0 where the step did not time itself).", "",
             "| time | study | command | status | wall [s] | commit | note |", "|---|---|---|---|---|---|---|"]
    for r in log:
        lines.append(f"| {r['time']} | {r['study']} | `{r['command']}` | {r['status']} | {r['wall_s']} | "
                     f"{r['code']['commit'][:7]}{'+' if r['code'].get('dirty') else ''} | {str(r.get('note', '')).replace('|', '/')[:160]} |")
    tl = os.path.join(HV2, "test_logs")
    if os.path.isdir(tl):
        lines += ["", "## Test logs", ""]
        for f in sorted(os.listdir(tl)):
            txt = open(os.path.join(tl, f), encoding="utf-8", errors="replace").read().strip().splitlines()
            lines.append(f"- `handover_v2/test_logs/{f}`: {txt[-1] if txt else '(empty)'}")
    lines += ["", "`+` after a commit hash marks a run made while the working tree had uncommitted changes (for the earliest "
              "runs this flag also counted new, not yet committed v2 scripts). All v2 scripts were committed afterwards; "
              "the git history shows their exact content at each commit. The estimator freeze and the fresh-test "
              "manifest were made from a clean tree (commit ac31e80). The commit shown is the one current when the step "
              "finished and was logged; long-running steps loaded their code when they started. cwvalid/estimate_v2.py "
              "has a single version (written once, never edited, committed in 3e96825), so every feature value was "
              "computed with that version; cwvalid/estimate.py (variant A) is unchanged since the v1 commit faee43e."]
    rl = os.path.join(HV2, "RUN_LOG_v2.md")
    open(rl, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    rows.append({"path": "handover_v2/RUN_LOG_v2.md", "role": "v2 report / log / provenance", "bytes": os.path.getsize(rl),
                 "sha256": C.sha256(rl)})
    pd.DataFrame(rows).to_csv(os.path.join(HV2, "IO_MANIFEST.csv"), index=False)
    print(f"manifest rows: {len(rows)}; log entries: {len(log)}")


if __name__ == "__main__":
    main()
