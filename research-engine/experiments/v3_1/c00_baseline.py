"""C00: immutable-baseline verification and environment record.

Checks every file listed in v3_1_correction/raw/original_v3_checksums.csv (taken 2026-10-14 10:57:18,
before any v3.1 change) against the current tree: every v3_extension file must be byte-identical
(the v3.0 tree is read-only in v3.1); code files may differ only on the v3.1 branch and every
difference is listed. Also re-verifies the hashes bound in raw/test_evaluation_record.json and the
plan/freeze hashes, and records the software environment and git revisions.

Writes v3_1_correction/raw/baseline_verification.csv, baseline_verification.json and
environment_record.json. Usage: python code/experiments/v3_1/c00_baseline.py [--tag LABEL]
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import v31paths as P

PLAN_SHA = "caa5f76c04c323d8526bc53f5017ddabca9cdd1c40ef7e22c8bce15c7a3112d8"
FREEZE_SHA = "56a5597dc7bbd5667c8bc66868ac5a204c218d15e75cf5dd8199c2a178a6b1ea"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", P.REPO, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def environment():
    import numpy
    import pandas
    import scipy
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "executable": sys.executable, "numpy": numpy.__version__, "scipy": scipy.__version__,
           "pandas": pandas.__version__, "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "cpu_count_logical": os.cpu_count(),
           "blas_note": "numpy's OpenBLAS is a DYNAMIC_ARCH build; the kernel selected at run time is not recorded",
           "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        cfg = numpy.show_config(mode="dicts")
        env["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
    except Exception as exc:  # older numpy
        env["numpy_blas"] = f"unavailable: {exc!r}"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    root = os.path.dirname(P.V3)
    snap = os.path.join(P.V31_RAW, "original_v3_checksums.csv")
    rows = []
    with open(snap, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = os.path.join(root, r["path"])
            now = sha(p) if os.path.exists(p) else None
            area = "v3_extension" if r["path"].startswith("v3_extension/") else "code"
            rows.append({"path": r["path"], "area": area, "sha256_baseline": r["sha256"], "sha256_now": now,
                         "status": "identical" if now == r["sha256"] else ("missing" if now is None else "changed")})
    v3_changed = [r for r in rows if r["area"] == "v3_extension" and r["status"] != "identical"]
    code_changed = [r for r in rows if r["area"] == "code" and r["status"] != "identical"]
    ter = json.load(open(os.path.join(P.V3, "raw", "test_evaluation_record.json"), encoding="utf-8"))
    bound = []
    for c, v in ter.get("test_results_sha256", {}).items():       # evaluate_test.py: results/<cell>_test.jsonl
        p = os.path.join(P.RES, f"{c}_test.jsonl")
        bound.append({"file": os.path.relpath(p, P.V3), "bound": v, "now": sha(p) if os.path.exists(p) else None})
    for k, v in (ter.get("outputs") or {}).items():                  # results/*.csv, raw/test_decisions.csv
        cands = [os.path.join(P.V3, "results", k), os.path.join(P.V3, "raw", k)]
        p = next((c for c in cands if os.path.exists(c)), cands[0])
        bound.append({"file": os.path.relpath(p, P.V3), "bound": v, "now": sha(p) if os.path.exists(p) else None})
    plan_now = sha(os.path.join(P.V3, "PROSPECTIVE_PLAN.json"))
    freeze_now = sha(os.path.join(P.V3, "raw", "calibration_freeze.json"))
    out = {"tag": a.tag, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "snapshot": os.path.relpath(snap, root),
           "snapshot_sha256": sha(snap), "n_files": len(rows),
           "v3_extension": {"n": sum(r["area"] == "v3_extension" for r in rows), "n_not_identical": len(v3_changed),
                            "not_identical": [r["path"] for r in v3_changed]},
           "code": {"n": sum(r["area"] == "code" for r in rows), "n_changed": len(code_changed),
                    "changed": [r["path"] for r in code_changed]},
           "plan_sha256": {"expected": PLAN_SHA, "now": plan_now, "ok": plan_now == PLAN_SHA},
           "freeze_sha256": {"expected": FREEZE_SHA, "now": freeze_now, "ok": freeze_now == FREEZE_SHA},
           "test_evaluation_record_bound_hashes": {"n": len(bound), "n_ok": sum(b["bound"] == b["now"] for b in bound),
                                                   "mismatch": [b for b in bound if b["bound"] != b["now"]]},
           "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
                   "v3_extension_head": git("rev-parse", "v3-extension"), "freeze_commit": git("rev-parse", "1a16fb1"),
                   "status_porcelain": git("status", "--porcelain").splitlines()}}
    with open(P.guard_v30(os.path.join(P.V31_RAW, f"baseline_verification_{a.tag}.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(P.guard_v30(os.path.join(P.V31_RAW, f"baseline_verification_{a.tag}.json")), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(P.guard_v30(os.path.join(P.V31_RAW, "environment_record.json")), "w", encoding="utf-8") as fh:
        json.dump(environment(), fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "git"}, indent=1))
    if v3_changed or not out["plan_sha256"]["ok"] or not out["freeze_sha256"]["ok"] or \
            out["test_evaluation_record_bound_hashes"]["mismatch"]:
        sys.exit("BASELINE VIOLATED")


if __name__ == "__main__":
    main()
