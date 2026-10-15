"""v3.1 package: code patch, full-release inventory (verified), correction manifest and a small review ZIP.

  v3_1_correction/patch/*.patch + PATCH_LOG.md   git format-patch 2aa8c44..HEAD (the v3.1 commits) with name-status/stat
  v3_1_correction/OUTPUT_INVENTORY_v31.csv       every file of the release tree (archive root, without .git/caches):
                                                 path, bytes, SHA-256, area, v3.1 status (v3.0 original unchanged /
                                                 v3.1 new / code changed in v3.1); the inventory is re-hashed after writing
                                                 and the run fails on any mismatch
  v3_1_correction/CORRECTION_MANIFEST.json       versions, baseline checks, code identity, items, deliverables with SHA-256
  ../CEAS_v3.1_Correction_Review.zip            v3_1_correction/, the changed and new code, the v3.1 tests, and the v3.0
                                                 documents needed to read them (plan, deviations, README, reports); the full
                                                 release (366 MB v3 tree + v3.1) is described by the inventory, not zipped
Every git call checks its return code; any failure stops the run before anything is written.
Usage: python code/experiments/v3_1/make_package_v31.py [--no-zip]
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

import v31paths as P

BASE_V30 = "2aa8c44"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
ZIP_NAME = "CEAS_v3.1_Correction_Review.zip"
V30_DOCS = ["v3_extension/README.md", "v3_extension/PROSPECTIVE_PLAN.json", "v3_extension/PROTOCOL_DEVIATIONS.json",
            "v3_extension/reports/IMPLEMENTATION_REPORT.md", "v3_extension/reports/NONCOMPUTATIONAL_HANDOFF.md",
            "v3_extension/claims/claim_evidence_map.csv", "v3_extension/results/numerical_resolution.csv",
            "v3_extension/raw/calibration_freeze.json", "v3_extension/raw/test_evaluation_record.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", P.REPO, *a], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            yield os.path.join(dp, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    final = os.path.join(P.V31_RAW, "baseline_verification_final.json")
    if not os.path.exists(final):
        raise SystemExit("run code/experiments/v3_1/c00_baseline.py --tag final first (end-of-cycle check of the v3.0 tree)")
    fb = json.load(open(final, encoding="utf-8"))
    if fb["v3_extension"]["n_not_identical"] or not fb["plan_sha256"]["ok"] or not fb["freeze_sha256"]["ok"] or \
            fb["test_evaluation_record_bound_hashes"]["mismatch"]:
        raise SystemExit("the final baseline check failed; the v3.0 tree is not intact")
    head = git("rev-parse", "HEAD").strip()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("commit the v3.1 code first (tracked files are modified)")
    names = git("diff", "--name-status", BASE_V30, "HEAD").splitlines()
    log = git("log", "--format=%h %ad %s", "--date=iso", f"{BASE_V30}..HEAD")
    stat = git("diff", "--stat", BASE_V30, "HEAD")
    pdir = os.path.join(P.V31, "patch")
    os.makedirs(pdir, exist_ok=True)
    for f in os.listdir(pdir):
        if f.endswith(".patch"):
            os.remove(os.path.join(pdir, f))
    git("format-patch", f"{BASE_V30}..HEAD", "-o", pdir)
    with open(os.path.join(pdir, "PATCH_LOG.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# v3.1 code patch\n\nBase: v3.0 frozen release commit `{BASE_V30}` (branch v3-extension). Head: `{head}` "
                 "(branch v3.1-corrections).\n\n## Commits\n\n```\n" + log + "```\n\n## Files (name-status against the base)\n\n```\n" +
                 "\n".join(names) + "\n```\n\n## Diff stat\n\n```\n" + stat + "```\n")
    changed_code = {"code/" + l.split("\t")[-1] for l in names}
    snap = {}
    with open(os.path.join(P.V31_RAW, "original_v3_checksums.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            snap[r["path"]] = r["sha256"]
    rows = []
    for p in walk(P.ARCH):
        rel = os.path.relpath(p, P.ARCH).replace("\\", "/")
        if rel in ("v3_extension/v3_1_correction/OUTPUT_INVENTORY_v31.csv", "v3_extension/v3_1_correction/CORRECTION_MANIFEST.json") \
                or rel.endswith(".zip"):
            continue
        h = sha(p)
        if rel.startswith("v3_extension/v3_1_correction/"):
            st = "v3.1 new"
        elif rel in changed_code:
            st = "code changed or added in v3.1"
        elif rel in snap:
            st = "v3.0 original, unchanged" if snap[rel] == h else "CHANGED v3.0 FILE"
        else:
            st = "archive file (outside the v3.0 checksum snapshot)"
        area = rel.split("/")[0] if "/" in rel else "top level"
        rows.append((rel, os.path.getsize(p), h, area, st))
    bad = [r for r in rows if r[4] == "CHANGED v3.0 FILE"]
    if bad:
        raise SystemExit(f"v3.0 files changed: {[r[0] for r in bad][:10]}")
    # correction manifest (written first, then inventoried itself; the inventory cannot list its own hash)
    sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
    import campaign as CP  # engine identity of the current tree
    rep = json.load(open(os.path.join(P.V31_RES, "exit_correction_summary.json"), encoding="utf-8"))
    eqv = json.load(open(os.path.join(P.V31_RES, "worker_equivalence_check.json"), encoding="utf-8"))
    base_start = json.load(open(os.path.join(P.V31_RAW, "baseline_verification_start.json"), encoding="utf-8"))
    deliv = [r for r in rows if r[4] == "v3.1 new"]
    man = {"release": "CEAS v3 extension, correction release v3.1", "version": "v3.1-correction-1", "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
           "brief": "CLAUDE HANDOFF: CLOSE THE VERIFIED V3 AUDIT FINDINGS (14 Oct 2026), items C00-C07",
           "baseline": {"v30_commit": BASE_V30, "plan_sha256": base_start["plan_sha256"], "freeze_sha256": base_start["freeze_sha256"],
                        "checksum_snapshot": "v3_extension/v3_1_correction/raw/original_v3_checksums.csv (taken 2026-10-14 10:57:18, before any change)",
                        "verification_start": {k: base_start[k] for k in ("checked_at", "v3_extension", "test_evaluation_record_bound_hashes")},
                        "verification_final": {k: fb[k] for k in ("checked_at", "v3_extension", "test_evaluation_record_bound_hashes")}},
           "code": {"branch": "v3.1-corrections", "head": head, "commits": log.strip().splitlines(),
                    "engine_identity_scheme": "v3.1-relpath-sha256 (cwvalid3.serialization.engine_identity over campaign.ENGINE_FILES_V31)",
                    "engine_sha_at_C01_replay": rep["engine_sha_v31"], "engine_sha_at_C01_replay_commit": rep["v31_git_head"],
                    "engine_sha_head": CP.engine_version(),
                    "engine_change_after_replay": "independent-review fixes (failure paths, t = 0 exit sentinel, provenance fields); "
                                                  "one full replay chunk re-executed on the final engine reproduces the stored replay "
                                                  "records' physics hashes (results/worker_equivalence_check.json)",
                    "equivalence_check": {k: eqv.get(k) for k in ("n_cases", "n_with_diff", "n_hash_equal", "n_sample_vs_v30",
                                                                  "n_replay_chunk_rows", "engine_sha_v31", "git_head")},
                    "files_changed_vs_v30": names},
           "work_kinds": {"rebuild_from_stored_records": ["C02 contract scan", "C03 hash reconciliation", "C04 derived columns (--rederive)",
                                                          "C05 claim map", "C06 D1 addendum",
                                                          "C07 rebuild of exit-dependent outputs (overlay of the C01 replay)", "C07 cost wording"],
                          "targeted_replay": ["C01 341 held exit records (v3.1 engine + v3.0-engine fidelity pass)", "two R5 traces",
                                              "C04 numerical audit of the same 108 development cases",
                                              "C02 worker equivalence sample (40 test cases) and one C01 replay chunk (31 cases) on the final engine"],
                          "full_campaign": "not run"},
           "independent_review": "reports/INDEPENDENT_REVIEW_LOG.md (39 findings: 30 confirmed and fixed, 9 refuted)",
           "items": {"C00": "done", "C01": "done", "C02": "done", "C03": "done", "C04": "done", "C05": "done", "C06": "done", "C07": "done"},
           "optional_E01_E03": "not started (E01 recommended; a confirmatory version needs a frozen protocol, untouched data and approval to exceed the 48 CPU-h ceiling)",
           "deliverables": [{"path": r[0], "bytes": r[1], "sha256": r[2]} for r in deliv]}
    manp = os.path.join(P.V31, "CORRECTION_MANIFEST.json")
    with open(manp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    mrel = os.path.relpath(manp, P.ARCH).replace("\\", "/")
    rows = [r for r in rows if r[0] != mrel] + [(mrel, os.path.getsize(manp), sha(manp), "v3_extension", "v3.1 new")]
    inv = os.path.join(P.V31, "OUTPUT_INVENTORY_v31.csv")
    with open(inv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256", "area", "v31_status"])
        w.writerows(sorted(rows))
    # verify the inventory against the tree
    mism = [r[0] for r in rows if sha(os.path.join(P.ARCH, r[0])) != r[2]]
    listed = {r[0] for r in rows}
    unlisted = [os.path.relpath(p, P.ARCH).replace("\\", "/") for p in walk(P.ARCH)
                if os.path.relpath(p, P.ARCH).replace("\\", "/") not in listed and not p.endswith(".zip")
                and not p.endswith("OUTPUT_INVENTORY_v31.csv")]
    if mism or unlisted:
        raise SystemExit(f"inventory verification failed: {len(mism)} hash mismatches, {len(unlisted)} unlisted files")
    print(f"inventory: {len(rows)} files ({len(deliv)} v3.1 new); patches: {len([f for f in os.listdir(pdir) if f.endswith('.patch')])}")
    if a.no_zip:
        return
    out = os.path.join(os.path.dirname(P.ARCH), ZIP_NAME)
    include = []
    for p in walk(P.V31):
        include.append(p)
    for rel in sorted(changed_code):
        p = os.path.join(P.ARCH, rel)
        if os.path.exists(p):
            include.append(p)
    for rel in V30_DOCS:
        include.append(os.path.join(P.ARCH, rel))
    seen = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in include:
            rel = os.path.relpath(p, P.ARCH).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            z.write(p, "CEAS_v3.1_Correction_Review/" + rel)
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise SystemExit("zip test failed")
    print(f"review zip: {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(seen)} files) sha256 {sha(out)}")


if __name__ == "__main__":
    main()
