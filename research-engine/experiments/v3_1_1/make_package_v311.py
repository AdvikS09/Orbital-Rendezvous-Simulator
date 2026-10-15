"""v3.1.1 packaging: baseline checks, patch, manifest, inventory and the compact review ZIP.

1. Baseline (read-only): every v3_extension file in the v3.0 checksum snapshot (v3_1_correction/raw/original_v3_checksums.csv)
   and every row of the v3.1 release inventory (OUTPUT_INVENTORY_v31.csv) outside code/ must still match its SHA-256, and no
   file may have appeared in v3_extension outside v3_1_1_followup. Code rows of the v3.1 inventory are compared with the files
   changed by v3.1.1 (git diff v3.1..HEAD); every other code row must match. Results go to results/baseline_check_v311.json.
   The run stops on any mismatch.
2. patch/: git format-patch v3.1..HEAD (the code must be committed).
3. CORRECTION_MANIFEST_v3.1.1.json (engine identity, commits, checks, item status), then OUTPUT_INVENTORY_v311.csv over
   v3_1_1_followup (path, bytes, SHA-256; it cannot list itself), verified after writing.
4. CEAS_v3.1.1_Followup_Review.zip next to the v3.1 review ZIP (CEAS_v3_work/): the v3_1_1_followup folder, the complete code
   tree at HEAD (git archive, including the exact-line fixtures) and ZIP_README.md; ZIP_INVENTORY.csv lists every other
   member's SHA-256. The ZIP is a delta bundle: it does not contain the v3.0/v3.1 archive data.
Usage: python make_package_v311.py
"""

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile

import v311paths as P

sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
ZIP = os.path.join(os.path.dirname(P.ARCH), "CEAS_v3.1.1_Followup_Review.zip")
BASE_ZIP = os.path.join(os.path.dirname(P.ARCH), "CW_Rendezvous_CEAS_Extension_v3.zip")
V31_ZIP = os.path.join(os.path.dirname(P.ARCH), "CEAS_v3.1_Correction_Review.zip")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", P.REPO, *a], capture_output=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.decode(errors='replace')}")
    return r.stdout


def rel(p, root=P.ARCH):
    return os.path.relpath(p, root).replace("\\", "/")


def walk(root):
    for d, _, fs in os.walk(root):
        for f in fs:
            yield os.path.join(d, f)


def baseline():
    changed = set(git("diff", "--name-only", "v3.1..HEAD").decode().split())
    snap = list(csv.DictReader(open(os.path.join(P.V31, "raw", "original_v3_checksums.csv"), encoding="utf-8")))
    inv = list(csv.DictReader(open(os.path.join(P.V31, "OUTPUT_INVENTORY_v31.csv"), encoding="utf-8")))
    bad_snap = [r["path"] for r in snap if r["path"].startswith("v3_extension/")
                and (not os.path.exists(os.path.join(P.ARCH, r["path"])) or sha(os.path.join(P.ARCH, r["path"])) != r["sha256"])]
    bad_inv, code_changed, code_same = [], [], 0
    for r in inv:
        p = os.path.join(P.ARCH, r["path"])
        ok = os.path.exists(p) and sha(p) == r["sha256"]
        if r["path"].startswith("code/"):
            if r["path"][len("code/"):] in changed:
                code_changed.append(r["path"])
            elif not ok:
                bad_inv.append(r["path"])
            else:
                code_same += 1
        elif not ok:
            bad_inv.append(r["path"])
    listed = {r["path"] for r in inv} | {"v3_extension/v3_1_correction/OUTPUT_INVENTORY_v31.csv"}
    new_files = [rel(p) for p in walk(P.V3) if not P.P31.inside(p, P.V311) and rel(p) not in listed and not p.endswith(".zip")]
    out = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "v30_snapshot": {"source": "v3_extension/v3_1_correction/raw/original_v3_checksums.csv",
                            "n_v3_extension_rows": sum(r["path"].startswith("v3_extension/") for r in snap), "n_mismatch": len(bad_snap),
                            "mismatch": bad_snap},
           "v31_inventory": {"source": "v3_extension/v3_1_correction/OUTPUT_INVENTORY_v31.csv",
                             "sha256": sha(os.path.join(P.V31, "OUTPUT_INVENTORY_v31.csv")), "n_rows": len(inv),
                             "n_non_code_rows_matching": sum(1 for r in inv if not r["path"].startswith("code/")) - len([x for x in bad_inv if not x.startswith("code/")]),
                             "n_code_rows_unchanged": code_same, "code_rows_changed_by_v311": sorted(code_changed),
                             "n_mismatch_unexpected": len(bad_inv), "mismatch_unexpected": bad_inv},
           "v31_manifest_sha256": sha(os.path.join(P.V31, "CORRECTION_MANIFEST.json")),
           "new_files_in_v3_extension_outside_v3_1_1_followup": new_files,
           "files_changed_v31_to_head": sorted(changed)}
    os.makedirs(P.V311_RES, exist_ok=True)
    with open(P.guard(os.path.join(P.V311_RES, "baseline_check_v311.json")), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    if bad_snap or bad_inv or new_files:
        raise SystemExit(f"baseline check failed: {len(bad_snap)} v3.0, {len(bad_inv)} v3.1 mismatches, {len(new_files)} new files")
    return out


def main():
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("commit the code first (tracked files modified)")
    base = baseline()
    print("baseline ok:", {k: v for k, v in base["v31_inventory"].items() if k != "code_rows_changed_by_v311"})
    pd = os.path.join(P.V311, "patch")
    if os.path.isdir(pd):
        for f in os.listdir(pd):
            if f.endswith(".patch"):
                os.remove(P.guard(os.path.join(pd, f)))
    os.makedirs(P.guard(pd), exist_ok=True)
    names = git("format-patch", "v3.1..HEAD", "-o", pd).decode().splitlines()      # paths contain spaces
    import campaign as CP
    head = git("rev-parse", "HEAD").decode().strip()
    log = git("log", "--format=%h %ad %s", "--date=iso", "v3.1..HEAD").decode().strip().splitlines()
    rd = lambda f: json.load(open(os.path.join(P.V311_RES, f), encoding="utf-8"))  # noqa: E731
    npth, rec, c3 = rd("normal_path_old_new_summary.json"), rd("recovery_path_summary.json"), rd("contract3_vs_contract2_scan.json")
    eng = CP.engine_identity()
    man = {"release": "CEAS v3 extension, follow-up v3.1.1 (two C02 failure-path gaps)", "version": "v3.1.1-followup-1",
           "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
           "brief": "CLAUDE FOLLOW-UP: CLOSE TWO REMAINING v3.1 FAILURE-PATH GAPS (independent review, 14 October 2026), R01-R04",
           "independent_checks_zip": "CEAS_v3_1_Independent_Checks.zip was named in the brief but not supplied; its findings were "
                                     "reproduced independently (reports/R01_R02_reproduction_before_fix.txt)",
           "baseline": {"archive_zip": {"name": os.path.basename(BASE_ZIP), "bytes": os.path.getsize(BASE_ZIP), "sha256": sha(BASE_ZIP)},
                        "v31_review_zip": {"name": os.path.basename(V31_ZIP), "bytes": os.path.getsize(V31_ZIP), "sha256": sha(V31_ZIP)},
                        "v31_tag": git("rev-parse", "v3.1^{commit}").decode().strip(), "check": "results/baseline_check_v311.json",
                        "v31_inventory_sha256": base["v31_inventory"]["sha256"], "v31_manifest_sha256": base["v31_manifest_sha256"]},
           "code": {"branch": git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(), "head": head, "commits_since_v3.1": log,
                    "files_changed_since_v3.1": base["files_changed_v31_to_head"], "patch": [os.path.basename(n) for n in names],
                    "engine_identity_scheme": eng["scheme"], "engine_sha256": eng["sha256"], "engine_sha16": eng["sha16"],
                    "engine_sha16_v31": npth["engine_old"], "contract": "v3.1-contract-3"},
           "checks": {"normal_path_old_new": {k: npth[k] for k in ("n_cases", "n_old_new_physics_equal", "n_old_new_with_field_diff",
                                                                   "n_new_vs_reference_equal", "n_decision_flips_old_new", "engine_new")},
                      "recovery_path": {k: rec[k] for k in ("n_cases", "n_held", "n_continuous", "n_decisions", "n_decision_flips",
                                                            "n_availability_changes", "engine_sha")},
                      "contract3_vs_contract2": {k: c3[k] for k in ("n_records", "n_records_changed", "n_item_changes")},
                      "tests": "tests_logs/pytest_full_v311.txt; tests_logs/pytest_from_review_zip_v311.txt"},
           "items": {"R01": "repaired (code + regressions)", "R02": "repaired by option B (driver-level nominal-only recovery) plus a "
                     "primary-feature checkpoint before the optional A_naive pass; held features are still not checkpointed before "
                     "reference propagation, and every statement says so", "R03": "documentation corrected (copies in corrected_docs; "
                     "generators patched)", "R04": "exact-line fixtures in code/tests/fixtures/v31; compact delta bundle"},
           "scientific_results": "unchanged: no stored record, selection, threshold, test decision or claim status changes",
           "optional_E01_E03": "not started", "manuscript": "not edited"}
    mp = os.path.join(P.V311, "CORRECTION_MANIFEST_v3.1.1.json")
    with open(P.guard(mp), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    inv = os.path.join(P.V311, "OUTPUT_INVENTORY_v311.csv")
    rows = sorted((rel(p, P.V311), os.path.getsize(p), sha(p)) for p in walk(P.V311) if os.path.abspath(p) != os.path.abspath(inv))
    with open(P.guard(inv), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)
    bad = [r[0] for r in rows if sha(os.path.join(P.V311, r[0])) != r[2]]
    if bad:
        raise SystemExit(f"inventory verification failed: {bad}")
    # review ZIP
    if os.path.exists(ZIP):
        os.remove(ZIP)
    tar = git("archive", "--format=zip", "--prefix=code/", "HEAD")
    members = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z, zipfile.ZipFile(io.BytesIO(tar)) as src:
        for info in src.infolist():
            if info.is_dir():
                continue
            data = src.read(info)
            z.writestr(info.filename, data)
            members.append((info.filename, len(data), hashlib.sha256(data).hexdigest()))
        for p in walk(P.V311):
            arc = "v3_extension/v3_1_1_followup/" + rel(p, P.V311)
            data = open(p, "rb").read()
            z.writestr(arc, data)
            members.append((arc, len(data), hashlib.sha256(data).hexdigest()))
        readme = open(os.path.join(P.V311, "README.md"), "rb").read()
        z.writestr("ZIP_README.md", readme)
        members.append(("ZIP_README.md", len(readme), hashlib.sha256(readme).hexdigest()))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["member", "bytes", "sha256"])
        w.writerows(sorted(members))
        z.writestr("ZIP_INVENTORY.csv", buf.getvalue())
    with zipfile.ZipFile(ZIP) as z:
        got = {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist()}
    bad = [m for m, _, h in members if got.get(m) != h]
    if bad or set(got) != {m for m, _, _ in members} | {"ZIP_INVENTORY.csv"}:
        raise SystemExit(f"ZIP verification failed: {bad[:5]}")
    print(json.dumps({"zip": ZIP, "bytes": os.path.getsize(ZIP), "sha256": sha(ZIP), "members": len(got),
                      "inventory_rows": len(rows), "engine_sha16": eng["sha16"], "head": head}, indent=1))


if __name__ == "__main__":
    main()
