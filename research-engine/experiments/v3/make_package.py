"""Patch, change log, output inventory and package ZIP for the v3 extension.

  patch/        git format-patch of every v3 commit on top of the archived code (ef9d224)
  patch/CHANGELOG.md   commits, files added/modified under code/, confirmation that archived
                       modules are unchanged
  OUTPUT_INVENTORY.csv every file of v3_extension/ and the v3 code (path, bytes, SHA-256, role)
  ../CW_Rendezvous_CEAS_Extension_v3.zip (next to the archive root): the archive tree with the v3
                       additions, excluding the local .git directory, caches and the replay copy

Usage: python code/experiments/v3/make_package.py [--zip]

v3.1: git failures now raise, and the script refuses to run once the v3.0 package exists (it would rewrite
the preserved patch/, OUTPUT_INVENTORY.csv and ZIP); the v3.1 package is experiments/v3_1/make_package_v31.py.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
BASE = "ef9d224"
ROLE = {"methods": "derivations and specifications", "reports": "reports", "results": "result tables",
        "raw": "raw records, manifests, logs", "claims": "claim-evidence map", "manuscript_assets": "figures, tables, bib",
        "patch": "code patch and change log", "environment": "environment record"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    # v3.1 (C07): a failing git command used to return empty output silently (empty patch/changelog)
    r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    if os.path.exists(os.path.join(V3, "OUTPUT_INVENTORY.csv")):
        # v3.1: the v3.0 patch/, inventory and ZIP are preserved originals; this script would rewrite them
        sys.exit("the v3.0 package exists and is preserved; use experiments/v3_1/make_package_v31.py")
    pdir = os.path.join(V3, "patch")
    os.makedirs(pdir, exist_ok=True)
    for f in os.listdir(pdir):
        if f.endswith(".patch"):
            os.remove(os.path.join(pdir, f))
    git("format-patch", f"{BASE}..HEAD", "-o", pdir)
    log = git("log", "--format=%h %ad %s", "--date=iso", f"{BASE}..HEAD")
    stat = git("diff", "--stat", BASE, "HEAD")
    changed = git("diff", "--name-status", BASE, "HEAD").splitlines()
    touched_hist = [l for l in changed if not (l.split("\t")[-1].startswith(("cwvalid3/", "experiments/v3/", "tests/test_v3_")))]
    with open(os.path.join(pdir, "CHANGELOG.md"), "w", encoding="utf-8") as fh:
        fh.write("# v3 code change log\n\nBase: archived Online Resource 1 code committed as `ef9d224` (tree identical to the archive's `code/`).\n\n")
        fh.write("## Commits\n\n```\n" + log + "```\n\n## Files (name-status against the base)\n\n```\n" + "\n".join(changed) + "\n```\n\n")
        fh.write("## Archived modules\n\n")
        fh.write("No file outside `cwvalid3/`, `experiments/v3/` and `tests/test_v3_*` was added or changed.\n" if not touched_hist
                 else "Changed outside the v3 directories:\n\n" + "\n".join(touched_hist) + "\n")
        fh.write("\n## Diff stat\n\n```\n" + stat + "```\n")
    rows = []
    for root in (V3, os.path.join(REPO, "cwvalid3"), os.path.join(REPO, "experiments", "v3"), os.path.join(REPO, "tests")):
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
            for f in fn:
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, ARCH).replace("\\", "/")
                if rel.startswith("code/tests/") and not os.path.basename(p).startswith("test_v3_"):
                    continue
                if rel == "v3_extension/OUTPUT_INVENTORY.csv":
                    continue
                top = rel.split("/")[1] if rel.startswith("v3_extension/") and rel.count("/") > 1 else ("top level" if rel.startswith("v3_extension/") else "code")
                rows.append((rel, os.path.getsize(p), sha(p), ROLE.get(top, top)))
    with open(os.path.join(V3, "OUTPUT_INVENTORY.csv"), "w", encoding="utf-8") as fh:
        fh.write("path,bytes,sha256,role\n")
        for r in sorted(rows):
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print(f"{len(rows)} files inventoried; patch files: {len([f for f in os.listdir(pdir) if f.endswith('.patch')])}")
    if a.zip:
        out = os.path.join(os.path.dirname(ARCH), "CW_Rendezvous_CEAS_Extension_v3.zip")
        skip = (".git", "__pycache__", ".pytest_cache")
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for dp, dn, fn in os.walk(ARCH):
                dn[:] = [d for d in dn if d not in skip]
                for f in fn:
                    p = os.path.join(dp, f)
                    z.write(p, os.path.join("CW_Rendezvous_CEAS_Extension_v3", os.path.relpath(p, ARCH)))
                    n += 1
        print(f"zip: {out} ({os.path.getsize(out) / 1e6:.1f} MB, {n} files) sha256 {sha(out)}")


if __name__ == "__main__":
    main()
