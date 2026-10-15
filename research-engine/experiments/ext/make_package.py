"""Assemble the handover ZIP (CW_Rendezvous_Research_Handover.zip in the project folder).

Layout inside the archive (top folder CW_Rendezvous_Research_Handover/):
  README.md      copy of handover/README.md
  handover/      reports (README, research handover report, methods, verification,
                 baseline/corrections, screening, figures/tables guide, run manifest,
                 reference equation check)
  code/          git archive of the repository at HEAD (src, cwvalid, experiments, tests, ...)
  code_history.bundle   full git history incl. tag baseline-paper-final (git clone code_history.bundle)
  environment/   requirements-extension.txt, python_version.txt, pip_freeze.txt, code_version.txt
  results/       all result folders with the executed results; verification histories are
                 the production solver variant at the full 0.5 s grid (histories_production/);
                 the tight-tolerance reference-variant histories are left out for size
                 (regenerate with verify_numerics.py)
Reference PDFs and the manuscript are not included.

Usage: python experiments/ext/make_package.py
"""

import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.dirname(REPO)
RES = os.path.join(PROJ, "results")
OUTZIP = os.path.join(PROJ, "CW_Rendezvous_Research_Handover.zip")
TOP = "CW_Rendezvous_Research_Handover"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
PROD_PREFIX = "rtol1e-13__"


def git(*a, **kw):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, **kw)


def main():
    tmp = os.path.join(PROJ, "_package_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    # code
    arc = os.path.join(tmp, "code.zip")
    git("archive", "--format=zip", "-o", arc, "HEAD")
    bundle = os.path.join(tmp, "code_history.bundle")
    git("bundle", "create", bundle, "--all")
    head = git("rev-parse", "HEAD", text=True).stdout.strip()
    dirty = git("status", "--short", text=True).stdout.strip()
    # environment
    env = os.path.join(tmp, "environment")
    os.makedirs(env)
    py = os.path.join(REPO, ".venv", "Scripts", "python.exe")
    py = py if os.path.exists(py) else sys.executable
    open(os.path.join(env, "python_version.txt"), "w").write(subprocess.run([py, "--version"], capture_output=True, text=True).stdout)
    open(os.path.join(env, "pip_freeze.txt"), "w").write(subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    open(os.path.join(env, "requirements-extension.txt"), "w").write(
        "# Exact versions used for the extension runs (Python 3.14.6)\n"
        "numpy==2.4.6\nscipy==1.17.1\npandas==3.0.3\nmatplotlib==3.11.0\npytest==9.1.1\n")
    if os.path.exists(OUTZIP):
        os.remove(OUTZIP)
    n_files = 0
    with zipfile.ZipFile(OUTZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def add(path, arcname):
            nonlocal n_files
            z.write(path, f"{TOP}/{arcname}")
            n_files += 1
        # reports
        add(os.path.join(PROJ, "handover", "README.md"), "README.md")
        for f in sorted(os.listdir(os.path.join(PROJ, "handover"))):
            p = os.path.join(PROJ, "handover", f)
            if os.path.isfile(p):
                add(p, f"handover/{f}")
        add(os.path.join(RES, "reference_equation_check.md"), "handover/reference_equation_check.md")
        # code from git archive
        with zipfile.ZipFile(arc) as zc:
            for info in zc.infolist():
                if info.is_dir():
                    continue
                z.writestr(f"{TOP}/code/{info.filename}", zc.read(info.filename))
                n_files += 1
        z.writestr(f"{TOP}/code/requirements-extension.txt", open(os.path.join(env, "requirements-extension.txt")).read())
        add(bundle, "code_history.bundle")
        for f in os.listdir(env):
            add(os.path.join(env, f), f"environment/{f}")
        z.writestr(f"{TOP}/environment/code_version.txt",
                   f"commit {head}\n" + ("uncommitted changes present\n" + dirty if dirty else "clean working tree\n"))
        # results
        for root, dirs, files in os.walk(RES):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            relp = os.path.relpath(root, RES).replace("\\", "/")
            if relp.startswith("baseline_reproduction/engine_baseline") and not relp.startswith("baseline_reproduction/engine_baseline/outputs"):
                continue   # code copy of the baseline tag (in the git bundle); keep only its regenerated outputs
            if relp.startswith("verification/histories") or relp.startswith("verification/pass2/histories"):
                continue
            for f in files:
                add(os.path.join(root, f), f"results/{relp}/{f}".replace("/./", "/"))
        # verification histories: production variant at the full 0.5 s grid
        for sub in ("verification/histories", "verification/pass2/histories"):
            d = os.path.join(RES, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".npz"):
                    continue
                zz = np.load(os.path.join(d, f))
                keep = {k: zz[k] for k in zz.files if k.startswith(PROD_PREFIX)}
                outp = os.path.join(tmp, f)
                np.savez_compressed(outp, **keep)
                add(outp, f"results/{sub}_production/{f}")
        z.writestr(f"{TOP}/results/verification/HISTORIES_NOTE.txt",
                   "histories_production/ (and pass2/histories_production/) hold the full histories of every "
                   "verification case at the production solver setting (DOP853 rtol 1e-13, 0.5 s grid, keys "
                   f"prefixed {PROD_PREFIX}).\n"
                   "The tight-tolerance reference-variant histories are not included for size; their differences "
                   "from production are summarised in verification_variants.csv.\n"
                   "Regenerate with: python experiments/ext/verify_numerics.py "
                   "(and --cases-json ../results/verification/pass2_cases.json --tag pass2).\n")
    shutil.rmtree(tmp)
    size = os.path.getsize(OUTZIP) / 2**20
    print(f"wrote {OUTZIP} ({size:.1f} MB, {n_files} files, code commit {head})")


if __name__ == "__main__":
    main()
