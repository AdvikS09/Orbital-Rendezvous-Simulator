"""Assemble CW_Rendezvous_CEAS_Extension_v2.zip (in the project folder).

Layout (top folder CW_Rendezvous_CEAS_Extension_v2/):
  README.md                  how to read and reproduce (copy of handover_v2/README.md)
  handover_v2/               v2 reports: RESEARCH_EXTENSION_REPORT, MANUSCRIPT_INSERTION_MAP, PROVENANCE,
                             FIGURES_AND_TABLES_v2, RUN_LOG_v2, IO_MANIFEST.csv, EXCLUDED_FILES.csv, test_logs/
  handover_v1/               the v1 handover reports, unchanged
  code/                      git archive of the repository at HEAD (src, cwvalid, indverify, experiments, tests)
  code_history.bundle        full git history (tags baseline-paper-final; branches research-extension, v2-extension)
  environment/               requirements, pip freeze, interpreter version, code version
  results/                   v1 results (preserved; bulky v1 trajectory-history folders excluded, listed with
                             hashes in handover_v2/EXCLUDED_FILES.csv; they are in the v1 handover ZIP) and
                             results/v2/ complete (raw/ separate from derived/)
Virtual environments, caches and the baseline-tag code copy (in the git bundle) are excluded.

Usage: python experiments/v2/make_package_v2.py
"""

import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

OUTZIP = os.path.join(C.PROJ, "CW_Rendezvous_CEAS_Extension_v2.zip")
TOP = "CW_Rendezvous_CEAS_Extension_v2"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv"}
EXCLUDE_V1 = ["verification/histories", "verification/pass2/histories", "operational/histories", "histories",
              "mechanisms/common_command/histories"]


def git(*a, **kw):
    return subprocess.run(["git", "-C", C.REPO, *a], check=True, capture_output=True, **kw)


def main():
    tmp = os.path.join(C.PROJ, "_package_v2_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    arc = os.path.join(tmp, "code.zip")
    git("archive", "--format=zip", "-o", arc, "HEAD")
    bundle = os.path.join(tmp, "code_history.bundle")
    git("bundle", "create", bundle, "--all")
    rev = C.code_revision()
    py = os.path.join(C.REPO, ".venv", "Scripts", "python.exe")
    py = py if os.path.exists(py) else sys.executable
    env = C.environment()
    # excluded v1 history files, with hashes
    excl = []
    for sub in EXCLUDE_V1:
        d = os.path.join(C.RESULTS, sub)
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                excl.append({"path": os.path.relpath(p, C.PROJ).replace("\\", "/"), "bytes": os.path.getsize(p),
                             "sha256": C.sha256(p), "available_in": "CW_Rendezvous_Research_Handover.zip (v1 handover)"})
    pd.DataFrame(excl).to_csv(os.path.join(C.PROJ, "handover_v2", "EXCLUDED_FILES.csv"), index=False)
    if os.path.exists(OUTZIP):
        os.remove(OUTZIP)
    n = 0
    with zipfile.ZipFile(OUTZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def add(path, arcname):
            nonlocal n
            z.write(path, f"{TOP}/{arcname}")
            n += 1
        add(os.path.join(C.PROJ, "handover_v2", "README.md"), "README.md")
        for sub, dest in (("handover_v2", "handover_v2"), ("handover", "handover_v1")):
            base = os.path.join(C.PROJ, sub)
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for f in sorted(files):
                    p = os.path.join(root, f)
                    add(p, f"{dest}/{os.path.relpath(p, base)}".replace("\\", "/"))
        add(os.path.join(C.RESULTS, "reference_equation_check.md"), "handover_v1/reference_equation_check.md")
        with zipfile.ZipFile(arc) as zc:
            for info in zc.infolist():
                if not info.is_dir():
                    z.writestr(f"{TOP}/code/{info.filename}", zc.read(info.filename))
                    n += 1
        req = ("# Exact versions used (Python 3.14.6)\nnumpy==2.4.6\nscipy==1.17.1\npandas==3.0.3\nmatplotlib==3.11.0\npytest==9.1.1\n")
        z.writestr(f"{TOP}/code/requirements-extension.txt", req)
        z.writestr(f"{TOP}/environment/requirements-extension.txt", req)
        z.writestr(f"{TOP}/environment/pip_freeze.txt", subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
        z.writestr(f"{TOP}/environment/python_version.txt", subprocess.run([py, "--version"], capture_output=True, text=True).stdout)
        z.writestr(f"{TOP}/environment/code_version.txt",
                   f"commit {rev['commit']} (branch {rev['branch']})\n"
                   + ("tracked files had uncommitted changes\n" if rev["dirty"] else "no uncommitted changes to tracked files\n")
                   + f"platform {env['platform']}; {env['processor']}; {env['cpu_count']} logical CPUs\n")
        add(bundle, "code_history.bundle")
        skip_abs = {os.path.normcase(os.path.join(C.RESULTS, s)) for s in EXCLUDE_V1}
        for root, dirs, files in os.walk(C.RESULTS):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and os.path.normcase(os.path.join(root, d)) not in skip_abs]
            rel = os.path.relpath(root, C.RESULTS).replace("\\", "/")
            if rel.startswith("baseline_reproduction/engine_baseline") and not rel.startswith("baseline_reproduction/engine_baseline/outputs"):
                continue
            for f in files:
                add(os.path.join(root, f), f"results/{rel}/{f}".replace("/./", "/"))
    shutil.rmtree(tmp)
    size = os.path.getsize(OUTZIP) / 2**20
    print(f"wrote {OUTZIP} ({size:.1f} MB, {n} files, code {rev['commit'][:7]}, excluded {len(excl)} v1 history files)")


if __name__ == "__main__":
    main()
