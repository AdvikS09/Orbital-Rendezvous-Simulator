import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
ARCHIVE = os.path.dirname(CODE)
V3 = os.path.join(ARCHIVE, "v3")

COMMANDS = [
    ("readme_rebuild_tables", ["revision_2026_10_13/rebuildtab.py"]),
    ("readme_status_and_v2_tests", ["-m", "pytest", "code/tests/testrevstatus.py", "code/tests/testv2.py", "-q"]),
    ("readme_check_selected", ["revision_2026_10_13/checksel.py"]),
    ("readme_audit_setters", ["revision_2026_10_13_followup/setters.py"]),
    ("readme_estimator_fit_check", ["code/experiments/v2/estfit.py", "--check"]),
    ("readme_plot_readability", ["manuscript/plotread.py"]),
    ("readme_plot_v2_manuscript", ["manuscript/plotv2.py", "--extension", "."]),
    ("full_test_suite", ["-m", "pytest", "code/tests", "-q", "-p", "no:cacheprovider"]),
]
SKIPDIRS = {".git", "__pycache__", ".pytest_cache", "v3"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIPDIRS]
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = sha256(p)
    return out


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-dir", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    copy = os.path.abspath(a.copy_dir)
    if os.path.exists(copy):
        shutil.rmtree(copy)
    shutil.copytree(ARCHIVE, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "v3"))
    logdir = os.path.join(V3, "raw", "replay")
    os.makedirs(logdir, exist_ok=True)
    orig = snapshot(ARCHIVE)
    before = snapshot(copy)
    env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    runs = []
    for name, args in COMMANDS:
        if a.only and name not in a.only:
            continue
        cmd = [sys.executable] + args
        pre = snapshot(copy)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall = time.perf_counter() - t0
        post = snapshot(copy)
        changed = sorted(k for k in post if pre.get(k) != post[k])
        logpath = os.path.join(logdir, f"{name}.log")
        with open(logpath, "w", encoding="utf-8") as fh:
            fh.write("$ python " + " ".join(args) + "\n")
            fh.write(p.stdout)
            fh.write("\n--- stderr ---\n")
            fh.write(p.stderr)
        files = []
        for k in changed:
            o = orig.get(k)
            files.append({"path": k, "existed_in_archive": o is not None,
                          "identical_to_archive": (o == post[k]) if o is not None else None,
                          "sha256_replay": post[k], "sha256_archive": o})
        runs.append({"id": name, "command": "python " + " ".join(args), "cwd": "<archive copy root>",
                     "exit_code": p.returncode, "wall_s": wall, "stdout_tail": p.stdout[-1500:],
                     "stderr_tail": p.stderr[-1500:], "files_written": files,
                     "log": os.path.relpath(logpath, ARCHIVE).replace("\\", "/")})
        print(f"{name}: exit {p.returncode} in {wall:.1f}s; {len(changed)} files written "
              f"({sum(1 for f in files if f['identical_to_archive'] is False)} differ from archive)", flush=True)
    after = snapshot(copy)
    import numpy, scipy, pandas, matplotlib, pytest
    rep = {
        "scope": "Replay of the archive's documented reproduction commands plus the full archived test suite, "
                 "run in a disposable copy so historical files are not overwritten. This check was run during the "
                 "v3 campaign; earlier checks (22 tests, four setters) were recorded separately.",
        "executed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
                        "matplotlib": matplotlib.__version__, "pytest": pytest.__version__},
        "archive_checked_environment": "revision_2026_10_13/requirements-checked.txt: numpy 2.3.5, scipy 1.17.0, "
                                       "pandas 2.2.3, matplotlib 3.10.8, pytest 9.1.1 on Python 3.12",
        "runs": runs,
        "n_files_original_archive": len(orig),
        "files_created_or_changed_in_copy": sorted(k for k in after if before.get(k) != after[k]),
    }
    with open(os.path.join(V3, "results", "replay.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({r["id"]: r["exit_code"] for r in runs}))


if __name__ == "__main__":
    main()
