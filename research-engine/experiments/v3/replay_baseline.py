"""P00 baseline replay of Online Resource 1, executed in a disposable copy of the archive.

The archive's own reproduction commands (README.md, "SMALLEST REPRODUCTION") write into the
archive (revision_2026_10_13/generated/, selected_numerical_checks.json, the follow-up audit
JSON). To keep historical evidence untouched, the whole archive is copied to a scratch
directory, every command runs there from the copy's root, and every file the commands create
or change is compared byte-for-byte with the original archive. Nothing in the original
archive is written.

Usage: python code/experiments/v3/replay_baseline.py --copy-dir <scratch dir>
Output: v3_extension/results/baseline_replay.json (+ logs under v3_extension/raw/baseline_replay/)
"""

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
V3 = os.path.join(ARCHIVE, "v3_extension")

# Commands exactly as documented in the archive README (relative paths, run from the root),
# followed by the complete test suite, which the README does not list but which exercises
# every archived module.
COMMANDS = [
    ("readme_rebuild_tables", ["revision_2026_10_13/rebuild_tables.py"]),
    ("readme_status_and_v2_tests", ["-m", "pytest", "code/tests/test_revision_status.py", "code/tests/test_v2.py", "-q"]),
    ("readme_check_selected", ["revision_2026_10_13/check_selected.py"]),
    ("readme_audit_setters", ["revision_2026_10_13_followup/audit_setters.py"]),
    ("readme_estimator_fit_check", ["code/experiments/v2/estimator_fit.py", "--check"]),
    ("readme_plot_readability", ["manuscript/plot_readability.py"]),
    ("readme_plot_v2_manuscript", ["manuscript/plot_v2_manuscript.py", "--extension", "."]),
    ("full_test_suite", ["-m", "pytest", "code/tests", "-q", "-p", "no:cacheprovider"]),
]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "v3_extension"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = sha256(p)
    return out


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-dir", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    copy = os.path.abspath(a.copy_dir)
    if os.path.exists(copy):
        shutil.rmtree(copy)
    shutil.copytree(ARCHIVE, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "v3_extension"))
    logdir = os.path.join(V3, "raw", "baseline_replay")
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
        t0, c0 = time.perf_counter(), time.process_time()
        p = subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall = time.perf_counter() - t0
        post = snapshot(copy)
        changed = sorted(k for k in post if pre.get(k) != post[k])
        log_path = os.path.join(logdir, f"{name}.log")
        with open(log_path, "w", encoding="utf-8") as fh:
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
                     "log": os.path.relpath(log_path, ARCHIVE).replace("\\", "/")})
        print(f"{name}: exit {p.returncode} in {wall:.1f}s; {len(changed)} files written "
              f"({sum(1 for f in files if f['identical_to_archive'] is False)} differ from archive)", flush=True)
    after = snapshot(copy)
    import numpy, scipy, pandas, matplotlib, pytest
    rep = {
        "scope": "Replay of the archive's documented reproduction commands plus the full archived test suite, "
                 "run in a disposable copy so historical files are not overwritten. This is a check executed in this "
                 "v3 session; earlier records (e.g. 22 tests, four setters) are listed separately in INPUT_AUDIT.md.",
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
    with open(os.path.join(V3, "results", "baseline_replay.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({r["id"]: r["exit_code"] for r in runs}))


if __name__ == "__main__":
    main()
