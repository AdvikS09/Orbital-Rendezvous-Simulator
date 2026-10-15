"""Shared paths and helpers for the v2 extension (experiments/v2/).

Raw outputs of the v2 extension go to results/v2/raw/<study>/; derived tables and
figures go to results/v2/derived/. Nothing under results/ outside results/v2/ is
written by v2 scripts, so the v1 evidence is preserved unchanged.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.dirname(REPO)
RESULTS = os.path.join(PROJ, "results")
V2 = os.path.join(RESULTS, "v2")
RAW = os.path.join(V2, "raw")
DERIVED = os.path.join(V2, "derived")
TABLES = os.path.join(DERIVED, "tables")
FIGS = os.path.join(DERIVED, "figures")
FROZEN_V1 = os.path.join(RESULTS, "screening", "frozen", "screening_frozen.json")
FROZEN_V2_DIR = os.path.join(RAW, "frozen_v2")
FROZEN_V2 = os.path.join(FROZEN_V2_DIR, "estimator_v2_frozen.json")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", REPO, *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def code_revision():
    """Commit, branch, whether TRACKED files have uncommitted changes, and any untracked files."""
    untracked = [l[3:] for l in git("status", "--short", "--untracked-files=all").splitlines() if l.startswith("??")]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--short", "--untracked-files=no")), "untracked": untracked}


def environment():
    import numpy
    import pandas
    import scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}


def jsonable(v):
    import math

    import numpy as np
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def log_run(study, command, status, wall_s, note=""):
    """Append one line to results/v2/raw/run_log.jsonl (command, status, time)."""
    os.makedirs(RAW, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "study": study, "command": command, "status": status,
           "wall_s": round(float(wall_s), 2), "code": code_revision(), "note": note}
    with open(os.path.join(RAW, "run_log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
