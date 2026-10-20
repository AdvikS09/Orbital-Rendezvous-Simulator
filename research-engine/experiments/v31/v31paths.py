import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
V31 = os.path.join(V3, "v31")
V31RES = os.path.join(V31, "results")
V31RAW = os.path.join(V31, "raw")
V31REP = os.path.join(V31, "reports")
V31LOG = os.path.join(V31, "tests_logs")
V31ASSETS = os.path.join(V31, "assets")
V31CLAIMS = os.path.join(V31, "claims")
V311 = os.path.join(V3, "v311")
V311DOCS = os.path.join(V311, "corrected")
SPLITS = ("pilot", "dev", "cal", "test", "transfer_cohort")
CELLSALL = ("D0", "D1", "D2", "D3", "D4", "D5")


def resfiles():
    out = []
    for c in CELLSALL:
        for s in SPLITS:
            p = os.path.join(RES, f"{c}{s.replace('_', '')}.jsonl")
            if os.path.exists(p):
                out.append((c, s, p))
    p = os.path.join(RES, "D1calreruns.jsonl")
    if os.path.exists(p):
        out.append(("D1", "cal_reruns", p))
    return out


def inside(path, root):
    p, r = (os.path.normcase(os.path.realpath(os.path.abspath(x))) for x in (path, root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:
        return False


def inclosed(path):
    return inside(path, V3) and not inside(path, V31)


def guardv30(path):
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if inclosed(ap):
        raise PermissionError(f"refusing to write into the v3.0 tree: {ap}")
    return ap


def guarddocs(path):
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if inside(ap, V311):
        return ap
    if inside(ap, V31):
        raise PermissionError(f"refusing to overwrite the released v3.1 tree: {ap}")
    return guardv30(ap)
