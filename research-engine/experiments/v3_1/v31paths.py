"""Paths shared by the v3.1 correction scripts. Every v3.1 output goes under v3_extension/v3_1_correction;
the v3.0 tree (manifests, results, ledger, plan, freeze, deviations, reports) is read-only here."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
V31 = os.path.join(V3, "v3_1_correction")
V31_RES = os.path.join(V31, "results")
V31_RAW = os.path.join(V31, "raw")
V31_REP = os.path.join(V31, "reports")
V31_LOG = os.path.join(V31, "tests_logs")
V31_ASSETS = os.path.join(V31, "manuscript_assets")
V31_CLAIMS = os.path.join(V31, "claims")
# v3.1.1: the v3.1 release tree is inventoried (OUTPUT_INVENTORY_v31.csv) and kept as released. The document generators
# (c05 claims, c07 cost wording, make_snippets_v31, report_pdf) now write regenerated, corrected documents under
# v3_1_1_followup/corrected_docs unless --out names another folder outside the v3.0 and v3.1 trees.
V311 = os.path.join(V3, "v3_1_1_followup")
V311_DOCS = os.path.join(V311, "corrected_docs")
SPLITS = ("pilot", "dev", "cal", "test", "transfer_cohort")
CELLS_ALL = ("D0", "D1", "D2", "D3", "D4", "D5")


def result_files():
    """(cell, split, path) for every stored v3.0 result file, including the D1 calibration reruns."""
    out = []
    for c in CELLS_ALL:
        for s in SPLITS:
            p = os.path.join(RES, f"{c}_{s}.jsonl")
            if os.path.exists(p):
                out.append((c, s, p))
    p = os.path.join(RES, "D1_cal_reruns.jsonl")
    if os.path.exists(p):
        out.append(("D1", "cal_reruns", p))
    return out


def inside(path, root):
    """True if path is root or below it (case-normalised real paths; safe on case-insensitive filesystems)."""
    p, r = (os.path.normcase(os.path.realpath(os.path.abspath(x))) for x in (path, root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:      # different drives
        return False


def in_closed_v30(path):
    return inside(path, V3) and not inside(path, V31)


def guard_v30(path):
    """Refuse to write anywhere in the v3.0 tree outside v3_1_correction (and, since v3.1.1, any UNC or extended-length
    path, whose spelling can alias the protected trees)."""
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if in_closed_v30(ap):
        raise PermissionError(f"refusing to write into the v3.0 tree: {ap}")
    return ap


def guard_docs(path):
    """Regenerated documents (v3.1.1): allowed under v3_1_1_followup or outside v3_extension; refuse the v3.0 tree and
    the released v3.1 tree."""
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if inside(ap, V311):
        return ap
    if inside(ap, V31):
        raise PermissionError(f"refusing to overwrite the released v3.1 tree: {ap}")
    return guard_v30(ap)
