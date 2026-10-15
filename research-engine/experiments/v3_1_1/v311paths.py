"""Paths of the v3.1.1 follow-up (CLOSE TWO REMAINING v3.1 FAILURE-PATH GAPS, 14 Oct 2026). Every v3.1.1 output goes
under v3_extension/v3_1_1_followup. The v3.0 tree and the v3.1 release tree (v3_extension/v3_1_correction, inventoried
in OUTPUT_INVENTORY_v31.csv) are read-only here."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "experiments", "v3_1"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import v31paths as P31  # noqa: E402

ARCH, V3, MAN, RES, V31 = P31.ARCH, P31.V3, P31.MAN, P31.RES, P31.V31
V311 = os.path.join(V3, "v3_1_1_followup")
V311_RES = os.path.join(V311, "results")
V311_REP = os.path.join(V311, "reports")
V311_LOG = os.path.join(V311, "tests_logs")
V311_DOCS = os.path.join(V311, "corrected_docs")
FIXTURES = os.path.join(REPO, "tests", "fixtures", "v31")


def guard(path):
    """Write only under v3_1_1_followup or the repository's test fixtures; never into the v3.0 or v3.1 trees."""
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if P31.inside(ap, V311) or P31.inside(ap, FIXTURES):
        return ap
    if P31.inside(ap, V3):
        raise PermissionError(f"refusing to write outside v3_1_1_followup: {ap}")
    return ap
