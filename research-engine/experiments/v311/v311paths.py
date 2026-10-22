import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "experiments", "v31"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import v31paths as P31

ARCH, V3, MAN, RES, V31 = P31.ARCH, P31.V3, P31.MAN, P31.RES, P31.V31
V311 = os.path.join(V3, "v311")
V311RES = os.path.join(V311, "results")
V311REP = os.path.join(V311, "reports")
V311LOG = os.path.join(V311, "tests_logs")
V311DOCS = os.path.join(V311, "corrected")
FIXTURES = os.path.join(REPO, "tests", "fixtures", "v31")


def guard(path):
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise PermissionError(f"refusing a UNC or extended-length path: {ap}")
    if P31.inside(ap, V311) or P31.inside(ap, FIXTURES):
        return ap
    if P31.inside(ap, V3):
        raise PermissionError(f"refusing to write outside v311: {ap}")
    return ap
