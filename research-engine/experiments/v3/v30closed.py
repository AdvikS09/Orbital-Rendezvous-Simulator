import os
import sys

HARNESSENV = "CWVALID3_REBUILD_HARNESS"


def refuseoutsharness(script):
    if os.environ.get(HARNESSENV) == "1":
        return
    sys.exit(f"{os.path.basename(script)} writes into the frozen v3.0 tree; run it via "
             "code/experiments/v31/c07rebuild.py")
