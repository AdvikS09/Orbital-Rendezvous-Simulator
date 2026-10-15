"""v3.1: the v3.0 output tree (v3_extension, outside v3_1_correction) is a preserved release.

The v3.0 scripts that write into it (evaluation, atlas, cohort, assets, development summaries, cost, verification
and replay drivers) call refuse_outside_harness() first. It stops the run unless the v3.1 rebuild harness
(experiments/v3_1/c07_rebuild.py) is active; the harness sets HARNESS_ENV and redirects every write aimed at the
v3.0 tree to v3_1_correction/rebuilt, so the preserved files cannot be overwritten by running these scripts directly.
"""

import os
import sys

HARNESS_ENV = "CWVALID3_REBUILD_HARNESS"


def refuse_outside_harness(script):
    if os.environ.get(HARNESS_ENV) == "1":
        return
    sys.exit(f"{os.path.basename(script)} writes into the preserved v3.0 output tree; run it through "
             "code/experiments/v3_1/c07_rebuild.py (writes are redirected to v3_1_correction/rebuilt)")
