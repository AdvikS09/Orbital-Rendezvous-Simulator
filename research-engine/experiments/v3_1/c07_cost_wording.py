"""C07 cost wording (rebuild from stored results, no timing re-run): results/cost_break_even.csv mixes two
cost bases under one column name. D0 rows are expected online CPU per case including referral work at the
test referral fraction; the held rows (D3, D4) are batch-amortised screen-only costs that EXCLUDE referral work
(their note says so, the column name does not). v3.1 adds an explicit cost_basis column, moves the held values
into screen_only_cpu_s_per_case_lower_bound and leaves expected_cpu_s_per_case_screen empty for them, so no
reader can take a lower bound for an expected cost. The held values come from one 40-case batch per held cell,
summarised as the median of three repetitions of that
40-case batch, timed one after the other on a machine that was not otherwise idle, with no raw repetitions archived
(raw/cost_timings.csv holds D0 only); they were slightly above the direct comparator (0.268 vs 0.237 s and 0.135 vs
0.125 s per case). Only features A, B and C were timed in D0.

Writes cost_break_even_v31.csv into --out (default since v3.1.1: v3_1_1_followup/corrected_docs/results; the released
v3.1 copy in v3_1_correction/results is kept unchanged). Usage: python code/experiments/v3_1/c07_cost_wording.py [--out DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

import v31paths as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P.V311_DOCS, "results"))
    out = ap.parse_args().out
    c = pd.read_csv(os.path.join(P.V3, "results", "cost_break_even.csv"))
    held = c.workflow.str.contains("batch_amortised")
    c.insert(c.columns.get_loc("workflow") + 1, "cost_basis", np.where(
        held, "screen-only CPU per case, batch-amortised, EXCLUDING referral work (lower bound on the screen's online cost)",
        "expected online CPU per case including referral work at the test referral fraction"))
    c["screen_only_cpu_s_per_case_lower_bound"] = np.where(held, c.expected_cpu_s_per_case_screen, np.nan)
    c.loc[held, "expected_cpu_s_per_case_screen"] = np.nan
    c["online_saving_cpu_s_per_case_upper_bound"] = np.where(held, c.online_saving_cpu_s_per_case, np.nan)
    c.loc[held, "online_saving_cpu_s_per_case"] = np.nan
    c["v31_note"] = np.where(held, "held rows: screen-only cost of one 40-case batch per held cell, median of 3 repetitions (no dispersion archived); "
                                   "the saving shown is an upper bound because referrals would add cost; observed slightly negative",
                             "D0: features A, B, C timed; sep and sat not timed")
    os.makedirs(P.guard_docs(out), exist_ok=True)
    c.to_csv(P.guard_docs(os.path.join(out, "cost_break_even_v31.csv")), index=False)
    print(c[["cell", "workflow", "cost_basis", "expected_cpu_s_per_case_screen", "screen_only_cpu_s_per_case_lower_bound",
             "online_saving_cpu_s_per_case", "online_saving_cpu_s_per_case_upper_bound"]].to_string())


if __name__ == "__main__":
    main()
