import argparse
import os

import numpy as np
import pandas as pd

import v31paths as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P.V311DOCS, "results"))
    out = ap.parse_args().out
    c = pd.read_csv(os.path.join(P.V3, "results", "breakeven.csv"))
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
    os.makedirs(P.guarddocs(out), exist_ok=True)
    c.to_csv(P.guarddocs(os.path.join(out, "breakeven31.csv")), index=False)
    print(c[["cell", "workflow", "cost_basis", "expected_cpu_s_per_case_screen", "screen_only_cpu_s_per_case_lower_bound",
             "online_saving_cpu_s_per_case", "online_saving_cpu_s_per_case_upper_bound"]].to_string())


if __name__ == "__main__":
    main()
