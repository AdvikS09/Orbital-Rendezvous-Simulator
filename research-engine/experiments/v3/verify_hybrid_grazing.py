"""P06 supplement: search the verify_hybrid base cases for a grazing configuration (the sampled command
magnitude touching the cap from below after desaturation) and run the one-sided finite-difference
ladder on both sides of it. Rows are appended to results/hybrid_validation.csv (test
near_switch_grazing); a base case where no such boundary exists on the a_max scan is recorded as such.

Usage: python code/experiments/v3/verify_hybrid_grazing.py [--workers 4]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_hybrid as VH  # noqa: E402


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(lab, r, w, Ts, "grazing") for lab, r, w, am in VH.CASES for Ts in (1.0, 10.0)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(VH.near_switch, jobs):
            rows.extend(res)
    p = os.path.join(VH.OUT, "hybrid_validation.csv")
    df = pd.read_csv(p)
    df = df[df.test != "near_switch_grazing"]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(p, index=False)
    g = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(g.dropna(axis=1, how="all").to_string())


if __name__ == "__main__":
    main()
