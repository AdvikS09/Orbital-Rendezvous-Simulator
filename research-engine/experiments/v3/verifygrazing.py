import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verifyhybrid as VH


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(lab, r, w, Ts, "grazing") for lab, r, w, am in VH.CASES for Ts in (1.0, 10.0)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(VH.nearswitch, jobs):
            rows.extend(res)
    p = os.path.join(VH.OUT, "hybridvalidation.csv")
    df = pd.read_csv(p)
    df = df[df.test != "near_switch_grazing"]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(p, index=False)
    g = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(g.dropna(axis=1, how="all").to_string())


if __name__ == "__main__":
    main()
