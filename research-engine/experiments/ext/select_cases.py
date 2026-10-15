"""Select representative and boundary cases from the refined main grid.

Writes
  results/operational/cases.json          (~70 cases for the sampled-control /
                                           perturbation study)
  results/mechanisms/common_cases.json    (common-command propagation cases)
  results/histories_cases.json            (cases whose full histories are saved)

Boundary cases are the two endpoints of each refined bracket (last separation
on one side, first on the other) for the reference terminal-arrival boundary
and for the 0.1 m and 1 m peak-difference thresholds.
Usage: python experiments/ext/select_cases.py
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid.scenario import axis_scenario  # noqa: E402

BR = os.path.join(RU.RESULTS, "main_grid", "refine", "boundary_brackets.csv")


def endpoints(br, axis, wn, am, pred, which="both"):
    q = br[(br.meta_axis == axis) & (br.wn_ratio == wn) & (br.a_max == am) & (br.predicate == pred)].sort_values("sep_lo")
    if q.empty:
        return []
    b = q.iloc[0]
    out = []
    if which in ("both", "lo"):
        out.append((float(b.sep_lo), f"{pred} lower bracket end"))
    if which in ("both", "hi"):
        out.append((float(b.sep_hi), f"{pred} upper bracket end"))
    return out


def scen(family, axis, sep, wn, am, label):
    return axis_scenario(family, axis, sep, wn, am, extra_meta={"label": label})


def main():
    br = pd.read_csv(BR)
    ops = []
    for axis in ("+R", "-R", "+V", "+H"):
        for am in (0.01, 0.05, 0.10):
            for sep, lab in endpoints(br, axis, 6.0, am, "nl_terminal"):
                ops.append(scen("oper", axis, sep, 6.0, am, lab))
            for pred in ("disc_gt_1m", "disc_gt_0.1m"):
                for sep, lab in endpoints(br, axis, 6.0, am, pred, which="hi"):
                    ops.append(scen("oper", axis, sep, 6.0, am, lab))
    for axis in ("+R", "+V", "+H"):
        for wn in (4.0, 8.0):
            for sep, lab in endpoints(br, axis, wn, 0.05, "nl_terminal"):
                ops.append(scen("oper", axis, sep, wn, 0.05, lab))
    for axis in ("+R", "-R", "+V", "+H"):
        for sep in (1000.0, 3000.0):
            ops.append(scen("oper", axis, sep, 6.0, 0.05, "representative"))
    seen, uniq = set(), []
    for s in ops:
        if s.sid not in seen:
            seen.add(s.sid)
            uniq.append(s)
    os.makedirs(os.path.join(RU.RESULTS, "operational"), exist_ok=True)
    json.dump([s.to_dict() for s in uniq], open(os.path.join(RU.RESULTS, "operational", "cases.json"), "w"), indent=1)
    cc = [scen("common", "+R", 1000.0, 6.0, 0.05, "representative unsaturated")]
    cc += [scen("common", "+R", 3000.0, 6.0, 0.05, "representative saturated")]
    for axis in ("+R", "-R", "+V", "+H"):
        for sep, lab in endpoints(br, axis, 6.0, 0.05, "disc_gt_1m", which="hi"):
            cc.append(scen("common", axis, sep, 6.0, 0.05, lab))
    for sep, lab in endpoints(br, "+R", 6.0, 0.05, "nl_terminal"):
        cc.append(scen("common", "+R", sep, 6.0, 0.05, lab))
    cc.append(scen("common", "+H", 50000.0, 6.0, 0.05, "long saturation, both arrive"))
    cc.append(scen("common", "+R", 2500.0, 6.0, 0.01, "low cap"))
    os.makedirs(os.path.join(RU.RESULTS, "mechanisms"), exist_ok=True)
    json.dump([s.to_dict() for s in cc], open(os.path.join(RU.RESULTS, "mechanisms", "common_cases.json"), "w"), indent=1)
    print(f"operational cases: {len(uniq)}; common-command cases: {len(cc)}")


if __name__ == "__main__":
    main()
