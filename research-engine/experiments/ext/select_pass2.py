"""Select second-pass verification cases: disagreement-band points (both classes,
several directions), the agreeing points just outside each selected band, and
refined 0.1 m / 1 m threshold-crossing endpoints. Writes results/verification/pass2_cases.json.
Usage: python experiments/ext/select_pass2.py
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid.runner import load_results  # noqa: E402
from cwvalid.scenario import axis_scenario  # noqa: E402

MG = os.path.join(RU.RESULTS, "main_grid")
d = pd.concat([load_results(os.path.join(MG, s)) for s in ("grid", "refine", "refine_bands")], ignore_index=True)
d = d[d.meta_velocity == "zero"].drop_duplicates(["meta_axis", "wn_ratio", "a_max", "sep_m"])
picks = []
fams = [("+R", 6.0, 0.05), ("-R", 6.0, 0.05), ("-R", 6.0, 0.10), ("+R", 4.0, 0.01), ("-V", 4.0, 0.10), ("+H", 4.0, 0.01)]
for ax, w, a in fams:
    g = d[(d.meta_axis == ax) & (d.wn_ratio == w) & (d.a_max == a)].sort_values("sep_m").reset_index(drop=True)
    dis = g.index[g.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
    if len(dis) == 0:
        continue
    i_mid = dis[len(dis) // 2]
    for i, why in ((i_mid, "inside disagreement band"), (dis[0] - 1, "just below band"), (dis[-1] + 1, "just above band")):
        if 0 <= i < len(g):
            picks.append((ax, float(g.sep_m[i]), w, a, f"{why} ({g.outcome_terminal_cw[i]})"))
br = pd.read_csv(os.path.join(MG, "refine", "boundary_brackets.csv"))
for ax, pred in (("+R", "disc_gt_0.1m"), ("+V", "disc_gt_0.1m"), ("+H", "disc_gt_1m"), ("-R", "disc_gt_1m")):
    b = br[(br.meta_axis == ax) & (br.wn_ratio == 6.0) & (br.a_max == 0.05) & (br.predicate == pred)].iloc[0]
    picks.append((ax, float(b.sep_lo), 6.0, 0.05, f"{pred} lower bracket end"))
    picks.append((ax, float(b.sep_hi), 6.0, 0.05, f"{pred} upper bracket end"))
cases = [axis_scenario("verify2", ax, s, w, a, extra_meta={"label": lab}).to_dict() for ax, s, w, a, lab in picks]
json.dump(cases, open(os.path.join(RU.RESULTS, "verification", "pass2_cases.json"), "w"), indent=1)
print(len(cases)); [print(c["meta"]["axis"], round(c["meta"]["sep_m"], 1), c["wn_ratio"], c["a_max"], c["meta"]["label"]) for c in cases]
