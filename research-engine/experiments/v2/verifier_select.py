"""Preselect the benchmark cases for the independent verifier (before any comparison).

Deterministic selection rules applied to the stored production results; the case list,
the reasons and a SHA-256 of the list are written before the independent verifier is
run on any case. Selection uses stored production results (nonlinear labels included),
which is appropriate for choosing difficult verification cases.

Rules (each yields the listed number of cases):
  R1 (6) two-orbit falsely favourable CW cases (mechanisms/horizon2, outcome_terminal_cw == cw_only)
  R2 (2) one-orbit quadratic-model label mismatches (grid, refine, refine_bands; outcome_terminal_so != agree)
  R3 (2) +radial and -radial saturated converging cases: main grid, 6n, 0.05 m/s^2, separation nearest 3 km
  R4 (2) mixed-direction cases with nonzero initial velocity from the broad test set (sorted by sid):
         the first in which both loops arrive and the first in which both fail
  R5 (2) close-range unsaturated cases: main grid, 6n, 0.05 m/s^2, +along-track nearest 500 m and
         -cross-track nearest 800 m (below the 1,085 m zero-velocity saturation distance)
  R6 (3) saturated cases nearest the 1 m peak-discrepancy threshold: boundary-refinement points at
         6n, 0.05 m/s^2 for +radial, +along-track, +cross-track (|log(peak/1 m)| minimal)
  R7 (5) difficult arrival-boundary cases from the capture-limit bisection (refine_bands): smallest
         |reference dwell margin|; smallest |CW dwell margin|; the cw_only point with the largest CW
         margin; the nl_only point with the largest reference margin; the -along-track point with the
         smallest |reference margin|
  R8 (1) strongly saturated long approach: main grid +cross-track 50 km at 6n, 0.05 m/s^2
  R9 (1) a two-orbit quadratic-model numerical failure (horizon2, so_status == failed; smallest
         separation, so the reference outcome is as close to the capture limit as possible)

Output: results/v2/raw/verifier/cases.json, selection.csv, selection_record.json
Usage:  python experiments/v2/verifier_select.py
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

from cwvalid.runner import load_results  # noqa: E402

OUT = os.path.join(C.RAW, "verifier")


def scen(campaign):
    return {json.loads(l)["sid"]: json.loads(l) for l in open(os.path.join(C.RESULTS, campaign, "scenarios.jsonl")) if l.strip()}


def main():
    if os.path.exists(os.path.join(OUT, "selection_record.json")):
        sys.exit("selection already frozen; refusing to overwrite (delete deliberately to reselect)")
    sel = []

    def add(campaign, row, rule, reason):
        sel.append({"sid": row["sid"], "campaign": campaign, "rule": rule, "reason": reason})

    h2 = load_results(os.path.join(C.RESULTS, "mechanisms/horizon2"))
    for _, r in h2[h2.outcome_terminal_cw == "cw_only"].sort_values("sid").iterrows():
        add("mechanisms/horizon2", r, "R1", f"two-orbit falsely favourable CW ({r.meta_axis} {r.sep_m:.0f} m, {r.wn_ratio:g}n, {r.a_max:g})")
    one = {c: load_results(os.path.join(C.RESULTS, c)) for c in ("main_grid/grid", "main_grid/refine", "main_grid/refine_bands")}
    for c, d in one.items():
        for _, r in d[d.outcome_terminal_so.isin(["cw_only", "nl_only"])].sort_values("sid").iterrows():
            add(c, r, "R2", f"one-orbit quadratic mismatch {r.outcome_terminal_so} (SO margin {r.so_terminal_dwell_margin_s:+.3f} s, ref margin {r.eci_terminal_dwell_margin_s:+.3f} s)")
    g = one["main_grid/grid"]
    base = g[(g.wn_ratio == 6.0) & (g.a_max == 0.05)]
    for ax in ("+R", "-R"):
        q = base[base.meta_axis == ax]
        r = q.loc[(np.log(q.sep_m / 3000.0)).abs().idxmin()]
        add("main_grid/grid", r, "R3", f"{ax} saturated converging, {r.sep_m:.0f} m")
    t = load_results(os.path.join(C.RESULTS, "screening/test")).sort_values("sid")
    mx = t[(t.meta_axis == "mixed") & (t.meta_velocity == "random")]
    r = mx[mx.cw_terminal_arrived & mx.eci_terminal_arrived].iloc[0]
    add("screening/test", r, "R4", f"mixed direction, random velocity (nu {r.meta_nu:.2f}), both arrive")
    r = mx[~mx.cw_terminal_arrived & ~mx.eci_terminal_arrived].iloc[0]
    add("screening/test", r, "R4", f"mixed direction, random velocity (nu {r.meta_nu:.2f}), both fail")
    for ax, s0 in (("+V", 500.0), ("-H", 800.0)):
        q = base[base.meta_axis == ax]
        r = q.loc[(np.log(q.sep_m / s0)).abs().idxmin()]
        add("main_grid/grid", r, "R5", f"{ax} close range unsaturated, {r.sep_m:.0f} m (CW saturated {r.cw_sat_total_s:.0f} s)")
    rf = one["main_grid/refine"]
    rb = rf[(rf.wn_ratio == 6.0) & (rf.a_max == 0.05) & (rf.cw_sat_total_s > 0)]
    for ax in ("+R", "+V", "+H"):
        q = rb[rb.meta_axis == ax]
        r = q.loc[(np.log(q.cw_vs_eci_max_pos_disc_m)).abs().idxmin()]
        add("main_grid/refine", r, "R6", f"{ax} near 1 m threshold (peak {r.cw_vs_eci_max_pos_disc_m:.4f} m)")
    b = one["main_grid/refine_bands"].copy()
    used = {s["sid"] for s in sel}
    b = b[~b.sid.isin(used)]
    picks = [
        (b.loc[b.eci_terminal_dwell_margin_s.abs().idxmin()], "smallest |reference dwell margin|"),
        (b.loc[b.cw_terminal_dwell_margin_s.abs().idxmin()], "smallest |CW dwell margin|"),
        (b[b.outcome_terminal_cw == "cw_only"].sort_values("cw_terminal_dwell_margin_s").iloc[-1], "cw_only band point, largest CW margin"),
        (b[b.outcome_terminal_cw == "nl_only"].sort_values("eci_terminal_dwell_margin_s").iloc[-1], "nl_only band point, largest reference margin"),
    ]
    q = b[b.meta_axis == "-V"]
    picks.append((q.loc[q.eci_terminal_dwell_margin_s.abs().idxmin()], "-along-track band point, smallest |reference margin|"))
    for r, why in picks:
        if r.sid in {s["sid"] for s in sel}:
            continue
        add("main_grid/refine_bands", r, "R7", f"{why} ({r.meta_axis} {r.sep_m:.1f} m, {r.wn_ratio:g}n, {r.a_max:g}; CW {r.cw_terminal_dwell_margin_s:+.3f} s, ref {r.eci_terminal_dwell_margin_s:+.3f} s, {r.outcome_terminal_cw})")
    q = base[(base.meta_axis == "+H")]
    r = q.loc[q.sep_m.idxmax()]
    add("main_grid/grid", r, "R8", f"+H {r.sep_m:.0f} m strongly saturated long approach")
    f = h2[h2.so_status == "failed"].sort_values("sep_m").iloc[0]
    add("mechanisms/horizon2", f, "R9", f"two-orbit quadratic numerical failure ({f.meta_axis} {f.sep_m:.0f} m, {f.wn_ratio:g}n, {f.a_max:g})")
    S = pd.DataFrame(sel)
    if S.sid.duplicated().any():
        raise SystemExit("duplicate selection")
    cases = []
    cache = {}
    for s in sel:
        if s["campaign"] not in cache:
            cache[s["campaign"]] = scen(s["campaign"])
        d = dict(cache[s["campaign"]][s["sid"]])
        d["meta"] = dict(d.get("meta", {}), verifier_rule=s["rule"], source_campaign=s["campaign"])
        cases.append(d)
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(os.path.join(OUT, "selection.csv"), index=False)
    json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
    rec = {"selected_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_cases": len(cases), "code": C.code_revision(),
           "cases_sha256": C.sha256(os.path.join(OUT, "cases.json")),
           "selection_sha256": C.sha256(os.path.join(OUT, "selection.csv")),
           "note": "selected before any independent-verifier result existed"}
    C.write_json(os.path.join(OUT, "selection_record.json"), rec)
    print(S.to_string())
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
