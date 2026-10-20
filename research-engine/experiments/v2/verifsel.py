import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C

from cwvalid.runner import loadres

OUT = os.path.join(C.RAW, "verifier")


def scen(campaign):
    return {json.loads(l)["sid"]: json.loads(l) for l in open(os.path.join(C.RESULTS, campaign, "scenarios.jsonl")) if l.strip()}


def main():
    if os.path.exists(os.path.join(OUT, "selection.json")):
        sys.exit("selection already frozen (delete it to reselect)")
    sel = []

    def add(campaign, row, rule, reason):
        sel.append({"sid": row["sid"], "campaign": campaign, "rule": rule, "reason": reason})

    h2 = loadres(os.path.join(C.RESULTS, "mechanisms/horizon2"))
    for _, r in h2[h2.outcome_terminal_cw == "cw_only"].sort_values("sid").iterrows():
        add("mechanisms/horizon2", r, "R1", f"two-orbit falsely favourable CW ({r.meta_axis} {r.sep_m:.0f} m, {r.wn_ratio:g}n, {r.a_max:g})")
    one = {c: loadres(os.path.join(C.RESULTS, c)) for c in ("maingrid/grid", "maingrid/refine", "maingrid/refinebands")}
    for c, d in one.items():
        for _, r in d[d.outcome_terminal_so.isin(["cw_only", "nl_only"])].sort_values("sid").iterrows():
            add(c, r, "R2", f"one-orbit quadratic mismatch {r.outcome_terminal_so} (SO margin {r.so_terminal_dwell_margin_s:+.3f} s, ref margin {r.eci_terminal_dwell_margin_s:+.3f} s)")
    g = one["maingrid/grid"]
    base = g[(g.wn_ratio == 6.0) & (g.a_max == 0.05)]
    for ax in ("+R", "-R"):
        q = base[base.meta_axis == ax]
        r = q.loc[(np.log(q.sep_m / 3000.0)).abs().idxmin()]
        add("maingrid/grid", r, "R3", f"{ax} saturated converging, {r.sep_m:.0f} m")
    t = loadres(os.path.join(C.RESULTS, "screening/test")).sort_values("sid")
    mx = t[(t.meta_axis == "mixed") & (t.meta_velocity == "random")]
    r = mx[mx.cw_terminal_arrived & mx.eci_terminal_arrived].iloc[0]
    add("screening/test", r, "R4", f"mixed direction, random velocity (nu {r.meta_nu:.2f}), both arrive")
    r = mx[~mx.cw_terminal_arrived & ~mx.eci_terminal_arrived].iloc[0]
    add("screening/test", r, "R4", f"mixed direction, random velocity (nu {r.meta_nu:.2f}), both fail")
    for ax, s0 in (("+V", 500.0), ("-H", 800.0)):
        q = base[base.meta_axis == ax]
        r = q.loc[(np.log(q.sep_m / s0)).abs().idxmin()]
        add("maingrid/grid", r, "R5", f"{ax} close range unsaturated, {r.sep_m:.0f} m (CW saturated {r.cw_sat_total_s:.0f} s)")
    rf = one["maingrid/refine"]
    rb = rf[(rf.wn_ratio == 6.0) & (rf.a_max == 0.05) & (rf.cw_sat_total_s > 0)]
    for ax in ("+R", "+V", "+H"):
        q = rb[rb.meta_axis == ax]
        r = q.loc[(np.log(q.cw_vs_eci_max_pos_disc_m)).abs().idxmin()]
        add("maingrid/refine", r, "R6", f"{ax} near 1 m threshold (peak {r.cw_vs_eci_max_pos_disc_m:.4f} m)")
    b = one["maingrid/refinebands"].copy()
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
        add("maingrid/refinebands", r, "R7", f"{why} ({r.meta_axis} {r.sep_m:.1f} m, {r.wn_ratio:g}n, {r.a_max:g}; CW {r.cw_terminal_dwell_margin_s:+.3f} s, ref {r.eci_terminal_dwell_margin_s:+.3f} s, {r.outcome_terminal_cw})")
    q = base[(base.meta_axis == "+H")]
    r = q.loc[q.sep_m.idxmax()]
    add("maingrid/grid", r, "R8", f"+H {r.sep_m:.0f} m strongly saturated long approach")
    f = h2[h2.so_status == "failed"].sort_values("sep_m").iloc[0]
    add("mechanisms/horizon2", f, "R9", f"two-orbit quadratic numerical failure ({f.meta_axis} {f.sep_m:.0f} m, {f.wn_ratio:g}n, {f.a_max:g})")
    S = pd.DataFrame(sel)
    if S.sid.duplicated().any():
        raise SystemExit("duplicate selection")
    cas = []
    cache = {}
    for s in sel:
        if s["campaign"] not in cache:
            cache[s["campaign"]] = scen(s["campaign"])
        d = dict(cache[s["campaign"]][s["sid"]])
        d["meta"] = dict(d.get("meta", {}), verifier_rule=s["rule"], source_campaign=s["campaign"])
        cas.append(d)
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(os.path.join(OUT, "selection.csv"), index=False)
    json.dump(cas, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
    rec = {"selected_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_cases": len(cas), "code": C.coderev(),
           "cases_sha256": C.sha256(os.path.join(OUT, "cases.json")),
           "selection_sha256": C.sha256(os.path.join(OUT, "selection.csv")),
           "note": "selected before any independent-verifier result existed"}
    C.writejson(os.path.join(OUT, "selection.json"), rec)
    print(S.to_string())
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
