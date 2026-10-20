import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reportutil as RU

SC = os.path.join(RU.RESULTS, "screening")
METHODLABEL = {"sep_only": "separation only", "sat_only": "CW saturation only", "combined": "combined (D1 + dwell margin)",
               "traj_1m": "D1 <= threshold (1 m target)", "traj_0.1m": "D1 <= threshold (0.1 m target)",
               "outcome": "outcome rule", "ff_flag": "falsely-favourable flag"}


def fmtrate(r):
    if not r["n"]:
        return "0/0"
    return f"{int(r['k'])}/{int(r['n'])} ({100*r['rate']:.1f}%, CI {100*r['ci95_lo']:.1f}-{100*r['ci95_hi']:.1f}%)"


def main():
    RU.setup()
    frozen = json.load(open(os.path.join(SC, "frozen", "frozen.json")))
    dev = pd.read_csv(os.path.join(SC, "frozen", "devperformance.csv"))
    perf = pd.read_csv(os.path.join(SC, "performance.csv")) if os.path.exists(os.path.join(SC, "performance.csv")) else pd.DataFrame()
    P = frozen["params"]
    prow = []
    for m, pp in P.items():
        prow.append({"method": METHODLABEL.get(m, m), "rule": frozen["rules"].get(m, ""),
                     "frozen parameters": "; ".join(f"{k} = {v:.6g}" if isinstance(v, float) else f"{k} = {v}" for k, v in pp.items())})
    RU.writetable(pd.DataFrame(prow), "screeningfrozen", "Frozen screening rules and parameters (fitted on "
                  f"{frozen['n_dev']} development scenarios: main grid, refinement points and 200 mixed development "
                  f"cases; thresholds use a factor {frozen['K']} margin). Frozen at {frozen['frozen_at']}.", "tab:screen_frozen")
    rows = []
    allp = pd.concat([dev, perf], ignore_index=True)
    sets = [s for s in ("dev:all", "test:all", "stress:all", "band_test:all", "test:direction=mixed", "test:velocity=random",
                        "test:in_dev_domain", "test:outside_dev_domain") if s in set(allp["set"])]
    for m in ("sep_only", "sat_only", "combined", "traj_1m", "traj_0.1m", "outcome"):
        for s in sets:
            q = allp[(allp.method == m) & (allp["set"] == s)].set_index("metric")
            if q.empty:
                continue
            rows.append({"method": METHODLABEL[m], "target": q.target.iloc[0], "set": s,
                         "accepted": fmtrate(q.loc["accepted"]),
                         "incorrect accepts / accepted": fmtrate(q.loc["incorrect_accept_given_accepted"]),
                         "violations accepted": fmtrate(q.loc["accepted_among_violating"]),
                         "unnecessary referrals / referred": fmtrate(q.loc["unnecessary_referral_given_referred"]),
                         "missed reference failures": fmtrate(q.loc["missed_nl_failures"]),
                         "missed falsely favourable": fmtrate(q.loc["missed_false_favourable"])})
    t = pd.DataFrame(rows)
    RU.writetable(t, "screeningperformance", "Screening performance with exact denominators and Clopper-Pearson "
                  "95% intervals. 'incorrect accepts / accepted': accepted scenarios that violate the method's target; "
                  "'violations accepted': share of target-violating scenarios that were accepted; 'missed reference "
                  "failures': accepted scenarios in which the reference does not arrive, over all such scenarios. "
                  "Zero observed incorrect acceptances do not imply a zero future rate.", "tab:screen_perf")
    ff = allp[allp.method == "ff_flag"]
    if not ff.empty:
        r = []
        for s, q in ff.groupby("set"):
            q = q.set_index("metric")
            r.append({"set": s, "falsely favourable detected": fmtrate(q.loc["ff_detected_among_ff"]),
                      "CW arrivals flagged": fmtrate(q.loc["flagged_among_cw_arrivals"]),
                      "false alarms among true arrivals": fmtrate(q.loc["false_alarm_among_true_arrivals"])})
        RU.writetable(pd.DataFrame(r), "screeningflag", "Detection of falsely favourable CW arrival predictions "
                      "by the dwell-margin flag.", "tab:screen_ff")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    data = []
    for setname, col, mk in (("test", RU.BLUE, "o"), ("stress", RU.ORANGE, "s")):
        f = os.path.join(SC, f"decisions{setname.replace('_', '')}.csv")
        if not os.path.exists(f):
            continue
        dd = pd.read_csv(f)
        axs[0].loglog(dd.disc_m, dd.f_d1, mk, ms=3, color=col, alpha=0.8, label=f"{setname} set", mfc="none" if setname == "stress" else col)
        for r_ in dd.itertuples():
            data.append({"panel": "a", "set": setname, "sid": r_.sid, "true_peak_m": r_.disc_m, "d1_m": r_.f_d1,
                         "accept_combined": r_.accept_combined, "adequate": r_.adequate})
    lim = [1e-6, 1e7]
    axs[0].plot(lim, lim, color=RU.MUTED, lw=0.8)
    axs[0].axvline(1, color=RU.MUTED, lw=0.6)
    axs[0].axhline(P["traj_1m"]["D1_max_m"], color=RU.INK2, lw=0.8, ls="--")
    axs[0].set_xlim(lim)
    axs[0].set_ylim(lim)
    axs[0].set_xlabel("true peak CW-to-reference difference [m]")
    axs[0].set_ylabel("CW-only first-order estimate D1 [m]")
    axs[0].set_title("(a) estimate vs reference (held-out sets)")
    axs[0].legend(fontsize=6.5, loc="upper left")
    q = perf[(perf.metric == "accepted") & (perf["set"] == "test:all")] if not perf.empty else pd.DataFrame()
    if not q.empty:
        order = ["sep_only", "sat_only", "combined"]
        q = q.set_index("method").loc[[m for m in order if m in set(q.method)]]
        bad = perf[(perf.metric == "incorrect_accept_given_accepted") & (perf["set"] == "test:all")].set_index("method").loc[q.index]
        y = np.arange(len(q))
        axs[1].barh(y, 100 * q["rate"], color=RU.BLUE, height=0.42)
        axs[1].errorbar(100 * q["rate"], y, xerr=[100 * (q["rate"] - q.ci95_lo), 100 * (q.ci95_hi - q["rate"])],
                        fmt="none", ecolor=RU.INK, lw=0.8, capsize=2)
        for yi, (m, r_) in zip(y, q.iterrows()):
            b = bad.loc[m]
            data.append({"panel": "b", "set": "test", "method": m, "accepted_k": int(r_["k"]), "n": int(r_["n"]),
                         "accept_rate": r_["rate"], "ci95_lo": r_["ci95_lo"], "ci95_hi": r_["ci95_hi"],
                         "incorrect_accepted_k": int(b["k"])})
            axs[1].text(2, yi + 0.33, f"{int(r_['k'])}/{int(r_['n'])} accepted, {int(b['k'])} in error",
                        va="center", fontsize=6.5, color=RU.INK2)
        axs[1].set_yticks(y)
        axs[1].set_yticklabels([METHODLABEL[m] for m in q.index], fontsize=7)
        axs[1].set_xlim(0, 100)
        axs[1].set_xlabel("test scenarios accepted [%] (95% CI)")
        axs[1].set_title("(b) acceptance on the broad test set")
    fig.tight_layout()
    RU.save(fig, "screening", pd.DataFrame(data))
    print(t.to_string())


if __name__ == "__main__":
    main()
