import math

import numpy as np
import pandas as pd


def famkeycols():
    return ["meta_axis", "wn_ratio", "a_max", "meta_velocity", "horizon_orbits", "unrestricted"]


def predic(ref="eci", thresh=(0.1, 1.0)):
    P = {}
    for thr in thresh:
        P[f"disc_gt_{thr:g}m"] = (lambda d, thr=thr: d[f"cw_vs_{ref}_max_pos_disc_m"] > thr)
    P["cw_terminal"] = lambda d: d["cw_terminal_arrived"].astype(bool)
    P["nl_terminal"] = lambda d: d[f"{ref}_terminal_arrived"].astype(bool)
    P["cw_legacy"] = lambda d: d["cw_legacy_arrived"].astype(bool)
    P["nl_legacy"] = lambda d: d[f"{ref}_legacy_arrived"].astype(bool)
    P["terminal_disagree"] = lambda d: d["outcome_terminal_cw"].isin(["cw_only", "nl_only"])
    return P


def findbrack(df, preds, tol):
    rows = []
    keys = famkeycols()
    for fam, g in df.groupby(keys, dropna=False):
        g = g.sort_values("sep_m")
        seps = g["sep_m"].to_numpy()
        for pname, pf in preds.items():
            v = pf(g).to_numpy()
            ch = np.where(v[1:] != v[:-1])[0]
            for i in ch:
                lo, hi = seps[i], seps[i + 1]
                rows.append({**dict(zip(keys, fam)), "predicate": pname, "sep_lo": lo, "sep_hi": hi,
                             "val_lo": bool(v[i]), "val_hi": bool(v[i + 1]),
                             "rel_width": hi / lo - 1.0, "resolved": hi / lo - 1.0 <= tol,
                             "sid_lo": g["sid"].iloc[i], "sid_hi": g["sid"].iloc[i + 1],
                             "n_changes_in_family": len(ch)})
    return pd.DataFrame(rows)


def newpoints(brackets, roundidx, firstroundpoints=4):
    out = []
    for _, b in brackets[~brackets["resolved"]].iterrows():
        fam = tuple(b[c] for c in famkeycols())
        lo, hi = math.log(b["sep_lo"]), math.log(b["sep_hi"])
        if roundidx == 0:
            k = firstroundpoints
            pts = [math.exp(lo + (hi - lo) * j / (k + 1)) for j in range(1, k + 1)]
        else:
            pts = [math.exp(0.5 * (lo + hi))]
        for p in pts:
            out.append((fam, p))
    seen = set()
    dedup = []
    for fam, p in out:
        key = (fam, float(f"{p:.9g}"))
        if key not in seen:
            seen.add(key)
            dedup.append((fam, float(f"{p:.9g}")))
    return dedup
