"""Adaptive refinement of separation intervals around boundaries.

For each scenario family (all main-grid settings except separation) and each
boolean predicate (threshold exceedance, arrival status, outcome
disagreement), every adjacent pair of separations with different predicate
values is a bracket. Brackets wider than the tolerance are refined: on first
refinement a bracket receives several interior log-spaced points (to expose
multiple crossings inside one grid cell, since monotonicity is not assumed);
afterwards each remaining bracket is bisected geometrically. Iteration stops
when every bracket satisfies hi/lo - 1 <= tol or the round limit is reached.
All brackets (not just the first) are reported.
"""

import math

import numpy as np
import pandas as pd


def family_key_cols():
    return ["meta_axis", "wn_ratio", "a_max", "meta_velocity", "horizon_orbits", "unrestricted"]


def predicates(ref="eci", thresholds=(0.1, 1.0)):
    P = {}
    for thr in thresholds:
        P[f"disc_gt_{thr:g}m"] = (lambda d, thr=thr: d[f"cw_vs_{ref}_max_pos_disc_m"] > thr)
    P["cw_terminal"] = lambda d: d["cw_terminal_arrived"].astype(bool)
    P["nl_terminal"] = lambda d: d[f"{ref}_terminal_arrived"].astype(bool)
    P["cw_legacy"] = lambda d: d["cw_legacy_arrived"].astype(bool)
    P["nl_legacy"] = lambda d: d[f"{ref}_legacy_arrived"].astype(bool)
    P["terminal_disagree"] = lambda d: d["outcome_terminal_cw"].isin(["cw_only", "nl_only"])
    return P


def find_brackets(df, preds, tol):
    """Return a DataFrame of all brackets (one row per predicate change)."""
    rows = []
    keys = family_key_cols()
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


def new_points(brackets, round_index, first_round_points=4):
    """Separations to add for unresolved brackets: a multi-point interior scan
    in round 0 (exposes several crossings inside one grid cell), geometric
    bisection afterwards."""
    out = []
    for _, b in brackets[~brackets["resolved"]].iterrows():
        fam = tuple(b[c] for c in family_key_cols())
        lo, hi = math.log(b["sep_lo"]), math.log(b["sep_hi"])
        if round_index == 0:
            k = first_round_points
            pts = [math.exp(lo + (hi - lo) * j / (k + 1)) for j in range(1, k + 1)]
        else:
            pts = [math.exp(0.5 * (lo + hi))]
        for p in pts:
            out.append((fam, p))
    # deduplicate (same family, same separation to 1e-9 relative)
    seen = set()
    dedup = []
    for fam, p in out:
        key = (fam, float(f"{p:.9g}"))
        if key not in seen:
            seen.add(key)
            dedup.append((fam, float(f"{p:.9g}")))
    return dedup


__all__ = ["family_key_cols", "predicates", "find_brackets", "new_points"]
