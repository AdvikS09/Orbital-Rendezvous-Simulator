"""P03 development-only stability of the legacy smallest-violating-feature threshold.

Legacy rule (experiments/v2/estimator_fit.py): threshold = K x min{feature of development cases that
violate the target}, K = 0.8, features A = d1, B = d1u, C = cres, sep, sat; targets 1 m and 0.1 m.
Data: the archived development set only (main_grid/grid, refine, refine_bands, screening/dev; 3,839
cases), with the archive's own labels (label_frame) so the full-data thresholds reproduce the frozen
ones exactly. Optionally the new v3 D0 development split (iid deployment draws) is used to show how
each padding performs on an independent development sample; no calibration/test data are read.

Families for group-aware resampling: designed sweep cases (grid, refine, refine_bands) are grouped
by (axis, velocity rule, omega_n/n, a_max), i.e. one separation sweep and its refinements form one
unit; every random development case is its own unit.

Analyses (rows of results/calibration_stability.csv):
  full_data          recomputed threshold vs frozen value, setter case
  group_bootstrap    B = 2000 family-level resamples: threshold quantiles, setter frequencies
  leave_one_family   threshold with each family removed (only families that change it are listed,
                     plus a summary row)
  leave_one_regime   campaign, axis, velocity rule and saturation class removed in turn
  padding_grid       K in {0.5, ..., 1.0}: dev acceptance and, if available, v3 D0 dev acceptance
                     and accepted adverse cases
  setters            the five smallest violating features per feature/target with parameters
Usage: python code/experiments/v3/p03_stability.py [--boot 2000]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "v2"))
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")

import estimator_fit as EF  # noqa: E402

FEATS = ("d1", "d1u", "cres", "sep", "sat")
NAMES = {"d1": "A", "d1u": "B", "cres": "C", "sep": "sep", "sat": "sat"}
TARGETS = {"traj_ok_1": "1m", "traj_ok_01": "0.1m"}
PADS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_dev():
    parts = []
    for c in EF.DEV:
        F, Lb, D, _ = EF.load_set(c)
        sc = pd.DataFrame([json.loads(l) for l in open(os.path.join(EF.C.RESULTS, c, "scenarios.jsonl")) if l.strip()])
        F = F.merge(Lb[["sid", "traj_ok_1", "traj_ok_01", "disc_m"]], on="sid")
        meta = pd.json_normalize(sc["meta"])
        F["campaign"] = c
        F["axis"] = meta["axis"].values
        F["velocity"] = meta["velocity"].values if "velocity" in meta else "zero"
        F["wn_ratio"] = sc["wn_ratio"].values
        F["a_max"] = sc["a_max"].values
        parts.append(F)
    F = pd.concat(parts, ignore_index=True)
    designed = F.campaign != "screening/dev"
    F["family"] = np.where(designed, "fam:" + F.axis.astype(str) + "|" + F.velocity.astype(str) + "|wn" +
                           F.wn_ratio.round(6).astype(str) + "|am" + F.a_max.round(6).astype(str), "rand:" + F.sid)
    F["sat_class"] = np.where(F.sat <= 0, "unsaturated", np.where(F.sat >= 1 - 1e-12, "fully_saturated", "partly_saturated"))
    return F


def legacy(F, feat, target, K=0.8):
    v = F[feat].to_numpy(float)
    bad = ~F[target].to_numpy(bool) & np.isfinite(v)
    if not bad.any():
        return np.nan, None
    i = np.where(bad)[0][np.argmin(v[bad])]
    return K * v[i], F.sid.iloc[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    F = load_dev()
    frozen = json.load(open(EF.C.FROZEN_V2))["rules"]
    rows = []
    fams = F.family.unique()
    print(f"development cases {len(F)}; families {len(fams)} "
          f"(designed {sum(f.startswith('fam:') for f in fams)}, random {sum(f.startswith('rand:') for f in fams)})")
    # optional independent iid development sample (v3 D0 dev)
    v3dev = None
    p = os.path.join(V3, "results", "dev", "dev_cases.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        v3dev = d[(d.cell == "D0") & (d.split == "dev")]
    rng = np.random.default_rng(20261013)
    gidx = {g: np.where(F.family.to_numpy() == g)[0] for g in fams}
    for feat in FEATS:
        for target, tl in TARGETS.items():
            thr, setter = legacy(F, feat, target)
            fz = frozen.get(f"{'traj1' if tl == '1m' else 'traj01'}_{feat}", {})
            fval = fz.get("threshold") if fz.get("status") == "ok" else None
            acc = int((F[feat].to_numpy(float) <= thr).sum()) if np.isfinite(thr) else 0
            rows.append({"analysis": "full_data", "method": NAMES[feat], "criterion": tl, "item": "all", "K": 0.8,
                         "threshold": thr, "frozen_threshold": fval,
                         "matches_frozen": (fval is not None and thr == fval), "setter_sid": setter,
                         "dev_accepted": acc, "dev_n": len(F),
                         "note": "" if fval is not None else str(fz.get("status", ""))[:200]})
            # group bootstrap
            ths, setters = [], {}
            for b in range(a.boot):
                pick = rng.choice(fams, size=fams.size, replace=True)
                idx = np.concatenate([gidx[g] for g in pick])
                t, s = legacy(F.iloc[idx], feat, target)
                ths.append(t)
                setters[s] = setters.get(s, 0) + 1
            ths = np.array(ths, float)
            top = sorted(setters.items(), key=lambda x: -x[1])[:5]
            rows.append({"analysis": "group_bootstrap", "method": NAMES[feat], "criterion": tl, "item": f"B={a.boot}",
                         "K": 0.8, "threshold": thr, "boot_q05": np.nanquantile(ths, 0.05), "boot_q50": np.nanquantile(ths, 0.5),
                         "boot_q95": np.nanquantile(ths, 0.95), "boot_min": np.nanmin(ths), "boot_max": np.nanmax(ths),
                         "boot_frac_equal_full": float(np.mean(ths == thr)),
                         "setter_frequencies": json.dumps({str(k): v / a.boot for k, v in top})})
            # leave one family out
            changed = []
            for g in fams:
                keep = np.ones(len(F), bool)
                keep[gidx[g]] = False
                t, s = legacy(F[keep], feat, target)
                if not (t == thr):
                    changed.append((g, t, s))
            for g, t, s in changed:
                rows.append({"analysis": "leave_one_family", "method": NAMES[feat], "criterion": tl, "item": g, "K": 0.8,
                             "threshold": t, "threshold_full": thr, "rel_change": (t - thr) / thr if thr else np.nan,
                             "setter_sid": s})
            rows.append({"analysis": "leave_one_family_summary", "method": NAMES[feat], "criterion": tl,
                         "item": f"{len(fams)} families", "K": 0.8, "threshold": thr,
                         "n_families_changing_threshold": len(changed),
                         "max_rel_change": max((abs((t - thr) / thr) for _, t, _ in changed), default=0.0)})
            # leave one regime out
            for col in ("campaign", "axis", "velocity", "sat_class"):
                for val in sorted(F[col].astype(str).unique()):
                    keep = F[col].astype(str).to_numpy() != val
                    t, s = legacy(F[keep], feat, target)
                    rows.append({"analysis": "leave_one_regime", "method": NAMES[feat], "criterion": tl,
                                 "item": f"{col}={val}", "n_removed": int((~keep).sum()), "K": 0.8, "threshold": t,
                                 "threshold_full": thr, "rel_change": (t - thr) / thr if thr else np.nan, "setter_sid": s})
            # padding grid
            for K in PADS:
                t, s = legacy(F, feat, target, K)
                r = {"analysis": "padding_grid", "method": NAMES[feat], "criterion": tl, "item": f"K={K}", "K": K,
                     "threshold": t, "dev_accepted": int((F[feat].to_numpy(float) <= t).sum()),
                     "dev_violations_accepted": int(((F[feat].to_numpy(float) <= t) & ~F[target].to_numpy(bool)).sum()),
                     "dev_n": len(F)}
                if v3dev is not None and len(v3dev):
                    col = {"d1": "f_A", "d1u": "f_B", "cres": "f_C", "sep": "f_sep", "sat": "f_sat"}[feat]
                    lab = "label_1m" if tl == "1m" else "label_01m"
                    f = v3dev[col].to_numpy(float)
                    acc = np.isfinite(f) & (f <= t)
                    adverse = v3dev[lab].isin(["violation", "exit", "failure", "unresolved"]).to_numpy()
                    r.update(v3_D0_dev_n=len(v3dev), v3_D0_dev_accepted=int(acc.sum()),
                             v3_D0_dev_adverse_accepted=int((acc & adverse).sum()))
                rows.append(r)
            # setters
            v = F[feat].to_numpy(float)
            bad = ~F[target].to_numpy(bool) & np.isfinite(v)
            order = np.where(bad)[0][np.argsort(v[bad])][:5]
            for rank, i in enumerate(order):
                c = F.iloc[i]
                rows.append({"analysis": "setters", "method": NAMES[feat], "criterion": tl, "item": f"rank{rank + 1}",
                             "setter_sid": c.sid, "feature_value": v[i], "truth_disc_m": c.disc_m, "campaign": c.campaign,
                             "axis": c.axis, "velocity": c.velocity, "wn_ratio": c.wn_ratio, "a_max": c.a_max,
                             "sep_m": c.sep, "sat_fraction": c.sat, "family": c.family})
    out = pd.DataFrame(rows)
    out.insert(0, "population", "archived development set (3,839 cases; designed families + 200 random)")
    out.to_csv(os.path.join(V3, "results", "calibration_stability.csv"), index=False)
    s = out[out.analysis.isin(["full_data", "group_bootstrap", "leave_one_family_summary"])]
    print(s[["analysis", "method", "criterion", "threshold", "matches_frozen", "boot_q05", "boot_q95",
             "boot_frac_equal_full", "n_families_changing_threshold", "max_rel_change"]].to_string())


if __name__ == "__main__":
    main()
