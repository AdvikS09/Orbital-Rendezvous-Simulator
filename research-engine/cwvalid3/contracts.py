import math

CONTRVER = "v3.1-contract-3"
# A branch that ends within 1e-6 s of the planned horizon counts as having reached it
HORTOLS = 1e-6
LABELS = ("ok", "violation", "exit", "failure", "unresolved", "interrupted")
ADVLABELS = ("violation", "exit", "failure", "unresolved", "interrupted")
TRUTHSTATUSES = ("completed_valid", "physical_domain_exit", "numerical_failure", "unresolved", "interrupted")
WORKERSTATUSES = ("ok", "reference_failed", "exception", "worker_crash", "missing")
OPS = ("worker_crash", "missing")
FEATSTAGEDONE = ("ok", "reference_failed")
KINDS = ("continuous", "held")
HELDCELLS = ("D3", "D4")

STATUSMAP = {"continuous": {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"},
             "held": {"completed": "completed", "domain_exit": "domain_exit", "numerical_failure": "numerical_failure"}}
ENDFIELD = {"continuous": "{b}_t_final_s", "held": "{b}_t_end_s"}
PAIRFIELDS = {"continuous": {"disc": "{p}_max_pos_disc_m", "cover": "{p}_common_horizon_s"},
              "held": {"disc": "{p}_peak_m", "cover": "{p}_common_end_s"}}
ARRFIELDS = {"continuous": (("cw_terminal_arrived", "eci_terminal_arrived", "nerm_terminal_arrived"),
                            ("cw_terminal_dwell_margin_s", "eci_terminal_dwell_margin_s")),
             "held": (("cw_arrived", "eci_arrived", "nerm_arrived"), ("cw_dwell_margin_s", "eci_dwell_margin_s"))}
FEATFIELDS = {
    "continuous": {"A": ("d1_peak_m", "d1_status"), "B": ("d1u_peak_m", "d1u_status"), "C": ("cres_peak_m", "cres_status"),
                   "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
    "held": {"A_H": ("A_H_peak_m", "A_H_status"), "B_H": ("B_H_peak_m", "B_H_status"),
             "A_naive": ("A_naive_peak_m", "A_naive_status"), "C_H": ("C_H_peak_m", "C_H_status"),
             "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
}
FEATRANGE = {"sat": (0.0, 1.0)}


def kindof(r, kind=None):
    if kind is not None:
        if kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}")
        return kind
    return "held" if r.get("cell") in HELDCELLS else "continuous"


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def nonneg(x):
    return finite(x) and float(x) >= 0.0


def horprob(r):
    T, n = r.get("horizon_s"), r.get("n_radps")
    orbits = r.get("horizon_orbits", 1.0)
    if not (finite(T) and float(T) > 0):
        return f"horizon_s {T!r} missing or not positive"
    if not (finite(n) and float(n) > 0 and finite(orbits) and float(orbits) > 0):
        return f"cannot check the horizon: n_radps {n!r}, horizon_orbits {orbits!r}"
    planned = float(orbits) * 2.0 * math.pi / float(n)
    if abs(float(T) - planned) > 1e-9 * planned:
        return f"horizon_s {T} differs from the planned horizon {planned}"
    return ""


def branch(r, b, kind):
    T = r.get("horizon_s")
    raw = r.get(f"{b}_status")
    st = STATUSMAP[kind].get(raw) if isinstance(raw, str) else None
    tend = r.get(ENDFIELD[kind].format(b=b))
    if st is None:
        return "unresolved", tend, f"{b}: unrecognised or missing status {raw!r}"
    if kind == "continuous" and st == "completed" and r.get(f"{b}_completed_horizon") is not True:
        return "unresolved", tend, f"{b}: status ok but completed_horizon is {r.get(f'{b}_completed_horizon')!r}"
    if kind == "continuous" and st == "completed" and r.get(f"{b}_numerical_failure") is True:
        return "unresolved", tend, f"{b}: status ok but numerical_failure flag set"
    if st == "completed":
        if not (finite(T) and finite(tend)):
            return "unresolved", tend, f"{b}: completed without finite horizon/end time"
        if float(tend) < float(T) - HORTOLS:
            return "unresolved", tend, f"{b}: completed status but ended at {tend} < horizon {T}"
    return st, tend, ""


def base(**kw):
    out = {"disc_m": math.nan, "u_num_m": math.nan, "cw_status": None, "ref_status": None, "cross_status": None,
           "ref_t_end_s": math.nan, "contract": CONTRVER}
    out.update(kw)
    return out


# Conservative: a trajectory that cannot be confirmed as completed or as a physical exit is
# unresolved or interrupted, never completed.
def truth_record(r, kind=None):
    kind = kindof(r, kind)
    ws = r.get("worker_status")
    if ws in OPS:
        return base(truth_status="interrupted", reason=f"operational interruption ({ws})")
    if ws == "exception":
        return base(truth_status="numerical_failure", reason="worker exception before the record was complete")
    if ws not in FEATSTAGEDONE:
        return base(truth_status="unresolved", reason=f"unrecognised worker_status {ws!r}")
    c, _, rcw = branch(r, "cw", kind)
    ref, refend, rref = branch(r, "eci", kind)
    cc, _, rcc = branch(r, "nerm", kind)
    pf = PAIRFIELDS[kind]
    disc = r.get(pf["disc"].format(p="cw_vs_eci"))
    discx = r.get(pf["disc"].format(p="cw_vs_nerm"))
    disc = float(disc) if nonneg(disc) else math.nan
    discx = float(discx) if nonneg(discx) else math.nan
    out = base(disc_m=disc, u_num_m=abs(disc - discx) if (math.isfinite(disc) and math.isfinite(discx)) else math.nan,
               cw_status=c, ref_status=ref, cross_status=cc, ref_t_end_s=float(refend) if finite(refend) else math.nan)
    hp = horprob(r)
    if hp:
        return {**out, "truth_status": "unresolved", "reason": hp}
    if ws == "ok" and (r.get("failed_stage") or r.get("reference_error") or r.get("error")):
        return {**out, "truth_status": "unresolved", "reason": "record marked ok but carries an error or failed stage"}
    if ws == "reference_failed":
        return {**out, "truth_status": "unresolved",
                "reason": "reference/cross-check stage raised: " + str(r.get("reference_error", ""))[:200]}
    if "numerical_failure" in (c, ref, cc):
        return {**out, "truth_status": "numerical_failure", "reason": f"cw={c} ref={ref} cross={cc}"}
    if "unresolved" in (c, ref, cc):
        return {**out, "truth_status": "unresolved", "reason": ";".join(x for x in (rcw, rref, rcc) if x)}
    if ref == "domain_exit" and cc == "domain_exit":
        return {**out, "truth_status": "physical_domain_exit", "reason": "reference and cross-check below the altitude floor"}
    if ref != cc:
        return {**out, "truth_status": "unresolved", "reason": f"reference {ref} vs cross-check {cc}"}
    if c != "completed":
        return {**out, "truth_status": "unresolved", "reason": f"cw={c}"}
    if not math.isfinite(out["u_num_m"]):
        return {**out, "truth_status": "unresolved", "reason": "missing, negative or non-finite discrepancy"}
    T = float(r["horizon_s"])
    for p in ("cw_vs_eci", "cw_vs_nerm"):
        cov = r.get(pf["cover"].format(p=p))
        if not finite(cov) or float(cov) < T - HORTOLS:
            return {**out, "truth_status": "unresolved", "reason": f"{p} discrepancy covers only [0, {cov}] of [0, {T}]"}
    return {**out, "truth_status": "completed_valid", "reason": ""}


def critlabel(tr, tol, band_factor=10.0, band_floor_m=1e-6):
    st = tr["truth_status"]
    if st == "physical_domain_exit":
        return "exit"
    if st == "numerical_failure":
        return "failure"
    if st == "interrupted":
        return "interrupted"
    if st != "completed_valid":
        return "unresolved"
    band = max(band_factor * tr["u_num_m"], band_floor_m)
    d = tr["disc_m"]
    if d > tol + band:
        return "violation"
    if d < tol - band:
        return "ok"
    return "unresolved"


def arrlabel(r, tr, kind=None, dwell_ambiguity_s=0.5):
    kind = kindof(r, kind)
    if tr["truth_status"] == "physical_domain_exit":
        return {"arrival_status": "not_applicable", "cw_arrived": None, "ref_arrived": None, "reason": "reference exit"}
    if tr["truth_status"] != "completed_valid":
        return {"arrival_status": "unresolved", "cw_arrived": None, "ref_arrived": None, "reason": tr["truth_status"]}
    keys, mkeys = ARRFIELDS[kind]
    vals = [r.get(k) for k in keys]
    margins = [r.get(k) for k in mkeys]
    if not all(isinstance(v, bool) for v in vals):
        return {"arrival_status": "unresolved", "cw_arrived": None, "ref_arrived": None,
                "reason": "missing or non-Boolean arrival field: " + ",".join(k for k, v in zip(keys, vals) if not isinstance(v, bool))}
    cwa, refa, cca = vals
    for m, a in zip(margins, (cwa, refa)):
        if a and not finite(m):
            return {"arrival_status": "unresolved", "cw_arrived": cwa, "ref_arrived": refa, "reason": "arrived without finite dwell margin"}
    if refa != cca or any(finite(m) and abs(float(m)) < dwell_ambiguity_s for m in margins):
        return {"arrival_status": "unresolved", "cw_arrived": cwa, "ref_arrived": refa,
                "reason": "cross-check disagreement or dwell margin within ambiguity"}
    return {"arrival_status": "arrived" if refa else "completed_non_arrival", "cw_arrived": cwa, "ref_arrived": refa, "reason": ""}


# Uses the nominal CW branch only. A reference failure after the feature exists must not turn an
# accepted case into a referral.
def featavail(r, method, kind=None):
    kind = kindof(r, kind)
    if method not in FEATFIELDS[kind]:
        raise ValueError(f"method {method!r} is not defined for {kind} records")
    ws = r.get("worker_status")
    recov = ws == "worker_crash" and r.get("feature_stage") == "complete" and (
        r.get("feature_checkpoint_recovered") is True or r.get("feature_recovery") == "driver_nominal_recovery")
    if ws not in FEATSTAGEDONE and not recov:
        return math.nan, "unavailable", f"worker_status {ws!r}"
    hp = horprob(r)
    if hp:
        return math.nan, "unavailable", hp
    c, _, why = branch(r, "cw", kind)
    if c != "completed":
        return math.nan, "unavailable", f"nominal CW branch not completed through T ({c}{': ' + why if why else ''})"
    key, st = FEATFIELDS[kind][method]
    if st is not None and r.get(st) != "ok":
        return math.nan, "unavailable", f"estimator status {r.get(st)!r}"
    v = r.get(key)
    if not finite(v):
        return math.nan, "unavailable", "missing or non-finite feature"
    lo, hi = FEATRANGE.get(method, (0.0, math.inf))
    if not (lo <= float(v) <= hi):
        return math.nan, "unavailable", f"feature {v} outside [{lo}, {hi}]"
    return float(v), "ok", ""


def feature_value(r, method, kind=None):
    return featavail(r, method, kind)[0]


def labelcounts(accepted, labels):
    vallabels(labels)
    out = {}
    for lab in LABELS:
        out[f"acc_{lab}"] = sum(1 for a, x in zip(accepted, labels) if a and x == lab)
        out[f"ref_{lab}"] = sum(1 for a, x in zip(accepted, labels) if (not a) and x == lab)
    checklabelpart(out, sum(1 for a in accepted if a))
    return out


def checklabelpart(counts, n_accepted):
    s = sum(counts.get(f"acc_{lab}", 0) for lab in LABELS)
    if s != n_accepted:
        raise ValueError(f"label partition violated: sum over labels {s} != n_accepted {n_accepted}")


def vallabels(labels):
    bad = sorted({str(x) for x in labels if x not in LABELS})
    if bad:
        raise ValueError(f"unknown labels: {bad}")


# Every non-ok label, including interrupted and unresolved, counts as adverse when accepted
def evalrule(values, labels, threshold):
    import numpy as np
    from cwvalid3.calibration import cptwosided, ucb
    v = np.asarray(values, float)
    labels = list(labels)
    vallabels(labels)
    acc = (np.isfinite(v) & (v <= threshold)) if threshold is not None else np.zeros(v.size, bool)
    cnt = labelcounts([bool(a) for a in acc], labels)
    n = int(acc.sum())
    adverse = sum(cnt[f"acc_{lab}"] for lab in ADVLABELS)
    known = cnt["acc_violation"] + cnt["acc_exit"]
    out = {"n_planned": int(v.size), "n_feature_unavailable": int((~np.isfinite(v)).sum()), "n_accepted": n,
           "n_referred": int(v.size - n), "coverage": float(acc.mean()) if v.size else math.nan, **cnt,
           "known_adverse_accepted": known, "conservative_adverse_accepted": adverse,
           "cond_violation_rate_conservative": adverse / n if n else math.nan, "contract": CONTRVER}
    lo, hi = cptwosided(adverse, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_conservative_lo"], out["cp95_two_sided_conservative_hi"] = lo, hi
    out["one_sided_95_ucb_conservative"] = ucb(adverse, n, 0.05) if n else math.nan
    return out


def isadverse(label):
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    return label in ADVLABELS
