"""v3.1 record contracts: explicit status whitelists, required fields, coverage-through-T checks,
type-checked arrival Booleans, separately derived feature availability and an exhaustive label set.

Why (CORRECTION_REPORT C02): the v3.0 labelling functions (cwvalid3/calibration.py, kept unchanged as
the frozen definition) were written for records produced by the v3.0 workers and trusted their shape.
Adversarial inputs showed that
  - missing arrival Booleans became a false/false non-arrival instead of unresolved;
  - unrecognised status strings could map to completed (continuous 'ok' was the only check);
  - a reference with a short horizon could pass when only CW completion was checked;
  - an unknown label entered the accepted denominator without entering the adverse numerator;
  - a reference or metric exception before feature persistence left the feature unavailable;
  - a worker crash (operational) was labelled as a numerical failure.
Only the last one touches stored v3.0 records (43 D1 calibration worker_crash records; see
v3_1_correction/results/existing_record_contract_impact.csv). The v3.1 contract:
  * a record whose required fields are missing, mistyped, negative or non-finite never yields
    completed_valid; unknown or malformed statuses are 'unresolved' (adverse if accepted);
  * a branch is complete only if its status is whitelisted as completed AND its end time reaches the
    horizon T (tolerance HORIZON_TOL_S); a completed_valid case also needs both pair discrepancies to
    cover [0, T];
  * worker_crash and missing records are 'interrupted' (operational), never a numerical or physical
    outcome; they stay adverse if accepted and are reported separately;
  * an exception in the reference/cross-check stage after the feature was stored ('reference_failed')
    gives 'unresolved' truth with the feature kept, as PROSPECTIVE_PLAN.json ('exclusions') specifies:
    "post-acceptance reference failures stay accepted with unresolved truth";
  * feature availability depends only on the worker reaching the feature stage, the nominal CW branch
    and the estimator status/value, never on the reference outcome;
  * labels are drawn from LABELS; counts must partition n_accepted exactly (check_label_partition); the v3.1
    evaluate_rule below is the partition-checked counterpart of the frozen calibration.evaluate_rule;
  * the horizon must be positive and equal the planned horizon (horizon_orbits x 2 pi / n_radps, default one
    orbit); an 'ok' record that carries an error, a failed stage or a branch numerical_failure flag is unresolved;
  * worker 'exception' is a numerical failure, as in the frozen specification ("numerical failure of the ...
    worker"); failed_stage records where it happened (including setup errors). It is adverse either way;
  * a worker_crash record whose feature stage was recovered keeps the feature; truth stays 'interrupted' (adverse if
    accepted). Recovered means: merged from a checkpoint of the same run session and engine
    (feature_checkpoint_recovered), or (v3.1.1) computed by the driver-level nominal-only recovery without reference
    branches (feature_recovery 'driver_nominal_recovery'). Per-feature failures inside a completed worker are carried
    by the method's own status field (e.g. 'feature_extraction_failed', 'naive_feature_failed'), so they remove only
    that method; feature_errors is diagnostic and does not change truth.

Versions: v3.1-contract-2 (v3.1); v3.1-contract-3 (v3.1.1: adds the driver-level recovery to the recovered condition;
identical to -2 for every record without feature_recovery, i.e. every stored v3.0/v3.1 record).

Record kinds and status vocabularies:
  continuous (D0-D2, D5; cwvalid production / cwvalid3.continuous): 'ok' | 'terminated' | 'failed'
  held (D3, D4; cwvalid3.sampled.HeldBatch): 'completed' | 'domain_exit' | 'numerical_failure'
"""

import math

CONTRACT_VERSION = "v3.1-contract-3"
HORIZON_TOL_S = 1e-6
LABELS = ("ok", "violation", "exit", "failure", "unresolved", "interrupted")
ADVERSE_LABELS = ("violation", "exit", "failure", "unresolved", "interrupted")
TRUTH_STATUSES = ("completed_valid", "physical_domain_exit", "numerical_failure", "unresolved", "interrupted")
WORKER_STATUSES = ("ok", "reference_failed", "exception", "worker_crash", "missing")
OPERATIONAL = ("worker_crash", "missing")
FEATURE_STAGE_DONE = ("ok", "reference_failed")
KINDS = ("continuous", "held")
HELD_CELLS = ("D3", "D4")

STATUS_MAP = {"continuous": {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"},
              "held": {"completed": "completed", "domain_exit": "domain_exit", "numerical_failure": "numerical_failure"}}
END_FIELD = {"continuous": "{b}_t_final_s", "held": "{b}_t_end_s"}
PAIR_FIELDS = {"continuous": {"disc": "{p}_max_pos_disc_m", "cover": "{p}_common_horizon_s"},
               "held": {"disc": "{p}_peak_m", "cover": "{p}_common_end_s"}}
ARRIVAL_FIELDS = {"continuous": (("cw_terminal_arrived", "eci_terminal_arrived", "nerm_terminal_arrived"),
                                 ("cw_terminal_dwell_margin_s", "eci_terminal_dwell_margin_s")),
                  "held": (("cw_arrived", "eci_arrived", "nerm_arrived"), ("cw_dwell_margin_s", "eci_dwell_margin_s"))}
FEATURE_FIELDS = {
    "continuous": {"A": ("d1_peak_m", "d1_status"), "B": ("d1u_peak_m", "d1u_status"), "C": ("cres_peak_m", "cres_status"),
                   "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
    "held": {"A_H": ("A_H_peak_m", "A_H_status"), "B_H": ("B_H_peak_m", "B_H_status"),
             "A_naive": ("A_naive_peak_m", "A_naive_status"), "C_H": ("C_H_peak_m", "C_H_status"),
             "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
}
FEATURE_RANGE = {"sat": (0.0, 1.0)}          # closed interval; every other feature must be >= 0


def kind_of(r, kind=None):
    """Record kind. Pass it explicitly when known (the campaign cell defines it); the fallback infers
    it from the cell name. A record whose end-time fields contradict the kind is caught by _branch."""
    if kind is not None:
        if kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}")
        return kind
    return "held" if r.get("cell") in HELD_CELLS else "continuous"


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _nonneg(x):
    return _finite(x) and float(x) >= 0.0


def horizon_problem(r):
    """'' if the record's horizon is positive and equals the planned horizon, else the reason."""
    T, n = r.get("horizon_s"), r.get("n_radps")
    orbits = r.get("horizon_orbits", 1.0)
    if not (_finite(T) and float(T) > 0):
        return f"horizon_s {T!r} missing or not positive"
    if not (_finite(n) and float(n) > 0 and _finite(orbits) and float(orbits) > 0):
        return f"cannot check the horizon: n_radps {n!r}, horizon_orbits {orbits!r}"
    planned = float(orbits) * 2.0 * math.pi / float(n)
    if abs(float(T) - planned) > 1e-9 * planned:
        return f"horizon_s {T} differs from the planned horizon {planned}"
    return ""


def _branch(r, b, kind):
    """Normalised (status, end_time, reason) for branch b ('cw', 'eci', 'nerm')."""
    T = r.get("horizon_s")
    raw = r.get(f"{b}_status")
    st = STATUS_MAP[kind].get(raw) if isinstance(raw, str) else None
    t_end = r.get(END_FIELD[kind].format(b=b))
    if st is None:
        return "unresolved", t_end, f"{b}: unrecognised or missing status {raw!r}"
    if kind == "continuous" and st == "completed" and r.get(f"{b}_completed_horizon") is not True:
        return "unresolved", t_end, f"{b}: status ok but completed_horizon is {r.get(f'{b}_completed_horizon')!r}"
    if kind == "continuous" and st == "completed" and r.get(f"{b}_numerical_failure") is True:
        return "unresolved", t_end, f"{b}: status ok but numerical_failure flag set"
    if st == "completed":
        if not (_finite(T) and _finite(t_end)):
            return "unresolved", t_end, f"{b}: completed without finite horizon/end time"
        if float(t_end) < float(T) - HORIZON_TOL_S:
            return "unresolved", t_end, f"{b}: completed status but ended at {t_end} < horizon {T}"
    return st, t_end, ""


def _base(**kw):
    out = {"disc_m": math.nan, "u_num_m": math.nan, "cw_status": None, "ref_status": None, "cross_status": None,
           "ref_t_end_s": math.nan, "contract": CONTRACT_VERSION}
    out.update(kw)
    return out


def truth_record(r, kind=None):
    """Conservative truth status of the full-horizon trajectory predicate for one record."""
    kind = kind_of(r, kind)
    ws = r.get("worker_status")
    if ws in OPERATIONAL:
        return _base(truth_status="interrupted", reason=f"operational interruption ({ws})")
    if ws == "exception":
        return _base(truth_status="numerical_failure", reason="worker exception before the record was complete")
    if ws not in FEATURE_STAGE_DONE:
        return _base(truth_status="unresolved", reason=f"unrecognised worker_status {ws!r}")
    cw, _, rcw = _branch(r, "cw", kind)
    ref, ref_end, rref = _branch(r, "eci", kind)
    cc, _, rcc = _branch(r, "nerm", kind)
    pf = PAIR_FIELDS[kind]
    disc = r.get(pf["disc"].format(p="cw_vs_eci"))
    discx = r.get(pf["disc"].format(p="cw_vs_nerm"))
    disc = float(disc) if _nonneg(disc) else math.nan
    discx = float(discx) if _nonneg(discx) else math.nan
    out = _base(disc_m=disc, u_num_m=abs(disc - discx) if (math.isfinite(disc) and math.isfinite(discx)) else math.nan,
                cw_status=cw, ref_status=ref, cross_status=cc, ref_t_end_s=float(ref_end) if _finite(ref_end) else math.nan)
    hp = horizon_problem(r)
    if hp:
        return {**out, "truth_status": "unresolved", "reason": hp}
    if ws == "ok" and (r.get("failed_stage") or r.get("reference_error") or r.get("error")):
        return {**out, "truth_status": "unresolved", "reason": "record marked ok but carries an error or failed stage"}
    if ws == "reference_failed":      # PROSPECTIVE_PLAN.json 'exclusions': stays accepted with unresolved truth
        return {**out, "truth_status": "unresolved",
                "reason": "reference/cross-check stage raised: " + str(r.get("reference_error", ""))[:200]}
    if "numerical_failure" in (cw, ref, cc):
        return {**out, "truth_status": "numerical_failure", "reason": f"cw={cw} ref={ref} cross={cc}"}
    if "unresolved" in (cw, ref, cc):
        return {**out, "truth_status": "unresolved", "reason": ";".join(x for x in (rcw, rref, rcc) if x)}
    if ref == "domain_exit" and cc == "domain_exit":
        return {**out, "truth_status": "physical_domain_exit", "reason": "reference and cross-check below the altitude floor"}
    if ref != cc:
        return {**out, "truth_status": "unresolved", "reason": f"reference {ref} vs cross-check {cc}"}
    if cw != "completed":
        return {**out, "truth_status": "unresolved", "reason": f"cw={cw}"}
    if not math.isfinite(out["u_num_m"]):
        return {**out, "truth_status": "unresolved", "reason": "missing, negative or non-finite discrepancy"}
    T = float(r["horizon_s"])
    for p in ("cw_vs_eci", "cw_vs_nerm"):
        cov = r.get(pf["cover"].format(p=p))
        if not _finite(cov) or float(cov) < T - HORIZON_TOL_S:
            return {**out, "truth_status": "unresolved", "reason": f"{p} discrepancy covers only [0, {cov}] of [0, {T}]"}
    return {**out, "truth_status": "completed_valid", "reason": ""}


def criterion_label(tr, tol, band_factor=10.0, band_floor_m=1e-6):
    """Label of the predicate disc <= tol; same band rule as the frozen v3.0 definition."""
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


def arrival_label(r, tr, kind=None, dwell_ambiguity_s=0.5):
    """Arrival agreement; missing or mistyped Booleans and incomplete branches are unresolved."""
    kind = kind_of(r, kind)
    if tr["truth_status"] == "physical_domain_exit":
        return {"arrival_status": "not_applicable", "cw_arrived": None, "ref_arrived": None, "reason": "reference exit"}
    if tr["truth_status"] != "completed_valid":
        return {"arrival_status": "unresolved", "cw_arrived": None, "ref_arrived": None, "reason": tr["truth_status"]}
    keys, mkeys = ARRIVAL_FIELDS[kind]
    vals = [r.get(k) for k in keys]
    margins = [r.get(k) for k in mkeys]
    if not all(isinstance(v, bool) for v in vals):
        return {"arrival_status": "unresolved", "cw_arrived": None, "ref_arrived": None,
                "reason": "missing or non-Boolean arrival field: " + ",".join(k for k, v in zip(keys, vals) if not isinstance(v, bool))}
    cw_a, ref_a, cc_a = vals
    for m, a in zip(margins, (cw_a, ref_a)):
        if a and not _finite(m):
            return {"arrival_status": "unresolved", "cw_arrived": cw_a, "ref_arrived": ref_a, "reason": "arrived without finite dwell margin"}
    if ref_a != cc_a or any(_finite(m) and abs(float(m)) < dwell_ambiguity_s for m in margins):
        return {"arrival_status": "unresolved", "cw_arrived": cw_a, "ref_arrived": ref_a,
                "reason": "cross-check disagreement or dwell margin within ambiguity"}
    return {"arrival_status": "arrived" if ref_a else "completed_non_arrival", "cw_arrived": cw_a, "ref_arrived": ref_a, "reason": ""}


def feature_availability(r, method, kind=None):
    """(value or NaN, 'ok' | 'unavailable', reason) from the nominal CW branch and its estimator only.

    Truth/reference outcomes are deliberately not consulted: a reference failure after the screen
    feature exists must not turn an accepted case into a referral (worker_status 'reference_failed'
    keeps the feature)."""
    kind = kind_of(r, kind)
    if method not in FEATURE_FIELDS[kind]:
        raise ValueError(f"method {method!r} is not defined for {kind} records")
    ws = r.get("worker_status")
    recovered = ws == "worker_crash" and r.get("feature_stage") == "complete" and (
        r.get("feature_checkpoint_recovered") is True or r.get("feature_recovery") == "driver_nominal_recovery")
    if ws not in FEATURE_STAGE_DONE and not recovered:
        return math.nan, "unavailable", f"worker_status {ws!r}"
    hp = horizon_problem(r)
    if hp:
        return math.nan, "unavailable", hp
    cw, _, why = _branch(r, "cw", kind)
    if cw != "completed":
        return math.nan, "unavailable", f"nominal CW branch not completed through T ({cw}{': ' + why if why else ''})"
    key, st = FEATURE_FIELDS[kind][method]
    if st is not None and r.get(st) != "ok":
        return math.nan, "unavailable", f"estimator status {r.get(st)!r}"
    v = r.get(key)
    if not _finite(v):
        return math.nan, "unavailable", "missing or non-finite feature"
    lo, hi = FEATURE_RANGE.get(method, (0.0, math.inf))
    if not (lo <= float(v) <= hi):
        return math.nan, "unavailable", f"feature {v} outside [{lo}, {hi}]"
    return float(v), "ok", ""


def feature_value(r, method, kind=None):
    return feature_availability(r, method, kind)[0]


def label_counts(accepted, labels):
    """acc_/ref_ counts over LABELS; raises on an unknown label (no silent denominator-only case)."""
    validate_labels(labels)
    out = {}
    for lab in LABELS:
        out[f"acc_{lab}"] = sum(1 for a, x in zip(accepted, labels) if a and x == lab)
        out[f"ref_{lab}"] = sum(1 for a, x in zip(accepted, labels) if (not a) and x == lab)
    check_label_partition(out, sum(1 for a in accepted if a))
    return out


def check_label_partition(counts, n_accepted):
    """n_accepted must equal the sum of accepted counts over LABELS."""
    s = sum(counts.get(f"acc_{lab}", 0) for lab in LABELS)
    if s != n_accepted:
        raise ValueError(f"label partition violated: sum over labels {s} != n_accepted {n_accepted}")


def validate_labels(labels):
    bad = sorted({str(x) for x in labels if x not in LABELS})
    if bad:
        raise ValueError(f"unknown labels: {bad}")


def evaluate_rule(values, labels, threshold):
    """v3.1 counterpart of calibration.evaluate_rule: counts over LABELS with the partition identity enforced;
    every non-ok label (including 'interrupted' and 'unresolved') counts as adverse when accepted."""
    import numpy as np
    from cwvalid3.calibration import cp_two_sided, ucb
    v = np.asarray(values, float)
    labels = list(labels)
    validate_labels(labels)
    acc = (np.isfinite(v) & (v <= threshold)) if threshold is not None else np.zeros(v.size, bool)
    cnt = label_counts([bool(a) for a in acc], labels)
    n = int(acc.sum())
    adverse = sum(cnt[f"acc_{lab}"] for lab in ADVERSE_LABELS)
    known = cnt["acc_violation"] + cnt["acc_exit"]
    out = {"n_planned": int(v.size), "n_feature_unavailable": int((~np.isfinite(v)).sum()), "n_accepted": n,
           "n_referred": int(v.size - n), "coverage": float(acc.mean()) if v.size else math.nan, **cnt,
           "known_adverse_accepted": known, "conservative_adverse_accepted": adverse,
           "cond_violation_rate_conservative": adverse / n if n else math.nan, "contract": CONTRACT_VERSION}
    lo, hi = cp_two_sided(adverse, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_conservative_lo"], out["cp95_two_sided_conservative_hi"] = lo, hi
    out["one_sided_95_ucb_conservative"] = ucb(adverse, n, 0.05) if n else math.nan
    return out


def is_adverse(label):
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    return label in ADVERSE_LABELS


__all__ = ["CONTRACT_VERSION", "HORIZON_TOL_S", "LABELS", "ADVERSE_LABELS", "TRUTH_STATUSES", "WORKER_STATUSES",
           "kind_of", "truth_record", "criterion_label", "arrival_label", "feature_availability", "feature_value",
           "label_counts", "check_label_partition", "validate_labels", "is_adverse", "evaluate_rule", "horizon_problem"]
