"""Frozen labelling, candidate rules, exact one-sided binomial calibration and held-out evaluation (P04).

Labelling (frozen in PROSPECTIVE_PLAN.json before any calibration label is computed)
  truth_status   completed_valid      CW and reference completed the full horizon, cross-check agrees
                 physical_domain_exit reference AND cross-check left the physical domain (altitude < 100 km)
                 numerical_failure    reference, cross-check or worker failed
                 unresolved           reference and cross-check disagree on status
  criterion tol: violation if disc > tol + band, ok if disc < tol - band, unresolved otherwise,
                 band = max(BAND_FACTOR * |disc_ref - disc_crosscheck|, BAND_FLOOR_M)
  adverse (certification, conservative) = violation OR physical_domain_exit OR numerical_failure OR
                 unresolved; a reference failure after acceptance stays an accepted case with
                 unresolved truth (never converted into a referral).
  feature available iff the worker finished, the CW branch completed the horizon and the feature
                 status is ok with a finite value; otherwise the case is referred.

Calibration for one family (cell, method, criterion) with M protected candidates in total:
  n = accepted calibration cases, b = accepted adverse cases,
  U = BetaQuantile(1 - delta/M; b + 1, n - b) (U = 1 if n = 0 or b = n; for b = 0, 1 - (delta/M)^(1/n)).
  Select the admissible candidate (U <= alpha) with the most calibration acceptances; ties -> smaller
  threshold, then candidate order. No admissible candidate -> abstain (refer all).
"""

import math

import numpy as np
from scipy.stats import beta, binom

BAND_FACTOR = 10.0
BAND_FLOOR_M = 1e-6
DWELL_AMBIGUITY_S = 0.5
QUANTILES = tuple(round(0.05 * k, 2) for k in range(1, 17))   # 0.05 ... 0.80


# ------------------------------------------------------------------------------------------
# labelling
# ------------------------------------------------------------------------------------------

def _st_cont(s):
    return {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}.get(s, "numerical_failure")


def truth_record(r):
    """Normalised truth fields from a raw campaign record (continuous or held)."""
    held = "cw_vs_eci_peak_m" in r or r.get("cell") in ("D3", "D4")
    if r.get("worker_status") != "ok":
        return {"truth_status": "numerical_failure", "reason": f"worker {r.get('worker_status')}", "disc_m": math.nan,
                "u_num_m": math.nan, "ref_status": None, "cross_status": None, "cw_status": None,
                "ref_t_end_s": math.nan}
    if held:
        cw, ref, cc = r["cw_status"], r["eci_status"], r["nerm_status"]
        disc, discx = r.get("cw_vs_eci_peak_m"), r.get("cw_vs_nerm_peak_m")
        t_end = r.get("eci_t_end_s")
    else:
        cw, ref, cc = _st_cont(r["cw_status"]), _st_cont(r["eci_status"]), _st_cont(r["nerm_status"])
        if r["cw_status"] == "ok" and not r.get("cw_completed_horizon", True):
            cw = "numerical_failure"
        disc, discx = r.get("cw_vs_eci_max_pos_disc_m"), r.get("cw_vs_nerm_max_pos_disc_m")
        t_end = r.get("eci_t_final_s")
    disc = math.nan if disc is None else float(disc)
    discx = math.nan if discx is None else float(discx)
    u = abs(disc - discx) if (math.isfinite(disc) and math.isfinite(discx)) else math.nan
    out = {"disc_m": disc, "u_num_m": u, "ref_status": ref, "cross_status": cc, "cw_status": cw,
           "ref_t_end_s": math.nan if t_end is None else float(t_end)}
    if cw == "numerical_failure" or ref == "numerical_failure" or cc == "numerical_failure":
        out.update(truth_status="numerical_failure", reason=f"cw={cw} ref={ref} cross={cc}")
    elif ref == "domain_exit" and cc == "domain_exit":
        out.update(truth_status="physical_domain_exit", reason="reference and cross-check below 100 km altitude")
    elif ref != cc:
        out.update(truth_status="unresolved", reason=f"reference {ref} vs cross-check {cc}")
    elif cw != "completed":
        out.update(truth_status="unresolved", reason=f"cw={cw}")
    elif not math.isfinite(disc) or not math.isfinite(u):
        out.update(truth_status="unresolved", reason="non-finite discrepancy")
    else:
        out.update(truth_status="completed_valid", reason="")
    return out


def criterion_label(tr, tol):
    """'ok' | 'violation' | 'exit' | 'failure' | 'unresolved' for the full-horizon predicate disc <= tol."""
    st = tr["truth_status"]
    if st == "physical_domain_exit":
        return "exit"
    if st == "numerical_failure":
        return "failure"
    if st != "completed_valid":
        return "unresolved"
    band = max(BAND_FACTOR * tr["u_num_m"], BAND_FLOOR_M)
    d = tr["disc_m"]
    if d > tol + band:
        return "violation"
    if d < tol - band:
        return "ok"
    return "unresolved"


def arrival_label(r, tr):
    """Secondary arrival outcome: CW arrival label vs reference label (descriptive)."""
    held = r.get("cell") in ("D3", "D4")
    if tr["truth_status"] != "completed_valid":
        return {"arrival_status": "not_applicable" if tr["truth_status"] == "physical_domain_exit" else "unresolved",
                "cw_arrived": None, "ref_arrived": None}
    if held:
        cw_a, ref_a, cc_a = r.get("cw_arrived"), r.get("eci_arrived"), r.get("nerm_arrived")
        m_cw, m_ref = r.get("cw_dwell_margin_s"), r.get("eci_dwell_margin_s")
    else:
        cw_a, ref_a, cc_a = r.get("cw_terminal_arrived"), r.get("eci_terminal_arrived"), r.get("nerm_terminal_arrived")
        m_cw, m_ref = r.get("cw_terminal_dwell_margin_s"), r.get("eci_terminal_dwell_margin_s")
    amb = (ref_a != cc_a) or any(m is not None and math.isfinite(float(m)) and abs(float(m)) < DWELL_AMBIGUITY_S
                                 for m in (m_cw, m_ref))
    if amb:
        return {"arrival_status": "unresolved", "cw_arrived": cw_a, "ref_arrived": ref_a}
    return {"arrival_status": "arrived" if ref_a else "completed_non_arrival", "cw_arrived": bool(cw_a),
            "ref_arrived": bool(ref_a)}


FEATURE_FIELDS = {
    "continuous": {"A": ("d1_peak_m", "d1_status"), "B": ("d1u_peak_m", "d1u_status"), "C": ("cres_peak_m", "cres_status"),
                   "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
    "held": {"A_H": ("A_H_peak_m", "A_H_status"), "B_H": ("B_H_peak_m", "B_H_status"),
             "A_naive": ("A_naive_peak_m", "A_naive_status"), "C_H": ("C_H_peak_m", "C_H_status"),
             "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
}


def feature_value(r, method, kind):
    """Feature value or NaN when unavailable (unavailable -> referral)."""
    if r.get("worker_status") != "ok":
        return math.nan
    cw_ok = (r.get("cw_status") == "completed") if kind == "held" else (r.get("cw_status") == "ok" and r.get("cw_completed_horizon"))
    if not cw_ok:
        return math.nan
    key, st = FEATURE_FIELDS[kind][method]
    v = r.get(key)
    if v is None:
        return math.nan
    if st is not None and r.get(st) != "ok":
        return math.nan
    v = float(v)
    return v if math.isfinite(v) else math.nan


# ------------------------------------------------------------------------------------------
# candidates, bounds, selection
# ------------------------------------------------------------------------------------------

def dev_quantile_thresholds(values, qs=QUANTILES):
    """Threshold giving acceptance fraction p on the development set: the ceil(p n)-th smallest finite
    feature value (n counts every development case, so unavailable features lower the reachable fraction)."""
    v = np.asarray(values, float)
    n = v.size
    fin = np.sort(v[np.isfinite(v)])
    out = []
    for p in qs:
        need = int(math.ceil(p * n - 1e-12))
        if need < 1 or need > fin.size:
            out.append((p, None))
        else:
            out.append((p, float(fin[need - 1])))
    return out


def build_candidates(dev_values, historical=None):
    """Ordered, deduplicated candidate list [{'cid','threshold','source'}]; ties keep the first source."""
    cands = []
    if historical is not None and math.isfinite(historical):
        cands.append({"threshold": float(historical), "source": "historical_v2_frozen"})
    for p, t in dev_quantile_thresholds(dev_values):
        if t is not None:
            cands.append({"threshold": t, "source": f"dev_q{p:.2f}"})
    seen, out = set(), []
    for c in cands:
        if c["threshold"] in seen:
            continue
        seen.add(c["threshold"])
        out.append(c)
    for k, c in enumerate(out):
        c["cid"] = k
    return out


def ucb(b, n, level):
    """One-sided exact (Clopper-Pearson) upper confidence limit with error level `level`."""
    if n == 0 or b >= n:
        return 1.0
    if b == 0:
        return 1.0 - level ** (1.0 / n)
    return float(beta.ppf(1.0 - level, b + 1, n - b))


def cp_two_sided(b, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if b == 0 else float(beta.ppf(a / 2, b, n - b + 1))
    hi = 1.0 if b == n else float(beta.ppf(1 - a / 2, b + 1, n - b))
    return lo, hi


def n_min_zero_adverse(alpha, delta, M):
    """Smallest n with 1 - (delta/M)^(1/n) <= alpha (zero accepted adverse cases)."""
    return int(math.ceil(math.log(delta / M) / math.log(1.0 - alpha) - 1e-12))


def n_min_for_b(b, alpha, delta, M, nmax=10 ** 6):
    lo, hi = b + 1, nmax
    if ucb(b, hi, delta / M) > alpha:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ucb(b, mid, delta / M) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


def accept_mask(values, threshold):
    v = np.asarray(values, float)
    return np.isfinite(v) & (v <= threshold)


def calibrate_family(values, adverse, candidates, alpha, level):
    """Evaluate every candidate on calibration data and select per the frozen rule.

    values: feature (NaN = unavailable -> referred); adverse: bool array (conservative adverse)."""
    adverse = np.asarray(adverse, bool)
    rows = []
    for c in candidates:
        acc = accept_mask(values, c["threshold"])
        n, b = int(acc.sum()), int((acc & adverse).sum())
        rows.append({**c, "n_accepted": n, "n_adverse": b, "ucb": ucb(b, n, level), "admissible": ucb(b, n, level) <= alpha})
    adm = [r for r in rows if r["admissible"]]
    if not adm:
        return rows, None
    best = sorted(adm, key=lambda r: (-r["n_accepted"], r["threshold"], r["cid"]))[0]
    return rows, best


def evaluate_rule(values, labels, threshold):
    """Single held-out evaluation of a frozen threshold (labels from criterion_label)."""
    v = np.asarray(values, float)
    labels = np.asarray(labels, object)
    acc = accept_mask(v, threshold) if threshold is not None else np.zeros(v.size, bool)
    avail = np.isfinite(v)
    out = {"n_planned": int(v.size), "n_feature_unavailable": int((~avail).sum()), "n_accepted": int(acc.sum()),
           "n_referred": int((~acc).sum()), "coverage": float(acc.mean()) if v.size else math.nan}
    for lab in ("ok", "violation", "exit", "failure", "unresolved"):
        out[f"acc_{lab}"] = int((acc & (labels == lab)).sum())
        out[f"ref_{lab}"] = int((~acc & (labels == lab)).sum())
    n = out["n_accepted"]
    known_bad = out["acc_violation"] + out["acc_exit"]
    cons_bad = known_bad + out["acc_failure"] + out["acc_unresolved"]
    out["known_adverse_accepted"] = known_bad
    out["conservative_adverse_accepted"] = cons_bad
    out["cond_violation_rate_known"] = known_bad / n if n else math.nan
    out["cond_violation_rate_conservative"] = cons_bad / n if n else math.nan
    lo, hi = cp_two_sided(known_bad, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_known_lo"], out["cp95_two_sided_known_hi"] = lo, hi
    lo, hi = cp_two_sided(cons_bad, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_conservative_lo"], out["cp95_two_sided_conservative_hi"] = lo, hi
    out["one_sided_95_ucb_conservative"] = ucb(cons_bad, n, 0.05) if n else math.nan
    return out


def mcnemar_exact(b01, b10):
    """Two-sided exact McNemar p-value on discordant pairs."""
    m = b01 + b10
    if m == 0:
        return 1.0
    k = min(b01, b10)
    return float(min(1.0, 2.0 * binom.cdf(k, m, 0.5)))


def paired_bootstrap_diff(acc_a, acc_b, good, n_boot=10000, seed=0):
    """Paired case-level bootstrap of coverage difference and net useful acceptance difference
    (accepted-and-good minus accepted-and-bad, per case)."""
    acc_a, acc_b, good = (np.asarray(x, bool) for x in (acc_a, acc_b, good))
    n = acc_a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    cov = (acc_a[idx].mean(1) - acc_b[idx].mean(1))
    nua = acc_a & good
    nub = acc_b & good
    bad_a = acc_a & ~good
    bad_b = acc_b & ~good
    net = ((nua[idx].sum(1) - bad_a[idx].sum(1)) - (nub[idx].sum(1) - bad_b[idx].sum(1))) / n
    return {"coverage_diff": float(acc_a.mean() - acc_b.mean()),
            "coverage_diff_ci95": [float(np.quantile(cov, 0.025)), float(np.quantile(cov, 0.975))],
            "net_useful_diff_per_case": float(((nua.sum() - bad_a.sum()) - (nub.sum() - bad_b.sum())) / n),
            "net_useful_diff_ci95": [float(np.quantile(net, 0.025)), float(np.quantile(net, 0.975))],
            "n_boot": n_boot, "seed": seed}


__all__ = ["truth_record", "criterion_label", "arrival_label", "feature_value", "FEATURE_FIELDS", "QUANTILES",
           "dev_quantile_thresholds", "build_candidates", "ucb", "cp_two_sided", "n_min_zero_adverse", "n_min_for_b",
           "accept_mask", "calibrate_family", "evaluate_rule", "mcnemar_exact", "paired_bootstrap_diff",
           "BAND_FACTOR", "BAND_FLOOR_M"]
