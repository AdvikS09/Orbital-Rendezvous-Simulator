import math

import numpy as np
from scipy.stats import beta, binom

BANDFACTOR = 10.0
BANDFLOORM = 1e-6
DWELLAMBS = 0.5
QUANT = tuple(round(0.05 * k, 2) for k in range(1, 17))


def stcont(s):
    return {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}.get(s, "numerical_failure")


def truth_record(r):
    held = "cw_vs_eci_peak_m" in r or r.get("cell") in ("D3", "D4")
    if r.get("worker_status") != "ok":
        return {"truth_status": "numerical_failure", "reason": f"worker {r.get('worker_status')}", "disc_m": math.nan,
                "u_num_m": math.nan, "ref_status": None, "cross_status": None, "cw_status": None,
                "ref_t_end_s": math.nan}
    if held:
        c, ref, cc = r["cw_status"], r["eci_status"], r["nerm_status"]
        disc, discx = r.get("cw_vs_eci_peak_m"), r.get("cw_vs_nerm_peak_m")
        tend = r.get("eci_t_end_s")
    else:
        c, ref, cc = stcont(r["cw_status"]), stcont(r["eci_status"]), stcont(r["nerm_status"])
        if r["cw_status"] == "ok" and not r.get("cw_completed_horizon", True):
            c = "numerical_failure"
        disc, discx = r.get("cw_vs_eci_max_pos_disc_m"), r.get("cw_vs_nerm_max_pos_disc_m")
        tend = r.get("eci_t_final_s")
    disc = math.nan if disc is None else float(disc)
    discx = math.nan if discx is None else float(discx)
    u = abs(disc - discx) if (math.isfinite(disc) and math.isfinite(discx)) else math.nan
    out = {"disc_m": disc, "u_num_m": u, "ref_status": ref, "cross_status": cc, "cw_status": c,
           "ref_t_end_s": math.nan if tend is None else float(tend)}
    if c == "numerical_failure" or ref == "numerical_failure" or cc == "numerical_failure":
        out.update(truth_status="numerical_failure", reason=f"cw={c} ref={ref} cross={cc}")
    elif ref == "domain_exit" and cc == "domain_exit":
        out.update(truth_status="physical_domain_exit", reason="reference and cross-check below 100 km altitude")
    elif ref != cc:
        out.update(truth_status="unresolved", reason=f"reference {ref} vs cross-check {cc}")
    elif c != "completed":
        out.update(truth_status="unresolved", reason=f"cw={c}")
    elif not math.isfinite(disc) or not math.isfinite(u):
        out.update(truth_status="unresolved", reason="non-finite discrepancy")
    else:
        out.update(truth_status="completed_valid", reason="")
    return out


def critlabel(tr, tol):
    st = tr["truth_status"]
    if st == "physical_domain_exit":
        return "exit"
    if st == "numerical_failure":
        return "failure"
    if st != "completed_valid":
        return "unresolved"
    band = max(BANDFACTOR * tr["u_num_m"], BANDFLOORM)
    d = tr["disc_m"]
    if d > tol + band:
        return "violation"
    if d < tol - band:
        return "ok"
    return "unresolved"


def arrlabel(r, tr):
    held = r.get("cell") in ("D3", "D4")
    if tr["truth_status"] != "completed_valid":
        return {"arrival_status": "not_applicable" if tr["truth_status"] == "physical_domain_exit" else "unresolved",
                "cw_arrived": None, "ref_arrived": None}
    if held:
        cwa, refa, cca = r.get("cw_arrived"), r.get("eci_arrived"), r.get("nerm_arrived")
        mcw, mref = r.get("cw_dwell_margin_s"), r.get("eci_dwell_margin_s")
    else:
        cwa, refa, cca = r.get("cw_terminal_arrived"), r.get("eci_terminal_arrived"), r.get("nerm_terminal_arrived")
        mcw, mref = r.get("cw_terminal_dwell_margin_s"), r.get("eci_terminal_dwell_margin_s")
    amb = (refa != cca) or any(m is not None and math.isfinite(float(m)) and abs(float(m)) < DWELLAMBS
                               for m in (mcw, mref))
    if amb:
        return {"arrival_status": "unresolved", "cw_arrived": cwa, "ref_arrived": refa}
    return {"arrival_status": "arrived" if refa else "completed_non_arrival", "cw_arrived": bool(cwa),
            "ref_arrived": bool(refa)}


FEATFIELDS = {
    "continuous": {"A": ("d1_peak_m", "d1_status"), "B": ("d1u_peak_m", "d1u_status"), "C": ("cres_peak_m", "cres_status"),
                   "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
    "held": {"A_H": ("A_H_peak_m", "A_H_status"), "B_H": ("B_H_peak_m", "B_H_status"),
             "A_naive": ("A_naive_peak_m", "A_naive_status"), "C_H": ("C_H_peak_m", "C_H_status"),
             "sep": ("sep_m", None), "sat": ("feature_sat_fraction", None)},
}


def feature_value(r, method, kind):
    if r.get("worker_status") != "ok":
        return math.nan
    cwok = (r.get("cw_status") == "completed") if kind == "held" else (r.get("cw_status") == "ok" and r.get("cw_completed_horizon"))
    if not cwok:
        return math.nan
    key, st = FEATFIELDS[kind][method]
    v = r.get(key)
    if v is None:
        return math.nan
    if st is not None and r.get(st) != "ok":
        return math.nan
    v = float(v)
    return v if math.isfinite(v) else math.nan


# Threshold accepting a fraction p of the development set: the ceil(p n)-th smallest finite value.
# n counts every case, so unavailable features lower the fraction that can be reached.
def devquantthr(values, qs=QUANT):
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


def buildcands(devvalues, histor=None):
    cands = []
    if histor is not None and math.isfinite(histor):
        cands.append({"threshold": float(histor), "source": "historical_v2_frozen"})
    for p, t in devquantthr(devvalues):
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


# One-sided exact (Clopper-Pearson) upper confidence limit
def ucb(b, n, level):
    if n == 0 or b >= n:
        return 1.0
    if b == 0:
        return 1.0 - level ** (1.0 / n)
    return float(beta.ppf(1.0 - level, b + 1, n - b))


def cptwosided(b, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if b == 0 else float(beta.ppf(a / 2, b, n - b + 1))
    hi = 1.0 if b == n else float(beta.ppf(1 - a / 2, b + 1, n - b))
    return lo, hi


# Smallest n with 1 - (delta/M)^(1/n) <= alpha: enough calibration cases to certify a candidate
# that accepts no adverse case
def nadv(alpha, delta, M):
    return int(math.ceil(math.log(delta / M) / math.log(1.0 - alpha) - 1e-12))


def nminforb(b, alpha, delta, M, nmax=10 ** 6):
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


def acceptmask(values, threshold):
    v = np.asarray(values, float)
    return np.isfinite(v) & (v <= threshold)


def calfam(values, adverse, candidates, alpha, level):
    adverse = np.asarray(adverse, bool)
    rows = []
    for c in candidates:
        acc = acceptmask(values, c["threshold"])
        n, b = int(acc.sum()), int((acc & adverse).sum())
        rows.append({**c, "n_accepted": n, "n_adverse": b, "ucb": ucb(b, n, level), "admissible": ucb(b, n, level) <= alpha})
    adm = [r for r in rows if r["admissible"]]
    if not adm:
        return rows, None
    best = sorted(adm, key=lambda r: (-r["n_accepted"], r["threshold"], r["cid"]))[0]
    return rows, best


def evalrule(values, labels, threshold):
    v = np.asarray(values, float)
    labels = np.asarray(labels, object)
    acc = acceptmask(v, threshold) if threshold is not None else np.zeros(v.size, bool)
    avail = np.isfinite(v)
    out = {"n_planned": int(v.size), "n_feature_unavailable": int((~avail).sum()), "n_accepted": int(acc.sum()),
           "n_referred": int((~acc).sum()), "coverage": float(acc.mean()) if v.size else math.nan}
    for lab in ("ok", "violation", "exit", "failure", "unresolved"):
        out[f"acc_{lab}"] = int((acc & (labels == lab)).sum())
        out[f"ref_{lab}"] = int((~acc & (labels == lab)).sum())
    n = out["n_accepted"]
    knownbad = out["acc_violation"] + out["acc_exit"]
    consbad = knownbad + out["acc_failure"] + out["acc_unresolved"]
    out["known_adverse_accepted"] = knownbad
    out["conservative_adverse_accepted"] = consbad
    out["cond_violation_rate_known"] = knownbad / n if n else math.nan
    out["cond_violation_rate_conservative"] = consbad / n if n else math.nan
    lo, hi = cptwosided(knownbad, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_known_lo"], out["cp95_two_sided_known_hi"] = lo, hi
    lo, hi = cptwosided(consbad, n) if n else (math.nan, math.nan)
    out["cp95_two_sided_conservative_lo"], out["cp95_two_sided_conservative_hi"] = lo, hi
    out["one_sided_95_ucb_conservative"] = ucb(consbad, n, 0.05) if n else math.nan
    return out


# Exact two-sided McNemar test on the discordant pairs
def mcnexact(b01, b10):
    m = b01 + b10
    if m == 0:
        return 1.0
    k = min(b01, b10)
    return float(min(1.0, 2.0 * binom.cdf(k, m, 0.5)))


# Both methods are scored on the same resampled case indices, so the interval is for the paired
# difference rather than for two independent proportions.
def pairbootdiff(acca, accb, good, n_boot=10000, seed=0):
    acca, accb, good = (np.asarray(x, bool) for x in (acca, accb, good))
    n = acca.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    cov = (acca[idx].mean(1) - accb[idx].mean(1))
    nua = acca & good
    nub = accb & good
    bada = acca & ~good
    badb = accb & ~good
    net = ((nua[idx].sum(1) - bada[idx].sum(1)) - (nub[idx].sum(1) - badb[idx].sum(1))) / n
    return {"coverage_diff": float(acca.mean() - accb.mean()),
            "coverage_diff_ci95": [float(np.quantile(cov, 0.025)), float(np.quantile(cov, 0.975))],
            "net_useful_diff_per_case": float(((nua.sum() - bada.sum()) - (nub.sum() - badb.sum())) / n),
            "net_useful_diff_ci95": [float(np.quantile(net, 0.025)), float(np.quantile(net, 0.975))],
            "n_boot": n_boot, "seed": seed}
