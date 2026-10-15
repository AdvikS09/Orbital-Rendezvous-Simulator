"""Step 7 - transparent screening for when CW needs nonlinear verification.

Decision per scenario: ACCEPT (rely on the CW closed-loop prediction for the
stated target) or REFER (run nonlinear verification).

Screening inputs come from a separate CW-only workflow per scenario
(`cw_only_features`): the CW closed loop with the production solver, its
diagnostics, and a first-order CW-only discrepancy estimate (cwvalid/estimate.py:
the CW model's gravity residual evaluated along the CW trajectory and
propagated through the linearised CW closed loop). No nonlinear propagation
enters any input. Nonlinear (inertial reference) results are used only as labels.

Targets (labels from the inertial nonlinear reference):
  traj_ok_1, traj_ok_01  peak CW-to-reference difference <= 1 m / 0.1 m
  outcome_ok             CW and reference agree on the primary terminal-arrival label
  false_fav              CW predicts terminal arrival, reference does not
  adequate               traj_ok_1 AND outcome_ok   (what the combined ACCEPT asserts)

Rules (thresholds fitted on development data only, then frozen):
  sep_only    target adequate : accept iff rho0 <= s_max
  sat_only    target adequate : accept iff CW saturated time / horizon <= tau
  combined    target adequate : accept iff CW arrives AND CW dwell margin >= m_min
                                AND first-order estimate D1 <= D_max
  traj_1m     target traj_ok_1 : accept iff D1 <= D1_max
  traj_0.1m   target traj_ok_01: accept iff D1 <= D01_max
  outcome     target outcome_ok: accept iff (CW arrives AND margin >= m_out) OR
                                 (CW does not arrive AND CW final range >= f_out)
  ff_flag     detection of false_fav among CW arrivals: flag iff margin < m_ff

Fitting: for an "accept iff x <= t" rule, t = k * (smallest x among development
cases violating the target), k = 0.8; for "accept iff x >= t", t = (largest x
among violating cases) / k. Each frozen rule is then checked to accept no
violating development case. The frozen file is hashed (SHA-256) and the test
and stress sets refuse to run without it.

Usage (repository root):
  python experiments/ext/screening.py features  # CW-only screening inputs for dev campaigns
  python experiments/ext/screening.py fit       # fit on development data, write freeze file
  python experiments/ext/screening.py evaluate  # evaluate the frozen rules on test and stress
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(os.path.dirname(ROOT), "results")
RES = os.path.join(RESULTS, "screening")
FROZEN = os.path.join(RES, "frozen", "screening_frozen.json")
DEV_SOURCES = ("main_grid/grid", "main_grid/refine", "main_grid/refine_bands", "screening/dev")
K = 0.8

from cwvalid.runner import load_results  # noqa: E402

REF = "eci"


# ---------------------------------------------------------------------------
# CW-only screening workflow
# ---------------------------------------------------------------------------

def cw_only_features(scn_d):
    """Run the CW-only screening workflow for one scenario (no nonlinear model)."""
    from cwvalid.estimate import first_order_discrepancy
    from cwvalid.experiment import RunConfig, make_law, run_branches, evaluate
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(scn_d)
    t0 = time.perf_counter()
    cfg = RunConfig(models=("cw",), reference="cw", crosscheck=None)
    sims, _ = run_branches(scn, cfg)
    row, _ = evaluate(scn, cfg, sims)
    est = first_order_discrepancy(sims["cw"], make_law(scn), scn.n, scn.R, scn.horizon)
    keep = ["sid", "sep_m", "x0", "y0", "z0", "vx0", "vy0", "vz0", "wn_ratio", "a_max", "horizon_s", "n_radps", "R_m",
            "cw_sat_total_s", "cw_sat_initial_s", "cw_terminal_arrived", "cw_terminal_dwell_margin_s",
            "cw_terminal_arrival_time_s", "cw_final_pos_m", "cw_q2_impulse_all_mps", "cw_q2_impulse_sat_mps",
            "cw_q2_peak_mps2", "cw_max_range_over_R", "cw_status"]
    out = {k: row.get(k) for k in keep}
    out.update(est)
    out["screen_wall_s"] = time.perf_counter() - t0
    return out


def compute_screen_features(campaign_dir, workers=7):
    """CW-only features for every scenario of a campaign. Results are cached one
    JSON line per scenario in screen_features.jsonl as they complete (resumable)."""
    from concurrent.futures import as_completed
    path = os.path.join(campaign_dir, "screen_features.jsonl")
    scns = [json.loads(l) for l in open(os.path.join(campaign_dir, "scenarios.jsonl")) if l.strip()]
    have = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                try:
                    r = json.loads(line)
                    have[r["sid"]] = r
                except json.JSONDecodeError:
                    pass
    todo = [s for s in scns if s["sid"] not in have]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, open(path, "a") as fh:
            futs = [ex.submit(cw_only_features, s) for s in todo]
            for k, f in enumerate(as_completed(futs)):
                r = f.result()
                r = {kk: (bool(v) if isinstance(v, np.bool_) else (float(v) if isinstance(v, np.floating) else v)) for kk, v in r.items()}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                have[r["sid"]] = r
                if (k + 1) % 200 == 0:
                    print(f"[features] {campaign_dir}: {k + 1}/{len(todo)}", flush=True)
    S = pd.DataFrame([have[s["sid"]] for s in scns if s["sid"] in have])
    S.to_csv(os.path.join(campaign_dir, "screen_features.csv"), index=False)
    return S


def features(S):
    f = pd.DataFrame(index=S.index)
    sep = S["sep_m"].astype(float)
    n = S["n_radps"].astype(float)
    T = S["horizon_s"].astype(float)
    f["sep_m"] = sep
    f["eps"] = sep / S["R_m"].astype(float)
    f["radial_frac"] = S["x0"].abs() / sep
    f["nu"] = np.sqrt(S["vx0"] ** 2 + S["vy0"] ** 2 + S["vz0"] ** 2) / (n * sep)
    f["beta"] = S["wn_ratio"].astype(float)
    f["alpha"] = S["a_max"].astype(float) / (n**2 * sep)
    f["sat_frac"] = S["cw_sat_total_s"].astype(float) / T
    f["cw_arrived"] = S["cw_terminal_arrived"].astype(bool)
    f["cw_margin_s"] = S["cw_terminal_dwell_margin_s"].astype(float)
    f["cw_final_m"] = S["cw_final_pos_m"].astype(float)
    f["q2_sat"] = S["cw_q2_impulse_sat_mps"].astype(float)
    f["d1"] = S["d1_peak_m"].astype(float)
    return f


def labels(D):
    lab = pd.DataFrame(index=D.index)
    disc = D[f"cw_vs_{REF}_max_pos_disc_m"].astype(float)
    cw_ok = D["cw_terminal_arrived"].astype(bool)
    nl_ok = D[f"{REF}_terminal_arrived"].astype(bool)
    lab["disc_m"] = disc
    lab["traj_ok_1"] = disc <= 1.0
    lab["traj_ok_01"] = disc <= 0.1
    lab["outcome_ok"] = cw_ok == nl_ok
    lab["false_fav"] = cw_ok & ~nl_ok
    lab["nl_fail"] = ~nl_ok
    lab["adequate"] = lab["traj_ok_1"] & lab["outcome_ok"]
    lab["ambiguous"] = D["ambiguous"].astype(bool)
    return lab


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

RULES = {
    "sep_only": ("adequate", lambda F, p: F.sep_m <= p["s_max_m"]),
    "sat_only": ("adequate", lambda F, p: F.sat_frac <= p["tau"]),
    "combined": ("adequate", lambda F, p: F.cw_arrived & (F.cw_margin_s >= p["m_min_s"]) & (F.d1 <= p["D_max_m"])),
    "traj_1m": ("traj_ok_1", lambda F, p: F.d1 <= p["D1_max_m"]),
    "traj_0.1m": ("traj_ok_01", lambda F, p: F.d1 <= p["D01_max_m"]),
    "outcome": ("outcome_ok", lambda F, p: (F.cw_arrived & (F.cw_margin_s >= p["m_out_s"]))
                | (~F.cw_arrived & (F.cw_final_m >= p["f_out_m"]))),
}
RULE_TEXT = {
    "sep_only": "accept iff rho0 <= s_max_m",
    "sat_only": "accept iff CW saturated time / horizon <= tau",
    "combined": "accept iff CW terminal arrival AND CW dwell margin >= m_min_s AND first-order CW-only estimate D1 <= D_max_m",
    "traj_1m": "accept (peak difference <= 1 m) iff D1 <= D1_max_m",
    "traj_0.1m": "accept (peak difference <= 0.1 m) iff D1 <= D01_max_m",
    "outcome": "accept (CW outcome label) iff (CW arrives AND margin >= m_out_s) OR (CW does not arrive AND CW final range >= f_out_m)",
    "ff_flag": "flag a CW arrival as possibly falsely favourable iff CW dwell margin < m_ff_s",
}


def _upper(values, bad):
    """Threshold for 'accept iff x <= t': K * min over violating cases."""
    vb = values[bad & np.isfinite(values)]
    return float(K * vb.min()) if vb.size else float(np.nanmax(values))


def _lower(values, bad):
    """Threshold for 'accept iff x >= t': max over violating cases / K."""
    vb = values[bad & np.isfinite(values)]
    return float(vb.max() / K) if vb.size else float(np.nanmin(values))


def fit(F, L):
    p = {}
    bad = ~L.adequate.to_numpy()
    p["sep_only"] = {"s_max_m": _upper(F.sep_m.to_numpy(), bad)}
    p["sat_only"] = {"tau": _upper(F.sat_frac.to_numpy(), bad)}
    p["traj_1m"] = {"D1_max_m": _upper(F.d1.to_numpy(), ~L.traj_ok_1.to_numpy())}
    p["traj_0.1m"] = {"D01_max_m": _upper(F.d1.to_numpy(), ~L.traj_ok_01.to_numpy())}
    mism = ~L.outcome_ok.to_numpy()
    arr = F.cw_arrived.to_numpy()
    m_out = _lower(F.cw_margin_s.to_numpy(), mism & arr)
    f_out = _lower(F.cw_final_m.to_numpy(), mism & ~arr)
    p["outcome"] = {"m_out_s": max(m_out, 0.0), "f_out_m": f_out}
    p["combined"] = {"m_min_s": p["outcome"]["m_out_s"], "D_max_m": p["traj_1m"]["D1_max_m"]}
    ff = L.false_fav.to_numpy()
    p["ff_flag"] = {"m_ff_s": _lower(F.cw_margin_s.to_numpy(), ff & arr)}
    for name, (target, rule) in RULES.items():
        n_bad = int((rule(F, p[name]).to_numpy() & ~L[target].to_numpy()).sum())
        if n_bad:
            raise RuntimeError(f"fitted rule {name} accepts {n_bad} development cases violating {target}")
    return p


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def cp(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return (float(ci.low), float(ci.high))


def rate(k, n):
    lo, hi = cp(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "ci95_lo": lo, "ci95_hi": hi}


def evaluate_rule(acc, ok, L):
    acc = np.asarray(acc, bool)
    ok = np.asarray(ok, bool)
    nlf = L.nl_fail.to_numpy()
    ff = L.false_fav.to_numpy()
    return {
        "accepted": rate(acc.sum(), acc.size),
        "incorrect_accept_given_accepted": rate((acc & ~ok).sum(), acc.sum()),
        "accepted_among_violating": rate((acc & ~ok).sum(), (~ok).sum()),
        "unnecessary_referral_given_referred": rate((~acc & ok).sum(), (~acc).sum()),
        "referred_among_satisfying": rate((~acc & ok).sum(), ok.sum()),
        "missed_nl_failures": rate((acc & nlf).sum(), nlf.sum()),
        "missed_false_favourable": rate((acc & ff).sum(), ff.sum()),
    }


def evaluate_flag(flag, L, F):
    flag = np.asarray(flag, bool)
    arr = F.cw_arrived.to_numpy()
    ff = L.false_fav.to_numpy()
    return {"ff_detected_among_ff": rate((flag & ff).sum(), ff.sum()),
            "flagged_among_cw_arrivals": rate((flag & arr).sum(), arr.sum()),
            "false_alarm_among_true_arrivals": rate((flag & arr & ~ff).sum(), (arr & ~ff).sum())}


def flat(method, target, set_name, ev, extra=None):
    return [{"method": method, "target": target, "set": set_name, "metric": m, **r, **(extra or {})} for m, r in ev.items()]


def load_set(path, require_complete=True, workers=7):
    sids = [json.loads(l)["sid"] for l in open(os.path.join(path, "scenarios.jsonl")) if l.strip()]
    D = load_results(path)
    D = D[D["sid"].isin(set(sids))].drop_duplicates("sid", keep="last")
    missing = sorted(set(sids) - set(D["sid"]))
    if missing:
        msg = f"{path}: {len(missing)} of {len(sids)} scenarios have no paired result"
        if require_complete:
            raise SystemExit(msg + "; refusing to fit/evaluate on an incomplete set")
        print("WARNING:", msg)
    S = compute_screen_features(path, workers)
    common = [s for s in D["sid"] if s in set(S["sid"])]
    M = D.set_index("sid").loc[common].reset_index()
    S = S.set_index("sid").loc[common].reset_index()
    # consistency: the CW-only screening run must reproduce the paired run's CW branch
    diff = int((S["cw_terminal_arrived"].astype(bool).to_numpy() != M["cw_terminal_arrived"].astype(bool).to_numpy()).sum())
    if diff:
        print(f"WARNING: {diff} scenarios where the CW-only run and the paired run's CW branch disagree on arrival")
    info = {"n_defined": len(sids), "n_loaded": len(M), "n_missing": len(missing), "cw_label_mismatch": diff}
    return M, S, info


def cmd_features(workers):
    for sub in DEV_SOURCES:
        p = os.path.join(RESULTS, sub)
        if not os.path.exists(os.path.join(p, "scenarios.jsonl")):
            print(f"[features] {sub}: not available yet, skipped")
            continue
        t0 = time.perf_counter()
        S = compute_screen_features(p, workers)
        print(f"[features] {sub}: {len(S)} scenarios ({time.perf_counter() - t0:.0f}s)")


def cmd_fit(workers):
    parts = []
    info = {}
    for sub in DEV_SOURCES:
        M, S, info[sub] = load_set(os.path.join(RESULTS, sub), workers=workers)
        F = features(S)
        L = labels(M)
        F["source"] = sub
        parts.append((F, L, M))
    F = pd.concat([p[0] for p in parts], ignore_index=True)
    L = pd.concat([p[1] for p in parts], ignore_index=True)
    params = fit(F.drop(columns="source"), L)
    rows = []
    for name, (target, rule) in RULES.items():
        acc = rule(F, params[name])
        rows += flat(name, target, "dev:all", evaluate_rule(acc, L[target], L))
        for src in DEV_SOURCES:
            idx = F.source == src
            rows += flat(name, target, f"dev:{src}", evaluate_rule(acc[idx], L[target][idx], L[idx]))
    rows += flat("ff_flag", "false_fav", "dev:all", evaluate_flag(F.cw_margin_s < params["ff_flag"]["m_ff_s"], L, F))
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "frozen", "development_performance.csv"), index=False)
    d1err = (F.d1 / L.disc_m).replace([np.inf, -np.inf], np.nan)
    frozen = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": REF, "K": K,
              "rules": RULE_TEXT, "params": params, "development": info,
              "n_dev": int(len(F)), "n_dev_violating": {t: int((~L[t]).sum()) for t in ("adequate", "traj_ok_1", "traj_ok_01", "outcome_ok")},
              "n_dev_false_fav": int(L.false_fav.sum()),
              "d1_over_true_quantiles_dev": {q: float(d1err.quantile(q)) for q in (0.01, 0.5, 0.99)},
              "inputs": "CW-only run (production solver) + first-order CW-only estimate; no nonlinear propagation"}
    json.dump(frozen, open(FROZEN, "w"), indent=2)
    print(json.dumps(frozen, indent=2))
    print("SHA-256", sha256(FROZEN))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def stress_candidates(stress_spec, base_spec, sample_one, frozen_path):
    """Boundary-focused stress set from CW-only information: candidates drawn from
    the test distribution are kept when the frozen combined rule's inputs lie
    near its thresholds (D1 within a factor 3 of D_max, or CW dwell margin within
    +/-600 s of m_min, or CW failing with final range within a factor 3 of f_out)."""
    fr = json.load(open(frozen_path))["params"]
    rng = np.random.default_rng(stress_spec["seed"])
    keep, tried = [], 0
    batch = 56
    # candidates are drawn in seed order and evaluated in parallel batches; selection
    # follows candidate index order, so the result equals the sequential procedure
    with ProcessPoolExecutor(max_workers=7) as ex:
        while len(keep) < stress_spec["N"] and tried < stress_spec["max_candidates"]:
            cands = [sample_one(rng, base_spec, "stress", tried + i) for i in range(batch)]
            feats = list(ex.map(cw_only_features, [c.to_dict() for c in cands]))
            for i, (scn, f) in enumerate(zip(cands, feats)):
                if len(keep) >= stress_spec["N"]:
                    break
                d1, m, arr, fin = f["d1_peak_m"], f["cw_terminal_dwell_margin_s"], bool(f["cw_terminal_arrived"]), f["cw_final_pos_m"]
                near = ((fr["combined"]["D_max_m"] / 3 <= d1 <= 3 * fr["combined"]["D_max_m"])
                        or (arr and m is not None and abs(m - fr["combined"]["m_min_s"]) <= 600.0)
                        or ((not arr) and fr["outcome"]["f_out_m"] / 3 <= fin <= 3 * fr["outcome"]["f_out_m"]))
                if near:
                    scn.meta["stress_candidate_index"] = tried + i
                    keep.append(scn)
            tried += batch
            print(f"[stress] {tried} candidates screened, {len(keep)} kept", flush=True)
    return keep, dict(stress_spec, candidates_tried=tried, rule="near frozen combined/outcome thresholds")


def cmd_evaluate(workers):
    frozen = json.load(open(FROZEN))
    h = sha256(FROZEN)
    p = frozen["params"]
    rows = []
    for set_name in ("test", "stress", "band_test"):
        path = os.path.join(RES, set_name)
        if not os.path.exists(os.path.join(path, "results.jsonl")):
            print(f"[evaluate] {set_name}: no results, skipped")
            continue
        sd = json.load(open(os.path.join(path, "set_definition.json")))
        if sd.get("frozen_sha256_at_run") != h:
            raise SystemExit(f"{set_name} was run with a different frozen file; refusing")
        M, S, info = load_set(path, workers=workers)
        F, L = features(S), labels(M)
        print(f"[evaluate] {set_name}: {info}")
        cats = {"all": np.ones(len(M), bool),
                "direction=axis": (M["meta_axis"] != "mixed").to_numpy(),
                "direction=mixed": (M["meta_axis"] == "mixed").to_numpy()}
        for v in sorted(M["meta_velocity"].unique()):
            cats[f"velocity={v}"] = (M["meta_velocity"] == v).to_numpy()
        if "meta_in_dev_domain" in M:
            cats["in_dev_domain"] = M["meta_in_dev_domain"].astype(bool).to_numpy()
            cats["outside_dev_domain"] = ~cats["in_dev_domain"]
        for name, (target, rule) in RULES.items():
            acc = rule(F, p[name]).to_numpy()
            for cname, idx in cats.items():
                rows += flat(name, target, f"{set_name}:{cname}", evaluate_rule(acc[idx], L[target].to_numpy()[idx], L[idx]))
        flag = (F.cw_margin_s < p["ff_flag"]["m_ff_s"]).to_numpy()
        for cname, idx in cats.items():
            rows += flat("ff_flag", "false_fav", f"{set_name}:{cname}", evaluate_flag(flag[idx], L[idx], F[idx]))
        out = M[["sid", "sep_m", "meta_axis", "meta_velocity", "wn_ratio", "a_max"]].copy()
        for name, (target, rule) in RULES.items():
            out[f"accept_{name}"] = rule(F, p[name]).to_numpy()
        out = pd.concat([out.reset_index(drop=True), L.reset_index(drop=True), F.add_prefix("f_").reset_index(drop=True)], axis=1)
        out.to_csv(os.path.join(RES, f"decisions_{set_name}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "screening_performance.csv"), index=False)
    print(f"[evaluate] frozen sha256 {h}; wrote screening_performance.csv ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "fit", "evaluate"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    {"features": cmd_features, "fit": cmd_fit, "evaluate": cmd_evaluate}[a.cmd](a.workers)


if __name__ == "__main__":
    main()
