"""P01 similarity verification (a code check, not a new result).

Each base case at 400 km is mapped to 600 km and 800 km with every nondimensional quantity equal:
  rho0' = rho0 R'/R,  rho_dot0' = rho_dot0 (n'R')/(nR),  omega_n'/n' = omega_n/n,
  a_max' = a_max (n'^2 R')/(n^2 R),  horizon = one orbit,  held period T_s' = T_s n/n'.
Both the base and the image are propagated with the production configuration (continuous: cwvalid
production functions; held: cwvalid3.sampled.HeldBatch). Compared after scaling, on matched
nondimensional times tau_j = n t_j (the base 0.5 s grid mapped to the image):
  max_j |q_cw - q'_cw|, max_j |q_ref - q'_ref| (q = rho/R), d/R, feature/R (A, B, C or A_H, B_H, C_H),
  and the terminal-arrival label with criteria scaled (1 m R'/R, 0.01 m/s n'R'/(nR), 300 s n/n').
Declared tolerance: the similarity holds to numerical accuracy when every nondimensional state and
discrepancy difference is <= 10 x (numerical noise of base + image) + 1e-15, the noise being the
reference/cross-check difference |q_eci - q_nerm| of each run. Features are compared twice: with the
production identity (output grid fixed at 2 s or 0.5 s, which does not scale with 1/n, so a small
sampling difference is expected) and on matched nondimensional output times (estimator rtol 1e-11,
absolute tolerance scaled with R), where the declared tolerance is 1 micrometre (in base units) plus
1e-7 relative. Cases where a branch reaches the 100 km floor are
reported separately: the floor is fixed in metres and breaks the map by construction.

Usage: python code/experiments/v3/similarity_checks.py [--workers 6]
Output: v3_extension/results/similarity_checks.csv
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
OUT = os.path.join(ARCH, "v3_extension", "results")

from cwvalid.constants import MU_EARTH, R_EARTH  # noqa: E402

BASES = [  # (label, rel0 at 400 km, wn_ratio, a_max) chosen to span unsaturated, switching and saturated regimes
    ("+R_3km_unsat", [3000.0, 0, 0, 0, 0, 0], 6.0, 0.05),
    ("-R_12km_switch", [-12000.0, 0, 0, 0, 0, 0], 8.0, 0.01),
    ("+H_20km", [0, 0, 20000.0, 0, 0, 0], 6.0, 0.05),
    ("mixed_8km_vel", [3000.0, -8000.0, 2000.0, 1.0, -2.0, 0.5], 5.0, 0.02),
    ("+V_500m", [0, 500.0, 0, 0, 0, 0], 6.0, 0.05),
    ("mixed_40km_satdiverge", [-25000.0, 30000.0, 8000.0, 20.0, -10.0, 5.0], 3.5, 0.008),
]
ALTS = (400_000.0, 600_000.0, 800_000.0)


def nm(alt):
    R = R_EARTH + alt
    return math.sqrt(MU_EARTH / R ** 3), R


def image(rel0, wn_ratio, a_max, alt):
    n0, R0 = nm(ALTS[0])
    n1, R1 = nm(alt)
    r = np.array(rel0, float)
    pos, vel = r[:3] * R1 / R0, r[3:] * (n1 * R1) / (n0 * R0)
    return list(np.concatenate((pos, vel))), wn_ratio, a_max * (n1 * n1 * R1) / (n0 * n0 * R0)


def arrival_scaled(states_fn, T, n_ratio, R_ratio, nR_ratio, t_grid):
    """Arrival label with tolerances scaled from (1 m, 0.01 m/s, 300 s) by the image factors."""
    from scipy.optimize import brentq
    pt, vt, dw = 1.0 * R_ratio, 0.01 * nR_ratio, 300.0 / n_ratio
    S = states_fn(t_grid)
    g = np.maximum(np.linalg.norm(S[:, :3], axis=1) / pt, np.linalg.norm(S[:, 3:], axis=1) / vt) - 1.0
    if g[-1] >= 0:
        return False, math.nan
    bad = np.where(g >= 0)[0]
    if bad.size == 0:
        te = 0.0
    else:
        j = int(bad[-1])

        def gf(s):
            x = states_fn(np.array([s]))[0]
            return float(max(np.linalg.norm(x[:3]) / pt, np.linalg.norm(x[3:]) / vt) - 1.0)
        a0, a1 = t_grid[j], t_grid[j + 1]
        te = float(brentq(gf, a0, a1, xtol=1e-9)) if gf(a0) * gf(a1) < 0 else 0.5 * (a0 + a1)
    return bool(T - te >= dw), (T - te - dw)


def run_continuous(args):
    label, rel0, wn, am, alt = args
    from cwvalid.estimate_v2 import estimator_variants
    from cwvalid.experiment import RunConfig, evaluate, make_law, run_branches
    from cwvalid.scenario import Scenario
    s = Scenario("v3sim", tuple(rel0), wn, am, 1.0, alt, meta={"variant": f"sim:{label}"})
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    sims, _ = run_branches(s, cfg)
    row, _ = evaluate(s, cfg, sims)
    f = estimator_variants(sims["cw"], make_law(s), s.n, s.R, s.horizon)
    n0, R0 = nm(ALTS[0])
    from cwvalid3.continuous import estimator_matrix
    from cwvalid3.controllers import pd_law
    dt_m = 2.0 * n0 / s.n          # the base 2 s output grid mapped to this altitude
    ml = pd_law(s.wn_ratio * s.n, s.a_max)
    Fm = {k: estimator_matrix(sims["cw"], ml, s.n, s.R, s.horizon, k, dt_out=dt_m, rtol=1e-11, atol=1e-11 * s.R / R0)["peak_m"]
          for k in ("A", "B", "C")}
    tau = np.arange(0, 2 * math.pi + 1e-12, 0.5 * n0)        # base 0.5 s grid in tau
    tau = np.append(tau[tau < 2 * math.pi - 1e-12], 2 * math.pi)
    t = tau / s.n
    out = {"label": label, "alt": alt, "n": s.n, "R": s.R, "status": {b: sims[b].status for b in sims},
           "q_cw": sims["cw"].states(t) / np.r_[[s.R] * 3, [s.n * s.R] * 3],
           "q_eci": sims["eci"].states(np.minimum(t, sims["eci"].t_final)) / np.r_[[s.R] * 3, [s.n * s.R] * 3],
           "q_nerm": sims["nerm"].states(np.minimum(t, sims["nerm"].t_final)) / np.r_[[s.R] * 3, [s.n * s.R] * 3],
           "disc": row["cw_vs_eci_max_pos_disc_m"], "disc_x": row["cw_vs_nerm_max_pos_disc_m"],
           "t_end_eci": sims["eci"].t_final, "T": s.horizon,
           "F": {"A": f["d1_peak_m"], "B": f["d1u_peak_m"], "C": f["cres_peak_m"]}, "Fm": Fm}
    R_ratio, n_ratio = s.R / R0, s.n / n0
    nR_ratio = (s.n * s.R) / (n0 * R0)
    for b in ("cw", "eci"):
        if sims[b].status == "ok":
            out[f"arr_{b}"] = arrival_scaled(sims[b].states, s.horizon, n_ratio, R_ratio, nR_ratio, t)
        else:
            out[f"arr_{b}"] = (None, math.nan)
    return out


def run_held(args):
    label, rel0, wn, am, alt, Ts0 = args
    from cwvalid3.held_metrics import case_summary
    from cwvalid3.sampled import HeldBatch, pd_matrix
    n0, R0 = nm(ALTS[0])
    n, R = nm(alt)
    Ts = Ts0 * n0 / n
    hb = HeldBatch([rel0], pd_matrix(wn * n)[None], [am], alt, Ts, h=Ts / 5.0, models=("cw", "nerm", "eci"),
                   estimators=("A_H", "B_H", "C_H")).run()
    sm, cv = case_summary(hb, 0, pairs=(("cw", "eci"), ("cw", "nerm")))
    tau = np.arange(0, 2 * math.pi + 1e-12, 0.5 * n0)
    tau = np.append(tau[tau < 2 * math.pi - 1e-12], 2 * math.pi)
    t = tau / n
    Y = hb.dense(0, t, cv.Fr, cv.Fl)
    sc = np.r_[[R] * 3, [n * R] * 3]
    out = {"label": label + f"_held{Ts0:g}s", "alt": alt, "n": n, "R": R, "status": {b: hb.status[b][0] for b in ("cw", "nerm", "eci")},
           "q_cw": cv.rel_from_rows("cw", Y) / sc, "q_eci": cv.rel_from_rows("eci", Y) / sc, "q_nerm": cv.rel_from_rows("nerm", Y) / sc,
           "disc": sm["cw_vs_eci_peak_m"], "disc_x": sm["cw_vs_nerm_peak_m"], "t_end_eci": hb.t_end["eci"][0], "T": hb.T,
           "F": {"A_H": sm["A_H_peak_m"], "B_H": sm["B_H_peak_m"], "C_H": sm["C_H_peak_m"]}, "Ts": Ts,
           "Fm": {e: float(np.max(np.linalg.norm(cv.rel_from_rows(e, Y)[:, :3], axis=1))) for e in ("A_H", "B_H", "C_H")}}
    R_ratio, n_ratio, nR_ratio = R / R0, n / n0, (n * R) / (n0 * R0)
    for b in ("cw", "eci"):
        if hb.status[b][0] == "completed":
            out[f"arr_{b}"] = arrival_scaled(lambda tt, b=b: cv.rel_from_rows(b, hb.dense(0, tt, cv.Fr, cv.Fl)), hb.T,
                                             n_ratio, R_ratio, nR_ratio, t)
        else:
            out[f"arr_{b}"] = (None, math.nan)
    return out


def compare(base, img, kind):
    nb = float(np.nanmax(np.linalg.norm(base["q_eci"][:, :3] - base["q_nerm"][:, :3], axis=1)))
    ni = float(np.nanmax(np.linalg.norm(img["q_eci"][:, :3] - img["q_nerm"][:, :3], axis=1)))
    tol_state = 10.0 * (nb + ni) + 1e-15
    floor = any(v in ("terminated", "domain_exit") for v in list(base["status"].values()) + list(img["status"].values()))
    r = {"case": base["label"], "kind": kind, "alt_image_km": img["alt"] / 1000.0, "R_ratio": img["R"] / base["R"],
         "status_base": str(base["status"]), "status_image": str(img["status"]), "altitude_floor_involved": floor,
         "max_dq_cw": float(np.max(np.linalg.norm(base["q_cw"][:, :3] - img["q_cw"][:, :3], axis=1))),
         "max_dq_eci": float(np.nanmax(np.linalg.norm(base["q_eci"][:, :3] - img["q_eci"][:, :3], axis=1))),
         "noise_q_base": nb, "noise_q_image": ni, "declared_tol_q": tol_state,
         "disc_over_R_base": base["disc"] / base["R"], "disc_over_R_image": img["disc"] / img["R"],
         "d_disc_over_R": abs(base["disc"] / base["R"] - img["disc"] / img["R"]),
         "disc_base_m": base["disc"], "disc_image_m": img["disc"]}
    feats_ok = True
    for k in base["F"]:
        # production feature identity: output grid fixed in seconds (does not scale with 1/n)
        fb, fi = base["F"][k] / base["R"], img["F"][k] / img["R"]
        r[f"{k}_over_R_base"], r[f"{k}_over_R_image"] = fb, fi
        r[f"{k}_rel_diff_fixed_grid"] = abs(fb - fi) / max(abs(fb), 1e-300)
        # same estimator sampled on matched nondimensional times: tests the similarity map itself
        mb, mi = base["Fm"][k] / base["R"], img["Fm"][k] / img["R"]
        r[f"{k}_rel_diff_matched_grid"] = abs(mb - mi) / max(abs(mb), 1e-300)
        tol = (1e-6 + 1e-7 * abs(mb) * base["R"]) / base["R"]   # 1 micrometre + 1e-7 relative, base units
        r[f"{k}_matched_within_tol"] = abs(mb - mi) <= tol
        feats_ok &= bool(r[f"{k}_matched_within_tol"])
    for b in ("cw", "eci"):
        r[f"arrival_{b}_base"], r[f"arrival_{b}_image"] = base[f"arr_{b}"][0], img[f"arr_{b}"][0]
        r[f"dwell_margin_{b}_base_tau"] = base[f"arr_{b}"][1] * base["n"]
        r[f"dwell_margin_{b}_image_tau"] = img[f"arr_{b}"][1] * img["n"]
    r["labels_equal"] = all(r[f"arrival_{b}_base"] == r[f"arrival_{b}_image"] for b in ("cw", "eci"))
    r["state_within_tol"] = r["max_dq_cw"] <= tol_state and r["max_dq_eci"] <= tol_state
    r["disc_within_tol"] = r["d_disc_over_R"] <= tol_state
    r["features_matched_grid_within_tol"] = bool(feats_ok)
    r["similarity_verified"] = bool(r["state_within_tol"] and r["disc_within_tol"] and feats_ok and r["labels_equal"]
                                    and not floor)
    return r


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs_c = [(lab, *image(rel0, wn, am, alt), alt) for lab, rel0, wn, am in BASES for alt in ALTS]
    jobs_h = [(lab, *image(rel0, wn, am, alt), alt, 10.0) for lab, rel0, wn, am in BASES[:4] for alt in ALTS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        rc = list(ex.map(run_continuous, jobs_c))
        rh = list(ex.map(run_held, jobs_h))
    rows = []
    for res, kind in ((rc, "continuous"), (rh, "held_Ts10s_at_400km")):
        by = {}
        for x in res:
            by.setdefault(x["label"], {})[x["alt"]] = x
        for lab, d in by.items():
            for alt in ALTS[1:]:
                rows.append(compare(d[ALTS[0]], d[alt], kind))
    df = pd.DataFrame(rows)
    df["wall_total_s"] = time.perf_counter() - t0
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "similarity_checks.csv"), index=False)
    print(df[["case", "kind", "alt_image_km", "altitude_floor_involved", "max_dq_cw", "max_dq_eci", "declared_tol_q",
              "d_disc_over_R", "labels_equal", "features_matched_grid_within_tol", "similarity_verified"]].to_string())


if __name__ == "__main__":
    main()
