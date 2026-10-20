import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import expm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = os.path.join(os.path.dirname(ROOT), "results", "reproduction")

from src import controller, cw, orbit, perturbations, validation

R = orbit.DEFRAD
MU = orbit.MUEARTH
N = cw.cwmeanmotion(R)
T = orbit.orbperiod(R)
TGT0 = orbit.circorbitstate(R, MU, np.radians(51.6))
FM = perturbations.ForceMod.twobody()


def logcross(seps, diffs, level=1.0):
    seps = np.asarray(seps, float)
    diffs = np.asarray(diffs, float)
    o = np.argsort(seps)
    seps, diffs = seps[o], diffs[o]
    for i in range(1, len(seps)):
        if diffs[i - 1] < level <= diffs[i]:
            lx0, lx1 = np.log(seps[i - 1]), np.log(seps[i])
            ly0, ly1 = np.log(diffs[i - 1]), np.log(diffs[i])
            return float(np.exp(lx0 + (np.log(level) - ly0) * (lx1 - lx0) / (ly1 - ly0))), (seps[i - 1], seps[i])
    return float("nan"), (np.nan, np.nan)


def peak(sep, a_max, samples, u=(1.0, 0, 0), wn=6.0):
    rel0 = np.concatenate((np.array(u) * sep, np.zeros(3)))
    times = np.linspace(0.0, T, samples)
    ctrl = controller.PDCtrl.critdamped(wn * N, max_accel=a_max)
    c = validation.cmpfid(TGT0, rel0, times, FM, N, controller=ctrl)
    cwm = validation.rdvmetrics(times, c["cw_states"], c["cw_controls"]).iloc[0]
    hfm = validation.rdvmetrics(times, c["hf_states"], c["hf_controls"]).iloc[0]
    return {"peak_m": float(c["summary"]["max_pos_error_m"].iloc[0]),
            "cw_conv_s": float(cwm["convergence_time_s"]), "hf_conv_s": float(hfm["convergence_time_s"]),
            "cw_arrived": bool(np.isfinite(cwm["convergence_time_s"]) and cwm["final_pos_error_m"] < 1.0),
            "hf_arrived": bool(np.isfinite(hfm["convergence_time_s"]) and hfm["final_pos_error_m"] < 1.0),
            "cw_max_acc": float(cwm["max_control_accel_mps2"]), "hf_max_acc": float(hfm["max_control_accel_mps2"])}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    A, B = cw.cwsystemmats(N)
    wn = 6 * N
    K = np.hstack((wn**2 * np.eye(3), 2 * wn * np.eye(3)))
    ev = np.linalg.eigvals(A - B @ K)
    rows = []
    for e in sorted(ev, key=lambda z: (-abs(z.real), z.imag)):
        if e.imag < 0:
            continue
        zeta = -e.real / abs(e)
        rows.append({"real_per_s": e.real, "imag_per_s": e.imag, "abs_over_n": abs(e) / N, "zeta": zeta,
                     "settling_2pct_s": 4.0 / -e.real})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "T2eigen.csv"), index=False)
    tt = np.linspace(0, 2 * T, 8001)
    st = orbit.propagate(orbit.twobodyderivs, TGT0, tt)
    validation.checkorbit(tt, st, MU, analperiod=T).to_csv(os.path.join(OUT, "T5invariants.csv"), index=False)
    s0 = np.array([200.0, -300.0, 80.0, 0, 0, 0])
    ctrlu = controller.PDCtrl.critdamped(wn, max_accel=None)
    Acl = A - B @ K
    r = []
    for k in (1001, 2001, 4001, 8001, 16001):
        times = np.linspace(0, T, k)
        sts, _ = cw.propcw(s0, times, N, ctrlu)
        exact = np.array([expm(Acl * t) @ s0 for t in times[:: max(1, (k - 1) // 1000)]])
        err = np.abs(sts[:: max(1, (k - 1) // 1000), :3] - exact[:, :3]).max()
        r.append({"samples": k, "dt_s": T / (k - 1), "max_pos_err_m": float(err)})
    pd.DataFrame(r).to_csv(os.path.join(OUT, "S31aexpm.csv"), index=False)
    r = []
    for sep in (5000.0, 14000.0):
        for k in (1001, 2001, 4001, 8001):
            r.append({"sep_m": sep, "samples": k, **peak(sep, 0.05, k)})
    pd.DataFrame(r).to_csv(os.path.join(OUT, "S31bsteps.csv"), index=False)
    seps12 = np.geomspace(50.0, 50_000.0, 12)
    r, onset = [], []
    for am in (0.01, 0.05, 0.10):
        pk = [peak(s, am, 2001)["peak_m"] for s in seps12]
        for s, p in zip(seps12, pk):
            r.append({"a_max": am, "stage": "sweep12", "sep_m": s, "peak_m": p})
        c12, (lo, hi) = logcross(seps12, pk)
        sub = np.geomspace(lo, hi, 9) if np.isfinite(lo) else []
        pks = [peak(s, am, 2001)["peak_m"] for s in sub]
        for s, p in zip(sub, pks):
            r.append({"a_max": am, "stage": "subsweep", "sep_m": s, "peak_m": p})
        csub, br = logcross(sub, pks) if len(sub) else (np.nan, (np.nan, np.nan))
        onset.append({"a_max": am, "onset_12pt_m": c12, "bracket12_lo_m": lo, "bracket12_hi_m": hi,
                      "onset_refined_m": csub, "bracket_sub_lo_m": br[0], "bracket_sub_hi_m": br[1]})
    pd.DataFrame(r).to_csv(os.path.join(OUT, "T10sweeps.csv"), index=False)
    pd.DataFrame(onset).to_csv(os.path.join(OUT, "T10onsets.csv"), index=False)
    print(pd.DataFrame(onset))
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
