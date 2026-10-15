"""Computation benchmark (run on an otherwise idle machine, single process).

For 30 seeded main-grid scenarios, times with time.perf_counter():
  * propagation only, per model (CW, second-order, NERM, inertial ECI), production
    DOP853 settings, 5 repetitions, median per scenario;
  * metric extraction for one branch on the 0.5 s grid;
  * screening workflow cost = CW propagation + CW diagnostics + first-order estimate D1
    versus a full paired evaluation (CW + inertial reference + diagnostics).
Hardware and software versions are recorded. Pure-Python right-hand sides are
used for every model, so relative costs reflect formulation cost under the same
implementation style; absolute times depend on the machine.

Output: results/benchmark/benchmark_runs.csv, benchmark_summary.csv, environment.json
Usage: python experiments/ext/benchmark.py
"""

import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402
from main_grid import grid_scenarios  # noqa: E402

from cwvalid import metrics as M  # noqa: E402
from cwvalid.experiment import RunConfig, make_cmd, run_scenario  # noqa: E402
from cwvalid.runner import software_versions  # noqa: E402
from screening import cw_only_features  # noqa: E402
from cwvalid.simulate import SolverConfig, simulate  # noqa: E402

REPS = 5


def main():
    out = os.path.join(RU.RESULTS, "benchmark")
    os.makedirs(out, exist_ok=True)
    env = software_versions()
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        env["ram_gb"] = m.ullTotalPhys / 2**30
    except Exception:
        pass
    env["machine"] = platform.machine()
    env["timer"] = "time.perf_counter, single process"
    json.dump(env, open(os.path.join(out, "environment.json"), "w"), indent=2)
    scns = grid_scenarios()
    rng = np.random.default_rng(424242)
    pick = [scns[i] for i in rng.choice(len(scns), 30, replace=False)]
    solver = SolverConfig()
    cfg = RunConfig()
    rows = []
    for s in pick:
        cmd = make_cmd(s, cfg)
        for m in ("cw", "so", "nerm", "eci"):
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                sim = simulate(m, s.rel0, cmd, s.horizon, n=s.n, R=s.R, solver=solver)
                ts.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            tr = M.make_trace(sim, M.common_grid(s.horizon, 0.5))
            M.branch_diagnostics(sim, tr, s.horizon, M.OutcomeCriteria())
            t_diag = time.perf_counter() - t0
            rows.append({"sid": s.sid, "sep_m": s.separation, "axis": s.meta["axis"], "wn_ratio": s.wn_ratio,
                         "a_max": s.a_max, "model": m, "propagation_median_s": float(np.median(ts)),
                         "propagation_min_s": float(np.min(ts)), "nfev": sim.nfev, "nsteps": sim.nsteps,
                         "diagnostics_s": t_diag})
        # screening workflow: CW-only run + diagnostics vs full paired evaluation
        ts_cw, ts_pair = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            cw_only_features(s.to_dict())   # the actual screening workflow: CW run + diagnostics + D1
            ts_cw.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            run_scenario(s, RunConfig(models=("cw", "eci"), reference="eci", crosscheck=None))
            ts_pair.append(time.perf_counter() - t0)
        rows.append({"sid": s.sid, "sep_m": s.separation, "axis": s.meta["axis"], "wn_ratio": s.wn_ratio,
                     "a_max": s.a_max, "model": "workflow", "screening_cw_only_s": float(np.median(ts_cw)),
                     "full_pair_s": float(np.median(ts_pair))})
        print(f"  {s.sid} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "benchmark_runs.csv"), index=False)
    prop = df[df.model != "workflow"].groupby("model").agg(
        median_propagation_s=("propagation_median_s", "median"), p90_propagation_s=("propagation_median_s", lambda x: x.quantile(0.9)),
        median_nfev=("nfev", "median"), median_diagnostics_s=("diagnostics_s", "median")).reset_index()
    wf = df[df.model == "workflow"]
    prop.loc[len(prop)] = {"model": "screening workflow (CW run + diagnostics + D1 estimate)", "median_propagation_s": wf.screening_cw_only_s.median(),
                           "p90_propagation_s": wf.screening_cw_only_s.quantile(0.9), "median_nfev": np.nan, "median_diagnostics_s": np.nan}
    prop.loc[len(prop)] = {"model": "full paired evaluation (CW + inertial reference)", "median_propagation_s": wf.full_pair_s.median(),
                           "p90_propagation_s": wf.full_pair_s.quantile(0.9), "median_nfev": np.nan, "median_diagnostics_s": np.nan}
    prop.to_csv(os.path.join(out, "benchmark_summary.csv"), index=False)
    RU.write_table(prop, "tab_benchmark", "Wall-clock cost per one-orbit closed-loop branch (median over 30 seeded "
                   "main-grid scenarios, median of 5 repetitions each), single process, production DOP853 settings, "
                   f"{env.get('processor', '')}; plus the screening workflow (CW run and diagnostics) and a full "
                   "paired evaluation (median of 3 repetitions).", "tab:benchmark",
                   fmt={"median_propagation_s": "{:.3f}", "p90_propagation_s": "{:.3f}", "median_nfev": "{:.0f}",
                        "median_diagnostics_s": "{:.3f}"})
    print(prop.to_string())


if __name__ == "__main__":
    main()
