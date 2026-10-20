import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "outputs", "figures")
DAT = os.path.join(ROOT, "outputs", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

from src import cw, orbit, plotting, validation


def main():
    radius = orbit.DEFRAD
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)

    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
    times = np.linspace(0.0, period, 4001)

    kpvalues = [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4]
    kdvalues = [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2]

    df = validation.gainsweep2(
        rel0, times, n, kpvalues, kdvalues, max_accel=None, pos_tol=1.0
    )
    csvpath = os.path.join(DAT, "gainsweep.csv")
    df.to_csv(csvpath, index=False)

    print(f"PD gain sweep ({len(kpvalues)} Kp x {len(kdvalues)} Kd = {len(df)} runs), "
          f"n = {n:.4e} rad/s\n")
    print(df.round(6).to_string(index=False))

    conv = df.dropna(subset=["convergence_time_s"])
    if not conv.empty:
        best = conv.loc[conv["convergence_time_s"].idxmin()]
        print(f"\nFastest converging: Kp={best.kp:.2e}, Kd={best.kd:.2e}, "
              f"t_conv={best.convergence_time_s:.1f} s, delta_v={best.delta_v_mps:.3f} m/s")

    fig = plotting.plotgaincmp(
        df, os.path.join(FIG, "gainsweep.png"),
        metrics=("final_pos_error_m", "convergence_time_s", "delta_v_mps"),
        title="PD gain sweep: metrics over Kp x Kd grid",
    )
    print(f"\nWrote: {os.path.relpath(csvpath, ROOT)}")
    print(f"Wrote: {os.path.relpath(fig, ROOT)}")


if __name__ == "__main__":
    main()
