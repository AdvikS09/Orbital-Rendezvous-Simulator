import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "outputs", "figures")
DAT = os.path.join(ROOT, "outputs", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

from src import controller, cw, orbit, plotting, validation


def main():
    radius = orbit.DEFRAD
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)

    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])

    omegan = 6.0 * n
    maxaccel = 0.05
    pdctrl = controller.PDCtrl.critdamped(omegan, max_accel=maxaccel)

    times = np.linspace(0.0, period, 4001)
    states, controls = cw.propcw(rel0, times, n, pdctrl)

    poserr = validation.poserrseries(states)
    velerr = validation.velerrseries(states)
    metr = validation.rdvmetrics(times, states, controls, pos_tol=1.0)

    history = pd.DataFrame({
        "t_s": times,
        "x_m": states[:, 0], "y_m": states[:, 1], "z_m": states[:, 2],
        "vx_mps": states[:, 3], "vy_mps": states[:, 4], "vz_mps": states[:, 5],
        "ax_mps2": controls[:, 0], "ay_mps2": controls[:, 1], "az_mps2": controls[:, 2],
        "pos_err_m": poserr, "vel_err_mps": velerr,
    })
    histpath = os.path.join(DAT, "rendezvousseries.csv")
    metrpath = os.path.join(DAT, "rendezvousmetrics.csv")
    history.to_csv(histpath, index=False)
    metr.to_csv(metrpath, index=False)

    print(f"CW rendezvous: n = {n:.6e} rad/s, period = {period:.1f} s")
    print(f"omega_n = {omegan:.4e} rad/s (= {omegan/n:.1f} n), max_accel = {maxaccel} m/s^2\n")
    print(metr.T.to_string(header=False))

    figs = [
        plotting.plotlvlhtraj(
            [("chaser", states)], os.path.join(FIG, "rendezvoustrajectory.png"),
            title="CW rendezvous: relative trajectory (LVLH)"),
        plotting.plotposcomps(
            times, states, os.path.join(FIG, "rendezvousposition.png"),
            title="CW rendezvous: relative position vs time"),
        plotting.plotvelcomps(
            times, states, os.path.join(FIG, "rendezvousvelocity.png"),
            title="CW rendezvous: relative velocity vs time"),
        plotting.plotctrlacc(
            times, controls, os.path.join(FIG, "rendezvouscontrol.png"),
            title="CW rendezvous: control acceleration vs time"),
    ]
    print()
    for p in (histpath, metrpath, *figs):
        print(f"Wrote: {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
