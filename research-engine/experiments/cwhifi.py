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

from src import controller, cw, orbit, perturbations, plotting, validation


def main():
    radius = orbit.DEFRAD
    inc = np.radians(51.6)
    mu = orbit.MUEARTH
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)
    target0 = orbit.circorbitstate(radius, mu, inc)
    times = np.linspace(0.0, period, 4001)

    t2 = np.linspace(0.0, 2.0 * period, 8001)
    tbstates = orbit.propagate(perturbations.ForceMod.twobody().derivatives, target0, t2)
    verify = validation.checkorbit(t2, tbstates, mu, analperiod=period)
    checkpath = os.path.join(DAT, "orbitvalidation.csv")
    verify.to_csv(checkpath, index=False)
    print("Two-body orbit validation (RK4, 2 orbits):")
    print(verify.to_string(index=False))

    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
    pdctrl = controller.PDCtrl.critdamped(6.0 * n, max_accel=0.05)
    fmhf = perturbations.ForceMod.j2_drag(area=1.0, mass=100.0)
    cmp = validation.cmpfid(target0, rel0, times, fmhf, n, controller=pdctrl)

    ts = pd.DataFrame({
        "t_s": times,
        "pos_error_m": cmp["pos_error"],
        "vel_error_mps": cmp["vel_error"],
    })
    tspath = os.path.join(DAT, "hifiseries.csv")
    ts.to_csv(tspath, index=False)
    print("\nControlled rendezvous: CW vs high-fidelity (two-body + J2 + drag):")
    print(cmp["summary"].to_string(index=False))

    errfig = plotting.ploterrgrowth(
        times, cmp["pos_error"], cmp["vel_error"],
        os.path.join(FIG, "hifierror.png"),
        title="CW vs high-fidelity error over time (controlled rendezvous)",
    )

    rel0drift = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    mod = [
        perturbations.ForceMod.twobody(),
        perturbations.ForceMod.j2only(),
        perturbations.ForceMod.dragonly(area=1.0, mass=100.0),
        perturbations.ForceMod.j2_drag(area=1.0, mass=100.0),
    ]
    series = []
    summ = []
    for fm in mod:
        c = validation.cmpfid(target0, rel0drift, times, fm, n, controller=None)
        series.append((fm.label, c["pos_error"]))
        summ.append(c["summary"])
    toggles = pd.concat(summ, ignore_index=True)
    togglpath = os.path.join(DAT, "hifitoggles.csv")
    toggles.to_csv(togglpath, index=False)
    print("\nFree-drift CW vs high-fidelity divergence by force-model toggle "
          "(1 km along-track offset, 1 orbit):")
    print(toggles.to_string(index=False))

    togglfig = plotting.ploterrcurves(
        times, series, os.path.join(FIG, "hifitoggles.png"),
        ylabel="CW vs high-fidelity position error [m]",
        title="Model divergence by perturbation toggle (free drift)", logy=True,
    )

    print()
    for p in (checkpath, tspath, togglpath, errfig, togglfig):
        print(f"Wrote: {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
