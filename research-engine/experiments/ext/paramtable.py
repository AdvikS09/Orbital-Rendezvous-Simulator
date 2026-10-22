import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import reportutil as RU
from maingrid import AMAX, CONFIG, SEPS, WN

from cwvalid.constants import (DEFALT, DEFRAD, J2EARTH,
                               MUEARTH, OMEGAEARTH, REARTH, mean_motion, period)


def main():
    n = mean_motion(DEFRAD)
    T = period(DEFRAD)
    sv = CONFIG["solver"]
    rows = [
        ("Earth gravitational parameter", "mu", f"{MUEARTH:.10g} m^3/s^2", "src/orbit.py (manuscript Table 3)"),
        ("Earth equatorial radius", "R_E", f"{REARTH:.1f} m", "src/orbit.py"),
        ("J2", "J2", f"{J2EARTH:.9g}", "src/orbit.py"),
        ("Earth rotation rate", "omega_E", f"{OMEGAEARTH:.8g} rad/s", "src/orbit.py"),
        ("Chief orbit", "", f"circular, altitude {DEFALT/1e3:.0f} km, R = {DEFRAD:.0f} m", "manuscript"),
        ("Inclination (inertial branch)", "i", "51.6 deg (no effect on two-body runs; matters with J2 or drag)", "manuscript"),
        ("Mean motion", "n", f"{n:.10g} rad/s", "sqrt(mu/R^3)"),
        ("Orbital period / main horizon", "T", f"{T:.4f} s", "2 pi / n"),
        ("PD gains", "Kp, Kd", "Kp = wn^2, Kd = 2 wn (isotropic)", "manuscript eq."),
        ("Bandwidth ratios", "wn/n", ", ".join(f"{w:g}" for w in WN), "study grid"),
        ("Acceleration caps", "a_max", ", ".join(f"{a:g}" for a in AMAX) + " m/s^2", "study grid"),
        ("Saturation", "", "direction-preserving norm scaling", "src/controller.py"),
        ("Separations (main grid)", "rho0", f"40 log-spaced, {SEPS[0]:.0f} m to {SEPS[-1]:.0f} m (ratio {SEPS[1]/SEPS[0]:.4f})", "study grid"),
        ("Directions (main grid)", "", "+/-radial, +/-along-track, +/-cross-track; zero LVLH velocity", "study grid"),
        ("Position tolerance", "p_tol", f"{CONFIG['crit']['pos_tol']} m", "study assumption"),
        ("Relative-speed tolerance", "v_tol", f"{CONFIG['crit']['vel_tol']} m/s", "study assumption"),
        ("Minimum remaining dwell", "", f"{CONFIG['crit']['dwell']} s", "study assumption"),
        ("Legacy position-only grid", "", f"{CONFIG['crit']['legacy_samples']} samples per orbit", "manuscript sweep"),
        ("Trajectory thresholds", "", "0.1 m and 1 m peak CW-to-reference difference", "manuscript criteria"),
        ("Integrator (production)", "", f"scipy DOP853, rtol {sv['rtol']:g}, atol {sv['atol_rel_pos']:g} m / {sv['atol_rel_vel']:g} m/s (relative), "
         f"{sv['atol_eci_pos']:g} m / {sv['atol_eci_vel']:g} m/s (inertial), {sv['atol_dv']:g} m/s (Delta-v), max step {sv['max_step']:g} s, "
         + ("event-located saturation switches with restart" if sv.get("sat_events") else "no switch events"), "convergence study"),
        ("Metric grid", "", f"uniform {CONFIG['dt_eval']} s + horizon end; extremes and crossings root-refined on dense output", "verification"),
        ("Primary nonlinear reference", "", "inertial two-body propagation of both spacecraft, LVLH reconstruction", "manuscript reference"),
        ("Cross-check formulation", "", "exact nonlinear relative equations in LVLH (NERM)", "new"),
        ("Altitude floor (domain check)", "", "chaser altitude 100 km terminates the NERM and inertial branches", "new"),
        ("Drag model (operational study)", "", "exponential, rho0 3.9e-12 kg/m^3 at 400 km, H 60 km; Cd 2.2, A 1 m^2, m 100 kg", "manuscript Table 3"),
    ]
    df = pd.DataFrame(rows, columns=["quantity", "symbol", "value", "source"])
    RU.writetable(df, "parameters", "Model and experiment parameters of the extension study.", "tab:parameters")
    sat = []
    for w in WN:
        for a in AMAX:
            kp = (w * n) ** 2
            sat.append({"wn/n": w, "a_max [m/s^2]": a, "Kp [1/s^2]": kp, "Kd [1/s]": 2 * w * n,
                        "a_max/Kp [m]": a / kp})
    RU.writetable(pd.DataFrame(sat), "saturationdistance",
                  "Gains and zero-velocity initial saturation distance a_max/Kp for the main-grid controller settings.",
                  "tab:satdist", fmt={"Kp [1/s^2]": "{:.4e}", "Kd [1/s]": "{:.4e}", "a_max/Kp [m]": "{:.1f}"})
    print(pd.DataFrame(sat).to_string())


if __name__ == "__main__":
    main()
