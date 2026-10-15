"""Terminal position AND velocity assessment.

The main study classifies an approach by position only ("sustained position
arrival": the relative position stays within 1 m for the rest of the orbit).
A rendezvous assessment should also report the relative velocity, because a
position gate alone does not distinguish a controlled arrival from a high-speed
pass through the 1 m ball.

For representative cases (the three offset directions at separations spanning the
converging regime, the onset, and the non-arrival regime) this script extracts,
from the nonlinear-reference closed loop (the physically meaningful trajectory):

  * relative speed at first 1 m entry;
  * final relative speed;
  * maximum relative speed while inside the 1 m ball;
  * closest approach;
  * position-only classification (sustained arrival) and a combined
    position-and-velocity classification for two illustrative terminal-speed
    gates (0.10 and 0.05 m/s, order-of-magnitude docking contact values from
    Fehse, Automated Rendezvous and Docking of Spacecraft, 2003).

Output (project root):
  TERMINAL_POSITION_VELOCITY_ASSESSMENT.csv

Run:  python experiments/run_terminal_assessment.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)

from src import controller, cw, orbit, perturbations, validation


def main():
    radius = orbit.DEFAULT_RADIUS
    mu = orbit.MU_EARTH
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)
    target0 = orbit.circular_orbit_state(radius, mu, np.radians(51.6))
    times = np.linspace(0.0, period, 2001)
    omega_n = 6.0 * n
    a_max = 0.05
    pos_tol = 1.0
    fm = perturbations.ForceModel.two_body()
    ctrl = controller.PDController.critically_damped(omega_n, max_accel=a_max)

    v_gates = [0.10, 0.05]  # illustrative terminal relative-speed gates [m/s]

    cases = {
        "radial": [500.0, 1000.0, 2000.0, 4000.0, 5500.0, 8000.0, 14000.0],
        "along-track": [1000.0, 4000.0, 7700.0, 14000.0, 27000.0],
        "cross-track": [1000.0, 5000.0, 10000.0, 30000.0, 50000.0],
    }
    unit = {"radial": np.array([1., 0, 0]),
            "along-track": np.array([0, 1., 0]),
            "cross-track": np.array([0, 0, 1.])}

    rows = []
    for dname, seps in cases.items():
        for sep in seps:
            rel0 = np.concatenate((unit[dname] * sep, np.zeros(3)))
            rel_states, controls, _, _ = validation.high_fidelity_relative(
                target0, rel0, times, fm, controller=ctrl)
            perr = validation.position_error_series(rel_states)
            verr = validation.velocity_error_series(rel_states)
            inside = perr < pos_tol
            conv_t = validation.convergence_time(times, perr, pos_tol)
            arrived_pos = np.isfinite(conv_t) and perr[-1] < pos_tol
            if inside.any():
                first = int(np.argmax(inside))  # first True
                v_entry = float(verr[first])
                t_entry = float(times[first])
                v_max_inside = float(verr[inside].max())
            else:
                v_entry = np.nan; t_entry = np.nan; v_max_inside = np.nan
            row = {
                "direction": dname, "sep_m": float(sep),
                "closest_approach_m": float(perr.min()),
                "final_pos_err_m": float(perr[-1]),
                "arrived_position": bool(arrived_pos),
                "t_first_1m_entry_s": t_entry,
                "v_at_first_1m_entry_mps": v_entry,
                "v_max_inside_1m_mps": v_max_inside,
                "v_final_mps": float(verr[-1]),
            }
            for g in v_gates:
                row[f"arrived_pos_and_v_lt_{g:g}"] = bool(
                    arrived_pos and np.isfinite(verr[-1]) and verr[-1] < g)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJ, "TERMINAL_POSITION_VELOCITY_ASSESSMENT.csv"), index=False)

    with pd.option_context("display.float_format", lambda v: f"{v:.4g}",
                           "display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    # Agreement between position-only and combined classification.
    for g in v_gates:
        col = f"arrived_pos_and_v_lt_{g:g}"
        agree = int((df["arrived_position"] == df[col]).sum())
        print(f"\nposition-only vs (position AND v<{g} m/s): agree on {agree}/{len(df)} cases; "
              f"disagreements: {int((df['arrived_position'] != df[col]).sum())}")
    print("\nMax final relative speed among position-arrived cases: %.3e m/s"
          % df.loc[df.arrived_position, "v_final_mps"].max())


if __name__ == "__main__":
    main()
