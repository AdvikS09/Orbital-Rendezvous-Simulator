import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)

from src import cw, lvlh, orbit


def main():
    radius = orbit.DEFAULT_RADIUS
    mu = orbit.MU_EARTH
    n = cw.cw_mean_motion(radius)
    target0 = orbit.circular_orbit_state(radius, mu, np.radians(51.6))
    a_target = orbit.rv_to_elements(target0, mu)["a"]

    separations = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0]
    rows = []
    for x0 in separations:
        for kind, ydot0 in (("zero_velocity", 0.0), ("bounded_motion", -2.0 * n * x0)):
            rel0 = np.array([x0, 0.0, 0.0, 0.0, ydot0, 0.0])
            chaser0 = lvlh.lvlh_to_inertial(target0, rel0)
            el = orbit.rv_to_elements(chaser0, mu)
            a_c = el["a"]
            da = a_c - a_target
            rows.append({
                "kind": kind, "x0_m": x0, "ydot0_mps": ydot0,
                "a_target_m": a_target, "a_chaser_m": a_c,
                "delta_a_m": da,
                "delta_a_over_x0": da / x0,
                "delta_a_over_x0sq_over_r": da / (x0 * x0 / radius),
                "chaser_ecc": el["e"],
                "period_diff_s": el["period"] - orbit.orbital_period(a_target, mu),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJ, "ENERGY_MATCHING_SMA.csv"), index=False)

    with pd.option_context("display.float_format", lambda v: f"{v:.6g}",
                           "display.width", 200, "display.max_columns", 20):
        for kind in ("zero_velocity", "bounded_motion"):
            sub = df[df.kind == kind][["x0_m", "delta_a_m", "delta_a_over_x0",
                                       "delta_a_over_x0sq_over_r", "chaser_ecc",
                                       "period_diff_s"]]
            print(f"\n=== {kind} ===")
            print(sub.to_string(index=False))
    print("\nInterpretation:")
    print("  zero_velocity   : delta_a / x0        ~ constant  -> FIRST-order SMA mismatch")
    print("  bounded_motion  : delta_a / (x0^2/r)  ~ constant  -> SECOND-order residual only")


if __name__ == "__main__":
    main()
