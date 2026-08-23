"""Compact gain-and-horizon robustness of the central result.

The central separation sweep uses one controller bandwidth (omega_n = 6 n) and a
one-orbit horizon. This experiment repeats the direction-resolved CW-to-reference
divergence for a small grid of controller bandwidths and integration horizons, to
test whether the two headline conclusions survive:

  1. the directional ranking (radial worst, then along-track, then cross-track);
  2. feedback masking (the one-metre agreement crossing sits far beyond the
     open-loop CW validity range).

Grid:
  omega_n / n in {4, 6, 8}        (controller bandwidth relative to orbit rate)
  horizon     in {0.5, 1, 2}      orbits
  direction   in {radial, along-track, cross-track}
  separation  : 16 log-spaced points, 50 m .. 50 km

For every cell the one-metre CW-to-reference crossing is estimated by log-log
interpolation of the peak trajectory difference against initial separation.
Sampling is held at ~2000 samples/orbit (the converged regime from the
convergence study), so the omega_n=6 n, 1-orbit cells reproduce the main sweep.

Outputs (written to the project root, one level above the platform):
  ROBUSTNESS_GAIN_HORIZON.csv          full per-(dir,gain,horizon,separation) grid
  ROBUSTNESS_GAIN_HORIZON_crossings.csv one-metre crossing per (dir,gain,horizon)

Run:  python experiments/run_robustness_gain_horizon.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)  # "Orbital Mechanics Research"

from src import controller, cw, orbit, perturbations, validation


def log_crossing(seps, diffs, level=1.0):
    """First upward crossing of ``level`` in log-log; None if never/always."""
    seps = np.asarray(seps, float)
    diffs = np.asarray(diffs, float)
    order = np.argsort(seps)
    seps, diffs = seps[order], diffs[order]
    if np.all(diffs >= level):
        return ("below_range", float(seps[0]))       # crosses below smallest tested
    if np.all(diffs < level):
        return ("above_range", float(seps[-1]))      # never reaches level in range
    for i in range(1, len(seps)):
        if diffs[i - 1] < level <= diffs[i]:
            lx0, lx1 = np.log(seps[i - 1]), np.log(seps[i])
            ly0, ly1 = np.log(diffs[i - 1]), np.log(diffs[i])
            lx = lx0 + (np.log(level) - ly0) * (lx1 - lx0) / (ly1 - ly0)
            return ("crossing", float(np.exp(lx)))
    return ("above_range", float(seps[-1]))


def main():
    radius = orbit.DEFAULT_RADIUS
    mu = orbit.MU_EARTH
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)
    target0 = orbit.circular_orbit_state(radius, mu, np.radians(51.6))
    a_max = 0.05
    pos_tol = 1.0
    fm = perturbations.ForceModel.two_body()

    directions = {
        "radial": np.array([1.0, 0.0, 0.0]),
        "along-track": np.array([0.0, 1.0, 0.0]),
        "cross-track": np.array([0.0, 0.0, 1.0]),
    }
    bandwidths = [4.0, 6.0, 8.0]          # omega_n / n
    horizons = [0.5, 1.0, 2.0]            # orbits
    separations = np.geomspace(50.0, 50_000.0, 16)

    grid_rows = []
    for wratio in bandwidths:
        omega_n = wratio * n
        ctrl = controller.PDController.critically_damped(omega_n, max_accel=a_max)
        zeta = 1.0  # gains built from critically_damped_gains (open-loop 2nd order)
        for h in horizons:
            N = int(round(2000 * h)) + 1
            times = np.linspace(0.0, h * period, N)
            for dname, u in directions.items():
                for sep in separations:
                    rel0 = np.concatenate((u * sep, np.zeros(3)))
                    cmp = validation.compare_cw_high_fidelity(
                        target0, rel0, times, fm, n, controller=ctrl)
                    s = cmp["summary"].iloc[0]
                    hfm = validation.rendezvous_metrics(
                        times, cmp["hf_states"], cmp["hf_controls"], pos_tol=pos_tol).iloc[0]
                    grid_rows.append({
                        "omega_n_over_n": wratio,
                        "horizon_orbits": h,
                        "direction": dname,
                        "sep_m": float(sep),
                        "max_traj_diff_m": float(s["max_pos_error_m"]),
                        "final_traj_diff_m": float(s["final_pos_error_m"]),
                        "rel_max_diff": float(s["max_pos_error_m"] / sep),
                        "hf_conv_time_s": float(hfm["convergence_time_s"]),
                        "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
                        "hf_max_accel_mps2": float(hfm["max_control_accel_mps2"]),
                        "hf_saturated": bool(hfm["max_control_accel_mps2"] >= 0.999 * a_max),
                    })

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(PROJ, "ROBUSTNESS_GAIN_HORIZON.csv"), index=False)

    # One-metre crossing per (direction, bandwidth, horizon).
    cross_rows = []
    for wratio in bandwidths:
        for h in horizons:
            for dname in directions:
                sub = grid[(grid.omega_n_over_n == wratio) &
                           (grid.horizon_orbits == h) &
                           (grid.direction == dname)]
                kind, val = log_crossing(sub.sep_m.values, sub.max_traj_diff_m.values)
                cross_rows.append({
                    "omega_n_over_n": wratio, "horizon_orbits": h, "direction": dname,
                    "crossing_kind": kind,
                    "one_metre_crossing_km": (val / 1000.0) if kind == "crossing" else np.nan,
                    "note": {"crossing": "", "above_range": ">50 km (no crossing)",
                             "below_range": "<50 m"}[kind],
                })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(os.path.join(PROJ, "ROBUSTNESS_GAIN_HORIZON_crossings.csv"), index=False)

    pd.set_option("display.width", 160)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print("\n=== one-metre CW-to-reference crossing [km], by bandwidth x horizon ===")
        for dname in directions:
            piv = cross[cross.direction == dname].pivot(
                index="omega_n_over_n", columns="horizon_orbits",
                values="one_metre_crossing_km")
            print(f"\n{dname}:")
            print(piv.to_string())
        # ranking check per cell
        print("\n=== directional ranking per cell (should be radial < along < cross) ===")
        ok = True
        for wratio in bandwidths:
            for h in horizons:
                c = {d: cross[(cross.omega_n_over_n == wratio) & (cross.horizon_orbits == h) &
                              (cross.direction == d)].iloc[0] for d in directions}
                def key(d):
                    r = c[d]
                    return (np.inf if r.crossing_kind == "above_range"
                            else -np.inf if r.crossing_kind == "below_range"
                            else r.one_metre_crossing_km)
                r, a, cx = key("radial"), key("along-track"), key("cross-track")
                ranked = r <= a <= cx
                ok = ok and ranked
                print(f"  wn/n={wratio}, h={h}: radial={r:.3g} along={a:.3g} cross={cx:.3g}  "
                      f"{'RANK OK' if ranked else 'RANK CHANGED'}")
        print("\nDirectional ranking preserved in every cell:" , ok)
    print(f"\nWrote ROBUSTNESS_GAIN_HORIZON.csv ({len(grid)} rows) and crossings ({len(cross)} rows)")


if __name__ == "__main__":
    main()
