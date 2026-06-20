"""Validate the two-body model and compare the three integrators.

Propagates a circular 400 km LEO for two orbital periods with Euler, RK4 and
velocity-Verlet, then reports:

* specific-energy and angular-momentum drift (conservation),
* measured vs analytic orbital period,

and writes a comparison CSV plus an orbit-trajectory and conservation figure.

Run:  python scripts/run_orbit_validation.py
"""

import _bootstrap  # noqa: F401  (adds project root to sys.path)

from src import analysis, physics, plotting, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


def main():
    radius = DEFAULT_RADIUS
    period = physics.orbital_period(radius, MU_EARTH)
    state0 = physics.circular_orbit_state(radius, MU_EARTH)

    n_orbits = 2
    steps_per_orbit = 4000
    dt = period / steps_per_orbit
    t_span = (0.0, n_orbits * period)

    methods = ["euler", "rk4", "verlet"]
    results = {m: simulation.propagate_two_body(state0, t_span, dt, MU_EARTH, m) for m in methods}

    # --- Validation table ---
    table = analysis.integrator_comparison_table(results, MU_EARTH, true_period=period)
    table.insert(1, "analytic_period_s", period)
    csv_path = _bootstrap.result_path("integrator_validation.csv")
    table.to_csv(csv_path, index=False)

    print(f"Circular LEO: radius = {radius/1e3:.3f} km, analytic period = {period:.3f} s")
    print(f"Integrating {n_orbits} orbits at dt = {dt:.4f} s ({steps_per_orbit} steps/orbit)\n")
    print(table.to_string(index=False))

    # --- Figures ---
    orbit_fig = plotting.plot_orbit_trajectory(
        results["rk4"].states,
        _bootstrap.figure_path("orbit_trajectory.png"),
        title="Circular 400 km LEO (RK4, 2 orbits)",
    )

    drift_series = [
        (m, analysis.relative_drift(analysis.energy_series(results[m].states, MU_EARTH)))
        for m in methods
    ]
    cons_fig = plotting.plot_conservation(
        results["rk4"].t,
        drift_series,
        _bootstrap.figure_path("energy_drift.png"),
        title="Specific-energy drift by integrator (2 orbits)",
    )

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {orbit_fig}")
    print(f"Wrote: {cons_fig}")


if __name__ == "__main__":
    main()
