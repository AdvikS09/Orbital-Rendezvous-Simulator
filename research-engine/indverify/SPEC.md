# Independent closed-loop verifier: mathematical specification

This document is the only input for implementing the independent verifier (`indverify/`). The implementation must not import, copy or call anything from the production code (`cwvalid/`, `src/`, `experiments/`). It may use Python's standard library, NumPy and SciPy. It may read scenario definition files (JSON) produced by the production pipeline, because these are configuration, not code.

Units: SI (m, s, m/s, m/s², rad). All vectors are 3-component.

## 1. Constants and target orbit

- μ = 3.986004418e14 m³/s²
- R_E = 6 378 137.0 m
- Target (chief) orbit: circular, altitude h (default 400 000 m), radius R = R_E + h, mean motion n = sqrt(μ / R³), period T_orb = 2π / n.
- Inertial (ECI) placement: inclination i = 51.6° (two-body motion makes the result independent of i; it is fixed for definiteness). At t = 0 the target is at the ascending node on the ECI +X axis:
  r_t(0) = (R, 0, 0), v_t(0) = n R (0, cos i, sin i).
- Horizon: T = N_orb · T_orb, where N_orb = `horizon_orbits` from the scenario (1 or 2).

## 2. Scenario input

A scenario is a JSON object with at least: `sid` (string), `rel0` = (x, y, z, ẋ, ẏ, ż) initial relative state in the LVLH frame (m, m/s), `wn_ratio` (ω_n / n), `a_max` (m/s², or null for no cap), `horizon_orbits`, `altitude` (m).

## 3. LVLH (Hill) frame of the target

From the target inertial state (r, v):

- x̂ = r / |r| (radial, away from Earth)
- ẑ = (r × v) / |r × v| (orbit normal)
- ŷ = ẑ × x̂ (along-track)
- C = matrix with rows x̂ᵀ, ŷᵀ, ẑᵀ (maps ECI components to LVLH components)
- Frame angular velocity, two-body target, expressed in LVLH components: ω = (0, 0, |r × v| / |r|²)

Relative state from inertial states (target t, chaser c):

- ρ = C (r_c − r_t)
- ρ̇ = C (v_c − v_t) − ω × ρ        (rotating-frame derivative; ω × ρ is the transport term)

Inverse (used once, to build the chaser's initial inertial state from rel0):

- r_c = r_t + Cᵀ ρ
- v_c = v_t + Cᵀ (ρ̇ + ω × ρ)

## 4. Controller (identical law in every branch; each branch uses its own state)

- ω_n = wn_ratio · n, K_p = ω_n², K_d = 2 ω_n
- a_raw = −K_p ρ − K_d ρ̇   (LVLH components, desired state = origin at rest)
- if a_max is not null and |a_raw| > a_max: a = a_raw · a_max / |a_raw|  (direction preserved); otherwise a = a_raw
- Continuous feedback: the command is evaluated from the branch's current state at every right-hand-side evaluation.
- In the inertial branch the LVLH command is applied to the chaser as Cᵀ a, with C built from the current target state, and ρ, ρ̇ for the feedback are reconstructed from the current inertial states with Section 3.

## 5. Branch dynamics

**Inertial two-spacecraft reference (primary).** State (r_t, v_t, r_c, v_c), 12 components:

- r̈_t = −μ r_t / |r_t|³
- r̈_c = −μ r_c / |r_c|³ + Cᵀ a(ρ, ρ̇)

Physical-domain check: the inertial branch stops when the chaser altitude |r_c| − R_E falls below 100 000 m (locate the crossing with an event). Such a run is a physical-domain termination, not a numerical failure; it counts as "not arrived".

**CW branch.** LVLH state (x, y, z, ẋ, ẏ, ż):

- ẍ = 3n²x + 2nẏ + a_x
- ÿ = −2nẋ + a_y
- z̈ = −n²z + a_z

**Quadratic (second-order) branch.**

- ẍ = 3n²x + 2nẏ − (3n² / (2R)) (2x² − y² − z²) + a_x
- ÿ = −2nẋ + (3n² / R) x y + a_y
- z̈ = −n²z + (3n² / R) x z + a_z

No physical-domain check is applied to the CW and quadratic branches. If their integration fails (step-size underflow, non-finite state), report a numerical failure with its time; never infer an arrival or a physical outcome from a failed run.

## 6. Non-smooth control: saturation switches

The saturated law is continuous but its derivative jumps where |a_raw| = a_max. The integrator must not take steps across such a switch without locating it: use an event function g_sat = |a_raw(state)| − a_max, stop at each root, and restart the integration from the root state (the root itself must not be detected again immediately; e.g. require a minimum advance or alternate the sought direction). Record the switch times. Give up event restarts only after a large cap (e.g. 2000) and record that it happened.

## 7. Solvers and convergence

Implement at least:

1. SciPy `solve_ivp` with method `Radau` (primary), with the switch events of Section 6 and the altitude event of Section 5, dense output, relative tolerance `rtol` and absolute tolerances chosen per component (suggestion: relative-state components atol = rtol × 1 km and rtol × 1 m/s; inertial components atol = rtol × 1000 km and rtol × 1 km/s), maximum step `max_step`.
2. SciPy `solve_ivp` with method `DOP853` and the same events (secondary).
3. A self-written fixed-step classical fourth-order Runge-Kutta integrator with step `h` (no events; a step may straddle a switch), with interpolation between steps by cubic Hermite polynomials using the stored state derivatives.

Convergence must be demonstrated by repeating each case with a ladder of settings (e.g. Radau rtol 1e-8, 1e-9, 1e-10, 1e-11, 1e-12; DOP853 rtol 1e-11, 1e-12, 1e-13; RK4 h = 1, 0.5, 0.25 s) and reporting the change of every compared quantity between settings.

## 8. Assessed quantities

Let T be the horizon and t_end ≤ T the time the branch actually reached.

**Trajectory comparison.** Over the common interval [0, min(t_end,A, t_end,B)] of two branches A and B: peak position difference P = max_t |ρ_A(t) − ρ_B(t)| and its time; peak velocity difference; maximum difference of the applied-command norms. Evaluate on a uniform grid and refine the maximum locally (bounded scalar maximisation within one grid step either side of the grid maximum).

**Terminal state.** |ρ(t_end)| and |ρ̇(t_end)|.

**Terminal arrival (primary criterion).** Terminal set: |ρ| < 1 m AND |ρ̇| < 0.01 m/s. Define g(t) = max(|ρ(t)| / 1 m, |ρ̇(t)| / 0.01 m/s) − 1, so g < 0 means inside.

- A branch can arrive only if it completed the full horizon (t_end = T, no failure, no domain termination).
- It must be inside at T (g(T) < 0).
- Final-entry time t_entry: the last time g changes from ≥ 0 to < 0; if g < 0 throughout, t_entry = 0.
- Dwell = T − t_entry; arrived iff dwell ≥ 300 s; dwell margin = dwell − 300 s.

Implement two monitoring conventions and report both:

- **Production convention:** sample g on t_k = 0.5 k s for k = 0 … ⌊T/0.5⌋, with T appended if it is not already a grid point (if the last grid point is within 1e-9 s of T it is replaced by T). Inside at end iff g < 0 at the last sample. If all samples are inside, t_entry = 0; otherwise let j be the last sample with g ≥ 0 and find t_entry as the root of g on [t_j, t_{j+1}] with Brent's method (xtol 1e-6 s) using the continuous (dense-output) solution; if g does not change sign on that interval, use the midpoint.
- **Fine convention:** the same definition evaluated on a much finer grid (default 0.01 s) over the whole horizon, with every sign change of g located by root finding, so that brief excursions between 0.5 s samples are detected. Report the number of terminal-set entries and exits found by each convention.

**Saturation.** Total saturated time (time with |a_raw| > a_max) and the switch times.

**Statuses.** For every branch: completed / physical-domain termination (altitude) / numerical failure, with the termination time and reason.

## 9. Outputs

A function `run_case(scenario_dict, solver_settings) -> (summary_dict, history_dict)` and a command-line entry point that reads a JSON list of scenarios and writes one JSON line per (scenario, solver setting) plus an `.npz` history per scenario at the finest setting (time, ρ, ρ̇, applied command and its norm for each branch).

## 10. Self-checks to include as unit tests

- Frame round trip: rel → inertial → rel recovers rel0 to round-off.
- Transport term: ρ̇ from Section 3 equals the finite-difference derivative of ρ(t) along a two-body propagation of both spacecraft.
- Uncontrolled CW branch versus the closed-form CW solution over one orbit.
- Unsaturated controlled CW branch versus the matrix exponential of the closed-loop matrix.
- Saturation preserves direction and caps the magnitude.
- Arrival assessment on synthetic g(t) histories with known entry times, including an excursion between 0.5 s samples that only the fine convention detects.
- Small separations: inertial reference and CW branch agree to second order in separation (difference shrinks about 4× when the separation halves).
- Independence: no module of `indverify` imports `cwvalid`, `src` or `experiments`.
