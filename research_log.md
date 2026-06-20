# Research Log

## Phase 1 — Orbital rendezvous (CW / Hill + PD control)

### Goal
Build the foundation for spacecraft rendezvous: a verified 2D two-body
propagator, Clohessy–Wiltshire (Hill) linear relative motion, a PD rendezvous
controller, verification of CW against a non-linear truth model, and a simple
keep-out-zone avoidance scaffold. SI units throughout; numpy / scipy /
matplotlib / pandas only.

### Modelling decisions
* **Planar (2D) only** for Phase 1. State `[x, y, vx, vy]` everywhere; position
  is `state[:2]`, velocity `state[2:]`.
* **Reference orbit:** circular 400 km LEO. `a = R_EARTH + 400 km = 6 778.137 km`,
  `MU_EARTH = 3.986004418e14`. Analytic period `T = 2π√(a³/µ) = 5553.62 s`,
  mean motion `n = √(µ/a³) = 1.1314e-3 rad/s`.
* **LVLH frame:** `x` radial (R-bar, +out), `y` along-track (V-bar, +velocity).
  CW equations used:
  `ẍ − 2n ẏ − 3n² x = aₓ`, `ÿ + 2n ẋ = a_y`.
* **Truth model for CW verification:** rather than approximating, the deputy is
  propagated as an independent full two-body orbit (adaptive RK45, rtol/atol
  1e-12) and then differenced against the target in the *rotating* LVLH frame
  (with the `ω × r` transport term), giving an exact non-linear relative
  trajectory to compare CW against.

### Integrators
Implemented explicit Euler (baseline), classic RK4, and velocity-Verlet
(leapfrog, symplectic). Verlet operates on an acceleration-of-position function
(valid for conservative two-body gravity); RK4/Euler operate on the full
first-order derivative (also used for the velocity-dependent CW + control
dynamics).

**Validation (2 orbits, dt = T/4000 ≈ 1.39 s), `integrator_validation.csv`:**

| method | max energy drift (rel) | period rel err |
| --- | --- | --- |
| euler | 3.66e-2 | 1.49e-2 |
| rk4 | 6.08e-15 | 1.4e-13 |
| verlet | 1.53e-12 | 8.2e-7 |

Interpretation: Euler bleeds energy badly; RK4 is effectively exact at this step;
Verlet keeps energy *bounded* (symplectic) with a small along-track phase error —
exactly the expected trade-off.

### CW / Hill module
* Closed-form state-transition matrix `Φ(t)` implemented and used as ground
  truth. Numerical RK4 integration of the CW ODE matches `Φ(t) x₀` to ~1e-10 m
  over a full orbit (`cw_stm_validation.csv`).
* Sanity checks that fell out correctly: `Φ(0) = I`; a pure along-track offset is
  a CW equilibrium (stays put); a 100 m radial offset produces ≈ −3771 m
  (= −6π·x₀) of along-track drift per orbit.

### PD rendezvous controller
* `u = −Kp (r − r_target) − Kd (v − v_target)`, gains support scalar / per-axis /
  matrix form. Critically damped tuning helper: `Kp = ωₙ²`, `Kd = 2ωₙ`.
* Default `ωₙ = 6n` (an order of magnitude above the orbital rate) with a thrust
  saturation of 0.05 m/s². From `[150, −250] m`:
  final position error 1.2e-7 m, settling 1541 s, peak accel 0.0134 m/s²,
  Δv 1.85 m/s (`pd_rendezvous_metrics.csv`).

### CW vs non-linear breakdown (`cw_vs_nonlinear_error.csv`)
Pure radial offsets propagated one orbit:

| initial sep | max CW error | % of sep |
| --- | --- | --- |
| 100 m | 1.10 m | 1.1 % |
| 1 km | 109.7 m | 11.0 % |
| 10 km | 11.0 km | 110 % |
| 50 km | 281 km | 562 % |

Confirms CW is trustworthy for close-proximity ops (≤ ~hundreds of metres) and
degrades quickly beyond ~1 km, motivating the later non-linear targeting work.

### Obstacle / keep-out-zone avoidance (`obstacle_avoidance_metrics.csv`)
PD attraction + artificial-potential-field (FIRAS) repulsion. A plain PD approach
flies straight through a 30 m keep-out zone placed on its path (clearance
−28.8 m); adding APF repulsion (`η = 50`) keeps +8.5 m clearance and still
reaches the target, at the cost of Δv (1.83 → 4.13 m/s).

### Known limitations / next steps
* APF is susceptible to local minima if a zone sits exactly between chaser and
  target; the demo offsets the zone to give a clear go-around side.
* 2D only; no J2, drag, or out-of-plane motion yet.
* Continuous-thrust PD only — no impulsive / optimal targeting yet.

### Tests
`tests/` (stdlib `unittest`, no extra deps): two-body conservation & period,
orbital-element round-trips, integrator accuracy/symplecticity, CW-vs-STM,
CW-vs-non-linear, LVLH round-trip, PD convergence & saturation, APF clearance.
20 tests, all passing.
