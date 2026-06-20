# Research Log

This log documents the **current 3D orbital rendezvous research engine**:
CW-based relative-motion guidance with a PD controller, validated against a
higher-fidelity non-linear orbital model (two-body + J2 + atmospheric drag).
An earlier 2D prototype has been superseded and is archived (see the final
section and `old_project_backup/`).

---

## Current engine — 3D CW rendezvous + PD control + high-fidelity validation

### Goal
Implement CW-based rendezvous guidance in full 3D and quantify how trustworthy
the linear model is by comparing it against a higher-fidelity non-linear orbital
simulator. SI units throughout; dependencies limited to numpy / matplotlib /
pandas.

### Modelling decisions
* **3D throughout.** State is `[x, y, z, vx, vy, vz]` (position `state[:3]`,
  velocity `state[3:]`) in both the inertial (ECI) and LVLH frames.
* **LVLH frame:** `x` radial (R-bar, +out), `y` along-track (V-bar, +velocity),
  `z` cross-track (H-bar, orbit normal). The transform uses the rotating-frame
  angular velocity `ω = (r × v)/r²` and the `ω × ρ` transport term for velocity.
* **Reference orbit:** circular 400 km LEO. `a = R_EARTH + 400 km = 6 778.137 km`,
  `MU_EARTH = 3.986004418e14`. Analytic period `T = 5553.62 s`, mean motion
  `n = √(µ/a³) = 1.1314e-3 rad/s`. The high-fidelity scenarios use an inclined
  (i = 51.6°, ISS-like) inertial orbit so J2 and 3D geometry are exercised; the
  CW rendezvous/gain-sweep results are inclination-independent (pure CW depends
  only on `n`).
* **CW equations (3D):**
  `ẍ − 2n ẏ − 3n² x = aₓ` (radial), `ÿ + 2n ẋ = a_y` (along-track),
  `z̈ + n² z = a_z` (cross-track — a decoupled harmonic oscillator at the orbital
  period).
* **Integration:** a single hand-rolled **fixed-step RK4** is used everywhere
  (deterministic and reproducible). SciPy is not used.
* **What "higher-fidelity" means here:** the high-fidelity model differs from CW
  *only in the force model* (full non-linear gravity + J2 + drag). Both the CW and
  high-fidelity trajectories use the **same RK4 integrator at the same step**, so
  integration error is common to both and the measured discrepancy isolates
  *modelling* error rather than integration error. The orbit-validation result
  below confirms the shared integrator is itself trustworthy at this step.

### Core orbital mechanics (`src/orbit.py`)
Two-body dynamics, RK4 step and propagator, circular-orbit setup, 3D classical
orbital-element conversions, and energy / angular-momentum / period diagnostics.

**Two-body validation (RK4, 2 orbits), `orbit_validation.csv`:**

| metric | value |
| --- | --- |
| max specific-energy drift (relative) | 1.4e-14 |
| max angular-momentum drift (relative) | 6.6e-15 |
| measured vs analytic period (relative error) | 1.3e-13 |

3D orbital-element round-trips (RV → COE → RV) reproduce the state to ~1e-3 m.

### Perturbations (`src/perturbations.py`)
* **J2:** the standard zonal-J2 gradient acceleration. Being axisymmetric, it
  conserves the z-component of angular momentum (verified to < 1e-8 over two
  orbits) while the two-body specific energy oscillates at the J2-potential scale
  (~order J2, bounded — not secular).
* **Drag:** exponential atmosphere (ρ₀ ≈ 3.9e-12 kg/m³ at 400 km, scale height
  60 km, moderate solar activity); `a = −½ ρ (Cd·A/m) |v_rel| v_rel` with
  `v_rel = v − ω⊕ × r`. Default vehicle: Cd = 2.2, A = 1 m², m = 100 kg. Drag
  removes energy and decays the orbit (verified).
* **Toggles:** a `ForceModel` dataclass exposes the four required configurations —
  two-body, J2 only, drag only, J2 + drag — via `acceleration()` / `derivatives()`.

### CW / Hill module (`src/cw.py`)
* Closed-form 3D state-transition matrix `Φ(t)` (in-plane 4×4 block + cross-track
  2×2 oscillator block) used as ground truth. Numerical RK4 integration of the CW
  ODE matches `Φ(t) x₀` to ~4e-10 m over a full orbit for a representative state
  (the absolute figure scales with state magnitude).
* Sanity checks that fell out correctly: `Φ(0) = I`; a pure along-track offset is a
  CW equilibrium (stays put); cross-track is a clean oscillator at the orbital
  period, decoupled from the in-plane motion.

### PD rendezvous controller (`src/controller.py`)
* `u = −Kp (r − r_target) − Kd (v − v_target)`; gains support scalar / per-axis /
  matrix form. Critically damped tuning helper: `Kp = ωₙ²`, `Kd = 2ωₙ`. Optional
  thrust-acceleration saturation.
* Baseline run (`cw_rendezvous_metrics.csv`): from `[200, −300, 80] m`, with
  `ωₙ = 6n` and a 0.05 m/s² limit — final position error 1.5e-7 m, convergence
  ≈ 1591 s, peak acceleration 0.017 m/s², Δv 2.33 m/s.

### Gain-tuning study (`gain_sweep.csv`)
Sweeps Kp × Kd, running the rendezvous for each and recording final errors,
convergence time, peak control acceleration and Δv. Representative outcomes:

| | Kp | Kd | convergence | Δv |
| --- | --- | --- | --- | --- |
| fastest converging | 2.0e-4 | 3.0e-2 | ≈ 703 s | 3.9 m/s (peak 0.074) |
| cheapest converging | 4.6e-5 | 3.0e-2 | ≈ 3939 s | 1.82 m/s |

Low-Kp combinations are sluggish and several do not reach the 1 m tolerance within
the one-orbit horizon (blank convergence entries — reported, not hidden). The
sweep exposes the intended speed-vs-fuel trade-off: the fastest gains cost roughly
twice the Δv of the most fuel-efficient converging gains.

### CW vs high-fidelity validation (`cw_vs_high_fidelity_*.csv`)
Free-drift divergence of CW from the high-fidelity model (1 km along-track offset,
one orbit), by perturbation toggle:

| force model | max CW-vs-truth position error |
| --- | --- |
| two-body | 2.78 m |
| two-body + J2 | 5.81 m |
| two-body + drag | 2.78 m (+≈0.004 m) |
| two-body + J2 + drag | 5.82 m |

Findings:
* **J2 roughly doubles** the CW modelling error (2.78 → 5.81 m).
* With **identical ballistic coefficients**, drag is common-mode and nearly
  cancels in the relative frame (adds only ~0.004 m). This is partly *by
  construction* (equal target/deputy area and mass); a real differential-drag
  effect would require differing `Cd·A/m`. So for close formations of similar
  vehicles, **J2 is the dominant differential perturbation**.
* Under closed-loop control the same CW-vs-high-fidelity divergence is actively
  suppressed to ~0.017 m.

### Tests
`tests/` (stdlib `unittest`; also runnable under pytest) — **32 tests, all
passing**: orbit 6, perturbations 6, validation 6, controller 5, cw 5, lvlh 4.
Coverage: two-body conservation & period, 3D element round-trips, J2 (h_z
conservation) and drag (orbit decay), LVLH round-trips and the co-orbital
equilibrium, CW-vs-STM agreement, cross-track oscillator, PD convergence &
saturation, and the validation utilities (metrics, gain sweep,
CW-vs-high-fidelity).

### Known limitations / next steps
* CW assumes a circular reference; accuracy degrades with separation and
  perturbations. Validation is demonstrated over ≈ 1 orbit at separations ≤ 1 km.
* Single-scale-height exponential atmosphere (not NRLMSISE-class); differential
  drag not yet exercised (identical vehicles).
* Continuous-thrust PD with saturation only — no impulsive / LQR / MPC guidance.
* Perfect, noise-free navigation assumed (no estimator or sensor model).
* Fixed-step RK4 (no adaptive stepping); no collision / keep-out handling.
* Next: impulsive/optimal guidance, differential drag, a J2-aware relative model,
  a navigation/estimation layer with Monte-Carlo dispersions, eccentric references,
  and (later) an interactive front end.

---

## Archived: 2D prototype (superseded — see `old_project_backup/`)

The first iteration was a **planar (2D)** prototype with state `[x, y, vx, vy]`.
It has been **superseded** by the 3D engine above and is retained only for
reference under `old_project_backup/`. Its details differ from the current engine
and should not be taken as describing it. In brief, the 2D prototype:

* used three integrators (explicit Euler, RK4, and velocity-Verlet) and compared
  their energy conservation, whereas the current engine standardises on RK4;
* validated CW against a non-linear truth model propagated with **SciPy's adaptive
  RK45** (rtol/atol 1e-12) — the current engine instead uses its own fixed-step
  RK4 for both models and does not depend on SciPy;
* included a 2D obstacle / keep-out-zone avoidance demo (PD + artificial-potential-
  field repulsion), which is **not** part of the current engine;
* shipped 20 tests (vs 32 now) and a different set of experiment scripts and
  output files.

Nothing in the active project imports from `old_project_backup/`.
