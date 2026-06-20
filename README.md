# Orbital Rendezvous Research Engine

CW-based spacecraft rendezvous with a PD controller, validated against a
higher-fidelity (J2 + atmospheric drag) orbital model. The engine is **3D**, uses
**SI units** throughout, and depends only on **numpy / matplotlib / pandas**
(integration is a hand-rolled RK4 — deterministic and reproducible). No GUI / web
app / database — that comes later.

## State conventions

```
inertial (ECI) state :  [x, y, z, vx, vy, vz]
LVLH relative state  :  [x, y, z, vx, vy, vz]
                         x = radial (R-bar), y = along-track (V-bar), z = cross-track (H-bar)
position = state[:3], velocity = state[3:]
```

## Project structure

```
src/
  orbit.py          two-body dynamics, RK4 propagator, orbital elements, energy/period
  perturbations.py  J2 + atmospheric drag, ForceModel toggles (two-body / J2 / drag / J2+drag)
  lvlh.py           inertial <-> LVLH transforms, relative position/velocity
  cw.py             Clohessy-Wiltshire 3D dynamics, state-transition matrix, propagation
  controller.py     configurable PD rendezvous controller (Kp, Kd, saturation)
  validation.py     conservation checks, error metrics, gain sweep, CW-vs-high-fidelity
  plotting.py       headless matplotlib helpers
experiments/
  run_cw_rendezvous.py        closed-loop PD rendezvous
  run_gain_sweep.py           Kp x Kd tuning study
  run_cw_vs_high_fidelity.py  two-body validation + CW vs high-fidelity + toggle study
outputs/
  figures/          generated PNG plots
  data/             generated CSV tables
tests/              unittest suite (no extra dependencies)
requirements.txt
README.md
old_project_backup/ snapshot of the previous (2D) src/ and scripts/
```

## Setup

The repo ships with a `.venv` that already has the dependencies. To recreate it:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the experiments

Each experiment is self-contained and writes to `outputs/figures/` and
`outputs/data/` (created on demand). On Windows with the bundled venv:

```powershell
.venv\Scripts\python.exe experiments\run_cw_rendezvous.py
.venv\Scripts\python.exe experiments\run_gain_sweep.py
.venv\Scripts\python.exe experiments\run_cw_vs_high_fidelity.py
```

Run the tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Features

1. **Core orbital mechanics** — two-body dynamics, RK4 integration, 3D classical
   orbital-element conversion, and energy / angular-momentum / period validation.
2. **Perturbations** — J2 oblateness and exponential-atmosphere drag, selectable
   via `ForceModel` toggles: two-body only, J2 only, drag only, J2 + drag.
3. **LVLH / relative motion** — inertial↔LVLH conversion (with the rotating-frame
   transport term) and relative position / velocity helpers.
4. **CW module** — the 3D Clohessy-Wiltshire equations (in-plane Coriolis coupling
   + decoupled cross-track oscillator), a closed-form state-transition matrix,
   propagation, and a configurable initial relative state. Each term is commented.
5. **PD controller** — configurable `Kp` / `Kd` (scalar, per-axis, or matrix),
   optional thrust saturation, with position/velocity-error and control tracking.
6. **Gain-tuning study** — sweeps Kp × Kd and reports final position error, final
   velocity error, convergence time, max control acceleration and delta-v.
7. **CW vs high-fidelity validation** — runs the same scenario in CW and in the
   full non-linear model and compares position/velocity error growth over time.
8. **Plots** — LVLH relative trajectory, x/y/z position vs time, relative velocity
   vs time, control acceleration vs time, gain comparison, CW-vs-HF error.

## Representative results (400 km circular LEO, i = 51.6°)

* **Two-body validation (RK4, 2 orbits):** specific-energy drift ~1.4e-14,
  angular-momentum drift ~6.6e-15, measured period matches the analytic
  5553.62 s (≈ 92.6 min) to ~1e-13 relative error.
* **CW model:** numerical RK4 matches the closed-form 3D state-transition matrix
  to ~4e-10 m over one orbit; the cross-track axis is a clean oscillator at the
  orbital period.
* **PD rendezvous** (start [200, −300, 80] m, ωₙ = 6n, 0.05 m/s² limit): drives a
  369 m error to ~1.5e-7 m, converging in ≈ 1591 s with Δv ≈ 2.33 m/s.
* **Gain sweep:** low Kp is sluggish / non-converging; the fastest converging
  combination (Kp = 2e-4, Kd = 3e-2) settles in ≈ 702 s at Δv ≈ 3.9 m/s,
  illustrating the speed-vs-fuel trade-off.
* **CW vs high-fidelity** (1 km along-track free drift, 1 orbit): two-body
  divergence ≈ 2.78 m; adding **J2** roughly doubles it to ≈ 5.81 m; adding
  **drag** alone leaves it ≈ 2.78 m — common-mode drag largely cancels in
  relative motion for identical vehicles, so **J2 is the dominant differential
  perturbation**. Under closed-loop control the divergence is actively suppressed
  to ~0.02 m.

## Notes, assumptions, and limitations

* **Atmosphere:** a single-scale-height exponential model (ρ₀ = 3.9e-12 kg/m³ at
  400 km, H = 60 km) representing moderate solar activity — not a NRLMSISE-class
  model. Default drag vehicle: Cd = 2.2, A = 1 m², m = 100 kg.
* **Drag in relative motion:** with identical target/deputy ballistic
  coefficients, common-mode drag nearly cancels; differential drag (different
  Cd·A/m) would be needed to see a large drag effect on the relative trajectory.
* **CW reference:** assumes a circular reference orbit; the linearisation degrades
  as separation grows and as perturbations accumulate (quantified in experiment 3).
* **Controller:** continuous-thrust PD with magnitude saturation — no impulsive or
  optimal (e.g. LQR / two-impulse) targeting yet.
* **Integration:** fixed-step RK4 everywhere for reproducibility (no adaptive
  stepping). The high-fidelity comparison propagates the target (passive) and
  deputy (controlled) as a combined non-linear system.
* The previous 2D implementation is preserved under `old_project_backup/`.
```
