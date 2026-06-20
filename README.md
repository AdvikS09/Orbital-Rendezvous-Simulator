# Orbital Rendezvous Research Engine

A clean, reproducible Python testbed for **spacecraft rendezvous guidance** built
on the **Clohessy–Wiltshire (Hill) relative-motion equations** with a **PD
controller**, and **validated against a higher-fidelity non-linear orbital
model** (two-body + J2 oblateness + atmospheric drag).

The engine is fully 3D, uses **SI units** throughout, and depends only on
`numpy`, `matplotlib`, and `pandas`. Numerical integration is a hand-rolled
fixed-step RK4 so every result is deterministic and reproducible. There is no
GUI, web app, or database — the focus is correct physics and clear, citable
outputs.

---

## 1. Project goal

Implement and study **CW-based rendezvous guidance**: drive a *chaser*
spacecraft to a *target* on a circular Low-Earth orbit using a Proportional–
Derivative (PD) controller acting on the linearised relative-motion dynamics, and
then **quantify how trustworthy that linear guidance is** by comparing it against
a higher-fidelity non-linear orbital model.

In one sentence: *CW + PD gives you cheap, analytic rendezvous guidance — this
project measures where that approximation holds and where it breaks.*

## 2. Research motivation

Autonomous proximity operations — rendezvous, docking, inspection, on-orbit
servicing, and debris removal — require guidance laws that are **simple enough to
run on board** yet **accurate enough to be safe**.

The Clohessy–Wiltshire equations are the workhorse model for close-proximity
relative motion about a circular orbit: they are *linear* and have a *closed-form
solution* (a state-transition matrix), which makes them cheap and analytically
tractable. But they assume a perfectly circular reference orbit and neglect:

* orbital **non-linearity** (they are a first-order expansion),
* Earth **oblateness (J2)** — the dominant *secular gravitational* perturbation
  in LEO, and
* **atmospheric drag**.

Real missions must know *when* the CW assumption is good enough and *when* a
richer model or controller is needed. As a working adequacy criterion, this study
treats CW guidance as acceptable while the relative position error stays within a
few percent of the inter-spacecraft separation over the proximity horizon; the
experiments measure where that holds. Concretely, the engine lets you:

1. implement CW relative dynamics and a PD rendezvous controller,
2. tune the controller and characterise the speed-vs-fuel trade-off, and
3. **validate** the linear model by measuring its divergence from a non-linear
   two-body + J2 + drag simulator as a function of separation and perturbation.

## 3. Method and conventions

State vectors are 6-dimensional and SI throughout:

```
Inertial (ECI):  [x, y, z, vx, vy, vz]      Earth-Centred Inertial
LVLH (relative): [x, y, z, vx, vy, vz]      x = radial (R-bar),
                                            y = along-track (V-bar),
                                            z = cross-track (H-bar)
position = state[:3],  velocity = state[3:]
```

* **Dynamics model:** point-mass gravity, optional J2 and exponential-atmosphere
  drag, integrated with fixed-step RK4.
* **Relative motion:** the chaser is expressed in the target's rotating LVLH
  frame; CW propagation uses the linearised equations and their state-transition
  matrix.
* **Guidance:** `u = -Kp (r - r_target) - Kd (v - v_target)` with optional thrust
  (acceleration-magnitude) saturation.
* **Validation — what "higher-fidelity" means here:** the high-fidelity model
  differs from CW *only in the force model* (full non-linear gravity + J2 + drag);
  both the CW and high-fidelity trajectories use the **same RK4 integrator at the
  same step size**. This is deliberate: it makes integration error common to both,
  so the measured discrepancy isolates *modelling* error rather than integration
  error. The orbit-validation check (energy drift ≈ 1.4 × 10⁻¹⁴ over two orbits,
  Section 10) confirms the shared integrator is itself trustworthy at this step.

## 4. Repository structure

```
app.py              Streamlit MVP interface (thin UI; no physics — calls src/scenarios.py)
src/
  orbit.py          two-body dynamics, RK4 propagator, orbital elements, energy/period
  perturbations.py  J2 + atmospheric drag, ForceModel toggles (two-body/J2/drag/J2+drag)
  lvlh.py           inertial <-> LVLH transforms, relative position/velocity
  cw.py             Clohessy-Wiltshire 3D dynamics, state-transition matrix, propagation
  controller.py     configurable PD rendezvous controller (Kp, Kd, saturation)
  validation.py     conservation checks, error metrics, gain sweep, CW-vs-high-fidelity
  scenarios.py      frontend-agnostic scenario API (plain-data results for any UI)
  plotting.py       headless matplotlib helpers
experiments/
  run_cw_rendezvous.py        closed-loop PD rendezvous (trajectory + time histories)
  run_gain_sweep.py           Kp x Kd tuning study
  run_cw_vs_high_fidelity.py  two-body validation + CW vs high-fidelity + perturbation toggles
outputs/            generated on first experiment run (git-ignored, not shipped)
  figures/          generated PNG plots
  data/             generated CSV tables
tests/              unittest / pytest suite (32 tests)
old_project_backup/ archived snapshot of the earlier 2D prototype
requirements.txt
pytest.ini
README.md
```

## 5. Module reference

| Module | Responsibility |
| --- | --- |
| `orbit` | Constants (`MU_EARTH`, `R_EARTH`, `J2_EARTH`, …), two-body acceleration, RK4 step and propagator, circular-orbit setup, energy / angular-momentum / period, 3D classical orbital-element conversions. |
| `perturbations` | `j2_acceleration`, exponential-atmosphere `drag_acceleration`, and a `ForceModel` dataclass with `two_body()`, `j2_only()`, `drag_only()`, `j2_drag()` toggles exposing `acceleration()` / `derivatives()`. |
| `lvlh` | Rotation matrix and angular velocity of the LVLH frame; `inertial_to_lvlh` / `lvlh_to_inertial` (with the rotating-frame transport term); `relative_position` / `relative_velocity`. |
| `cw` | CW mean motion, state-space matrices `(A, B)`, `cw_derivatives`, closed-form `cw_state_transition`, analytic and RK4 propagation with a configurable initial relative state. |
| `controller` | `PDController` (scalar / per-axis / matrix gains, optional saturation), `critically_damped` factory, gain helpers. |
| `validation` | Energy/period verification, position/velocity/control error series, delta-v, convergence time, `gain_sweep`, high-fidelity relative propagation, and `compare_cw_high_fidelity`. |
| `plotting` | Headless (Agg) helpers: LVLH 3D trajectory, position/velocity components, control acceleration, error growth, gain-sweep heatmaps. |

## 6. Installation

**Prerequisite — Python 3.11+** (tested on 3.14). Install from
[python.org](https://www.python.org/downloads/); on Windows tick *“Add python.exe
to PATH”* during install. Verify with `python --version` (use `python`, or the
`py` launcher, on Windows — `python3` is not available there by default).

**Clone and enter the project**, then run all commands from the project root:

```powershell
git clone <repo-url>
cd "Orbital Stability Platform"
```

**Create the virtual environment and install dependencies (required first step).**
The `.venv/` is git-ignored, so it does *not* exist in a fresh clone — you must
create it:

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Once the environment is **activated**, every command below uses a plain `python`.
(If you prefer not to activate it, replace `python` with `.venv\Scripts\python.exe`
on Windows or `.venv/bin/python` on macOS/Linux.)

Runtime dependencies (`requirements.txt`): `numpy`, `matplotlib`, `pandas`.
Results in this README were generated with **Python 3.14.6, numpy 2.4.6,
matplotlib 3.11.0, pandas 3.0.3**.

## 7. Running the tests

The tests need the runtime dependencies installed (above); the *test runner* is
flexible. The standard-library `unittest` needs no extra package:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

`pytest` is optional and **not** in `requirements.txt` — install it first
(`python -m pip install pytest`), then simply:

```powershell
python -m pytest
```

Either reports **32 passing tests** (orbit 6, perturbations 6, validation 6,
controller 5, cw 5, lvlh 4), covering two-body conservation and period, 3D
orbital-element round-trips, J2 (angular-momentum conservation) and drag (orbit
decay), LVLH round-trips, CW-vs-state-transition-matrix agreement, PD convergence
and saturation, and the validation utilities. (`pytest.ini` scopes collection to
`tests/` so bare `pytest` does not clash with the archived
`old_project_backup/tests` package.)

## 8. Running the experiments

Each experiment adds the project root to `sys.path` automatically, so the working
directory does not matter; run them with your venv's `python`. They write plots to
`outputs/figures/` and tables to `outputs/data/` (both created on first run).

```powershell
python experiments\run_cw_rendezvous.py
python experiments\run_gain_sweep.py
python experiments\run_cw_vs_high_fidelity.py
```

| Experiment | What it does | Outputs |
| --- | --- | --- |
| `run_cw_rendezvous.py` | Closed-loop PD rendezvous from a configurable initial relative state. | `cw_rendezvous_lvlh_trajectory.png`, `cw_rendezvous_position.png`, `cw_rendezvous_velocity.png`, `cw_rendezvous_control.png`; `cw_rendezvous_timeseries.csv`, `cw_rendezvous_metrics.csv` |
| `run_gain_sweep.py` | Sweeps a grid of Kp and Kd values, running the rendezvous for each. | `gain_sweep_comparison.png`; `gain_sweep.csv` (also printed to the terminal) |
| `run_cw_vs_high_fidelity.py` | Two-body orbit validation, controlled CW-vs-high-fidelity comparison, and a perturbation-toggle free-drift study. | `cw_vs_high_fidelity_error.png`, `cw_vs_high_fidelity_toggles.png`; `orbit_validation.csv`, `cw_vs_high_fidelity_timeseries.csv`, `cw_vs_high_fidelity_toggles.csv` |

## Interactive app (Streamlit MVP)

An interactive Streamlit interface sits on top of the engine for exploration and
demos. It is a **thin frontend**: it contains no physics — all computation goes
through `src/scenarios.py`, which returns plain dicts / DataFrames — so the
backend can be reused if the UI is later replaced by a polished web frontend.

Streamlit and Plotly are pinned in `requirements.txt`. Install and launch:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

It opens in your browser (default http://localhost:8501) with two modes:

* **Learning Mode** — seven guided presets (basic circular orbit, J2 effect, drag
  effect, CW relative motion, CW + PD rendezvous, gain tuning, CW vs
  high-fidelity), each with a plain-English explanation, editable inputs, Run /
  Reset buttons, interactive Plotly plots, key metrics, and an automatic
  interpretation of the result.
* **Research Mode** — full control of the initial relative state (x/y/z, vx/vy/vz),
  target altitude, inclination, duration, timestep, Kp/Kd, J2/drag toggles, and
  model choice (CW only / high-fidelity only / comparison), with relative-
  trajectory, position, velocity and control-acceleration plots, error metrics
  (final position/velocity error, convergence time, delta-v), and a CSV download.

## 9. Interpreting the outputs

The experiments must be run first (Section 8) — `outputs/` is git-ignored and
generated on demand. Open the PNGs in `outputs/figures/` with any image viewer and
the CSVs in `outputs/data/` with any spreadsheet or text tool.

* **`cw_rendezvous_lvlh_trajectory.png`** — the chaser's path in the target LVLH
  frame (target at the origin). A successful rendezvous curves into the origin.
* **`cw_rendezvous_position.png` / `_velocity.png`** — radial/along-track/
  cross-track components vs time; both should decay smoothly to zero.
* **`cw_rendezvous_control.png`** — commanded acceleration components and
  magnitude vs time; the peak indicates the required thrust authority and the
  area under `|u|` relates to fuel use (delta-v).
* **`cw_rendezvous_metrics.csv`** — one-row summary: initial/final position error,
  final velocity error, convergence time, peak control acceleration, delta-v.
* **`gain_sweep_comparison.png` / `gain_sweep.csv`** — final position error,
  convergence time, and delta-v over the Kp × Kd grid. Reading these together
  exposes the **speed-vs-fuel trade-off** and which gains fail to converge (blank
  convergence-time cells).
* **`cw_vs_high_fidelity_error.png`** — position and velocity error between the CW
  and high-fidelity models over time for the controlled scenario.
* **`cw_vs_high_fidelity_toggles.png` / `_toggles.csv`** — how far CW drifts from
  the truth model under each perturbation toggle (two-body / J2 / drag / J2+drag),
  isolating which effect dominates the modelling error.
* **`orbit_validation.csv`** — energy / angular-momentum drift and measured-vs-
  analytic period, confirming the integrator is trustworthy.

## 10. Representative results

Reference orbit: 400 km circular LEO. *The CW rendezvous and gain-sweep results
are inclination-independent* — pure CW depends only on mean motion. *Only the
CW-vs-high-fidelity results use the i = 51.6° inertial orbit* (where J2 and the 3D
geometry matter).

| Quantity | Result |
| --- | --- |
| Orbital period (analytic) | 5553.62 s (≈ 92.6 min) |
| Two-body energy drift (RK4, 2 orbits) | ≈ 1.4 × 10⁻¹⁴ (relative) |
| Measured-vs-analytic period error | ≈ 1.3 × 10⁻¹³ (relative) |
| CW numeric vs state-transition matrix¹ | ≈ 4 × 10⁻¹⁰ m over one orbit |
| PD rendezvous (start [200, −300, 80] m) | 369 m → 1.5 × 10⁻⁷ m, converges ≈ 1591 s, Δv ≈ 2.33 m/s |
| Gain sweep — fastest converging | Kp = 2×10⁻⁴, Kd = 3×10⁻² → ≈ 703 s, **but** Δv ≈ 3.9 m/s (peak 0.074 m/s²) |
| Gain sweep — cheapest converging | Kp = 4.6×10⁻⁵, Kd = 3×10⁻² → ≈ 3939 s, Δv ≈ 1.82 m/s |
| CW vs high-fidelity, 1 km along-track free drift, 1 orbit | two-body 2.78 m; **+J2 5.81 m**; +drag 2.78 m; J2+drag 5.82 m |

¹ Measured for a representative initial relative state; this absolute figure
scales with the state magnitude (it is the integrator-vs-closed-form agreement,
not a physical error).

Two headline findings:

* **J2 roughly doubles** the CW modelling error (2.78 m → 5.81 m).
* With **identical ballistic coefficients**, drag is common-mode and nearly
  cancels in the relative frame — it adds only ≈ 0.004 m of extra error, confirming
  expected behaviour. This is partly *by construction* (the toggle study uses equal
  target/deputy area and mass); a real differential-drag effect would require
  differing `Cd·A/m`, which the engine does not yet exercise. So for close
  formations of similar vehicles, **J2 is the dominant differential perturbation**.

The gain-sweep rows show the trade-off the study is designed to expose: pushing
for the fastest convergence costs roughly twice the delta-v of the most
fuel-efficient converging gains.

## 11. Current limitations and assumptions

* **Linearised reference:** CW assumes a circular reference orbit; accuracy
  degrades with separation and accumulated perturbations.
* **Validation scope:** the CW-vs-truth comparison is demonstrated over ≈ 1 orbit
  at separations up to ≈ 1 km; CW error grows with separation beyond that range,
  which is not characterised here.
* **Non-converging gains:** some combinations in the gain sweep do not reach the
  1 m tolerance within the simulation horizon (blank `convergence_time` entries);
  these are reported, not hidden.
* **Atmosphere:** a single-scale-height exponential density model (ρ₀ ≈
  3.9 × 10⁻¹² kg/m³ at 400 km, H = 60 km, moderate solar activity) — not a
  NRLMSISE-class model.
* **Differential drag:** with identical target/deputy ballistic coefficients,
  drag is common-mode and mostly cancels in the relative frame; capturing a large
  drag effect would require differing `Cd·A/m`.
* **Controller:** continuous-thrust PD with magnitude saturation only — no
  impulsive, LQR, or MPC guidance, and no thrust-direction or duty-cycle limits.
* **Perfect navigation:** full, noise-free state knowledge is assumed; there is no
  sensor model, estimator, or measurement noise.
* **Integration:** fixed-step RK4 (no adaptive stepping or error control); the
  high-fidelity reference shares this integrator and step (see Section 3).
* **No collision/keep-out handling** in this engine (the earlier 2D prototype had
  an obstacle-avoidance demo; it is archived in `old_project_backup/`).

## 12. Next steps

* Add impulsive and optimal guidance (two-impulse CW targeting, LQR, MPC) for
  comparison against PD.
* Introduce differential drag and a J2-aware relative-motion model.
* Add a navigation/estimation layer (e.g. an EKF) and Monte-Carlo dispersion runs
  to study robustness under noise and uncertainty.
* Extend the validation to eccentric reference orbits, larger separations, and
  longer horizons.
* Build an interactive front end on top of the engine (planned, not yet started).

## 13. References and provenance

The relative-motion and perturbation models follow standard astrodynamics:

* Clohessy, W. H., & Wiltshire, R. S. (1960). *Terminal Guidance System for
  Satellite Rendezvous.* Journal of the Aerospace Sciences — the CW equations and
  state-transition matrix.
* Vallado, D. A. *Fundamentals of Astrodynamics and Applications* — orbital-element
  conversions and the zonal-J2 perturbation.
* Curtis, H. D. *Orbital Mechanics for Engineering Students* — two-body dynamics
  and relative motion.

This research engine supersedes an earlier 2D prototype, preserved unmodified
under `old_project_backup/` (nothing in the active project imports from it).

Licensed under the MIT License — see [`LICENSE`](LICENSE).
