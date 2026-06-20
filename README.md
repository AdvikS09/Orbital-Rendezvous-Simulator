# Orbital Stability Platform — Phase 1: Orbital Rendezvous

A small, modular Python platform for 2D orbital dynamics and spacecraft
rendezvous. Phase 1 covers two-body propagation with multiple integrators,
Clohessy–Wiltshire (Hill) relative motion, a PD rendezvous controller,
verification of the linear CW model against a higher-fidelity non-linear model,
and a simple keep-out-zone avoidance scaffold.

All quantities are **SI** (metres, seconds, kilograms, Newtons). The only
third-party dependencies are **numpy, scipy, matplotlib, pandas**. No GUI, web
app, or database.

## State-vector convention

Everything uses the same planar layout:

```
state = [x, y, vx, vy]          position = state[:2], velocity = state[2:]
```

* **Inertial two-body frame** — `x, y` are Earth-centred inertial coordinates.
* **LVLH / Hill frame** (relative motion) — `x` is radial (R-bar, +outward),
  `y` is along-track (V-bar, +velocity direction), centred on a target on a
  circular reference orbit.

## Module layout (`src/`)

| Module | Contents |
| --- | --- |
| `constants.py` | SI constants (`MU_EARTH`, `R_EARTH`, …) and default 400 km LEO geometry |
| `physics.py` | 2D two-body dynamics, circular-orbit setup, specific energy / angular momentum, period & mean motion |
| `orbital_elements.py` | Cartesian ↔ planar classical elements (a, e, ω, ν) |
| `integrators.py` | Euler, RK4, and velocity-Verlet (leapfrog) steppers + `integrate` driver |
| `cw.py` | Clohessy–Wiltshire dynamics, closed-form state-transition matrix, exact LVLH frame transforms |
| `controllers.py` | PD rendezvous controller + artificial-potential-field keep-out avoidance |
| `simulation.py` | High-level drivers: `propagate_two_body`, `simulate_cw`, `propagate_nonlinear_relative` (truth model) |
| `analysis.py` | Conservation tracking, period measurement, error metrics, validation tables (pandas) |
| `plotting.py` | Headless (Agg) matplotlib helpers |

## Setup

The repository ships with a `.venv` that already has the dependencies. To
recreate it from scratch:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running

Each script is self-contained and writes figures to `outputs/figures/` and CSV
tables to `outputs/tables/` (git-ignored, created on demand). On Windows with the
bundled venv:

```powershell
.venv\Scripts\python.exe scripts\run_orbit_validation.py        # two-body + integrator comparison
.venv\Scripts\python.exe scripts\run_cw_rendezvous.py           # natural CW motion + STM validation
.venv\Scripts\python.exe scripts\run_pd_rendezvous.py           # closed-loop PD rendezvous
.venv\Scripts\python.exe scripts\run_cw_vs_nonlinear.py         # CW vs non-linear truth model
.venv\Scripts\python.exe scripts\run_obstacle_avoidance_demo.py # keep-out-zone avoidance
```

Run the tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## What each script produces

| Script | Figures | Table |
| --- | --- | --- |
| `run_orbit_validation.py` | `orbit_trajectory.png`, `energy_drift.png` | `integrator_validation.csv` |
| `run_cw_rendezvous.py` | `cw_relative_trajectory.png` | `cw_stm_validation.csv` |
| `run_pd_rendezvous.py` | `pd_relative_trajectory.png`, `pd_position_error.png`, `pd_velocity_error.png`, `pd_control_effort.png` | `pd_rendezvous_metrics.csv` |
| `run_cw_vs_nonlinear.py` | `cw_vs_nonlinear_error.png`, `cw_vs_nonlinear_trajectory.png` | `cw_vs_nonlinear_error.csv` |
| `run_obstacle_avoidance_demo.py` | `obstacle_avoidance_trajectory.png` | `obstacle_avoidance_metrics.csv` |

## Representative results (400 km circular LEO)

* **Analytic period:** 5553.62 s (≈ 92.6 min).
* **Integrator conservation** (2 orbits): Euler drifts ~3.7 % in energy; RK4 is
  at machine precision (~6e-15); velocity-Verlet is symplectic with bounded
  energy error (~1e-12) — see `integrator_validation.csv`.
* **CW model:** numerical RK4 matches the closed-form CW state-transition matrix
  to ~1e-10 m. A 100 m radial offset produces ≈ −3771 m of along-track secular
  drift per orbit (the textbook `−6π·x₀`).
* **PD rendezvous:** drives a 291 m initial offset to < 1 µm, settling in
  ≈ 1541 s with a total Δv of ≈ 1.85 m/s.
* **CW vs non-linear:** linearisation error grows from ~1 % of the separation at
  100 m to >100 % at 10 km over one orbit — quantifying where CW stops being
  trustworthy.
* **Obstacle avoidance:** a plain PD path penetrates a keep-out zone by 28.8 m,
  whereas the PD + artificial-potential-field controller keeps +8.5 m clearance
  and still reaches the target (Δv 1.83 → 4.13 m/s for the detour).

## Roadmap (beyond Phase 1)

* Higher-fidelity dynamics: J2 oblateness, drag; 3D (out-of-plane) motion.
* Optimal / impulsive rendezvous (two-impulse CW targeting, LQR).
* APF local-minimum handling and multi-obstacle corridors.
