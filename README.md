# Orbital Rendezvous Simulator

A reproducible Python testbed for **spacecraft rendezvous guidance**, built on
the **Clohessy–Wiltshire (Hill) relative-motion equations** with a **PD
controller**, and compared against a **higher-fidelity non-linear orbital model**
(two-body + J2 oblateness + atmospheric drag) — plus a browser-based platform
that makes it usable by students, teachers, engineers and researchers without
installing anything.

The engine is fully 3D, SI units throughout, and depends only on `numpy`,
`matplotlib` and `pandas`. Numerical integration is a hand-rolled fixed-step RK4
with no random numbers anywhere, so every result is deterministic and
reproducible.

> **Status.** Open-source research and educational software. It is **not**
> NASA-approved, NASA-validated, flight-certified or operational mission
> software, and must not be used for operational flight decisions. It has **not**
> been benchmarked against an external tool such as NASA GMAT or Ansys/AGI STK —
> the repository provides the *workflow* to carry out such a comparison, but no
> benchmark result ships with it. See [KNOWN_GAPS.md](KNOWN_GAPS.md).

---

## Contents

1. [The research question](#1-the-research-question)
2. [Two ways to use this](#2-two-ways-to-use-this)
3. [Installation](#3-installation)
4. [Running the platform locally](#4-running-the-platform-locally)
5. [Running the tests](#5-running-the-tests)
6. [Running the experiments](#6-running-the-experiments)
7. [Repository structure](#7-repository-structure)
8. [Method and conventions](#8-method-and-conventions)
9. [Representative results](#9-representative-results)
10. [Limitations and assumptions](#10-limitations-and-assumptions)
11. [Deployment](#11-deployment)
12. [Documentation](#12-documentation)
13. [References and licence](#13-references-and-licence)

---

## 1. The research question

Autonomous proximity operations — rendezvous, docking, inspection, on-orbit
servicing, debris removal — need guidance laws **simple enough to run on board**
yet **accurate enough to be safe**.

The Clohessy–Wiltshire equations are the workhorse model for close-proximity
relative motion about a circular orbit. They are *linear* and have a
*closed-form solution*, which makes them cheap and analytically tractable. But
they assume a perfectly circular reference orbit and neglect orbital
non-linearity, Earth oblateness (J2), and atmospheric drag.

In one sentence: *CW + PD gives you cheap, analytic rendezvous guidance — this
project measures where that approximation holds and where it breaks.*

Concretely, the engine lets you implement CW relative dynamics and a PD
rendezvous controller, characterise the speed-versus-fuel trade-off, and
**quantify** the linear model's divergence from a non-linear two-body + J2 +
drag simulator as a function of separation, direction and perturbation.

## 2. Two ways to use this

### The pilot platform — `pilot_app.py`

A browser-based interface for education and research. Guided activities for
learners, full parameter control for professionals, anonymous usage analytics
for running a pilot, and an administrator dashboard. No Python knowledge needed
by participants.

```bash
streamlit run pilot_app.py
```

### The engine — `src/`

The physics, importable and scriptable, with no platform code involved:

```python
from src import scenarios as sc

result = sc.run_cw_rendezvous([200, -300, 80], [0, 0, 0])
print(result["metrics"])
```

`src/scenarios.py` is a frontend-agnostic API returning plain dicts and
DataFrames. It contains no plotting, no Streamlit and no file I/O.

The original research MVP, `app.py`, is also still present and still runs
(`streamlit run app.py`).

## 3. Installation

**Prerequisite — Python 3.11+** (developed and tested on 3.14.6). On Windows
tick *"Add python.exe to PATH"* during install, and use `python` or the `py`
launcher — `python3` is not available there by default.

```bash
git clone <repo-url>
cd "Orbital Stability Platform"
```

Create the virtual environment (`.venv/` is git-ignored, so a fresh clone does
not have one):

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

For a byte-for-byte reproducible environment use the full pinned closure
instead:

```bash
python -m pip install -r requirements-lock.txt
```

The results in this README were produced with **Python 3.14.6, numpy 2.4.6,
pandas 3.0.3, matplotlib 3.11.0**.

## 4. Running the platform locally

```bash
streamlit run pilot_app.py
```

Opens at <http://localhost:8501>. With no configuration it uses a local SQLite
database at `pilot_data/pilot.db` and flags every record as development data, so
you cannot accidentally contaminate real pilot figures while trying things out.

To see the administrator dashboard locally:

```bash
python scripts/hash_password.py          # prints PILOT_ADMIN_PASSWORD_HASH=...
```

Set `PILOT_ADMIN_USERNAME` and that hash in your environment, restart, then open
<http://localhost:8501/?view=admin>.

Copy `.env.example` to `.env` for the full set of options — every one has a
working default, and the file explains what each is for.

## 5. Running the tests

```bash
python -m pytest -q
```

**251 tests.** The engine's own tests also run under the standard library with
nothing extra installed:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

| Area | File | Tests |
| --- | --- | --- |
| Two-body dynamics, RK4, elements | `test_orbit.py` | 6 |
| J2 and drag | `test_perturbations.py` | 6 |
| LVLH transforms | `test_lvlh.py` | 4 |
| Clohessy–Wiltshire | `test_cw.py` | 5 |
| PD controller | `test_controller.py` | 5 |
| Control implementation | `test_control_implementation.py` | 6 |
| Validation utilities | `test_validation.py` | 6 |
| External-ephemeris plumbing | `test_external_validation.py` | 8 |
| Collision detection | `test_collision.py` | 7 |
| **Numerical regression baseline** | `test_regression_baseline.py` | 11 |
| **Integrator convergence** | `test_integrator_convergence.py` | 9 |
| Platform database | `test_platform_database.py` | 11 |
| Analytics and privacy | `test_platform_analytics.py` | 30 |
| Input validation | `test_platform_validation.py` | 31 |
| Runner, rate limiting, auth | `test_platform_runner.py` | 38 |
| Educational content | `test_platform_content.py` | 41 |
| End-to-end user journey | `test_ui_flow.py` | 27 |

Two of these deserve mention. **`test_regression_baseline.py`** pins the
published numerical outputs at `rtol=1e-9` against a captured baseline — if it
fails, a result the research depends on has moved, and the right response is to
find out why rather than to regenerate the baseline.
**`test_platform_content.py`** asserts that every number quoted to a learner in
the guided activities is actually what the engine produces.

## 6. Running the experiments

Nine standalone scripts write figures to `outputs/figures/` and tables to
`outputs/data/` (both git-ignored, created on first run). Each adds the project
root to `sys.path`, so the working directory does not matter.

```bash
python experiments/run_cw_rendezvous.py
python experiments/run_gain_sweep.py
python experiments/run_cw_vs_high_fidelity.py
```

| Experiment | What it does |
| --- | --- |
| `run_cw_rendezvous.py` | Closed-loop PD rendezvous from a configurable initial relative state |
| `run_gain_sweep.py` | Kp × Kd tuning study |
| `run_cw_vs_high_fidelity.py` | Two-body validation, CW-vs-reference comparison, perturbation toggles |
| `run_separation_sweep.py` | Direction-resolved divergence, 50 m to 50 km — the central experiment |
| `run_convergence_study.py` | Time-step convergence of the reported metrics |
| `run_robustness_gain_horizon.py` | Do the conclusions survive different bandwidths and horizons? |
| `run_radial_energy_matching.py` | Energy-matched vs zero-velocity radial offsets |
| `run_energy_matching_audit.py` | Analytical audit of the bounded-motion condition |
| `run_terminal_assessment.py` | Terminal position *and* velocity assessment |

## 7. Repository structure

```
pilot_app.py        Pilot platform entry point (Streamlit)
app.py              Original research MVP (still works)

src/                THE VALIDATED ENGINE - not modified by the platform
  orbit.py          two-body dynamics, RK4, orbital elements, energy/period
  perturbations.py  J2 + drag, ForceModel toggles
  lvlh.py           inertial <-> LVLH transforms
  cw.py             CW dynamics, closed-form state-transition matrix
  controller.py     PD rendezvous controller (Kp, Kd, saturation)
  validation.py     conservation checks, error metrics, gain sweep, comparison
  external_validation.py  export/import + compare vs an external tool
  collision.py      keep-out-zone / debris detection (detection only)
  scenarios.py      frontend-agnostic scenario API  <- the platform boundary
  plotting.py       headless matplotlib helpers

pilot/              THE PLATFORM - no physics
  config.py         environment-driven settings
  models.py         database schema (anonymous by construction)
  db.py             engine, versioned migrations, health checks
  analytics.py      anonymous identity, events, metric queries
  validation.py     input bounds and warnings
  ratelimit.py      per-session quotas
  auth.py           administrator credentials
  runner.py         the single validate -> guard -> run -> record path
  export.py         result CSVs and reproducible scenario configs
  content/          scenario catalogue, guided activities, assessments
  ui/               Streamlit pages

experiments/        nine standalone research scripts
tests/              251 tests
docs/               architecture, data dictionary, privacy, deployment, guides
scripts/            init_db, hash_password, seed_pilot_codes, purge_expired
old_project_backup/ archived 2D prototype (nothing active imports from it)
```

## 8. Method and conventions

States are 6-dimensional and SI throughout:

```
Inertial (ECI):  [x, y, z, vx, vy, vz]   Earth-Centred Inertial
LVLH (relative): [x, y, z, vx, vy, vz]   x = radial (R-bar)
                                         y = along-track (V-bar)
                                         z = cross-track (H-bar)
position = state[:3],  velocity = state[3:]
```

* **Dynamics:** point-mass gravity, optional J2 and exponential-atmosphere drag,
  integrated with fixed-step RK4.
* **Relative motion:** the chaser is expressed in the target's rotating LVLH
  frame; CW propagation uses the linearised equations and their state-transition
  matrix.
* **Guidance:** `u = -Kp (r - r_target) - Kd (v - v_target)` with optional
  thrust-magnitude saturation.
* **What "higher-fidelity" means here:** the reference model differs from CW
  *only in the force model* (full non-linear gravity + J2 + drag). Both use the
  **same RK4 integrator at the same step size**, deliberately, so integration
  error is common to both and cancels — the measured discrepancy therefore
  isolates *modelling* error rather than integration error. The orbit-validation
  check confirms the shared integrator is itself trustworthy at these steps.

## 9. Representative results

Reference orbit: 400 km circular LEO. The CW rendezvous and gain-sweep results
are **inclination-independent** — pure CW depends only on mean motion. Only the
CW-vs-reference results use the i = 51.6° inertial orbit, where J2 and the 3D
geometry matter.

| Quantity | Result |
| --- | --- |
| Orbital period (analytic) | 5553.62 s (≈ 92.6 min) |
| Two-body energy drift (RK4, 2 orbits, 4001 samples) | ≈ 1.1 × 10⁻¹³ (relative) |
| Angular-momentum drift | ≈ 5.5 × 10⁻¹⁴ (relative) |
| Measured-vs-analytic period error | ≈ 2.3 × 10⁻¹² (relative) |
| CW numeric vs closed-form STM¹ | ≈ 3.0 × 10⁻⁸ m over one orbit |
| PD rendezvous from [200, −300, 80] m | 369 m → 1.5 × 10⁻⁷ m, converges ≈ 1595 s, Δv ≈ 2.33 m/s, peak 0.017 m/s² |
| CW vs reference, 1 km along-track free drift, 1 orbit | two-body 2.78 m; **+J2 5.81 m**; +drag 2.78 m; J2+drag 5.82 m |

¹ For a representative initial relative state; this absolute figure scales with
the state magnitude. It is integrator-vs-closed-form agreement, not a physical
error.

Every value above is asserted by `tests/test_regression_baseline.py`.

**Two headline findings:**

* **J2 roughly doubles** the CW modelling error (2.78 m → 5.81 m).
* With **identical ballistic coefficients**, drag is common-mode and nearly
  cancels in the relative frame — it adds only ≈ 0.004 m. This is partly *by
  construction*: the toggle study uses equal target and chaser area and mass. A
  real differential-drag effect would require differing `Cd·A/m`, which the
  engine does not yet exercise. So for close formations of similar vehicles,
  **J2 is the dominant differential perturbation**.

A third result worth knowing: the same 1 km offset placed **radially** instead
of along-track gives roughly forty times the error, because a radial offset
excites a large along-track drift. Model validity is not one number — it depends
on what you ask of the model.

## 10. Limitations and assumptions

* **Linearised reference.** CW assumes a circular reference orbit; accuracy
  degrades with separation, and fastest for a radial offset.
* **Validation scope.** CW-vs-reference is characterised out to 50 km; the
  published agreement figures are for separations up to ≈ 1 km over one orbit.
* **No external benchmark.** The reference model is more faithful than CW, not
  ground truth. It has not been compared against GMAT, STK or anything else.
* **Atmosphere.** Single-scale-height exponential model (ρ₀ ≈ 3.9 × 10⁻¹² kg/m³
  at 400 km, H = 60 km, moderate solar activity) — not NRLMSISE-class. Real
  density varies by an order of magnitude with solar activity, which is not
  modelled.
* **Differential drag.** Target and chaser share `Cd·A/m`, so drag is
  common-mode by construction.
* **Guidance.** Continuous-thrust PD with magnitude saturation only — no
  impulsive, LQR or MPC guidance, no thrust-direction or duty-cycle limits.
* **Perfect navigation.** Full, noise-free state knowledge. No sensor model,
  estimator or measurement noise, so nothing here says anything about robustness
  to navigation error.
* **Integration.** Fixed-step RK4, no adaptive stepping. The reference shares
  the integrator and step by design.
* **Collision handling is detection only.** No avoidance, path planning or
  manoeuvre generation, and no position uncertainty — it is not a
  conjunction-assessment tool.

`KNOWN_GAPS.md` records what is missing, provisional, or needs scientific
judgement.

## 11. Deployment

The platform is designed for **Streamlit Community Cloud** (free public URL)
with managed PostgreSQL, or for **Docker** on any host. Full instructions,
including why the SQLite fallback must not be used for a hosted pilot, are in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

```bash
docker build -t orbital-pilot .
docker run -p 8501:8501 --env-file .env orbital-pilot
```

The image runs the numerical regression tests during the build, so an image
whose engine outputs have moved cannot be produced.

## 12. Documentation

| Document | For |
| --- | --- |
| [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md) | What existed before the platform work, verified by running it |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | What was added, decisions taken, test results |
| [MANUAL_REVIEW_CHECKLIST.md](MANUAL_REVIEW_CHECKLIST.md) | Every journey to test by hand, with expected outcomes |
| [KNOWN_GAPS.md](KNOWN_GAPS.md) | Missing, provisional or unverified — read before a real pilot |
| [CHANGELOG.md](CHANGELOG.md) | Versioned release notes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How it fits together and why |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every stored field and every metric definition |
| [docs/PRIVACY.md](docs/PRIVACY.md) | What is recorded, what cannot be, retention |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Getting it onto a public URL |
| [docs/PILOT_COORDINATOR.md](docs/PILOT_COORDINATOR.md) | Running a pilot with a class or a team |
| [docs/PARTICIPANT.md](docs/PARTICIPANT.md) | Hand-out for participants |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Symptoms, causes, fixes |
| [docs/results_summary.md](docs/results_summary.md) | The physics write-up |
| [research_log.md](research_log.md) | Modelling decisions and their reasoning |

## 13. References and licence

The relative-motion and perturbation models follow standard astrodynamics:

* Clohessy, W. H., & Wiltshire, R. S. (1960). *Terminal Guidance System for
  Satellite Rendezvous.* Journal of the Aerospace Sciences — the CW equations
  and state-transition matrix.
* Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
  orbital-element conversions and the zonal-J2 perturbation.
* Curtis, H. D. *Orbital Mechanics for Engineering Students* — two-body dynamics
  and relative motion.

This engine supersedes an earlier 2D prototype, preserved unmodified under
`old_project_backup/`. Nothing in the active project imports from it.

Licensed under the MIT License — see [`LICENSE`](LICENSE).
