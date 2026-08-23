# Current State Audit

Audit of the repository **as found**, before any pilot-platform work. Everything
below was verified by reading the code and running it — nothing is taken from
documentation on trust. Where the documentation and the code disagree, the code
wins and the discrepancy is recorded.

* Repository: `Orbital Stability Platform` (remote `AdvikS09/Orbital-Rendezvous-Simulator`)
* Branch at audit time: `main` @ `5e0b6a2` "Add working Streamlit MVP interface"
* Audit date: 2026-08-23
* Environment: Windows 11, Python 3.14.6, numpy 2.4.6, pandas 3.0.3,
  matplotlib 3.11.0, plotly 6.8.0, streamlit 1.58.0, pytest 9.1.1

---

## 1. Existing architecture

The project is a **research engine with a thin Streamlit frontend**. It has a
genuinely clean layering, which is the single most important fact for this
project: the platform work can sit on top of it without touching physics.

```
app.py                     Streamlit MVP  ── presentation only, no physics
   │  calls
   ▼
src/scenarios.py           frontend-agnostic scenario API  ── orchestration only
   │  calls
   ▼
src/{orbit, perturbations, lvlh, cw, controller, validation,
     external_validation, collision}.py                    ── the engine
```

### 1.1 Module inventory (verified by reading each file)

| Module | Lines | What it actually contains |
| --- | --- | --- |
| `src/orbit.py` | 242 | Constants (`MU_EARTH`, `R_EARTH`, `J2_EARTH`, `OMEGA_EARTH`), two-body acceleration, `rk4_step` / `propagate`, circular-orbit setup, specific energy, angular momentum, classical orbital-element conversions both ways, `measure_period` by swept-angle interpolation. |
| `src/perturbations.py` | 148 | Zonal-J2 acceleration, exponential-atmosphere density, drag acceleration with Earth co-rotation, and a `ForceModel` dataclass exposing the four toggles (two-body / J2 / drag / J2+drag) plus `acceleration()` and `derivatives()`. |
| `src/lvlh.py` | 103 | LVLH rotation matrix, frame angular velocity, `inertial_to_lvlh` / `lvlh_to_inertial` including the `ω × ρ` transport term, relative position/velocity helpers. |
| `src/cw.py` | 146 | CW mean motion, `(A, B)` state-space matrices, `cw_derivatives`, the closed-form state-transition matrix `Φ(t)`, analytic propagation and RK4 propagation with optional control. |
| `src/controller.py` | 79 | `PDController` with scalar / per-axis / matrix gains, a `critically_damped` factory, and thrust-magnitude saturation. Callable, so it drops straight into the propagators. |
| `src/validation.py` | 221 | Energy / angular-momentum series and relative drift, `verify_orbit`, rendezvous error metrics (position, velocity, control, Δv, convergence time), `high_fidelity_relative` (full non-linear target+deputy propagation expressed in LVLH), `gain_sweep`, `compare_cw_high_fidelity`. |
| `src/external_validation.py` | 176 | CSV/JSON serialisation of an engine ephemeris, flexible-column CSV import with unit handling, and `compare_ephemerides` (interpolates onto the engine grid, reports max/final/mean/RMS). **Data plumbing only — see §4.1.** |
| `src/collision.py` | 94 | `minimum_distance` and `analyze`: keep-out-zone and debris minimum distances plus risk flags. **Detection only, no avoidance.** |
| `src/scenarios.py` | 370 | Nine functions returning plain dicts / DataFrames: `run_orbit`, `compare_orbit_perturbation`, `run_cw_relative`, `run_cw_rendezvous`, `run_gain_sweep`, `run_relative_scenario`, `build_validation_scenario`, `compare_external_ephemeris`, `run_collision_scenario`. Also a `MAX_STEPS = 200_000` clamp. |
| `src/plotting.py` | 172 | Headless matplotlib helpers used by the experiment scripts (not by the app). |
| `app.py` | 724 | Streamlit UI: Plotly figure builders, eight Learning-Mode presets, a Research Mode, and an External Validation mode. Contains no physics. |

### 1.2 Experiments

`experiments/` holds nine standalone scripts that write to `outputs/`. Three are
documented in the README (`run_cw_rendezvous`, `run_gain_sweep`,
`run_cw_vs_high_fidelity`); six more exist and are **not** in the README:
`run_convergence_study`, `run_separation_sweep`, `run_energy_matching_audit`,
`run_radial_energy_matching`, `run_robustness_gain_horizon`,
`run_terminal_assessment`. These represent substantial later research (a
direction-resolved separation sweep to 50 km, a gain/horizon robustness grid, a
bounded-motion audit, terminal position-and-velocity assessment).

### 1.3 Design properties worth preserving

* **No hidden state.** Every scenario function is pure: inputs in, plain data out.
* **Determinism.** A single hand-rolled fixed-step RK4 everywhere; no SciPy, no
  adaptive stepping, no RNG. Re-running gives bit-identical results.
* **Shared integrator for both models.** CW and the high-fidelity reference use
  the same RK4 at the same step, so the measured discrepancy isolates *modelling*
  error rather than integration error. This is a deliberate and correct choice.
* **SI throughout**, `[x, y, z, vx, vy, vz]` state convention in both ECI and LVLH.

---

## 2. Existing simulator capabilities (verified working)

Confirmed by executing the code, not by reading docs:

| Capability | Status | Evidence |
| --- | --- | --- |
| Two-body orbital propagation (3D, inclined) | **Working** | Energy drift 1.1 × 10⁻¹³, angular-momentum drift 5.5 × 10⁻¹⁴, period error 2.3 × 10⁻¹² over two orbits. |
| J2 perturbation toggle | **Working** | Reproduces 5.81 m CW-vs-reference error vs 2.78 m two-body. |
| Atmospheric drag toggle (exponential atmosphere) | **Working** | Orbit decays; adds ≈ 0.004 m to relative error with matched ballistic coefficients. |
| Combined J2 + drag | **Working** | 5.82 m. |
| Orbital-element conversions (RV ↔ COE) | **Working** | Round-trip tested. |
| Energy / angular-momentum diagnostics | **Working** | `validation.verify_orbit`. |
| Clohessy–Wiltshire relative motion (3D) | **Working** | RK4 matches the closed-form `Φ(t)` to 3.0 × 10⁻⁸ m over an orbit. |
| CW closed-form state-transition matrix | **Working** | Reproduces the textbook −12π·x₀ per-orbit along-track drift. |
| PD-controlled rendezvous | **Working** | 369 m → 1.5 × 10⁻⁷ m, converges ≈ 1595 s, Δv 2.326 m/s, peak 0.017 m/s². |
| Thrust saturation | **Working** | Magnitude clip, direction preserved. |
| Gain sweep (Kp × Kd) | **Working** | 16-cell grid with speed-vs-fuel trade-off. |
| CW vs high-fidelity comparison | **Working** | Per-toggle divergence table. |
| LVLH transforms and visualisation | **Working** | Round-trip tested; 3D LVLH Plotly plot in the app. |
| Keep-out / debris **detection** | **Working (scaffold)** | Distances and risk flags only. |
| External-ephemeris export / import / compare | **Working (plumbing only)** | See §4.1 — no benchmark has been run. |
| Streamlit app | **Working** | Starts on :8501, HTTP 200, Learning Mode renders and runs. |

### 2.1 Capabilities in the brief that **do not exist**

These were checked for specifically and are **absent**. They must not be
advertised by the platform:

| Requested feature | Reality |
| --- | --- |
| **GMAT / STK benchmark results** | **No benchmark exists.** `src/external_validation.py` is the *workflow plumbing* to run one — export a scenario, import an external CSV, compute errors. No GMAT or STK run has been performed, and no external ephemeris is committed. `docs/results_summary.md` says so explicitly: "the high-fidelity model itself is still not validated against an external tool (still future work)". `outputs/` contains no external comparison. |
| **Numerical-integrator comparison** | The current engine has **only RK4**. Euler / RK4 / velocity-Verlet comparison existed in the archived 2D prototype (`old_project_backup/`) and was deliberately dropped. What *does* exist is a step-size **convergence** study (`experiments/run_convergence_study.py`). |
| **Chaos / resonance / stability tools** | **None.** No Lyapunov exponents, Poincaré sections, resonance analysis, or stability maps anywhere in `src/`. Grepped for all of these. |
| **Obstacle avoidance** | Detection only. The artificial-potential-field avoidance demo lives in the archived 2D prototype and is not part of the current engine. |
| **Differential drag** | Target and deputy always share `Cd·A/m`, so drag is common-mode by construction. |

---

## 3. Existing tests

**53 tests, all passing** (`python -m pytest`, 12.1 s).

| File | Tests | Covers |
| --- | --- | --- |
| `test_orbit.py` | 6 | Circular speed, 400 km period, RK4 energy/momentum conservation, measured period, zero eccentricity, element round-trip. |
| `test_perturbations.py` | 6 | J2 equatorial symmetry, toggle labels, two-body equivalence, J2 conserving hz with bounded energy, drag decay, zero drag without atmosphere. |
| `test_lvlh.py` | 4 | Rotation-matrix orthonormality, transform round-trip, helper consistency, along-track co-orbital equilibrium. |
| `test_cw.py` | 5 | Matrix shapes, `Φ(0) = I`, RK4-vs-STM agreement, along-track equilibrium, cross-track harmonic at the orbital period. |
| `test_controller.py` | 5 | Critical-damping gains, gain coercion, convergence to target, saturation, cross-track convergence. |
| `test_validation.py` | 6 | Orbit verification, zero Δv uncontrolled, convergence-time semantics, CW≈high-fidelity at small separation, perturbations increasing divergence, gain-sweep shape. |
| `test_control_implementation.py` | 6 | Control-vector round-trip through LVLH, rotation magnitude preservation, saturation direction, zero-state zero-command, P and D term isolation, finite-difference check of the applied acceleration. |
| `test_external_validation.py` | 8 | CSV round-trip, JSON validity, known-shift metrics, km units, column autodetection, cross-grid interpolation, no-overlap error, scenario self-consistency. |
| `test_collision.py` | 7 | Minimum distance, keep-out violation and clearance, debris collision and clearance, summary columns, scenario integration. |

### 3.1 Gaps in the existing test suite

* **No regression lock on the published numbers.** Nothing asserted that the
  2.78 m / 5.81 m divergence figures or the 2.33 m/s Δv still hold. A refactor
  could have silently moved them.
* **No integrator order-of-accuracy test.** RK4 was asserted to be accurate, but
  never asserted to be *fourth order*.
* **No test that the reported metrics are step-size independent** (the
  convergence study exists as a script, but was not asserted).
* **No app-level test.** `app.py` was entirely untested.
* **No input-validation test.** The scenario API clamps `MAX_STEPS` but has no
  bounds on altitude, `dt`, gains, or separations, and no test of nonsense input.

**Added during this audit stage** (before any other change, so they lock the
pre-existing behaviour): `tests/test_regression_baseline.py` (11 tests) and
`tests/test_integrator_convergence.py` (9 tests). See §6.

---

## 4. Technical debt and risks

### 4.1 Documentation overstates validation status — **highest-priority risk**

The README's External Validation section and the app's "External Validation
(GMAT / STK)" mode both read as though external validation is an available
capability. It is a *workflow*, not a *result*. Someone skimming the app could
reasonably come away believing the engine has been benchmarked against NASA GMAT.
It has not.

`docs/results_summary.md` is honest about this; the README and app UI are not
explicit enough. For a platform that will be used by schools and professionals,
this needs unambiguous wording.

### 4.2 `requirements.txt` is a UTF-16 `pip freeze` dump

56 lines of transitive pins (`altair`, `blinker`, `gitdb`, `smmap`, `tzdata`, …)
in **UTF-16 LE with a BOM**. Two problems: direct dependencies are
indistinguishable from transitive ones, and the encoding is not what `pip` and
most tooling expect from a requirements file. It also pins `pytest`, contradicting
the README's claim that pytest is "not in requirements.txt".

### 4.3 README drift

| README claim | Reality |
| --- | --- |
| "47 passing tests (orbit 6, perturbations 6, validation 6, controller 5, cw 5, lvlh 4, external_validation 8, collision 7)" | 53 tests. The listed counts sum to 47 and omit `test_control_implementation.py` (6). |
| `research_log.md`: "32 tests, all passing" | Also stale. |
| README §12 "Build an interactive front end … (planned, not yet started)" | The Streamlit app exists and is described earlier in the same README. |
| README §4 lists three experiments | Nine exist. |
| README §11 "No collision/keep-out handling in this engine" | `src/collision.py` exists and is wired into the app. |

### 4.4 Git hygiene

* At audit time the working tree had **substantial uncommitted work**: `docs/`,
  six experiment scripts, `src/collision.py`, `src/external_validation.py`, and
  three test files — all untracked. A single mistaken `git clean` would have
  destroyed them. Committed as `3294749` before branching.
* `old_project_backup/` (the archived 2D prototype, 22 files) had been **deleted
  from the working tree** but not committed. Restored from `HEAD` during this
  audit — the README and `research_log.md` both still reference it.
* A stray git repository exists at `<project-root>` (the whole home directory),
  with no commits and a stale index. It is unrelated to this project and was
  left untouched, but it is a hazard: running `git` commands from a parent
  directory silently targets it. Recommend `rm -rf ~/.git` after confirming
  nothing depends on it — **not done here**, as it is outside this project.

### 4.5 No input validation at the API boundary

`src/scenarios.py` clamps step count to `MAX_STEPS = 200_000` and nothing else.
Each row below was **executed against the current code**, not inferred:

| Input | Observed behaviour | Severity |
| --- | --- | --- |
| `altitude_km = -7000` | **No error.** `circular_speed` takes `sqrt(mu/negative)` → `nan`, and the entire trajectory propagates as `NaN`. The caller gets a `(2001, 6)` array of `NaN` and the app plots an empty chart with no message. | **Silent wrong answer** |
| Relative state containing `NaN` | **No error.** Metrics come back `NaN`; results render as blanks. | **Silent wrong answer** |
| `duration_s = 1e7`, `dt_s = 1` | **No error.** Silently clamped to 200 000 s — a **20× shorter** simulation than requested, with nothing telling the user. | **Silent wrong answer** |
| `duration_s = -500` | **No error.** Returns 2 samples. | **Silent wrong answer** |
| `dt_s = 0` | `ZeroDivisionError` — raw traceback in the UI. | Crash |
| `altitude_km = -6378.137` (exactly zero radius) | `ZeroDivisionError` — raw traceback. | Crash |
| `mass = 0` with drag enabled | `ZeroDivisionError` — raw traceback. | Crash |
| `kp = 1e6`, `kd = 1e3` | Runs; the "rendezvous" diverges to 118 m with no stability warning. | Misleading |

The silent-wrong-answer rows are the serious ones. A crash is visible; a `NaN`
plot or a quietly-truncated horizon looks like a result. For a platform aimed at
students who cannot yet tell the difference, this must be fixed at the boundary.

### 4.6 Performance / resource risk for a shared deployment

The engine is pure-Python loops. Measured on this machine (Python 3.14.6):

| Call | Steps | Wall time |
| --- | --- | --- |
| `run_cw_rendezvous` (default, `dt = 5 s`, 1 orbit) | 1 110 | 0.07 s |
| `run_relative_scenario(model="comparison")`, `dt = 10 s` | 555 | 0.23 s |
| `run_orbit(400, 51.6, 2.0)` @ 2000 steps/orbit | 4 000 | 0.25 s |
| `run_gain_sweep` (16 cells, `dt = 10 s`) | 16 × 555 | 0.47 s |
| `run_relative_scenario(model="comparison")`, `dt = 1 s` | 5 553 | 2.31 s |
| `run_orbit(400, 51.6, 20.0)` @ 2000 steps/orbit | 40 000 | 2.50 s |

Typical interactive runs are sub-second, which is fine. The risk is the ceiling:
the high-fidelity comparison path costs ≈ 0.4 ms/step, so the existing
`MAX_STEPS = 200_000` clamp permits a **single request that runs for roughly 80
seconds** and blocks a Streamlit worker for its duration. Nothing in the UI stops
a user setting `dt = 0.1 s` over a long duration. With no rate limiting, a
handful of concurrent pilot users can exhaust a small hosted instance.

### 4.7 No platform infrastructure at all

Absent, and all required by the brief: any database, analytics, consent flow,
participant identity, admin area, secrets handling, rate limiting, CI workflow,
`.env` handling, deployment configuration, or containerisation.

### 4.8 Smaller items

* `src/lvlh.py` imports `MU_EARTH` purely to re-export it (`# noqa: F401`).
* `experiments/run_separation_sweep.py` line 66 contains a vestigial `if True:`.
* `app.py` has no error handling around scenario calls — an exception in any
  preset shows a raw Streamlit traceback to the user.
* `outputs/` is git-ignored but present on disk with ~1.3 MB of generated CSVs;
  `cw_rendezvous_timeseries.csv` alone is 1 MB.
* The app's `reset_prefix` calls `st.rerun()` inside a button handler, which
  works but couples UI state to widget keys by string prefix — fragile as the UI
  grows.

---

## 5. What must be preserved

Non-negotiable for the platform work:

1. **`src/` physics is frozen** unless a genuine bug is found. No rewrites.
2. **The published numbers must not move**: 5553.62 s period; 1.1 × 10⁻¹³ energy
   drift; 2.78 / 5.81 / 2.78 / 5.82 m per-toggle divergence; Δv 2.326 m/s; peak
   0.017 m/s²; convergence ≈ 1595 s.
3. **`src/scenarios.py` stays frontend-agnostic** — no Streamlit, no database, no
   file I/O. The platform layer sits *above* it.
4. **Determinism** — no RNG or adaptive stepping introduced into the engine path.
5. `experiments/`, `docs/results_summary.md`, `research_log.md` and
   `old_project_backup/` are research record and stay.

---

## 6. Regression protection added before any change

Written and run *before* touching anything else, against the engine as found:

* **`tests/test_regression_baseline.py`** (11 tests) — asserts the canonical
  outputs against `tests/baseline_engine_outputs.json`, captured from the engine
  at this commit, at `rtol = 1e-9`. Covers two-body conservation, mean motion,
  CW-vs-STM agreement, PD rendezvous metrics, all four perturbation toggles, the
  gain sweep, and the three scenario-API entry points the UI uses. Two tests
  additionally assert the *findings* as relationships (J2 roughly doubles the
  error; drag is nearly common-mode) so they survive a deliberate baseline
  refresh.
* **`tests/test_integrator_convergence.py`** (9 tests) — RK4 observed order of
  accuracy measured at 3.6–4.4 against the closed-form CW solution, energy drift
  falling with step size, circular-orbit closure, cross-track harmonic solution,
  along-track equilibrium, the −12π·x₀ textbook drift, and step-independence of
  the two published metrics (divergence and Δv).
* `tests/_capture_baseline.py` regenerates the baseline deliberately.

Suite after this stage: **73 tests, all passing.**
