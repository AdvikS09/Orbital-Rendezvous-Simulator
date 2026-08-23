# Migration inventory

Everything the Streamlit platform contains, where each item goes in the
React + FastAPI application, and what proves the migration preserved it.

Compiled by reading every source file in full — six parallel inventory passes
over the scenario catalogue, the physics engine, the Streamlit UI, the platform
services, the educational content and the validation evidence, followed by an
independent completeness audit against the repository. The enumerable sections
(scenario and parameter tables, engine API, event vocabulary, database schema)
are generated from the code itself rather than transcribed, so they cannot drift.

**Status key**

| Symbol | Meaning |
| --- | --- |
| `done` | Migrated and covered by a test |
| `partial` | Migrated, but with a documented limitation |
| `todo` | Inventoried, not yet migrated |
| `n/a` | Deliberately not migrated (reason given) |

---

## 0. What the audit found before anything was migrated

Six findings changed the plan. They are recorded here because each was a real
defect, not a migration inconvenience.

| # | Finding | Action |
| --- | --- | --- |
| 1 | **`Dockerfile` broke on the `orbital_core` rename.** `COPY src/ ./src/` copied only the compatibility shim, so `orbital_core/` never entered the image and the build-time regression gate would fail. | Fixed: both are copied. |
| 2 | **`tests/test_control_implementation.py` claimed GMAT validation as fact** — *"GMAT validates the nonlinear propagator and the LVLH relative-state reconstruction"*. No GMAT run has ever happened here. The only assertion of external validation as fact in the codebase. | Docstring corrected with a dated note. Tests unchanged — they were right; the claim about them was wrong. |
| 3 | **`app.py` external-validation copy read as a completed capability** (present tense, *"This workflow validates…"*). Not covered by the honesty-guard test, which only inspects the scenario catalogue. | Rewritten to lead with "No such comparison has been run for this repository." |
| 4 | **Ten parameters bypass the validator entirely.** `_VALIDATED_KEYS` covers 11 names; `keep_out_radius`, `collision_radius`, `debris_x/y/z`, `omega_n_factor` and five booleans/selects are unchecked. `debris_x/y/z` have **no bound at any layer** — Streamlit's `number_input` is called without `min`/`max` on purpose. | Bounded in the API schema layer. See §6.2. |
| 5 | **Three experiment scripts write outside the repository**, to `os.path.dirname(ROOT)`. None of their CSVs exists on disk, so results quoted from those studies have no saved artefact. | Recorded in `KNOWN_GAPS.md`; not changed, as fixing it alters research-script behaviour. |
| 6 | **The published two-body numbers differ between documents** because they were produced on different sample grids (8001 vs 4001). Only the 4001-sample set is regression-locked. | Parity fixtures pin the exact grid per item. See `ENGINE_PARITY_REPORT.md`. |

One reported finding was **wrong**: the audit flagged "two emoji already present in
`web-platform/backend/app/services/adapters.py`". Those are `m/s²` and `×` —
mathematical notation, not pictographs. Verified with a
`\p{Extended_Pictographic}` scan: **zero** pictographic characters in any new
file.

---

## 1. Architecture mapping

| Streamlit | React + FastAPI | Status |
| --- | --- | --- |
| `src/` (physics) | `orbital_core/` — same files, `git mv`, no line changed | `done` |
| `src/scenarios.py` | `orbital_core/scenarios.py`, called by both frontends | `done` |
| `pilot/content/*` | Same modules, read by the API | `done` |
| `pilot/validation.py` | Same module, called by `POST /simulations/{id}` | `done` |
| `pilot/runner.py` | Same module — one run path for both frontends | `done` |
| `pilot/analytics.py` | Same module, behind `app/analytics/` | `done` |
| `pilot/export.py` | Same module, served by `/exports/*` | `done` |
| `pilot/ui/figures.py` | `frontend/src/charts/` — Plotly.js from returned arrays | `done` |
| `pilot/ui/*` (pages) | `frontend/src/pages/` | see §4 |
| `pilot_app.py` (router) | `frontend/src/App.tsx` + React Router | `done` |
| `app.py` (legacy MVP) | Kept, unchanged in behaviour | `n/a` — regression reference |

**No physics was rewritten in JavaScript.** The frontend collects inputs, calls
the API and renders returned arrays. It has no propagator, no force model and no
integrator.

---

## 2. Physics engine

`orbital_core/`, 11 modules, unchanged. Every file moved with `git mv`;
`git log --follow` traces each back through the rename.

| Module | Public API | Migration |
| --- | --- | --- |
| `orbit` | 17 functions + 8 constants | Imported by API; not re-exposed directly |
| `perturbations` | `j2_acceleration`, `exponential_density`, `drag_acceleration`, `ForceModel` | Via scenario runners |
| `lvlh` | 6 transform functions | Via scenario runners |
| `cw` | 6 functions incl. closed-form `cw_state_transition` | Via scenario runners + convergence scenario |
| `controller` | `PDController`, `critically_damped_gains`, `saturate` | Via scenario runners |
| `validation` | 13 functions (conservation, metrics, gain sweep, comparison) | Via scenario runners |
| `external_validation` | 6 functions (export/import/compare) | `build_validation_scenario` exposed; upload path `todo` |
| `collision` | `minimum_distance`, `analyze` | Via `proximity_safety` |
| `scenarios` | **11 functions — the API boundary** | Called by `app/services/simulation.py` |
| `plotting` | 6 matplotlib helpers | `n/a` — experiment scripts only; the web app plots client-side |

### 2.1 Constants — asserted unchanged

| Constant | Value | Unit |
| --- | --- | --- |
| `G` | 6.67430 × 10⁻¹¹ | m³ kg⁻¹ s⁻² |
| `MU_EARTH` | 3.986004418 × 10¹⁴ | m³/s² |
| `R_EARTH` | 6 378 137.0 | m |
| `J2_EARTH` | 1.08262668 × 10⁻³ | — |
| `OMEGA_EARTH` | 7.2921159 × 10⁻⁵ | rad/s |
| `DEFAULT_ALTITUDE` | 400 000.0 | m |
| `DEFAULT_RADIUS` | 6 778 137.0 | m |

Integrator: fixed-step classical RK4, everywhere, unchanged. No adaptive
stepping, no RNG, no tolerance altered.

---

## 3. Scenarios — 11 scenarios, 100 parameters

Every scenario reaches the frontend through `GET /api/v1/scenarios/{id}` and runs
through `POST /api/v1/simulations/{id}`.

| # | Scenario | Params (adv.) | Result kind | Tracks | Engine call | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `two_body_orbit` | 3 (0) | `orbit` | 3 | `run_orbit` | `done` |
| 2 | `j2_effect` | 3 (0) | `orbit_comparison` | 3 | `compare_orbit_perturbation` | `done` |
| 3 | `drag_effect` | 6 (1) | `orbit_comparison` | 3 | `compare_orbit_perturbation` | `done` |
| 4 | `cw_natural_motion` | 9 (4) | `relative` | 3 | `run_cw_relative` | `done` |
| 5 | `pd_rendezvous` | 11 (4) | `relative` | 3 | `run_cw_rendezvous` | `done` |
| 6 | `gain_tuning` | 6 (1) | `gain_sweep` | 3 | `run_gain_sweep` | `done` |
| 7 | `cw_vs_reference` | 15 (4) | `comparison` | 3 | `run_relative_scenario` | `done` |
| 8 | `integrator_convergence` | 4 (0) | `convergence` | 2 | `cw.propagate_cw` vs `propagate_cw_analytic` | `done` |
| 9 | `proximity_safety` | 15 (4) | `collision` | 3 | `run_collision_scenario` | `done` |
| 10 | `external_validation_export` | 6 (0) | `external_export` | 2 | `build_validation_scenario` | `partial` |
| 11 | `research_workbench` | 22 (10) | `comparison` | 2 | `run_relative_scenario` | `done` |

Three scenarios are gated off the guided track (8, 10, 11) — preserved exactly.

### 3.1 Parameter fidelity

All 100 parameters are carried with **name, label, kind, default, minimum,
maximum, step, unit, help text, options and the advanced flag**, served from the
same `Param` objects the Streamlit app uses. Consequences preserved:

* Per-scenario label overrides — `integrator_convergence` says "Radial offset"
  where others say "Radial offset (R-bar, x)".
* Per-scenario advanced overrides — `drag_effect` makes `area`/`mass` visible
  and `cd` advanced, inverting the shared defaults.
* Four parameters have empty help text (`include_debris`, `debris_x/y/z`).
* `options` is non-empty for exactly one parameter (`research_workbench.model`).

### 3.2 Behaviour that is not in the parameter list

Recorded because a re-implementation would lose it:

| Hidden behaviour | Where | Effect |
| --- | --- | --- |
| `steps_per_orbit` hard-coded 1500 / 800 / 600 | scenario runners 1 / 2 / 3 | Not user-visible; materially affects accuracy |
| Gain grid `kp ∈ {1e-5, 4.6e-5, 1e-4, 2e-4}`, `kd ∈ {5e-3, 1e-2, 1.36e-2, 3e-2}`, `pos_tol = 1.0` | `orbital_core/scenarios.py` | The catalogue text says "4×4 grid" and "1 m tolerance"; these must stay in sync |
| `gain_tuning` ignores `max_accel` | `_run_gain_sweep` | Saturation disabled, as its stated assumption says |
| `pd_rendezvous` never passes `kp`/`kd` | `_run_pd_rendezvous` | Bandwidth only |
| `integrator_convergence` ignores `dt_s`/`duration_s` | `_run_convergence` | Builds its own six-point grid over one period |
| Convergence grid differs from the experiment script | 126–4001 in-app vs 251–16001 in `run_convergence_study.py` | Both are "the convergence study" in the docs |
| Debris always starts at rest | `_run_collision` never sets `debris_rel_vel` | Engine supports it; catalogue does not expose it |
| `DURATION` default 5553.62 s | shared `Param` | ≈ one orbit at 400 km; does **not** follow the altitude control |

`_run_convergence` is the only runner containing analysis logic in the content
layer rather than in `orbital_core`. It is called unchanged; the API adapts its
output.

---

## 4. User interface

### 4.1 Pages

| Streamlit stage | React route | Status |
| --- | --- | --- |
| `landing` | `/` | `done` |
| `consent` | `/consent` | `done` |
| `entry` | `/sign-in` | `done` |
| `track` | `/track` | `done` |
| `workspace` (guided) | `/activities/:activityId` | `done` |
| `workspace` (professional/free) | `/scenarios/:scenarioId` | `done` |
| `assessment_pre` / `assessment_post` | `/assessment/:phase` | `done` |
| `feedback` | `/feedback` | `done` |
| `confirmation` | `/done` | `done` |
| `?view=admin` | `/admin`, HTTP Basic on every admin endpoint | `done` |

The Streamlit admin area was reachable at an unauthenticated URL that then asked
for a password. In the new application **the API itself requires credentials**,
so the route cannot leak data before the form is answered.

### 4.2 Plots

All eight result kinds render from returned arrays via Plotly.js. Palette
(Okabe-Ito) and dash patterns carried over unchanged, so a series keeps the same
colour *and* the same line style across scenarios.

| Streamlit builder | React equivalent | Status |
| --- | --- | --- |
| `orbit_3d` | `PlotKind.orbit3d`, Earth drawn to scale | `done` |
| `lvlh_trajectory` | `PlotKind.scatter3d` + target marker | `done` |
| `series` / `components` | `PlotKind.line` | `done` |
| `control_history` | `PlotKind.line` with emphasised \|u\| trace | `done` |
| `dual_error` | `PlotKind.dual_axis` | `done` |
| `gain_heatmap` | `PlotKind.heatmap` | `done` |
| `proximity_3d` | `PlotKind.scatter3d` + keep-out sphere | `done` |
| `convergence_plot` | `PlotKind.log_log` + computed 4th-order reference line | `done` |

The `|u|` trace was styled by hard-coded index (`fig.data[3]`) in Streamlit;
the API now marks it explicitly, so reordering cannot silently restyle the wrong
series.

### 4.3 Emoji removal

**59 occurrences of 19 distinct emoji** across the Streamlit UI, all replaced by
1.5px-stroke SVG icons inheriting `currentColor`:

| Glyph | Count | Replacement |
| --- | --- | --- |
| ⚠ | 13 | `warning` icon |
| ✅ | 10 | `success` icon |
| 🚫 | 8 | `error` icon |
| 🛰 | 4 | `orbit` icon |
| 🚨 | 4 | `error` icon |
| ⬇ | 3 | `download` icon |
| ▶ | 2 | `play` icon |
| 🔬 | 2 | `flask` icon |
| 🔒 | 2 | `lock` icon |
| ℹ | 2 | `info` icon |
| 🎓 📐 📈 ⚖ 🔍 💡 🎉 🧪 👋 | 1 each | `book`, `ruler`, `chart`, `chart`, `target`, `info`, `success`, `flask`, `user` |

Icons are `aria-hidden`, because each sits beside a text label that already
carries the meaning. Screen readers previously announced "triangular ruler" for
📐, which was noise.

Mathematical characters are **kept**, being notation rather than decoration:
`Δ ² × → · — – ≈ − §` and the superscript digits used in scientific notation.

### 4.4 The number-formatting defect this migration fixes

The Streamlit interface rendered values with Python's `%g`, producing `2e-12`
and `1.1060555202205516e-13` on screen. `frontend/src/formatting/number.ts`
replaces this with a single utility supporting fixed, significant-figure,
scientific, percent, angle, duration, distance and velocity presentation, with
real Unicode superscripts.

`2e-12` now renders as `2.0 × 10⁻¹²`. 44 tests cover it, including one asserting
that **no output for any magnitude ever contains machine notation**, and one
asserting `rawValue()` — used for CSV and JSON — never contains a superscript,
a Unicode minus or a multiplication sign, so exports stay parseable.

---

## 5. Educational content

| Item | Count | Destination | Status |
| --- | --- | --- | --- |
| Guided activities | 3 | `GET /api/v1/activities` | `done` |
| Activity steps | 11 (4 / 3 / 4) | Same, with `params`, `predict_options`, `look_for`, `takeaway` verbatim | `done` |
| Assessment instrument | 1 (`orbital_basics` v1.0) | `GET /api/v1/assessments/{id}` | `done` |
| Assessment items | 10, across 5 topics | Same | `done` |
| Scenario `reading_guide` | 11 blocks | In `ScenarioDetail.reading_guide` and every `SimulationResult` | `done` |
| Scenario `explanation` | 11 blocks, 341–664 characters | `ScenarioDetail.explanation` | `done` |
| Assumptions / limitations | Per scenario | Shown beside every result and written into every export | `done` |

Correct answers are **withheld** from `GET /assessments/{id}` and returned only
with a submitted post-assessment, so the pre-assessment cannot be answered by
reading the API. Streamlit had no equivalent exposure because the data never left
the server.

The `DRAFT — not yet reviewed by a subject teacher` status on both content
modules is carried through to the API and displayed.

---

## 6. Platform services

### 6.1 Analytics — 15 event types, 9 tables

`EventType` values, `USER_TYPES`, `TRACKS`, `SAFE_DETAIL_KEYS`, the
`participant_hash` HMAC construction and every reporting query are **reused, not
reimplemented**, so both frontends record identical events against identical
metric definitions.

Bucket edges are load-bearing for chart continuity and are preserved exactly:

* altitude `<300km | 300-500km | 500-800km | 800-1200km | >1200km`
* separation `<100m | 100m-1km | 1-10km | 10-50km | >50km`
* steps `<500 | 500-2k | 2k-10k | 10k-40k | >40k`

Metric definitions are served by `GET /api/v1/admin/report` and written into
every export as `METRIC_DEFINITIONS.txt`. **A page view is never counted as a
user**: a visitor who signs in but completes no simulation is not a unique user.

### 6.2 Input validation

`pilot/validation.py` is called unchanged — the same bounds, the same verbatim
error and warning messages. Every silent-wrong-answer case it fixes is still
fixed, and `POST /simulations/{id}` cannot reach the engine without passing it.

The ten unvalidated parameters (§0 finding 4) are now bounded in the API schema
layer. That is an addition, not a change: the engine still receives whatever the
validator allows, and no existing accepted input is newly rejected.

### 6.3 Other services

| Service | Migration | Status |
| --- | --- | --- |
| Rate limiting | `pilot/ratelimit.py`, keyed by session token | `done` |
| Admin auth | `pilot/auth.py` PBKDF2 + lockout, via HTTP Basic | `done` |
| Config | `pilot/config.py`, same env vars | `done` |
| Migrations | `pilot/db.py`, run on API startup | `done` |
| Exports | `pilot/export.py` via `/exports/*` | `done` |
| Scripts | Unchanged, still work | `done` |

---

## 7. Validation evidence

**No GMAT or STK benchmark exists in this repository.** The workflow to run one
exists and is exposed; the result does not. Every place that implied otherwise
has been corrected (§0 findings 2 and 3).

The only committed numerical artefact is `tests/baseline_engine_outputs.json`.
All of `outputs/` is git-ignored, so a fresh clone has no published figures until
the experiment scripts are re-run.

| Evidence | Detail | Migration |
| --- | --- | --- |
| Regression baseline | 1 JSON, `RTOL = 1e-9`, `atol = 0` | Parity fixtures reuse the same tolerance |
| Convergence bounds | observed order 3.6–4.4; step-independence < 1e-3 | Reused |
| Absolute guards | energy drift < 1e-11, period error < 1e-9, CW-vs-STM < 1e-6 | Reused |
| Learner-content tolerances | 1595 s ±25, 2.33 m/s ±0.05, 0.017 m/s² ±0.001, 2.8/5.8 m ±0.15, −3770 m rel 0.02 | Reused |
| Honesty guard tests | Block "NASA-approved", "flight-certified" etc. in scenario text and require the landing-page disclaimers | Ported to the new stack |

Two published figures have **no** regression coverage and remain unguarded: the
≈110 m radial-offset divergence (recomputed ad hoc, never saved) and the 1.7 cm
closed-loop agreement. Recorded in `KNOWN_GAPS.md`.

The RK4 CW-vs-STM error hits a floating-point floor at ≈5.3 × 10⁻¹⁰ m. The
fourth-order test deliberately uses steps 125–1000 to stay above it; extending
that list would break the 3.6–4.4 window for reasons of arithmetic, not physics.

---

## 8. Deliberately not migrated

| Item | Reason |
| --- | --- |
| `app.py` (legacy MVP) | Kept and still runs — the regression reference the brief requires |
| `pilot/ui/*` (Streamlit pages) | Kept; the Streamlit app must remain functional |
| `orbital_core/plotting.py` | Matplotlib helpers for experiment scripts; the web app plots client-side |
| `old_project_backup/` | Archived 2D prototype, 21 files, excluded from collection and from the image. Includes an obstacle-avoidance demo with no counterpart in the current engine |
| `experiments/*` | Research scripts, unchanged and still runnable |

---

## 9. Parity tests required

| Area | Test | Tolerance |
| --- | --- | --- |
| Every scenario, default parameters | API result vs direct engine call | `rtol = 1e-9` |
| Time arrays, state vectors, control arrays | Element-wise | `rtol = 1e-9` |
| Scalar metrics | Per metric | `rtol = 1e-9` |
| Conservation metrics | Energy, angular momentum, period | `rtol = 1e-9` |
| Units and defaults | Exact match against `Param` | exact |
| Error handling | Each silent-wrong-answer case rejected | exact message |
| Number formatting | Python and TypeScript agree | exact string |
| Export contents | CSV values equal engine values | `rtol = 1e-12` |

Results in `ENGINE_PARITY_REPORT.md`.
