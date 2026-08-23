# Implementation report

What was built, what was decided and why, how it was tested, and what remains
limited.

* Branch: `pilot-platform`, six commits from `main` @ `3294749`
* 54 files added, 3 modified (`.gitignore`, `README.md`, `requirements.txt`)
* **0 files changed under `src/`, `app.py` or `experiments/`**
* Tests: 53 → 255, all passing
* Date: 2026-08-23

---

## 1. The constraint everything else follows from

The engine in `src/` produces validated numerical results cited across the
repository's research write-ups. Those numbers must not move.

So the first thing done — before any feature work — was to **lock them**:
`tests/test_regression_baseline.py` captures the engine's canonical outputs and
asserts them at `rtol = 1e-9`. Only then was anything else built, and it was all
built **above** `src/scenarios.py`, the frontend-agnostic boundary the engine
already provided.

**Result: `git diff main -- src/` is empty.** Re-running
`experiments/run_cw_vs_high_fidelity.py` and `experiments/run_cw_rendezvous.py`
after all the work produced CSV files **byte-identical** to the ones present
before it.

---

## 2. Stages

| Stage | Commit | What |
| --- | --- | --- |
| 0 | `3294749` | Committed pre-existing uncommitted research work; restored deleted `old_project_backup/` |
| 1 | `7c244a2` | Audit; regression and convergence tests |
| 2 | `7c50fb6` | Platform foundation: config, database, analytics, validation, safeguards, runner, export |
| 3–4 | `51d7dcb` | Educational content and the full participant interface |
| 5 | `c8247d8` | Requirements, scripts, CI, Docker, Streamlit config |
| 6 | `50abe42` | Documentation |
| — | `d6295ea` | Cohort fix found by end-to-end verification |

### Stage 0 — before touching anything

The working tree had **substantial untracked work**: `docs/`, six experiment
scripts, `src/collision.py`, `src/external_validation.py` and three test files —
one `git clean` from destruction. All committed first.

`old_project_backup/` (22 files, the archived 2D prototype) had been deleted
from the working tree but not committed, while `README.md` and `research_log.md`
still referenced it. Restored from `HEAD`. **Flagged in `KNOWN_GAPS.md` §7 in
case the deletion was intentional.**

---

## 3. Features added

### 3.1 Participant journey

Implemented as a stage machine in `pilot_app.py` over modules in `pilot/ui/`.

| Step | Module | Notes |
| --- | --- | --- |
| Landing | `ui/flow.py` | Purpose, audience, and seven physics limitations stated up front |
| Privacy and consent | `ui/flow.py` | Continue is **disabled** until consent is given |
| Anonymous sign-in | `ui/flow.py` | Pilot code, participant code, user type; email-shaped codes rejected |
| Track choice | `ui/flow.py` | Guided / professional / free; the suggestion is a nudge, not a gate |
| Simulation | `ui/workspace.py` | All three tracks converge on one run path |
| Results | `ui/results.py` | Numbers → plots → interpretation → assumptions and limitations |
| Downloads | `export.py` | Result CSVs, reproducible config JSON, caveats text |
| Feedback | `ui/feedback.py` | Questions adapt to user type |
| Assessments | `ui/assessment.py` | Optional pre/post for educational users |
| Confirmation | `ui/flow.py` | Says feedback *may* influence releases; promises nothing specific |

### 3.2 Scenarios — eleven, all real

`pilot/content/scenarios_catalog.py`. Each declares its parameters, assumptions,
limitations and a reading guide as **required dataclass fields**, so a scenario
cannot be added without them.

| Scenario | Engine capability |
| --- | --- |
| Two-body circular orbit | `run_orbit` |
| J2 effect | `compare_orbit_perturbation` |
| Atmospheric drag | `compare_orbit_perturbation` |
| Natural CW relative motion | `run_cw_relative` |
| PD-controlled rendezvous | `run_cw_rendezvous` |
| Gain tuning | `run_gain_sweep` |
| CW versus non-linear reference | `run_relative_scenario` |
| Integrator convergence | `cw.propagate_cw` vs `propagate_cw_analytic` |
| Proximity safety | `run_collision_scenario` |
| External-validation workflow | `build_validation_scenario` |
| Research workbench | `run_relative_scenario`, all parameters |

**Three capabilities in the brief were found not to exist and were not faked** —
GMAT/STK benchmark results, an integrator *comparison*, and chaos/resonance/
stability tools. See `CURRENT_STATE_AUDIT.md` §2.1 and `KNOWN_GAPS.md` §1. A
test (`test_no_scenario_claims_external_validation_as_a_result`) guards against
the software ever claiming a benchmark it has not run.

### 3.3 Educational content

Three activities (11 steps) using predict-then-run: the Run button is disabled
until a prediction is made. Ten-item matched pre/post instrument.

**Every number quoted to a learner is asserted against the engine** by
`tests/test_platform_content.py` — the 109-minute period at 1200 km, the
−12π·x₀ = −3770 m radial drift, the 1595 s / 2.33 m/s / 0.017 m/s² rendezvous
figures, and the 2.8 m → 5.8 m J2 gap. A wrong "expected value" teaches the
wrong lesson more effectively than none at all.

Both files carry a `review_status` of **DRAFT — not yet reviewed by a subject
teacher**, asserted by a test so it cannot be quietly dropped.

### 3.4 Analytics

Fifteen event types; `record_event` refuses anything outside the vocabulary.
Metric definitions implemented exactly as specified — see
`docs/DATA_DICTIONARY.md` §1.

### 3.5 Administrator dashboard

`/?view=admin`, credentials from environment secrets only. Filters by date,
organisation and user type. Reports organisations, cohorts, users, runs,
per-user counts, usage by every dimension, return rate, assessment change,
professional ratings separately, qualitative feedback, and bugs/requests. CSV
per table plus a ZIP bundle carrying `METRIC_DEFINITIONS.txt`.

---

## 4. Architecture decisions

### 4.1 Keep Streamlit

The repository already used it and `app.py` was already physics-free. Migrating
to FastAPI plus a JS frontend would mean rewriting every plot and form to reach
the same place, buying concurrency, layout control and a REST API — none of
which is the pilot's bottleneck. What the pilot needs is a public URL with no
install, which Streamlit Community Cloud provides free. The costs are real and
documented in `docs/ARCHITECTURE.md` §7.

### 4.2 Package named `pilot`, not `platform`

A top-level `platform` package would shadow the standard-library module of that
name and break NumPy, Streamlit and pytest on import.

### 4.3 Validation above the boundary, not inside `src/`

The audit found inputs producing **silently wrong answers**: negative altitude →
all-`NaN` trajectory with no error; over-long horizon → quietly truncated to a
fifth. Fixing that inside `src/` meant editing validated physics.

`pilot/validation.py` sits above the boundary instead. The engine is unchanged;
the platform refuses to ask it questions it answers badly. **Trade-off, stated
plainly:** anyone importing `src.scenarios` directly still gets the old
behaviour. `KNOWN_GAPS.md` §6.1.

### 4.4 One run path

Every simulation goes through `pilot/runner.run_scenario`: validate → rate-limit
→ run → record. Analytics cannot be bypassed by a UI change, and an engine
exception can never reach a participant as a traceback.

### 4.5 Anonymity structurally, not by policy

`HMAC-SHA256(salt, PILOT|PARTICIPANT)`. The schema has **no column** that could
hold a name, email, age, IP or user agent, and a test asserts it. Event detail
is filtered against an allow-list of bucketed keys.

### 4.6 SQLAlchemy Core with a hand-rolled migration runner

Core rather than ORM: the queries are aggregate reporting, which reads better as
SQL-shaped code. A numbered migration list rather than Alembic: one dependency
and one concept fewer, and the schema is small.

### 4.7 `is_demo` on every observation row

Set from `PILOT_ENV`; excluded from every report by default. This is why a fresh
development instance shows zeros — it is working correctly.

---

## 5. Database design

Eight tables: `organisations`, `pilot_codes`, `participants`, `sessions`,
`events`, `assessments`, `feedback`, `schema_migrations` (+ `admin_audit`).
Full field-by-field reference in `docs/DATA_DICTIONARY.md`.

Design points worth calling out:

* **`participants.participant_hash`** is the only identity. The code is never
  stored.
* **`events` denormalises** `user_type`, `track` and `organisation_id` so
  reporting queries do not need three joins.
* **`assessments`** is unique on `(participant, instrument, phase)` — retaking
  overwrites rather than duplicating.
* **`detail_json`** holds bands, not values.
* **Unregistered pilot codes are recorded** (see §7.1).

SQLite for local development, PostgreSQL for deployment, same code path.

---

## 6. Verification

### 6.1 Commands

```bash
# Full suite
python -m pytest -q                                    # 255 passed

# The guard that the physics has not moved
python -m pytest tests/test_regression_baseline.py -q   # 11 passed
python -m pytest tests/test_integrator_convergence.py -q # 9 passed

# Engine works with no platform dependencies
python -m unittest discover -s tests -p "test_orbit.py" # 6 passed

# Engine untouched
git diff main --stat -- src/                            # empty

# Experiments reproduce their committed outputs
python experiments/run_cw_vs_high_fidelity.py
python experiments/run_cw_rendezvous.py
diff <before> <after>                                   # identical

# Operational scripts
python scripts/init_db.py --check
python scripts/seed_pilot_codes.py --list
python scripts/purge_expired.py --dry-run

# The app
streamlit run pilot_app.py
```

### 6.2 Results

**255 tests, all passing, 47 s.**

| Area | File | Tests |
| --- | --- | --- |
| Engine (pre-existing, unchanged) | 9 files | 53 |
| Numerical regression baseline | `test_regression_baseline.py` | 11 |
| Integrator convergence | `test_integrator_convergence.py` | 9 |
| Platform database | `test_platform_database.py` | 11 |
| Analytics and privacy | `test_platform_analytics.py` | 34 |
| Input validation | `test_platform_validation.py` | 31 |
| Runner, rate limiting, auth | `test_platform_runner.py` | 38 |
| Educational content | `test_platform_content.py` | 41 |
| End-to-end user journey | `test_ui_flow.py` | 27 |

### 6.3 Live verification in a browser

Ran the app and walked the journey by hand: landing → consent gate → sign-in →
track → pre-assessment → guided activity with a real simulation → results with
plots and downloads → admin sign-in.

Confirmed against the database afterwards:

* Event sequence: `session_started`, `pilot_accessed`, `track_selected`,
  `assessment_completed`, `simulation_started`, `simulation_completed`,
  `scenario_category_used`, `feature_category_used`.
* `detail_json` held `{"model": "orbit", "steps_bucket": "2k-10k",
  "altitude_bucket_km": "300-500km", "warning_count": 0}` — **buckets only**.
* The participant code `BLUE-FALCON-42` appeared **nowhere** in the database.
* `is_demo = true` on every row; the dashboard showed zeros until demo
  inclusion was explicitly ticked.
* Admin rejected a wrong password with an attempt counter; no secret appeared
  anywhere on the page.
* Mobile viewport at 375 px: **zero horizontal overflow**.

### 6.4 End-to-end lifecycle check

A separate verification exercised a full two-day pilot through the real code
path — coordinator registers an organisation, two participants sign in (one
registered code, one not), simulations run, an invalid input is rejected,
assessments and feedback are submitted, one participant returns on a **separate
date**, then every metric, filter and export is checked. **All checks passed**,
including:

* returning user = 1, return rate 50 %, measured-impact users = 1;
* a rejected run counted as `simulation_rejected`, **not** as a failure;
* organisation, user-type and date filters each isolating correctly;
* ZIP export containing no participant code, no full hash and no session token;
* identical results from identical inputs (the reproducibility claim);
* no raw parameter values in any event detail.

---

## 7. Issues found and fixed

### 7.1 Unregistered cohorts collapsed together — found by verification

Two *different* unregistered pilot codes both mapped to the single
"Unregistered pilot code" organisation with `pilot_code_id` NULL, so a
coordinator who forgot to register could not segment their own data at all.

**Fix** (`d6295ea`): unregistered codes are recorded as `pilot_codes` rows under
that same organisation. The organisation stays literally "Unregistered pilot
code" — no institution is invented — but the cohort dimension survives. Added
`usage_by_pilot_code()` and a "Cohorts" table in the dashboard, plus 5 tests.

### 7.2 Sidebar run counter lagged by one render

`_sidebar()` ran before the stage, so it read `completed_runs` before the run
incremented it. Moved after the stage; Streamlit writes to the sidebar container
whenever it is called, so position in the file does not affect layout.

### 7.3 Deprecated Streamlit API past its removal date

41 uses of `use_container_width`, deprecated for removal after 2025-12-31.
Migrated to `width="stretch"` / `width="content"`.

### 7.4 Pre-existing repository issues

* `requirements.txt` was a 56-line **UTF-16** `pip freeze`. Rewritten as curated
  UTF-8 direct dependencies with the full closure in `requirements-lock.txt`.
  **Pinned versions unchanged**, so results are unaffected.
* README drift corrected: it claimed 47 tests (was 53), said the interactive
  frontend was "not yet started" while describing it, listed 3 experiments when
  9 exist, and said there was no collision handling when `src/collision.py`
  exists and is wired into the app.

---

## 8. Remaining limitations

Full inventory in `KNOWN_GAPS.md`. The ones that would change a decision:

| Limitation | Detail |
| --- | --- |
| **No GMAT/STK benchmark** | Does not exist and is not claimed. The workflow ships; the result does not |
| **No integrator comparison** | One integrator (RK4). A convergence study is exposed instead, correctly labelled |
| **No chaos/resonance/stability tools** | None exist in this engine |
| **Educational content unreviewed** | Numbers verified; reading level, curriculum fit and item quality are not |
| **Assessment design is confounded** | Same items before and after — part of any gain is test familiarity |
| **Not load-tested** | Single-user timings only |
| **Not deployed to a public host** | Postgres path, secrets injection and cold start unexercised |
| **Docker image not built** | Written and reviewed; no Docker daemon available |
| **CI not executed** | Secret-scan commands verified locally; the matrix has not run. Python 3.12 untested |
| **Validation does not protect `src/`** | Direct importers still get the old silent-`NaN` behaviour |
| **Rate limiter is per-process** | Multiple replicas multiply the effective limit |
| **Retention is not automatic** | `purge_expired.py` must be scheduled |
| **Accessibility partly unverified** | Chromium only; no screen-reader testing; contrast not measured |

---

## 9. Files

### Added — platform (`pilot/`, 6551 lines)

`config.py`, `models.py`, `db.py`, `analytics.py`, `validation.py`,
`ratelimit.py`, `auth.py`, `runner.py`, `export.py`;
`content/{scenarios_catalog,activities,assessments}.py`;
`ui/{common,figures,flow,workspace,results,assessment,feedback,admin}.py`;
plus `pilot_app.py`.

### Added — tests (2443 lines)

`conftest.py`, `test_regression_baseline.py`, `_capture_baseline.py`,
`baseline_engine_outputs.json`, `test_integrator_convergence.py`,
`test_platform_{database,analytics,validation,runner,content}.py`,
`test_ui_flow.py`.

### Added — operations

`scripts/{init_db,hash_password,seed_pilot_codes,purge_expired}.py`,
`.github/workflows/tests.yml`, `Dockerfile`, `.dockerignore`,
`.streamlit/config.toml`, `.env.example`, `requirements-lock.txt`.

### Added — documentation

`CURRENT_STATE_AUDIT.md`, `IMPLEMENTATION_REPORT.md`,
`MANUAL_REVIEW_CHECKLIST.md`, `KNOWN_GAPS.md`, `CHANGELOG.md`, and
`docs/{ARCHITECTURE,DATA_DICTIONARY,PRIVACY,DEPLOYMENT,PILOT_COORDINATOR,PARTICIPANT,TROUBLESHOOTING}.md`.

### Modified — three files

`.gitignore` (added `pilot_data/`, `.env`, `*.db`), `README.md` (rewritten and
corrected), `requirements.txt` (rewritten; versions unchanged).

### Not modified

**Every file under `src/`, `app.py`, and every file under `experiments/`.**
