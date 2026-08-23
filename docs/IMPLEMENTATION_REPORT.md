# Implementation report — custom web migration

What was built, what changed, what it cost, and what it is verified to do.

* Branch: `custom-web-migration`, four commits from `pilot-platform`
* 196 files added, 6 modified, 10 renamed
* **Zero lines of physics changed**
* Tests: 442 → 519 (255 engine, 187 backend, 77 frontend)
* Date: 2026-08-23

---

## 1. The constraint, and how it was met

The engine produces validated numerical results the research depends on. The
migration had to replace the presentation layer without touching them.

**Result:** `git diff pilot-platform --find-renames -M -- src/ orbital_core/`
shows **zero content change** in 9 of the 11 engine files. The two exceptions
are a module docstring (11 lines, naming both frontends instead of one) and
`src/__init__.py`, which became a compatibility shim. No constant, equation,
force model, integrator, tolerance or unit differs.

Independently confirmed three ways:

1. The 255-test engine suite passes unchanged, still importing `from src import
   …` — so it is testing the move, not rewritten imports.
2. Re-running `run_cw_vs_high_fidelity.py` and `run_cw_rendezvous.py` produces
   **byte-identical** CSV files.
3. 338,556 values compared between the new API and a direct engine call:
   maximum difference exactly zero. See `ENGINE_PARITY_REPORT.md`.

---

## 2. Phases

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | `docs/MIGRATION_INVENTORY.md` | Done |
| 2 | `orbital_core/` extraction | Done |
| 3 | `web-platform/` workspace | Done |
| 4 | FastAPI backend, 19 endpoints | Done |
| 5 | Content migration, emoji removal | Done |
| 6 | React frontend | Done |
| 7 | Plotting | Done |
| 8 | Exports | Done |
| 9 | Analytics and pilot features | Done |
| 10 | Numerical parity | Done — bit-exact |
| 11 | Automated testing | Done — 519 tests |
| 12 | Deployment | Done, **Docker build unverified** (no daemon available) |

---

## 3. Phase 1 — inventory

Six parallel passes read every source file in full, followed by an independent
completeness audit against the repository. 514 items inventoried; the audit
found real gaps and three live defects, all fixed:

1. **The Dockerfile was broken by the rename.** `COPY src/ ./src/` copied only
   the shim, so `orbital_core/` never entered the image and the build-time
   regression gate would have failed.
2. **A test docstring asserted GMAT validation as fact** — "GMAT validates the
   nonlinear propagator". No GMAT run has ever happened here. Corrected with a
   dated note; the tests themselves were right.
3. **`app.py` described the external-validation workflow as a completed
   capability** in the present tense, and was not covered by the honesty-guard
   test. Rewritten to lead with "No such comparison has been run".

One audit finding was **wrong** and is recorded as such: the reported emoji in
`adapters.py` were `m/s²` and `×` — mathematical notation.

---

## 4. Phase 2 — engine extraction

`git mv src orbital_core`, so all 11 files keep full history
(`git log --follow` traces each back to its original commit).

`src/` became a compatibility shim that **aliases the same module objects**:
`src.orbit is orbital_core.orbit` evaluates to `True`. Not a copy — there is
exactly one definition of every constant.

`pilot/` and `app.py` now import `orbital_core` directly. `tests/` and
`experiments/` deliberately still import `src`, which is what makes the suite a
regression check on the move rather than a test of rewritten imports.

`orbital_core` confirmed free of any `streamlit`, `plotly` or `fastapi` import.

**The legacy Streamlit app kept serving on :8501 throughout the refactor.**

---

## 5. Phase 4 — backend

19 endpoints under `/api/v1`. Pydantic models throughout.

| Method | Path |
| --- | --- |
| GET | `/health` |
| GET | `/scenarios`, `/scenarios/categories`, `/scenarios/{id}` |
| POST | `/scenarios/{id}/preflight` |
| POST | `/simulations/{id}` |
| POST | `/sessions`, `/sessions/track` |
| POST | `/feedback` |
| GET | `/assessments/{id}` |
| POST | `/assessments/{id}/submit` |
| GET | `/activities`, `/activities/{id}` |
| POST | `/activities/{id}/complete` |
| POST | `/exports/{id}/config`, `/exports/{id}/notes` |
| GET | `/admin/report`, `/admin/organisations`, `/admin/export` |

### 5.1 The key decision: import, do not duplicate

The API imports `orbital_core` **and** `pilot` — the scenario catalogue, input
validation, run path, analytics, exports and educational content. It does not
copy any of them.

There is therefore exactly one definition of each scenario, each bound, each
metric and each piece of teaching text, shared with the Streamlit application.
They cannot disagree, because there is nothing to disagree with.

### 5.2 Response shape

No Plotly figure is built in Python. Responses carry named numeric arrays plus
plot *metadata* — axis titles, units, series-to-axis mapping, colour index, dash
pattern — and React draws them. Every response includes inputs with units,
derived scalars, tables, warnings, interpretation text, solver information,
runtime and provenance.

---

## 6. Phase 10 — parity

**338,556 values compared. Maximum absolute and relative difference: zero.**

Tolerance `rtol = 1e-9`, taken from `tests/test_regression_baseline.py:31` —
the repository's only existing tolerance, not one invented here. Full detail,
including why the band was never consumed, in `ENGINE_PARITY_REPORT.md`.

---

## 7. Bugs found and fixed

All found by testing, none pre-existing in the physics.

| # | Bug | Found by | Consequence if shipped |
| --- | --- | --- | --- |
| 1 | `final_pos_error_m` emitted twice with unrelated meanings (chaser-to-target distance vs model gap) | parity test | A participant shown the wrong number under a plausible label |
| 2 | `POST /simulations/{id}` declared `response_model=SimulationResult` but returned `ErrorResponse` on failure, so FastAPI rejected its own error payload | 9 API tests | Every rejected input became a 500 |
| 3 | Repo-root `app.py` shadowed the backend `app` package when the root preceded the backend on `sys.path` | parity script | `'app' is not a package` in some run configurations, including Docker |
| 4 | Interpretation sentences used f-strings and printed `5e-13` | browser check | **The exact defect this migration was asked to fix**, still present |
| 5 | Charts did not reflow on resize: the oversized Plotly canvas propped its container open, so the `ResizeObserver` never fired | browser check at 375px | 426px of horizontal overflow after a tablet rotation |
| 6 | `HTTPException` responses carried no `error_class` | agent cross-check | Per-field messages lost exactly when they mattered |
| 7 | Plotly `purge` read a possibly-detached ref in cleanup | ESLint | A leaked WebGL context per chart |
| 8 | Dockerfile did not copy `orbital_core` | inventory audit | Legacy image build failure |
| 9 | Two false GMAT-validation claims in the codebase | inventory audit | A false scientific claim |

Guard tests were added for 1, 3, 4 and 5 so none can recur silently.

---

## 8. Files

### Added — backend (`web-platform/backend/`)

`app/main.py`, `app/bootstrap.py`, `app/__init__.py`;
`app/api/deps.py`, `app/api/v1/{scenarios,pilot,system}.py`;
`app/schemas/{common,scenarios,pilot}.py`;
`app/services/{catalogue,simulation,adapters,formatting}.py`;
`app/analytics/__init__.py`;
`tests/{conftest,test_engine_parity,test_api,test_formatting_parity,_parity_report}.py`;
`requirements.txt`, `pytest.ini`.

### Added — frontend (`web-platform/frontend/`, 89 files)

`src/formatting/number.ts`; `src/api/{types,client}.ts`;
`src/components/ui/` (9 primitives); `src/components/controls/` (6);
`src/components/results/` (8); `src/components/admin/` (2);
`src/charts/` (5); `src/pages/` (12); `src/state/session.tsx`;
`src/styles/{tokens,global}.css`; `src/{main,App}.tsx`;
`tests/{formatting,components}.test.tsx`; plus configs.

### Added — deployment

`web-platform/Dockerfile`, `docker-compose.yml`, `.dockerignore`,
`.env.example`.

### Added — documentation

`docs/{MIGRATION_INVENTORY,ENGINE_PARITY_REPORT,UI_MIGRATION_REPORT,IMPLEMENTATION_REPORT,MANUAL_REVIEW_CHECKLIST,KNOWN_GAPS}.md`.

### Renamed — 10 engine files

`src/*.py` → `orbital_core/*.py`, zero content change except one docstring.

### Modified — 6 files

| File | Change |
| --- | --- |
| `src/__init__.py` | Became a compatibility shim |
| `app.py` | Imports `orbital_core`; two misleading strings corrected |
| `tests/test_control_implementation.py` | False GMAT claim corrected (docstring only) |
| `Dockerfile` | Copies `orbital_core` |
| `requirements.txt` | Unchanged pins; note added |
| `.gitignore` | `node_modules`, `dist`, `app/static` |

### Not touched

Every file under `pilot/ui/`, `pilot_app.py`, `experiments/`,
`old_project_backup/`, and every engine module's content.

---

## 9. Commands used to test

```bash
# Engine suite (repository root) - 255 passed
python -m pytest -q

# Backend suite - 187 passed
cd web-platform/backend && python -m pytest -q

# Parity measurement
cd web-platform/backend && python tests/_parity_report.py

# Frontend - 77 passed
cd web-platform/frontend && npx vitest run
npx tsc --noEmit
npx eslint src --ext .ts,.tsx

# Production build
cd web-platform/frontend && npm run build

# Engine untouched
git diff pilot-platform --find-renames -M --stat -- src/ orbital_core/

# Experiments still reproduce their committed outputs
python experiments/run_cw_vs_high_fidelity.py
python experiments/run_cw_rendezvous.py
```

---

## 10. Test results

| Suite | Tests | Result |
| --- | ---: | --- |
| Engine (root) | 255 | Pass |
| Backend — engine parity | 116 | Pass |
| Backend — API behaviour | 58 | Pass |
| Backend — formatting parity | 13 | Pass |
| Frontend — formatting | 44 | Pass |
| Frontend — components | 24 | Pass |
| Frontend — chart traces | 9 | Pass |
| **Total** | **519** | **All pass** |

TypeScript: zero errors. ESLint: zero errors, zero warnings. Production build:
succeeds.

---

## 11. Verified by hand, in a browser

Served from a single origin at `http://127.0.0.1:8000` (built frontend copied
into the backend image layout, exactly as in production):

* Health reports engine 1.1.0, database reachable, 11 scenarios, no warnings.
* Landing page renders with the status and validation statements verbatim.
* Deep link `/scenarios/pd_rendezvous` resolves (client routing through the
  catch-all).
* An unknown `/api/v1/...` path returns JSON, not HTML.
* Running `two_body_orbit` renders 2 Plotly charts, metrics, interpretation,
  assumptions, limitations and downloads.
* **Zero machine notation and zero emoji on screen.** Energy drift reads
  `4.58 × 10⁻¹³`; the interpretation reads `4.6 × 10⁻¹³`.
* At 375 × 812: zero horizontal overflow, on fresh load and after resizing from
  desktop.

---

## 12. Remaining limitations

Full list in `KNOWN_GAPS.md`. The ones that would change a decision:

* **The Docker image has never been built** — no daemon was available. The
  Dockerfile is written and reviewed but unverified.
* **CI has never run.** Python 3.12 compatibility is unproven.
* **Not load-tested.** All timings are single-user.
* **Not deployed to a public host.** The PostgreSQL path is untested.
* **No screen-reader testing**, and contrast ratios are not measured.
* **Chromium only.** Firefox, Safari and Edge are unverified.
* **No end-to-end browser test suite.** The journeys were exercised by hand.
* **Still no GMAT/STK benchmark**, and none is claimed.
