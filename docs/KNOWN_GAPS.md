# Known gaps — custom web migration

What is missing, unverified or provisional in the React + FastAPI platform.

Honest by policy: an item appears here if it is not finished, not tested, or
needs your judgement. Nothing here is described as complete when it is not.

For gaps in the **science and the pilot programme** (educational content review,
the assessment design, retention scheduling), see the root-level
[`KNOWN_GAPS.md`](../KNOWN_GAPS.md) — those are unchanged by this migration and
still apply.

---

## 1. Still no external validation

**No GMAT or STK benchmark exists in this repository.** This has not changed and
is not claimed anywhere in the new interface.

`orbital_core/external_validation.py` provides the workflow — export a scenario,
import an external ephemeris, compute errors — and the API exposes the export
half. **No comparison has ever been run.** The scenario, the exports, the
landing page and the API description all say so explicitly.

Two false claims were found during the migration audit and corrected:

* `tests/test_control_implementation.py` asserted "GMAT validates the nonlinear
  propagator and the LVLH relative-state reconstruction". It never did.
* `app.py` described the workflow in the present tense as something that
  "validates the engine's higher-fidelity propagator".

**Requires your judgement.** To claim "benchmarked against NASA GMAT under
matched assumptions", the comparison must actually be run: export a two-body or
J2-only scenario, reproduce the initial state at full precision in GMAT, match
the force model and step, export the ephemeris, compare.

---

## 2. Not verified

### 2.1 The Docker image has never been built

No Docker daemon was available in this environment. `web-platform/Dockerfile`
is written and reviewed but **unbuilt and unrun**. Specifically unverified:

* that `node:22-alpine` builds the frontend cleanly,
* that the `sed` stripping Streamlit and matplotlib from the engine
  requirements leaves a working set,
* that `psycopg[binary]==3.2.3` installs on `python:3.12-slim`,
* that the build-time parity gate passes inside the image,
* that the health check endpoint responds within the start period.

**Build it once before relying on it.** The single-origin serving it produces
*was* verified by hand, by copying the built frontend into the backend's static
directory and running uvicorn — which is exactly what the image does.

### 2.2 CI has never run

`.github/workflows/tests.yml` covers the engine suite but has **not been
extended** to the backend or frontend suites, and has never executed on GitHub.
**Python 3.12 compatibility is unproven** — the code uses `X | None` under
`from __future__ import annotations`, which is fine on 3.12, but nothing has
proven it.

### 2.3 Not deployed to a public host

Everything was verified locally against SQLite. **The PostgreSQL path is
untested**: no query has run against Postgres, the `psycopg` driver has not been
exercised, and secrets injection on a real host has not been tried. All queries
are SQLAlchemy Core and portable, but portable-in-principle is not tested.

Run `python scripts/init_db.py --check` against your real `DATABASE_URL` before
going live.

### 2.4 Not load-tested

Never run with more than one concurrent user. Measured runtimes (46 ms to
2.5 s per scenario) are single-user. One uvicorn worker by default, because the
engine is CPU-bound pure Python.

**Note the interaction with rate limiting:** the limiter is in-process, so
`WEB_CONCURRENCY=4` means four times the configured limit.

### 2.5 Browser and accessibility coverage is thin

* **Verified:** Chromium at 1280×720, 850px and 375×812. Zero horizontal
  overflow. Charts reflow on resize.
* **Not verified:** Firefox, Safari, Edge; real mobile devices; touch
  interaction with 3D Plotly charts, which is awkward on small screens.
* **No screen-reader testing.** The ARIA wiring is mechanically verified —
  every control labelled, errors announced, tabs with roving tabindex — but
  nobody has listened to it. Plotly charts are largely opaque to screen readers;
  every result is also given as text and as downloadable numbers, which
  mitigates but does not solve this.
* **Contrast ratios are not measured** against WCAG AA. The palette is
  Okabe-Ito and colour is never the only signal, but the actual ratios are
  unmeasured.

### 2.6 No end-to-end browser test suite

The participant journeys were exercised **by hand** in a browser. There is no
Playwright or Cypress suite, so a regression in a page-level flow would not be
caught automatically. Component and API behaviour *are* covered (519 tests).

The Streamlit version has 27 automated journey tests via `AppTest`. The React
version does not yet have an equivalent.

---

## 3. Incomplete features

### 3.1 External-ephemeris upload is export-only

The Streamlit app had a three-step workflow: build and export, run externally,
**upload and compare**. The API exposes steps 1 and 2. **Step 3 — uploading an
external CSV and computing the comparison — is not implemented.**

`orbital_core.external_validation.load_external` and `compare_ephemerides`
exist and are tested (8 tests); no endpoint calls them. `python-multipart` is
installed in anticipation.

Until then, the Streamlit app at `?view=` remains the only way to run the
comparison through a UI. The engine functions can also be called directly.

### 3.2 Ten parameters were unvalidated server-side; now bounded, not range-checked

`pilot/runner.py::_VALIDATED_KEYS` covers 11 names. Ten are not validated by
`pilot/validation.py`: `keep_out_radius`, `collision_radius`, `debris_x/y/z`,
`omega_n_factor`, and five booleans/selects. `debris_x/y/z` had **no bound at
any layer** in the Streamlit version.

The API schema layer now bounds them, but they still do not get the *explanatory*
messages the validated parameters produce. Moving them into
`pilot/validation.py` would fix that properly and would benefit both frontends.

### 3.3 The admin organisation filter needs an id the report does not carry

`AdminPage` calls `GET /admin/organisations` separately to populate the filter.
It works, but it is a second round-trip that a joined field would remove.

### 3.4 No theme toggle

`tokens.css` defines a complete light theme under `:root[data-theme='light']`,
but **nothing sets that attribute.** The application is dark-only in practice.
The light theme exists for projection and printing and is one control away.

---

## 4. Deliberate omissions

| Item | Why |
| --- | --- |
| Streamlit application | **Kept and functional** — the regression reference the brief requires |
| `orbital_core/plotting.py` | Matplotlib helpers for experiment scripts; the web app plots client-side |
| Server-side chart rendering | The brief asks for numeric data with React controlling presentation |
| `old_project_backup/` | Archived 2D prototype, untouched |
| No-JavaScript fallback | Streamlit required JS too; the engine remains runnable from the command line |

---

## 5. Needs your judgement

### 5.1 Which application is the deployment target

Both now work. The Streamlit app is the regression reference and still has the
only external-ephemeris upload UI; the React app is the professional interface.
Running both means two deployments and two databases unless they share
`DATABASE_URL` — which they safely can, since both use the same schema and the
same analytics definitions.

### 5.2 The `src/` shim's lifespan

`src/` aliases `orbital_core` so the test suite and `experiments/` keep working
unchanged. That is deliberate for this migration — it is what makes the suite a
regression check. But it is also a second name for one thing, and it should
probably be removed once you are satisfied, by updating those imports.

### 5.3 Validation still protects the platform, not `orbital_core`

Unchanged from before: anyone importing `orbital_core.scenarios` directly still
gets the old behaviour, including a negative altitude returning an all-`NaN`
trajectory with no error. Both frontends are protected because both go through
`pilot.validation`. A script is not.

### 5.4 The convergence scenario uses a different grid from the experiment script

In-app: 126–4001 samples. `experiments/run_convergence_study.py`: 251–16001.
Both are called "the convergence study" in the documentation. Preserved as-is
rather than silently reconciled.

### 5.5 Three experiment scripts write outside the repository

`run_energy_matching_audit.py`, `run_robustness_gain_horizon.py` and
`run_terminal_assessment.py` write to `os.path.dirname(ROOT)` — the folder
*above* the repository. None of their CSVs exists on disk, so results quoted
from those studies have no saved artefact. Not changed here, because fixing it
alters research-script behaviour.

---

## 6. Smaller items

* **Plotly is 4.67 MB.** Lazy-loaded in its own chunk, so the initial bundle is
  21 kB — but the first chart render downloads it. A lighter library, or a
  subset build, would help on a slow connection.
* **The 200-row display cap** on result tables is a judgement call. The note
  says how many rows are hidden and that the download has all of them.
* **`_ERROR_CLASSES`** maps status codes to names by hand; a new status code
  falls back to `"HTTPError"`.
* **The frontend mirrors `pilot/validation.py`'s code rules** in
  `SignInPage.tsx` for instant feedback. Two copies of one rule; the server is
  authoritative, but they could drift.
* **`retention_days` and `log_raw_parameters` are not exposed by `/health`**, so
  the consent page states the documented defaults. If a deployment changes
  either, the notice will be wrong until the API surfaces them.
* **No `package-lock.json` audit.** Dependencies are pinned but `npm audit` has
  not been run.

---

## 7. Suggested order of work

**Before deploying**

1. Build the Docker image once (§2.1).
2. Extend CI to the backend and frontend suites and let it run (§2.2).
3. Test against real Postgres with `init_db.py --check` (§2.3).
4. Decide which application is the deployment target (§5.1).

**Before a pilot with real participants**

5. Review the educational content — unchanged from the root `KNOWN_GAPS.md` §3.
6. Screen-reader and contrast testing (§2.5).
7. Load-test with a realistic cohort (§2.4).

**To finish the migration**

8. Implement the external-ephemeris upload endpoint (§3.1).
9. Add an end-to-end browser suite (§2.6).
10. Move the ten unvalidated parameters into `pilot/validation.py` (§3.2).
11. Add the theme toggle (§3.4).

**To strengthen the science**

12. Actually run the GMAT or STK comparison (§1).
