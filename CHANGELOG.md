# Release notes

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/): the **engine** (`src/`) and the
**platform** (`pilot/`) are versioned separately, because a platform release
that leaves the physics untouched must be visibly distinguishable from one that
does not.

* Engine version: `orbital_core/__init__.py` → `__version__` (was `src/`)
* Platform version: `pilot/__init__.py` → `__version__`, and `PILOT_APP_VERSION`

---

## [2.0.0] — 2026-08-24 — Web platform (React + FastAPI)

A browser platform on a custom stack, replacing the Streamlit presentation
layer while reusing the validated engine unchanged. **The physics did not
change:** the engine files moved from `src/` to `orbital_core/` with `git mv`
(nine byte-identical, one docstring edit), and the new API reproduces the
engine's outputs bit-for-bit — 338,556 values compared across all scenarios
with a maximum difference of exactly zero (`docs/ENGINE_PARITY_REPORT.md`).

### Added

* **FastAPI backend** (`web-platform/backend/`) — a versioned `/api/v1` with 19
  endpoints. No physics; it imports `orbital_core` and `pilot`, so there is one
  definition of every scenario, bound and metric shared with the Streamlit app.
  Responses carry named numeric arrays plus plot metadata; no figure is rendered
  server-side.
* **React + TypeScript + Vite frontend** (`web-platform/frontend/`) — the full
  participant journey (landing, consent, sign-in, tracks, scenarios, guided
  activities, assessments, feedback) and a password-protected admin dashboard,
  reached at real URLs.
* **One shared number-formatting utility** rendering scientific notation with
  real Unicode superscripts (`2.0 × 10⁻¹²`), fixing the Streamlit interface's
  machine-notation display (`2e-12`). Presentation formatting never reaches an
  export.
* **Plotly.js charts** built in the browser from returned arrays, lazy-loaded so
  the initial bundle stays small.
* **Reproducible exports** — result CSVs (machine values), a scenario-config
  JSON, and a caveats file — plus an admin ZIP export with metric definitions.
* **Production Docker build** — multi-stage, single-origin, non-root, with the
  engine-parity suite as a build-time gate; `docker-compose.yml` for local dev.
* **Migration documentation** — `docs/MIGRATION_INVENTORY.md`,
  `docs/ENGINE_PARITY_REPORT.md`, `docs/UI_MIGRATION_REPORT.md`, plus web-platform
  `IMPLEMENTATION_REPORT`, `MANUAL_REVIEW_CHECKLIST` and `KNOWN_GAPS`.

### Changed

* `src/` → `orbital_core/` (history preserved). `src/` remains a compatibility
  shim aliasing the same module objects, so existing scripts and the test suite
  import unchanged.
* Corrected two overstated external-validation statements found during the
  migration audit: a test docstring and the legacy app's copy both implied GMAT
  validation had been performed. It has not; the wording now says so.
* **Regression-test tolerance made platform-robust.** The baseline comparison
  used `rtol = 1e-9`, which held only on the machine the baseline was captured
  on. CI on Linux (and even between two Linux runners) produced last-bit
  differences of ~1e-8 in the accumulated RK4 sums and BLAS matrix products —
  floating-point reality across CPUs/BLAS builds, not a physics change (values
  still agree to eight significant figures). The comparison now uses
  `rtol = 1e-6` plus an absolute floor `atol = 1e-11` for the machine-epsilon
  conservation drifts, which absorbs cross-platform variance while staying far
  tighter than any real regression. **No physics changed** — the engine
  reproduces bit-for-bit on a fixed machine. `requirements.txt` was also
  re-encoded from UTF-16 to UTF-8 so a fresh `pip install` works on Linux/macOS.

### Tests

* 519 automated tests: 255 engine/platform, 187 web API (including 116 parity),
  77 frontend (formatting, components, chart traces).

### Not changed

* The Streamlit platform (`pilot_app.py`, `app.py`) remains fully functional as
  a regression reference.

---

## [Platform 1.0.0] — 2026-08-23

First release of the pilot platform. **Engine unchanged at 1.1.0** — no file in
`src/` was modified, and the validated numerical results are identical.

### Added — participant experience

* Landing page stating purpose, audience and limitations, with the physics
  caveats spelled out before anyone starts.
* Privacy notice and explicit consent gate; the flow cannot be entered without it.
* Anonymous sign-in: organisation/pilot code, participant code, user type. The
  participant code is never stored — only an HMAC-SHA256 of it.
* Three tracks: guided educational activities, professional/research
  exploration, free simulation.
* Ten scenarios, each carrying its own assumptions, limitations and reading
  guide, all backed by tested engine capability.
* Three guided activities (11 steps) using a predict-then-run gate: the Run
  button stays locked until a prediction is made.
* Results views with headline numbers, plots, a plain-English interpretation,
  and assumptions and limitations shown alongside every result.
* Downloads: result CSVs, a reproducible scenario-configuration JSON, and a
  plain-text notes file carrying the caveats and any warnings raised.
* Configurable anonymous pre/post assessment (ten items) with per-item
  correctness and a confidence self-rating.
* Feedback form that adapts its questions to the participant's user type.
* Confirmation page explaining that feedback may influence later releases, and
  being explicit that no particular suggestion is promised.

### Added — analytics

* SQLAlchemy Core schema for organisations, pilot codes, anonymous participants,
  sessions, events, assessments and feedback.
* Versioned, idempotent migrations run automatically on startup.
* SQLite fallback for local development; PostgreSQL for deployment.
* Fifteen event types; anything outside the vocabulary is refused.
* Metric definitions implemented exactly as specified: unique / active /
  returning user, completed run, measured-impact user.
* `is_demo` on every observation table, set from `PILOT_ENV`, excluded from all
  reporting by default, so development records cannot be mistaken for pilot data.

### Added — administrator area

* Password-protected dashboard at `?view=admin`, credentials from environment
  secrets only.
* Filters by date range, organisation and user type.
* Reporting on organisations, users, runs, per-user counts, usage by date /
  organisation / scenario / user type / feature, return rate, pre/post
  assessment change, professional usefulness and usability, qualitative
  feedback, and bugs and feature requests.
* CSV export per table and a ZIP bundle including `METRIC_DEFINITIONS.txt`.
* System tab reporting database health, schema version and configuration
  warnings, with database passwords stripped.

### Added — verification and safety

* `tests/test_regression_baseline.py` (11 tests) locking the published numerical
  outputs at `rtol=1e-9` against a captured baseline.
* `tests/test_integrator_convergence.py` (9 tests) measuring RK4's observed
  order of accuracy and the step-independence of the published metrics.
* Platform tests: database (11), analytics and privacy (30), input validation
  (31), runner, rate limiting and auth (38), content (41), UI journey (27).
* Input validation above the engine boundary, fixing the silent-wrong-answer
  cases found in the audit.
* Per-session rate limiting and a predicted-cost check before any run.
* PBKDF2 admin credentials, constant-time comparison, lockout after five
  failures.
* GitHub Actions: full suite on Python 3.12/3.14 across Ubuntu and Windows, a
  dedicated numerical-regression job, and a secret scan.
* Dockerfile that runs the regression tests during build, so an image whose
  physics has moved cannot be produced.

### Added — documentation

`CURRENT_STATE_AUDIT.md`, `IMPLEMENTATION_REPORT.md`,
`MANUAL_REVIEW_CHECKLIST.md`, `KNOWN_GAPS.md`, and `docs/`: architecture, data
dictionary, privacy, deployment, coordinator guide, participant guide,
troubleshooting.

### Changed

* `requirements.txt` rewritten as curated UTF-8 direct dependencies. It was a
  56-line UTF-16 `pip freeze` in which direct and transitive dependencies were
  indistinguishable. Full closure moved to `requirements-lock.txt`. **Pinned
  versions are unchanged**, so results are unaffected.
* README rewritten to cover both the engine and the platform, and corrected: it
  claimed 47 tests when there were 53, said the interactive frontend was "not
  yet started" while describing it, listed three experiments when nine exist,
  and said there was no collision handling when `src/collision.py` exists.
* Migrated off `use_container_width`, deprecated for removal after 2025-12-31.

### Fixed

* Inputs that produced silently wrong answers rather than errors — negative
  altitude yielding an all-`NaN` trajectory, `NaN` propagating to `NaN` metrics,
  an over-long horizon being quietly truncated to a fifth of what was requested,
  negative durations returning two samples. All now rejected with an
  explanation. **The engine is unchanged**; the gate sits above it.
* `old_project_backup/` (22 files, the archived 2D prototype) had been deleted
  from the working tree but not committed. Restored from `HEAD`.
* Substantial uncommitted research work (`docs/`, six experiment scripts,
  `src/collision.py`, `src/external_validation.py`, three test files) committed
  before any platform work began, where a stray `git clean` would have destroyed
  it.

### Explicitly not included

Recorded in `KNOWN_GAPS.md` rather than faked:

* **GMAT/STK benchmark results.** The repository contains the workflow to run
  such a comparison; no comparison has ever been run, and none ships. The
  platform states this wherever the workflow appears.
* **Integrator comparison.** The engine has one integrator, RK4. What is
  exposed is a step-size convergence study, correctly labelled as such.
* **Chaos, resonance and stability tools.** None exist in this engine.

### Security notes

* No secret is committed. `.env` is git-ignored, `.env.example` holds
  placeholders only, and CI fails if a credential or database file is tracked.
* Tracebacks are never shown to participants.
* Logs contain event types, error class names and timings — never participant
  codes, orbital parameters, feedback text, or database passwords.

---

## [Engine 1.1.0] — before this work

The state audited in `CURRENT_STATE_AUDIT.md`: CW relative motion with a PD
controller, validated against a non-linear two-body + J2 + drag reference;
Streamlit MVP (`app.py`); nine experiment scripts; 53 passing tests.

Retained unchanged. Headline results: 400 km period 5553.62 s; two-body energy
drift ~1.1 × 10⁻¹³ over two orbits; CW-vs-reference free-drift divergence
2.78 m (two-body) and 5.81 m (with J2); PD rendezvous Δv 2.326 m/s converging in
about 1595 s.
