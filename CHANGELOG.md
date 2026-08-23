# Release notes

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/): the **engine** (`src/`) and the
**platform** (`pilot/`) are versioned separately, because a platform release
that leaves the physics untouched must be visibly distinguishable from one that
does not.

* Engine version: `src/__init__.py` → `__version__`
* Platform version: `pilot/__init__.py` → `__version__`, and `PILOT_APP_VERSION`

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
