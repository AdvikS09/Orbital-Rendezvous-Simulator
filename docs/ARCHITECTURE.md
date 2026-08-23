# Architecture

How the pilot platform is put together, and why.

## 1. The governing constraint

The simulation engine in `src/` produces validated numerical results that are
cited in `README.md`, `research_log.md` and `docs/results_summary.md`. Those
numbers must not move. Everything about this architecture follows from that.

The engine already had a clean boundary — `src/scenarios.py`, described in its
own docstring as "the backend boundary for any UI", returning plain dicts and
DataFrames with no plotting, no Streamlit and no file I/O. The platform was
built **on top of** that boundary rather than into it. No file in `src/` was
modified.

## 2. Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  pilot_app.py                              Streamlit entry point │
│  routes by session stage; ?view=admin for the dashboard          │
├─────────────────────────────────────────────────────────────────┤
│  pilot/ui/          flow · workspace · results · assessment ·    │
│                     feedback · admin · figures · common          │
│                     presentation only, no physics, no SQL        │
├─────────────────────────────────────────────────────────────────┤
│  pilot/runner.py    THE single simulation path:                  │
│                     validate → rate-limit → run → record         │
├───────────────┬─────────────────┬───────────────┬───────────────┤
│ validation.py │ ratelimit.py    │ analytics.py  │ export.py     │
│ input bounds  │ per-session     │ anonymous     │ CSV + config  │
│ + warnings    │ quotas          │ events        │ JSON          │
├───────────────┴─────────────────┼───────────────┴───────────────┤
│  pilot/content/                 │  pilot/db.py · models.py       │
│  scenarios_catalog · activities │  SQLAlchemy Core, migrations   │
│  assessments                    │  SQLite (dev) / Postgres       │
├─────────────────────────────────┴───────────────────────────────┤
│  src/scenarios.py               ← the frozen boundary            │
├─────────────────────────────────────────────────────────────────┤
│  src/  orbit · perturbations · lvlh · cw · controller ·          │
│        validation · external_validation · collision              │
│        THE VALIDATED ENGINE — unmodified                         │
└─────────────────────────────────────────────────────────────────┘
```

Dependencies point downwards only. `src/` imports nothing from `pilot/`, so the
engine remains usable — and testable — with no platform code present at all.

## 3. Why Streamlit was kept

The repository already used Streamlit, and the existing `app.py` was a genuinely
thin frontend with no physics in it. Migrating to FastAPI plus a JavaScript
frontend would have meant rewriting every plot and every form to reach the same
place, for these gains: better concurrency, finer layout control, a REST API.

None of those are the pilot's bottleneck. What the pilot needs is a public URL
that works without installing Python, which Streamlit Community Cloud provides
free. The cost of staying is real and is documented in §7.

`app.py` is untouched and still runs — it is the original research MVP.
`pilot_app.py` is the new participant-facing entry point. Both call the same
`src/scenarios.py`.

## 4. Request path

Every simulation, from every track, follows exactly one path:

1. **`pilot/ui/workspace.py`** collects parameters through widgets generated
   from the scenario's declared `Param` list.
2. **`pilot/runner.validate_params`** bounds-checks them. Failure stops here and
   returns explanations, not a traceback.
3. **`pilot/ratelimit`** consumes one run from the session's allowance.
4. **`simulation_started`** is recorded.
5. The scenario's runner calls `src/scenarios.py`.
6. **`simulation_completed`** or **`simulation_failed`** is recorded, with the
   measured duration.
7. **`pilot/ui/results.py`** renders numbers, plots, an interpretation, the
   assumptions and limitations, and the downloads.

Consequences worth stating: analytics cannot be bypassed by a UI change; a
scenario cannot be added without assumptions and limitations, because they are
required fields on the dataclass; and an engine exception can never reach the
participant as a traceback.

## 5. Why input validation lives outside `src/`

The audit found that `src/scenarios.py` accepts inputs producing *silently wrong
answers* — a negative altitude yields an all-`NaN` trajectory with no error, an
over-long horizon is quietly truncated to a fifth of what was requested.

Fixing that inside `src/` would have meant editing validated physics code. The
gate went into `pilot/validation.py` instead, above the boundary. The engine
behaves exactly as it always did; the platform simply refuses to ask it
questions it answers badly. Anyone using `src/` directly still gets the old
behaviour, which is recorded in `KNOWN_GAPS.md` as a real remaining risk.

## 6. Anonymity by construction

Participant identity is `HMAC-SHA256(server_salt, PILOT_CODE|PARTICIPANT_CODE)`.

* The same person typing the same code on a later date is recognised, which is
  what makes "returning user" measurable.
* The stored value cannot be reversed to the code they chose.
* Two deployments with different salts produce different values, so records
  cannot be correlated across them.

This is reinforced structurally: the schema has **no column** that could hold a
name, email address, age, IP address or user agent, and a test asserts that
(`test_no_column_can_hold_an_email_or_a_name`). Analytics detail payloads are
filtered against an allow-list of derived, bucketed keys, so a participant's
orbital parameters cannot be reconstructed from the event log.

## 7. Known architectural limitations

| Limitation | Consequence | Mitigation |
| --- | --- | --- |
| Streamlit reruns the whole script on every interaction | Results are held in `st.session_state`; a very large result bundle costs memory per session | Step budget caps result size |
| One Python process, cooperative concurrency | A long simulation occupies a worker and other users wait | Predicted-cost check rejects runs over ~20 s before they start |
| Rate limiter is in-process | With multiple app processes each keeps its own counters, so the effective limit multiplies | Documented; a hosted Streamlit pilot runs a single process |
| SQLite fallback on an ephemeral filesystem | Data lost on restart | Production warning in the admin area; `DATABASE_URL` required for real pilots |
| No CSRF/session binding beyond Streamlit's own | A shared browser could resume another participant's session | Sessions are anonymous; nothing sensitive is exposed |
| The engine is pure-Python loops | Runtime scales linearly with steps | Cost model in `pilot/validation.py`, measured not guessed |

## 8. Extension points

* **A new scenario** — add a `Scenario` to `pilot/content/scenarios_catalog.py`
  with its parameters, assumptions and limitations. The UI, validation, cost
  model, analytics and downloads all follow automatically. A test asserts every
  scenario runs with its defaults.
* **A new activity** — add an `Activity` to `pilot/content/activities.py`. Tests
  assert every step references a real scenario and runs.
* **A new assessment** — add an `Instrument` to `pilot/content/assessments.py`.
  `instrument_id` and `instrument_version` are stored with every result, so
  revising one does not silently pool incomparable data.
* **A schema change** — append a numbered entry to `MIGRATIONS` in
  `pilot/db.py`. Never edit an applied migration.

## 9. Testing strategy

| Layer | Approach |
| --- | --- |
| Engine | Closed-form solutions, conservation laws, and a regression baseline pinned at `rtol=1e-9` |
| Integrator | Measured order of accuracy, plus step-independence of the two published metrics |
| Platform | Real temporary SQLite database per test, module caches reset around each |
| User journey | Streamlit's `AppTest` executes the real `pilot_app.py` and drives its widgets |
| Content | Every number quoted to a learner is asserted against the engine |
| Secrets | CI greps tracked files for credentials and database files |
