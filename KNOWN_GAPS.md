# Known gaps

What is missing, incomplete, provisional or unverified. Read this before
running a real pilot.

Ordered by how much it should affect a decision to use the platform.

---

## 1. Things the brief asked for that do not exist

Three capabilities implied by the pilot specification are **not present in this
repository**. They were checked for specifically, and are documented rather than
faked.

### 1.1 GMAT / STK benchmark results — do not exist

**Status: absent. Nothing here has been benchmarked against an external tool.**

`src/external_validation.py` provides the *plumbing* for such a comparison —
export a scenario, import an external tool's ephemeris, compute position and
velocity error. It has been there since before this work and is well tested
(8 tests). But **no GMAT or STK run has ever been performed**, no external
ephemeris is committed, and `outputs/` contains no comparison.

The repository's own physics write-up says so: *"the high-fidelity model itself
is still not validated against an external tool (still future work)"*
(`docs/results_summary.md`).

The platform therefore presents this as a **workflow you can carry out**, never
as a result. The scenario, the exports, the README and the landing page all say
explicitly that no benchmark ships with the software, and the project is not
affiliated with NASA or Ansys/AGI.

**Requires your judgement.** If you want the claim "benchmarked against NASA
GMAT under matched assumptions", you must actually run the comparison: export a
two-body or J2-only scenario from the platform, reproduce the initial state to
full precision in GMAT, propagate with a matched force model and step, export the
ephemeris, and compare. That would be a genuine contribution — and until it is
done, the claim cannot be made.

### 1.2 Numerical-integrator comparison — does not exist

The engine has **one** integrator: fixed-step classical RK4. There is no Euler,
Verlet, adaptive or symplectic scheme to compare against.

The archived 2D prototype (`old_project_backup/`) did compare Euler, RK4 and
velocity-Verlet, and used SciPy's adaptive RK45 as truth. That was deliberately
dropped when the 3D engine standardised on a single deterministic integrator.

What the platform exposes instead is a **step-size convergence study** — RK4
against the exact closed-form CW solution, measuring the observed order of
accuracy. That is real, is tested, and is labelled as what it is. It is not an
integrator comparison.

### 1.3 Chaos, resonance and stability tools — do not exist

No Lyapunov exponents, Poincaré sections, resonance analysis, stability maps or
bifurcation tooling anywhere in `src/`. Nothing of the kind has been written.

---

## 2. Needs your scientific judgement

### 2.1 The reference model is not ground truth

The CW-vs-reference comparison measures the linear model against a *more
faithful* model, not against reality. The reference has no higher-order gravity,
no third-body perturbations, no solar radiation pressure, and a crude
atmosphere — and, per §1.1, has never been checked against an independent tool.

Every number of the form "CW is wrong by 2.78 m" means "CW differs from *this
engine's* non-linear model by 2.78 m". The platform words it as "the gap between
the two models" rather than "the error", but a reader may still take it as
truth. **Whether that framing is adequate for your audience is your call.**

### 2.2 The drag result is partly an artefact of the setup

The finding that drag is nearly common-mode and adds only ~0.004 m of relative
error is true **for identical spacecraft**. The toggle study gives target and
chaser the same `Cd·A/m` by construction, so differential drag cannot appear.

Real formations of dissimilar vehicles would behave differently, and differential
drag is used deliberately for formation control. The engine cannot currently
exercise this: `high_fidelity_relative` applies one `ForceModel` to both bodies.
Supporting per-vehicle ballistic coefficients is a small change to `src/`, which
is why it was **not** made here.

### 2.3 The atmosphere model is calibrated at one altitude

Single-scale-height exponential, ρ₀ ≈ 3.9 × 10⁻¹² kg/m³ at 400 km, H = 60 km.
Real thermospheric density varies by an order of magnitude with solar activity,
local time and geomagnetic conditions.

The platform permits altitudes from 150 km and warns below 200 km, but any drag
result away from 400 km is indicative only. **Whether the 150 km floor is the
right cut-off is a judgement call** — it was chosen because below that a
spacecraft is effectively re-entering.

### 2.4 The validation bounds in `pilot/validation.py` are judgement calls

The following were chosen by reading the repository's own characterisation work,
not derived from anything:

| Bound | Value | Basis |
| --- | --- | --- |
| Minimum altitude | 150 km | Below this, re-entry; atmosphere model meaningless |
| Maximum altitude | 40 000 km | Beyond GEO the drag model is nonsense |
| CW "trustworthy" separation | 1 km | The published agreement figures are at ≤ 1 km |
| CW "characterised" separation | 50 km | `run_separation_sweep.py` runs to 50 km |
| Under-resolution warning | < 200 steps/orbit | Published results use 500–2000 |
| Maximum duration | 50 orbits | Practical, not physical |

These drive the warnings participants see. **Please review them.** They are all
single constants in one module.

### 2.5 The cost model is measured on one machine

`SECONDS_PER_STEP` in `pilot/validation.py` was measured on the development
machine (Windows, Python 3.14.6). A slower host will exceed the predicted time;
a faster one will reject runs it could have handled.

The step limit (`PILOT_MAX_SIMULATION_STEPS`) is the harder guarantee. If your
host is much slower, lower both limits.

---

## 3. Educational content needs teacher review

**Both the guided activities and the assessment instrument are marked DRAFT and
have not been reviewed by a subject teacher.**

### 3.1 What has been verified

* Every activity step runs successfully through the same validated path.
* **Every number quoted to a learner is asserted against the engine** by
  `tests/test_platform_content.py` — the 109-minute period at 1200 km, the
  −3770 m radial drift, the 1595 s / 2.33 m/s / 0.017 m/s² rendezvous figures,
  the 2.8 m → 5.8 m J2 gap, and the claim that radial is far worse than
  along-track.
* Each assessment item is answerable from the activities, has exactly one
  correct option, and carries an explanation.

### 3.2 What has not been verified — by anyone

* **Reading level.** Not measured. Some explanations are long.
* **Curriculum alignment.** Written against the physics, not against any
  specification.
* **Item quality.** No trial, no discrimination indices, no distractor analysis.
  Some distractors may be obviously wrong to a weak student, and some questions
  may be answerable by elimination without understanding.
* **Timing.** The 20–25 minutes per activity is an estimate, untested with real
  learners.
* **Age-appropriateness.** No judgement has been made about what age group this
  suits.

### 3.3 The assessment design has a built-in confound

The **same items** are used before and after. That is the most sensitive design
for small cohorts, but it means part of any measured gain comes from having seen
the questions already, not from learning. There is no control group and no
counterbalancing.

The dashboard states this next to the numbers. It is a real limitation of the
measurement, not a presentational one. **Do not report the gain as a
teaching-effect size.**

### 3.4 Specific items to look at

* `q4_drag` (drag makes a satellite go *faster*) — correct, but "counter-
  intuitive and correct" can read as "trick question".
* `q8_radial_drift` — the most valuable item and the hardest. May be too hard as
  a *pre*-test item.
* `q5_ballistic` — needs the idea that drag scales with area over mass. Covered
  in the activity, but a pre-test taker will likely guess.

Both content files are plain Python data structures; editing the text needs no
programming. **If you change assessment items, bump the instrument `version`**,
or before-and-after data from two different tests will silently pool.

---

## 4. Not verified

Honest inventory of what has *not* been checked.

### 4.1 Not load-tested

Never run with more than one concurrent user. All timings are single-user.

The architecture makes it predictable — one Streamlit process, one simulation at
a time — but "a class of 30 works fine" is **reasoning, not measurement**.

### 4.2 Not deployed to a public host

Everything was verified locally. Streamlit Community Cloud with managed
Postgres, the recommended configuration in `docs/DEPLOYMENT.md`, has **not been
exercised**. Untested consequences: PostgreSQL-specific SQL behaviour (all
queries are SQLAlchemy Core and portable, but only SQLite has actually run
them), the `psycopg` driver path, secrets injection, and cold-start behaviour.

`python scripts/init_db.py --check` against your real `DATABASE_URL` before
going live is the cheapest way to catch most of this.

### 4.3 Docker image not built

The `Dockerfile` is written and reviewed but **has not been built or run** — no
Docker daemon was available. Its dependency versions match the tested
environment and it runs the regression tests during build, but the build itself
is unverified. Build it once before relying on it.

### 4.4 CI workflow not executed

`.github/workflows/tests.yml` has not run on GitHub. The secret-scan shell
commands were executed locally and behave correctly; the matrix job (Python
3.12 on Ubuntu) has not. **Python 3.12 compatibility is untested** — the code
uses `X | None` annotations under `from __future__ import annotations`, which is
fine on 3.12, but nothing has proven it.

### 4.5 Browser and accessibility coverage is thin

* Verified: Chromium at 1280×720 and at 375×812 (no horizontal overflow).
* Not verified: Firefox, Safari, Edge; real mobile devices; touch interaction
  with 3D Plotly charts, which are awkward on small screens.
* **No screen-reader testing.** Plotly charts are largely opaque to screen
  readers; every result is also given as text and as downloadable numbers, which
  mitigates but does not solve this.
* Colour choices use the Okabe-Ito palette and pair colour with dash patterns,
  but contrast ratios have not been measured against WCAG.

### 4.6 Rate limiting is per-process

Counters live in memory. With multiple app replicas, each keeps its own, so the
effective limit multiplies by the replica count. Fine for a single-process
Streamlit deployment; wrong if you scale horizontally.

### 4.7 Retention is not automatic

`scripts/purge_expired.py` exists and works, but **nothing runs it**. The
consent page's retention promise is only true if a coordinator schedules it.

### 4.8 No backup or restore procedure has been exercised

`docs/DEPLOYMENT.md` §7 suggests `pg_dump`. Untested.

---

## 5. Engine limitations carried forward

Pre-existing, unchanged, and out of scope for platform work:

* **No impulsive, LQR or MPC guidance** — continuous-thrust PD only.
* **No navigation or estimation layer** — perfect noise-free state knowledge.
  Nothing here says anything about robustness to sensor error.
* **No eccentric reference orbits** — CW assumes circular. Real missions often
  are not.
* **No collision avoidance** — detection only. No path planning, no manoeuvre
  generation, no covariance, so it is not a conjunction-assessment tool.
* **Fixed-step RK4 only** — no adaptive stepping or error control.
* **No J2-aware relative-motion model** (e.g. Schweighart–Sedwick) to compare
  against plain CW.

---

## 6. Platform limitations

### 6.1 Validation protects the platform, not `src/`

The input validation that fixes the silent-wrong-answer cases lives in
`pilot/validation.py`, **above** the engine boundary. Anyone importing
`src.scenarios` directly still gets the old behaviour: a negative altitude still
returns an all-`NaN` trajectory with no error.

This was deliberate — fixing it inside `src/` would have meant editing validated
physics. **It remains a real trap for anyone scripting against the engine
directly.** Fixing it properly means adding guards to `src/` and re-verifying
the numerical baseline.

### 6.2 Free-text feedback is the one place personal data can enter

Everything else is anonymous by construction. Feedback boxes store text
verbatim. The form warns against personal information; it cannot enforce it.

### 6.3 Small cohorts are not truly anonymous

With a handful of participants, pilot code plus timing can narrow down who a row
belongs to. Inherent to any small pilot. Report at cohort level.

### 6.4 A shared browser can resume a session

Session state is Streamlit's. Someone returning to an un-closed browser tab
continues the previous participant's session. Nothing sensitive is exposed, but
their runs would be attributed to the earlier participant.

### 6.5 Consent is recorded, not evidenced

The consent version and timestamp are stored against the anonymous participant.
Because the record is anonymous, it cannot serve as proof that a *specific
person* consented. If your institution needs auditable individual consent, that
must be handled outside this platform.

---

## 7. Corrected during this work — check nothing was lost

* `old_project_backup/` (22 files, the archived 2D prototype) had been **deleted
  from the working tree** but not committed. Restored from `HEAD`. **Please
  confirm you did not intend to delete it.**
* Substantial uncommitted research work — `docs/`, six experiment scripts,
  `src/collision.py`, `src/external_validation.py`, three test files — was
  untracked and one `git clean` from destruction. Committed as `3294749` before
  any platform work.
* A stray git repository exists at `<project-root>` — your entire home
  directory, with no commits and a stale index referencing a
  `Desktop/Orbital Stability Platform` folder that no longer exists. It is
  unrelated to this project and was **left untouched**, but it is a hazard:
  running `git` from a parent directory silently targets it. Consider removing
  `~/.git` once you have confirmed nothing depends on it.

---

## 8. Suggested order of work

**Before a pilot with real participants**

1. Review the activities and assessment items (§3), or run without assessments.
2. Confirm your institution's ethics/safeguarding position (`docs/PRIVACY.md` §10).
3. Deploy to the real host and verify with `init_db.py --check` (§4.2).
4. Review the validation bounds (§2.4).
5. Schedule `purge_expired.py` (§4.7).

**Before claiming external validation**

6. Actually run the GMAT or STK comparison (§1.1).

**To strengthen the science**

7. Per-vehicle ballistic coefficients, so differential drag can be exercised (§2.2).
8. Eccentric reference orbits.
9. A navigation-error layer and Monte-Carlo dispersions.
10. Impulsive and optimal guidance for comparison against PD.

**To harden the platform**

11. Build and run the Docker image (§4.3); let CI run once (§4.4).
12. Load-test with a realistic cohort (§4.1).
13. Move input guards into `src/` and re-verify the baseline (§6.1).
