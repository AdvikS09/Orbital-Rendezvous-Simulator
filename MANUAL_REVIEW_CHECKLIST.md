# Manual review checklist

Everything to test by hand, with the expected outcome for each step.

Automated tests cover 255 cases including the full participant journey, so this
checklist deliberately concentrates on what a machine cannot judge: whether the
wording is honest, whether an explanation actually helps, and whether the thing
behaves sensibly when a real person does something unexpected.

**Setup for local review**

```bash
python -m pip install -r requirements.txt
python scripts/hash_password.py          # note the hash it prints
```

Then set `PILOT_ADMIN_USERNAME` and `PILOT_ADMIN_PASSWORD_HASH`, and:

```bash
streamlit run pilot_app.py
```

Tick each box. Anything marked **[JUDGEMENT]** needs your opinion, not a
pass/fail.

---

## A. Landing page

| # | Step | Expected |
| --- | --- | --- |
| A1 | Open the app | Landing page; title and one-line description of what it does |
| A2 | Read "What you can do here" | Eight capabilities, all of which the platform actually has |
| A3 | Read "What this is not" | Explicit: not NASA-approved, not NASA-validated, not flight-certified, not operational mission software, not for operational flight decisions |
| A4 | Read the validation statement | States clearly that it has **not** been benchmarked against GMAT or STK, and that no such comparison ships |
| A5 | Expand "Known limitations of the physics" | Seven limitations: linear model, imperfect reference, crude atmosphere, perfect navigation, one controller, no avoidance, identical spacecraft |
| A6 | **[JUDGEMENT]** | Could a reader come away believing this is validated, professional or flight software? If yes, the wording needs strengthening |
| A7 | Expand the footer "About this software" | Same status statement plus the privacy summary; reachable from every page |
| A8 | Press **Start** | Moves to the privacy notice |

---

## B. Privacy notice and consent

| # | Step | Expected |
| --- | --- | --- |
| B1 | Read the summary box | Says exactly what is and is not recorded; names email, IP, browser, exact age as never recorded |
| B2 | Confirm the "Continue" button | **Disabled** until the box is ticked |
| B3 | Try clicking Continue without ticking | Nothing happens; caption says the box must be ticked |
| B4 | Expand "The full detail" | Explains why data is collected, how the hashing works, that parameters are not stored, retention period, and how to withdraw |
| B5 | Check the withdrawal wording | Says records cannot be individually deleted because they are anonymous, and offers closing the page or running locally instead |
| B6 | Tick the box | Continue becomes enabled |
| B7 | Press **Back** | Returns to the landing page with nothing recorded |
| B8 | **[JUDGEMENT]** | Would this satisfy your institution's requirements? See `docs/PRIVACY.md` §10 |

---

## C. Sign-in

| # | Step | Expected |
| --- | --- | --- |
| C1 | Read the participant-code help | Tells the user to make one up, write it down, and not use a name or email |
| C2 | Enter participant code `alice@example.org` | Rejected: "That looks like an email address…" |
| C3 | Enter `ab` | Rejected: must be at least 3 characters |
| C4 | Enter `Alice Smith` | Rejected: letters, digits, hyphens and underscores only |
| C5 | Enter a 50-character code | Rejected: at most 40 characters |
| C6 | Leave the pilot code blank, submit | Rejected: "Enter the organisation or pilot code you were given" |
| C7 | Enter pilot `DEMO-A`, participant `BLUE-FALCON-42`, type Student | Proceeds; a blue notice says the code is not registered and the session is grouped under "Unregistered pilot code" |
| C8 | Check the sidebar | Shows an 8-character reference (e.g. `69d24108`) — **not** the code you typed |
| C9 | Confirm nothing was recorded before submitting | Caption says so; the participant row appears only after Continue |

**Edge cases**

| # | Step | Expected |
| --- | --- | --- |
| C10 | Lower-case `demo-a` / `blue-falcon-42` | Same participant as C7 — codes are case-insensitive |
| C11 | Same participant code, **different** pilot code | A **different** participant — identity is per-pilot |
| C12 | Reload mid-flow and sign in again with the same codes | Recognised as the same participant; no duplicate row |

---

## D. Track selection

| # | Step | Expected |
| --- | --- | --- |
| D1 | View the three cards | Guided, Professional/research, Free — each with a description and who it suits |
| D2 | Check the suggestion | "suggested for you" is on Guided for a student, Professional for an engineer/researcher |
| D3 | Confirm nothing is forced | All three are selectable regardless of user type |
| D4 | Choose **Guided** as a student | Goes to the pre-assessment |
| D5 | Choose **Free** | Goes straight to the workspace, no assessment |

---

## E. Guided activities

| # | Step | Expected |
| --- | --- | --- |
| E1 | Pick "What keeps a satellite up…" | Step 1 of 4, with a progress bar |
| E2 | Confirm the Run button is **locked** | Disabled; caption says a prediction is needed |
| E3 | Choose a prediction | Run becomes enabled |
| E4 | Press Run | Spinner, then results in about a second |
| E5 | Check "What to look for" | Appears **above** the results |
| E6 | Check the metrics | Period ≈ 92.6 min; energy drift and period error ~1e-13 |
| E7 | Check the altitude plot | Flat line at 400 km (axis range will be tiny) |
| E8 | Check the takeaway | Appears **after** the results |
| E9 | Press **Next step** | Step 2; previous results cleared |
| E10 | Step 2 at 1200 km | Period ≈ 109 min — longer than at 400 km |
| E11 | Step 3 (J2) | Divergence grows steadily rather than oscillating |
| E12 | Step 3 with inclination 0° | Different result — J2 depends on inclination |
| E13 | Step 4 (drag) | Altitude trends downwards |
| E14 | Step 4 with area doubled | Roughly double the decay |
| E15 | **Activity 2, step 1** (1 km along-track) | Chaser barely moves — under 1 m of drift |
| E16 | **Activity 2, step 2** (100 m radial) | Ends ≈ −3770 m along-track. **The most important step in the whole sequence** |
| E17 | Activity 2, step 3 (cross-track) | Clean sine wave, one cycle per orbit, returns to +200 m; in-plane stays zero |
| E18 | Activity 3, step 1 | Converges ≈ 1595 s, Δv ≈ 2.33 m/s, peak ≈ 0.017 m/s²; path is **curved**, not straight |
| E19 | Activity 3, step 2 (bandwidth 12) | Faster convergence, higher Δv and higher peak thrust |
| E20 | Activity 3, step 3 (gain sweep) | Two heatmaps; fastest and cheapest are **different** cells; some blanks |
| E21 | Activity 3, step 4, J2 off then on | ≈ 2.8 m, then ≈ 5.8 m |
| E22 | Press **Finish activity** | Offers the post-assessment |
| E23 | **[JUDGEMENT]** | Is the reading level right for your learners? Are the explanations too long? |
| E24 | **[JUDGEMENT]** | Do the predictions have a plausible-but-wrong option, or is the answer obvious? |

---

## F. Professional / research workspace

| # | Step | Expected |
| --- | --- | --- |
| F1 | Open the scenario dropdown | Eleven scenarios grouped by category |
| F2 | Confirm expert scenarios are present | "Integrator convergence" and "Research workbench" appear here but **not** in Guided |
| F3 | Check the cost estimate | "Estimated N integration steps (~X s)" updates as you change parameters |
| F4 | Expand "Advanced controls" | Gains, ballistic parameters, timestep — hidden by default |
| F5 | Run *CW versus a non-linear reference*, control **off**, J2 **on** | Max gap ≈ 5.8 m |
| F6 | Same with J2 off | ≈ 2.8 m |
| F7 | Same with control **on** | Gap collapses; the interpretation explains that feedback masks model error |
| F8 | Change 1 km along-track to 1 km **radial** | Far larger gap — tens of times |
| F9 | Run *Integrator convergence* | Mean observed order ≈ 4; log-log plot tracks the ideal 4th-order line |
| F10 | Run *Proximity safety* with control on | Keep-out violation flagged; explanation says the controller does not avoid anything |
| F11 | Same with control off | Free-drift pass; may or may not violate |
| F12 | Run *External-tool validation workflow* | CSV and JSON download buttons, plus a warning that no benchmark ships |
| F13 | Open the downloaded JSON | Initial state to full precision; force model, step and duration all recorded |
| F14 | Run *Research workbench* | Every parameter exposed; model selector works |
| F15 | Press **Reset** | Parameters return to defaults |

---

## G. Results, downloads, reproducibility

| # | Step | Expected |
| --- | --- | --- |
| G1 | After any run, check the order | Numbers → plots → interpretation → assumptions and limitations → downloads |
| G2 | Check assumptions and limitations | **Always shown**, side by side, for every scenario |
| G3 | Expand "How to read this" | Scenario-specific guidance |
| G4 | Download a results CSV | Opens in a spreadsheet; columns have units in their names |
| G5 | Download the scenario configuration JSON | Contains parameters, engine version, library versions, assumptions, limitations, a Python snippet, and a provenance note saying it is not NASA-validated |
| G6 | Download "Notes and caveats" | Plain text with assumptions, limitations, any warnings raised, and the status statement |
| G7 | **Reproducibility test** | Note the Δv. Reset, re-enter the same parameters, run again — **identical** to every digit |
| G8 | Inspect all three downloads | No participant code, no pilot code, no organisation, no session token |
| G9 | Check the download caption | Says results are numbers only and identify nobody |

---

## H. Input validation — the silent-wrong-answer cases

Each of these previously produced a wrong answer or a traceback rather than an
error. Use *Research workbench*.

| # | Input | Expected |
| --- | --- | --- |
| H1 | Altitude `-7000` | Red: "Altitude must be at least 150 km". **No run.** Previously: silent all-`NaN` trajectory |
| H2 | Timestep `0` | Red: "Timestep must be at least 0.01 s". Previously: `ZeroDivisionError` traceback |
| H3 | Duration `10000000` | Red: too many orbits or too many steps. Previously: silently truncated to a fifth |
| H4 | Duration `-500` | Red: rejected. Previously: returned 2 samples |
| H5 | Mass `0` with drag on | Red: "Mass must be at least 0.1 kg". Previously: `ZeroDivisionError` |
| H6 | All offsets `0`, all velocities `0` | Red: "nothing to simulate" |
| H7 | Timestep `0.05` over one orbit | Red: exceeds the step budget, with a note that the engine has no such limit locally |
| H8 | Separation `80000` m | **Amber warning**, run proceeds: CW is only characterised to 50 km |
| H9 | Separation `5000` m | Amber: published agreement is at ≤ 1 km |
| H10 | Timestep `100` s | Amber: only ~56 steps per orbit; published results use 500–2000 |
| H11 | Altitude `180` km | Amber: atmosphere model extrapolated below its calibration |
| H12 | Confirm every message | Says what is wrong **and** what to do about it |
| H13 | Confirm no tracebacks anywhere | A raw Python traceback must never appear |

---

## I. Rate limiting

| # | Step | Expected |
| --- | --- | --- |
| I1 | Press Run 21 times quickly | Around the 21st: "That is 20 simulations in under a minute… please wait N s" |
| I2 | Wait a minute, run again | Allowed |
| I3 | Check the sidebar caption | Shows remaining runs when the allowance gets low |
| I4 | Confirm a blocked run is not a failure | Admin → System → Recent events shows `simulation_rejected`, and the failed-run count does not rise |

---

## J. Assessments

| # | Step | Expected |
| --- | --- | --- |
| J1 | Enter the pre-assessment | Ten questions, no time limit, explains it measures the activities not the learner |
| J2 | Submit with some unanswered | Amber warning, submission allowed, unanswered marked incorrect |
| J3 | Try to retake | "You have already completed this one" |
| J4 | Press **Skip** | Goes to the workspace; nothing recorded |
| J5 | Complete the post-assessment | Shows before / after / change, plus every answer with explanations |
| J6 | Read the caveat under the change | Says one person's before-and-after is weak evidence, partly because the questions were seen twice |
| J7 | **[JUDGEMENT]** | Are the questions appropriate for your learners? Is anything ambiguous or answerable by elimination? |
| J8 | **[JUDGEMENT]** | Is the confidence question worth keeping? |

---

## K. Feedback and confirmation

| # | Step | Expected |
| --- | --- | --- |
| K1 | Open feedback as a **student** | Usefulness asks about *understanding orbital mechanics* |
| K2 | Open feedback as an **engineer/researcher** | Usefulness asks about *real engineering or research work*; clarity asks about assumptions and limitations |
| K3 | Check the caption | Warns against entering personal information |
| K4 | Submit | Confirmation page |
| K5 | Press **Skip feedback** | Confirmation page; nothing recorded |
| K6 | Read the confirmation page | Says feedback **may** influence later releases, and is explicit that no particular suggestion is promised |
| K7 | Check the return-visit prompt | Tells the participant to reuse the same code |
| K8 | Press "Finish and start a new session" | Returns to landing with state cleared |

---

## L. Return-visit measurement

**The one journey that needs two calendar days.** Automated tests cover it by
backdating the database; this confirms it for real.

| # | Step | Expected |
| --- | --- | --- |
| L1 | **Day 1:** sign in as `DEMO-A` / `RETURN-TEST`, complete a simulation | Recorded |
| L2 | Admin → headline | Returning users **0** |
| L3 | **Day 2:** sign in with the **same two codes** | Green: "Welcome back. Your earlier work is recognised." |
| L4 | Complete another simulation | Recorded against the same participant |
| L5 | Admin → headline | Returning users **1**; return rate rises |
| L6 | Admin → Simulations per user | The participant shows `active_days` = 2 |
| L7 | Same day, second session instead | **Not** counted as returning — returning means separate dates |

*Shortcut if you cannot wait:* run
`pytest tests/test_platform_analytics.py::test_returning_on_a_later_date_is_a_return_visit -q`.

---

## M. Administrator area

| # | Step | Expected |
| --- | --- | --- |
| M1 | Open `/?view=admin` with no credentials configured | "No administrator credentials are configured"; no report shown |
| M2 | Configure credentials, reload | Sign-in form |
| M3 | Wrong password | "Incorrect username or password. 4 attempt(s) remaining" |
| M4 | Wrong username, right password | Same message — does not reveal which half was wrong |
| M5 | Fail five times | Form locks; message says five minutes |
| M6 | Correct password **while locked** | Still refused |
| M7 | Wait five minutes (or restart), sign in correctly | Dashboard |
| M8 | Confirm no report leaked before sign-in | "Pilot administration" appears only after authentication |

**Reporting**

| # | Step | Expected |
| --- | --- | --- |
| M9 | Headline metrics | Organisations, unique/active/returning users, completed/failed runs, per-user, return rate, measured-impact, feedback count |
| M10 | Hover each metric | Tooltip gives the definition |
| M11 | **Development instance** | Everything zero, with a blue notice explaining `PILOT_ENV` |
| M12 | Tick "Include development / demo records" | Figures appear **and** an amber warning says these are not pilot results |
| M13 | Usage tab | Usage by date (chart), organisations, **cohorts by pilot code**, user type, scenario, feature |
| M14 | Two unregistered cohorts | Both under "Unregistered pilot code" as organisation, but **separate rows** in the cohorts table |
| M15 | Period filter | 7/30/90 days, all time, custom range |
| M16 | Organisation filter | Isolates that organisation's figures |
| M17 | User-type filter | Isolates that type |
| M18 | Learning impact tab | Only participants with **both** assessments; caveat about the repeated instrument |
| M19 | Feedback tab | Mean ratings, **engineers/researchers separately**, qualitative comments, bugs and requests |
| M20 | System tab | Database health, schema version, configuration warnings, settings, recent events |
| M21 | **Search the page for your admin password** | **Must not appear.** Nor the participant salt, nor a database password |
| M22 | Export tab → Download every table (ZIP) | Opens; contains every table plus `METRIC_DEFINITIONS.txt` |
| M23 | Open the ZIP's CSVs | Only `participant_ref` (8 chars) — no full hash, no codes, no session tokens |
| M24 | Open `METRIC_DEFINITIONS.txt` | Definitions **and** the filters that produced the export |
| M25 | Individual CSV buttons | Each table downloads on its own |
| M26 | Sign out | Returns to the sign-in form |

---

## N. Privacy and security

| # | Check | Expected |
| --- | --- | --- |
| N1 | `git grep -nE 'PILOT_ADMIN_PASSWORD=' -- ':!.env.example'` | No hits |
| N2 | `git grep -nE 'postgres.*:.*@' -- ':!.env.example' ':!tests/*' ':!*.md'` | No hits |
| N3 | `git ls-files \| grep -E '\.(db\|sqlite3?)$'` | No hits |
| N4 | `git ls-files \| grep '^\.env$'` | No hits (only `.env.example`) |
| N5 | Open `.env.example` | Placeholders only |
| N6 | Query the database for your participant code | **Not present** — only the hash |
| N7 | Inspect `events.detail_json` | Buckets only ("300-500km"), never a typed value |
| N8 | Inspect the schema | No column for name, email, age, IP or user agent |
| N9 | Check the application log | Event types, error class names, timings — no codes, parameters, feedback text or passwords |
| N10 | Break `DATABASE_URL`, restart | App still runs; sidebar warns tracking is unavailable; **no password in the error** |
| N11 | Set `PILOT_LOG_RAW_PARAMETERS=true` | Consent page adds a sentence; admin shows a standing warning |
| N12 | Force an engine error | Plain-language message, never a traceback |

---

## O. Deployment

| # | Check | Expected |
| --- | --- | --- |
| O1 | `python scripts/init_db.py --check` against the real `DATABASE_URL` | Reachable, schema up to date, **no password shown** |
| O2 | Deploy with `PILOT_ENV=production` | Admin → System shows `production` |
| O3 | Admin → System | **No configuration warnings** |
| O4 | Deploy *without* `DATABASE_URL` | Warning that SQLite will be wiped on restart |
| O5 | Deploy *without* `PILOT_PARTICIPANT_SALT` | Warning that returning participants will not be recognised |
| O6 | Restart the app | Data survives (proves Postgres, not SQLite) |
| O7 | `python scripts/seed_pilot_codes.py --list` | Registered organisations shown |
| O8 | Sign in with a registered code | Real organisation name, no "unregistered" notice |
| O9 | `python scripts/purge_expired.py --dry-run` | Counts, deletes nothing |
| O10 | `docker build -t orbital-pilot .` | Builds; regression tests pass during build. **Not yet verified — see KNOWN_GAPS §4.3** |
| O11 | Push and let CI run | All jobs green. **Not yet verified — see KNOWN_GAPS §4.4** |

---

## P. Accessibility and devices

| # | Check | Expected |
| --- | --- | --- |
| P1 | Resize to 375 px wide | No horizontal scrolling *(verified in Chromium)* |
| P2 | Real phone | Usable. 3D plots are fiddly on touch — expected |
| P3 | Tablet | Usable |
| P4 | Firefox, Safari, Edge | *(not yet verified — see KNOWN_GAPS §4.5)* |
| P5 | Keyboard only, no mouse | All controls reachable |
| P6 | Browser zoom 200 % | Layout holds |
| P7 | Screen reader | Every result is also stated as text. *(not tested — see KNOWN_GAPS §4.5)* |
| P8 | **[JUDGEMENT]** Colour | Series use the Okabe-Ito palette **and** dash patterns, so plots should read without colour. Contrast ratios not measured |

---

## Q. Content review — please do this before a classroom pilot

| # | Task | Why |
| --- | --- | --- |
| Q1 | Read all three activities in `pilot/content/activities.py` | Marked DRAFT; never reviewed by a teacher |
| Q2 | Read all ten items in `pilot/content/assessments.py` | Same |
| Q3 | Check reading level | Not measured |
| Q4 | Check curriculum alignment | Written against the physics, not a specification |
| Q5 | Check distractors | No trial, no discrimination analysis |
| Q6 | Consider whether `q8_radial_drift` is too hard as a **pre**-test item | It is the best question and the hardest |
| Q7 | Time an activity with a real learner | The 20–25 minute estimates are untested |
| Q8 | If you edit any item, **bump the instrument `version`** | Otherwise two different tests pool silently |

---

## R. Regression — confirm the science did not move

| # | Check | Expected |
| --- | --- | --- |
| R1 | `python -m pytest -q` | **255 passed** |
| R2 | `python -m pytest tests/test_regression_baseline.py -q` | 11 passed |
| R3 | `python -m pytest tests/test_integrator_convergence.py -q` | 9 passed |
| R4 | `git diff main --stat -- src/` | **Empty.** No engine file changed |
| R5 | `python experiments/run_cw_vs_high_fidelity.py` | Toggle table gives 2.78 / 5.81 / 2.78 / 5.82 m |
| R6 | `python experiments/run_cw_rendezvous.py` | Δv ≈ 2.33 m/s, converges ≈ 1591 s |
| R7 | Compare `outputs/data/*.csv` with any pre-existing copies | Identical |
| R8 | `python -m unittest discover -s tests -p "test_orbit.py"` | Passes — the engine still works with no platform dependencies |
