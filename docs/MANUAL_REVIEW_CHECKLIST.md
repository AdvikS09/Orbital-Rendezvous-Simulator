# Manual review checklist — custom web platform

What to test by hand in the React + FastAPI application, and what each step
should do.

519 automated tests cover the numbers, the API and the components, so this
concentrates on what a machine cannot judge: whether it looks right, reads
honestly, and behaves sensibly when a real person does something unexpected.

For the **Streamlit** application, use the root-level
[`MANUAL_REVIEW_CHECKLIST.md`](../MANUAL_REVIEW_CHECKLIST.md) — it still applies
and that application still runs.

**Setup**

```bash
cd web-platform/backend && uvicorn app.main:app --reload --port 8000
cd web-platform/frontend && npm run dev
```

Open <http://localhost:5173>. Items marked **[JUDGEMENT]** need your opinion,
not a pass/fail.

---

## A. The two defects this migration was asked to fix

| # | Check | Expected |
| --- | --- | --- |
| A1 | Run any scenario. Search the page for `e-` | **No match.** Every small value reads as `4.58 × 10⁻¹³`, never `4.58e-13` |
| A2 | Search for `e+` | No match |
| A3 | Read the interpretation sentence under the metrics | Numbers inside prose are formatted too — `4.6 × 10⁻¹³`, not `5e-13` |
| A4 | Scan the whole interface for emoji | **None.** Status is carried by a line icon plus a text word |
| A5 | Turn off colour (greyscale print preview) | Severity still readable: "Warning:", "Error:" precede the message |
| A6 | Check the superscript characters | Real Unicode (⁻¹³), not `^-13` or a raised span |
| A7 | Check the minus sign in a negative value | U+2212 (−), which aligns with digits, not a hyphen |
| A8 | **[JUDGEMENT]** | Are the exponents legible at your display size? |

---

## B. Numeric input

| # | Step | Expected |
| --- | --- | --- |
| B1 | Open a scenario with sliders | The value sits in its own cell **above** the track, never over the handle |
| B2 | Drag a slider to each extreme | The value never overlaps the min/max labels |
| B3 | Type `4.6e-5` into a gain field | Accepted; the run uses 4.6 × 10⁻⁵ |
| B4 | Type `abc` into a numeric field | Field marked invalid, message announced. **Not silently zeroed** |
| B5 | Check every field | The unit is beside the value, never inside it or overlapping |
| B6 | Compare digits down a results column | They line up — tabular numerals, right-aligned |
| B7 | Open "Advanced controls" | Gains, timestep and ballistic parameters, hidden by default |
| B8 | Enter altitude `-7000` and Run | Red message naming the bound. **Nothing is run** |
| B9 | Watch the estimate as you change the timestep | "Estimated N integration steps (~X s)" updates |
| B10 | Set a timestep that exceeds the budget | Run is disabled and the reason is given before you click |
| B11 | Hover a displayed number | The full-precision value appears in the tooltip |
| B12 | **[JUDGEMENT]** | Is the precision right? Too many digits, or too few? |

---

## C. Scenarios and results

| # | Step | Expected |
| --- | --- | --- |
| C1 | `/scenarios` | Eleven scenarios grouped by category |
| C2 | Open `two_body_orbit`, Run | Period ≈ 92.6 min; energy drift ≈ 10⁻¹³; flat altitude line |
| C3 | Expand "Background" | The full multi-paragraph explanation, not a summary |
| C4 | Check below every result | Assumptions **and** limitations, always visible, never collapsed |
| C5 | Expand "How to read this" | Scenario-specific guidance |
| C6 | Run `cw_vs_reference`, control off, J2 off then on | ≈ 2.78 m, then ≈ 5.81 m |
| C7 | Same scenario, control **on** | The gap collapses; the interpretation explains that feedback masks model error |
| C8 | Change 1 km along-track to 1 km **radial** | Far larger gap |
| C9 | Run `gain_tuning` | Two heatmaps; fastest and cheapest are different cells; blanks are explained |
| C10 | Run `integrator_convergence` | Mean observed order ≈ 4; log-log plot tracks the reference line |
| C11 | Run `proximity_safety` with control on | Keep-out violation flagged, with the explanation that the controller does not avoid |
| C12 | Run `external_validation_export` | CSV and JSON downloads, and a warning that **no benchmark ships** |
| C13 | Run `research_workbench` | All 22 parameters; 10 advanced |
| C14 | Set a separation of 80 km | **Amber warning**, run proceeds — CW is characterised only to 50 km |
| C15 | Set a timestep of 100 s | Amber: too few steps per orbit |
| C16 | Press Run twice with identical inputs | Identical numbers to the last digit |

---

## D. Plots

| # | Check | Expected |
| --- | --- | --- |
| D1 | 3D orbit plot | Earth to scale; axes labelled with units |
| D2 | Relative-motion plot | Target marker at the origin |
| D3 | Every axis | Carries its unit in the title |
| D4 | Legend | Present, series named |
| D5 | Hover a trace | Value and unit shown |
| D6 | Two series on one plot | Distinguished by **dash pattern as well as colour** |
| D7 | Same quantity across scenarios | Same colour |
| D8 | Modebar | PNG download works; filename names the plot |
| D9 | Compare against the Streamlit version | Same shape, same data, no added smoothing |
| D10 | Watch the first chart load | Brief spinner while Plotly loads, then the chart |
| D11 | **[JUDGEMENT]** | Do the colours read well on your display? |

---

## E. Exports

| # | Step | Expected |
| --- | --- | --- |
| E1 | Download a result CSV | Opens in a spreadsheet; columns carry units in their names |
| E2 | **Inspect the CSV values** | Machine values (`1.1060555202205516e-13`), **never** `1.11 × 10⁻¹³` |
| E3 | Check a negative value in the CSV | Plain hyphen, so it parses as a number |
| E4 | Read the CSV header comments | Scenario, timestamp, units, solver |
| E5 | Download the scenario configuration | Parameters, engine version, library versions, assumptions, limitations, a runnable snippet, and the provenance note |
| E6 | Download "Notes and caveats" | Assumptions, limitations, any warnings raised, and the status statement |
| E7 | Download the full result JSON | Complete payload |
| E8 | Search every download for identifying data | **No** participant code, pilot code, organisation or session token |
| E9 | Reproducibility | Re-enter the exported parameters; identical result |

---

## F. Participant journey

| # | Step | Expected |
| --- | --- | --- |
| F1 | Landing page | Purpose, audience, and a prominent "What this is not" |
| F2 | Read the status statement | Not NASA-approved, not flight-certified, not benchmarked against GMAT/STK |
| F3 | Expand the physics limitations | Seven items |
| F4 | Press Start | Privacy notice |
| F5 | Try Continue without ticking | Disabled, with a caption saying so |
| F6 | Read the privacy summary | What is and is not recorded |
| F7 | Enter participant code `alice@example.org` | Rejected: looks like an email address |
| F8 | Enter `ab` | Rejected: too short |
| F9 | Enter `Alice Smith` | Rejected: letters, digits, hyphens, underscores only |
| F10 | Sign in with `SELF` / `BLUE-FALCON-42` | Proceeds; notice that the code is not registered |
| F11 | Check the sidebar | An 8-character reference, **not** the code you typed |
| F12 | Track selection | Three cards; one marked suggested for your user type; all selectable |
| F13 | Start a guided activity | Step 1 with a progress indicator |
| F14 | Confirm Run is locked | Disabled until a prediction is chosen; caption says so |
| F15 | Choose a prediction, Run | Enabled; "What to look for" above, "Takeaway" below |
| F16 | **Activity 2, step 2** (100 m radial) | Ends ≈ −3770 m along-track. The best moment in the sequence |
| F17 | Finish the activity | Post-assessment offered |
| F18 | Feedback as a **student** | Usefulness asks about understanding |
| F19 | Feedback as an **engineer/researcher** | Usefulness asks about real engineering work; clarity asks about assumptions |
| F20 | Confirmation page | Feedback **may** influence releases; no promise made |

---

## G. Assessments

| # | Step | Expected |
| --- | --- | --- |
| G1 | Open the pre-assessment | Ten questions, no time limit |
| G2 | **Open DevTools, inspect the API response** | `correct_index` is `null` — not answerable by reading the API |
| G3 | Submit with some unanswered | Warning; submission allowed; unanswered marked incorrect |
| G4 | Complete the post-assessment | Before/after/change, then every answer with its explanation |
| G5 | Read the caveat | Says part of any gain comes from having seen the questions twice |
| G6 | Skip | Works on both phases |
| G7 | **[JUDGEMENT]** | Are the questions right for your learners? |

---

## H. Administration

| # | Step | Expected |
| --- | --- | --- |
| H1 | Open `/admin` with no credentials configured | "No administrator credentials are configured" |
| H2 | **Call `GET /api/v1/admin/report` directly with curl, no auth** | **401.** The API is protected, not merely the page |
| H3 | Configure credentials, reload | Sign-in form |
| H4 | Wrong password | Refused, attempt counter shown |
| H5 | Fail five times | Locked for five minutes |
| H6 | Correct password while locked | Still refused |
| H7 | Sign in correctly | Dashboard |
| H8 | **Inspect the DOM after sign-in** | The password appears nowhere |
| H9 | **Check localStorage and sessionStorage** | No credentials. Only an anonymous session token |
| H10 | Development instance | Everything zero, with a notice explaining `PILOT_ENV` |
| H11 | Tick "Include development / demo records" | Figures appear **and** a standing warning says these are not pilot results |
| H12 | Usage tab | By date, organisation, **cohort (pilot code)**, user type, scenario, feature |
| H13 | Hover a metric tile | The definition |
| H14 | Learning impact tab | Only participants with both assessments; the repeated-instrument caveat |
| H15 | Feedback tab | Mean ratings, a separate engineers/researchers block, comments, bugs and requests |
| H16 | Export tab | ZIP downloads, containing `METRIC_DEFINITIONS.txt` |
| H17 | Open the ZIP | Only 8-character refs — no full hash, no codes, no tokens |
| H18 | Sign out | Returns to the form |

---

## I. Errors and limits

| # | Step | Expected |
| --- | --- | --- |
| I1 | Stop the backend, press Run | "Could not reach the simulation server" — not a stack trace |
| I2 | Restart it, press Run | Recovers without a reload |
| I3 | Press Run 21 times quickly | Rate-limit message with a wait time |
| I4 | Confirm a blocked run is not a failure | Admin shows `simulation_rejected`; failed-run count unchanged |
| I5 | Any error at all | Plain language. **Never** a Python traceback |
| I6 | Visit `/api/v1/nonsense` | JSON error with `error_class`, not HTML |
| I7 | Visit `/not-a-page` | Redirects to the landing page |
| I8 | Block Plotly in DevTools, run a scenario | Callout saying the chart could not be drawn and the numbers are unaffected; page does not crash |

---

## J. Responsive and accessibility

| # | Check | Expected |
| --- | --- | --- |
| J1 | Resize to 375px | **No horizontal page scrolling** |
| J2 | **Run a scenario at desktop width, then resize to 375px** | Charts reflow. Zero overflow. *(This was a real bug; verify the fix.)* |
| J3 | Rotate a tablet | Same |
| J4 | At 375px | Sidebar becomes a drawer; panels stack, parameters first |
| J5 | A wide results table on mobile | Scrolls **inside its container**, not the page |
| J6 | Keyboard only, no mouse | Every control reachable; focus always visible |
| J7 | Tab to a wide table | It receives focus and arrow keys scroll it |
| J8 | Tab through the tabs | Arrow keys move between them |
| J9 | Press Tab from the top | A "Skip to main content" link appears |
| J10 | Browser zoom 200% | Layout holds |
| J11 | Enable reduced motion | Transitions stop; the spinner still indicates activity |
| J12 | Screen reader | Every control announced with its label and unit. *(Untested — see `KNOWN_GAPS.md` §2.5)* |
| J13 | Firefox, Safari, Edge | *(Untested — see `KNOWN_GAPS.md` §2.5)* |

---

## K. Deployment

| # | Check | Expected |
| --- | --- | --- |
| K1 | `curl /api/v1/health` | `status: ok`, engine version, database, scenario count |
| K2 | Health with a Postgres URL | **No password anywhere** in the response |
| K3 | `docker build -f web-platform/Dockerfile -t orbital-web .` | Builds; the parity gate passes during the build. *(Untested — `KNOWN_GAPS.md` §2.1)* |
| K4 | Run the image | API and frontend on one origin |
| K5 | `docker compose up --build` | Whole stack at :8000 with Postgres |
| K6 | Deep link in the built image | `/scenarios/pd_rendezvous` resolves |
| K7 | Search the built bundle for `localhost` | **No match.** All API calls are relative |
| K8 | Deploy without `DATABASE_URL` | Warning that SQLite will be wiped on restart |
| K9 | Deploy without `PILOT_PARTICIPANT_SALT` | Warning that returning participants will not be recognised |
| K10 | Set `PILOT_ENV=production` | Health reports production; records no longer flagged demo |
| K11 | Restart with Postgres | Data survives |
| K12 | `git grep -nE 'PILOT_ADMIN_PASSWORD=' -- ':!*.example'` | No hits |

---

## L. The Streamlit app must still work

The brief requires it as a regression reference.

| # | Check | Expected |
| --- | --- | --- |
| L1 | `streamlit run pilot_app.py` | Starts and serves |
| L2 | Run a simulation there | Works |
| L3 | `streamlit run app.py` | The legacy MVP still runs |
| L4 | `python -m pytest -q` at the repository root | **255 passed** |
| L5 | `git diff pilot-platform --find-renames -M --stat -- src/ orbital_core/` | Only a docstring and the shim |
| L6 | `python experiments/run_cw_vs_high_fidelity.py` | Byte-identical CSVs |
| L7 | Compare a metric between the two applications | Identical to the last digit |

---

## M. Content review

| # | Task | Why |
| --- | --- | --- |
| M1 | Read the landing and consent pages | Carried over verbatim; confirm nothing was lost |
| M2 | Compare a scenario explanation with `scenarios_catalog.py` | Should match exactly |
| M3 | Check assumptions and limitations on every scenario | Never empty |
| M4 | Read the external-validation warning | Must say no benchmark has been run |
| M5 | **[JUDGEMENT]** | Could a reader come away thinking this is validated or flight software? |
| M6 | **[JUDGEMENT]** | Is the tone right for a research tool? Any marketing language left? |
| M7 | **[JUDGEMENT]** | Is the information density right, or too dense? |
