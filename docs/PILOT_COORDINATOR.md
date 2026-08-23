# Pilot coordinator guide

For whoever is running a pilot — a teacher with a class, or someone trialling
the platform with an engineering team.

---

## 1. What you are actually piloting

An open-source orbital-mechanics simulator that runs in a browser, built around
one research question: **how far can you trust the simple linear model that
spacecraft rendezvous guidance actually uses?**

Be clear with participants about what it is not. It is research and educational
software. It is **not** NASA-approved, NASA-validated, flight-certified or
operational mission software, and it has **not** been benchmarked against an
external tool such as GMAT or STK. The app says all of this on its landing page;
please do not describe it more strongly than the app describes itself.

## 2. Before the first session

### 2.1 Deploy it

Follow [DEPLOYMENT.md](DEPLOYMENT.md). The three things that matter most:

* `PILOT_ENV=production` — otherwise every record is flagged as demo data and
  your reports show zeros.
* A **fixed** `PILOT_PARTICIPANT_SALT`, set before you start and never changed.
* A real Postgres `DATABASE_URL`, not the SQLite fallback.

### 2.2 Register your organisation

Optional, but it gives you a proper name in the dashboard and lets you filter
reports:

```bash
python scripts/seed_pilot_codes.py \
    --org-code SCHOOL-A --org-name "Example High School" --org-type school \
    --code SCHOOL-A-Y12 --label "Year 12 physics"

python scripts/seed_pilot_codes.py --list
```

Only register organisations that have actually agreed to take part.

Give each cohort its own code (`SCHOOL-A-Y12`, `SCHOOL-A-Y13`) if you want to
compare them.

### 2.3 Review the educational content — please actually do this

The guided activities and the assessment questions are marked
**DRAFT — not yet reviewed by a subject teacher**. They were written to match
what the simulator genuinely does, and every number quoted in them is checked
against the engine by an automated test. But nobody has checked them for
reading level, curriculum alignment, or whether the questions discriminate
usefully between students.

Read them before you use them:

* `pilot/content/activities.py` — three activities, 4 + 3 + 4 steps
* `pilot/content/assessments.py` — the ten-item instrument

Both are plain Python data structures; editing the text needs no programming.
If you change assessment items, bump `version` on the instrument so your
before-and-after data does not silently pool two different tests.

### 2.4 Check your institution's requirements

Anonymous aggregate usage data is usually a lighter-touch category than personal
data, but **whether you need ethics approval, safeguarding measures or parental
consent is a decision for your institution, not for this software.** Pilots
involving school-age children commonly require processes this platform does not
implement and cannot replace. Sort that out first.

[PRIVACY.md](PRIVACY.md) has the detail you will need for such an application.

## 3. Running a session

### 3.1 What to tell participants

* The URL.
* The pilot code (e.g. `SCHOOL-A-Y12`).
* **How to choose a participant code:** something memorable and *not personal* —
  not their name, not their email. `BLUE-FALCON-42` is ideal.
* **To write it down.** Coming back with the same code is what makes repeat use
  measurable, and it is one of the things the pilot is trying to measure.

The participant guide, [PARTICIPANT.md](PARTICIPANT.md), can be handed out
directly.

### 3.2 Suggested shape for a class

| Time | What |
| --- | --- |
| 5 min | Introduce the question: two spacecraft, close together, how do they move? |
| 5 min | Sign-in, and the pre-assessment |
| 20 min | Activity 1: *What keeps a satellite up, and what pulls it down* |
| 25 min | Activity 2: *Why you cannot just fly straight at another spacecraft* |
| 25 min | Activity 3: *Flying the rendezvous, and paying for it* |
| 10 min | Post-assessment and feedback |

Activities 1 and 2 stand alone; activity 3 assumes activity 2.

The single most valuable instruction you can give: **make them commit to a
prediction before pressing Run.** The app enforces this — the Run button stays
locked until a prediction is chosen — because a simulation you have not
predicted is an animation, not an experiment.

### 3.3 Moments worth stopping the class for

* **Activity 2, step 2.** A 100 m offset *above* the target becomes nearly 4 km
  of drift in one orbit. Almost everyone predicts wrong. It is the best moment
  in the whole sequence.
* **Activity 1, step 4.** Drag removes energy and the satellite ends up going
  *faster*. Worth dwelling on.
* **Activity 3, step 3.** The fastest gains and the cheapest gains are different
  cells. Engineering is trade-offs, not optimisation of one number.

### 3.4 For a professional or research cohort

Point them at **Professional / research exploration**. Suggest:

1. *CW versus a non-linear reference* with control **off** — the honest measure
   of linearisation error. Then turn control on and watch feedback mask it.
2. *Integrator convergence* — confirms the RK4 propagator is fourth-order, so
   the published metrics are not step-size artefacts.
3. *External-tool validation workflow* — the export half of a GMAT/STK
   comparison. Be clear that no such comparison ships with the software; if a
   participant runs one, that result would be a genuine contribution.

Ask them specifically about **missing physics** (differential drag, eccentric
references, navigation error, impulsive/optimal guidance). That is the most
actionable feedback the project can get.

## 4. Reading the dashboard

`https://YOUR-APP/?view=admin`

Definitions are in [DATA_DICTIONARY.md](DATA_DICTIONARY.md) §1 and in the
exported ZIP. The ones people misread:

* **Unique user** means *completed at least one simulation*. Someone who signed
  in and left is not counted. This is deliberate.
* **Returning user** means *two separate dates*. Two sessions the same afternoon
  is not returning.
* **Rejected runs are not failures.** A rejected run never reached the engine —
  the input was out of bounds or the rate limit was hit. High rejection usually
  means the guidance was unclear, not that the software is broken.
* **Demo records are excluded by default.** If your dashboard shows zeros,
  check `Environment: production` on the System tab before assuming nobody came.

### 4.1 Assessment results — the caveat that matters

The same instrument is used before and after. That is the most sensitive design
for small cohorts, but it means **part of any gain comes from having seen the
questions already**. There is no control group. Treat the numbers as indicative
of whether the activities did something, not as a measurement of how much.

With fewer than ~20 matched participants, report the direction and the
distribution, not a mean with a confidence interval.

## 5. Exporting

Admin → **Export** → *Download every table (ZIP)*. Contains every aggregate
table as CSV, plus `METRIC_DEFINITIONS.txt` recording the definitions and the
filters that produced them — so a spreadsheet cannot drift from its meaning.

Individual tables have their own CSV buttons.

Everything is anonymous and aggregate. Nothing in an export identifies a
participant.

## 6. Keeping your promises

* **Retention.** The consent page promises data is kept at most
  `PILOT_RETENTION_DAYS` (default 730). This is not automatic. Schedule
  `python scripts/purge_expired.py --confirm`.
* **Anonymity.** Do not attempt to map participant codes back to people, and do
  not keep a separate list that would let you. That list, not the database,
  would be the thing that de-anonymises your pilot.
* **Small cohorts.** With a handful of participants, timing plus cohort can
  narrow down who a row belongs to. Report at cohort level.

## 7. What to do with the feedback

The **Feedback** tab separates bugs and feature requests from general comments.
Bug reports are the most actionable — they name a scenario and what looked
wrong. Feature requests from the professional cohort are the best signal about
what physics to add next.

The confirmation page tells participants their feedback *may* influence later
releases, and is explicit that no promise is made about any particular
suggestion. Please keep that expectation when you talk to them.

## 8. If something goes wrong

[TROUBLESHOOTING.md](TROUBLESHOOTING.md). The first thing to check is always
Admin → System, which reports database health and every configuration warning.
