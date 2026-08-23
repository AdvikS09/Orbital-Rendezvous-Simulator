# Data dictionary and metric definitions

Every field the platform stores, and the exact definition of every number the
administrator dashboard reports.

Schema defined in [`pilot/models.py`](../pilot/models.py); metric queries in
[`pilot/analytics.py`](../pilot/analytics.py).

---

## 1. Metric definitions

These are used consistently in the dashboard, the CSV/ZIP exports, and this
document. The exported ZIP includes a copy so a spreadsheet cannot drift from
its definitions.

| Metric | Definition | Implementation |
| --- | --- | --- |
| **Unique user** | An anonymous participant who has completed **at least one** simulation. Deliberately *not* limited to the reporting period — being unique is a property of the participant, not of a window. | `COUNT(DISTINCT participant_id)` over `simulation_completed` events |
| **Active user** | An anonymous participant who completed a simulation **during the selected reporting period**. | Same, restricted to `occurred_date` in range |
| **Returning user** | An anonymous participant who used the platform on **at least two separate dates**. Two sessions on the same day is not returning. | `HAVING COUNT(DISTINCT occurred_date) >= 2` |
| **Completed run** | One simulation that finished successfully and displayed results. | One `simulation_completed` event |
| **Failed run** | One simulation that raised an error during execution. **Runs rejected by input validation or rate limiting are not failures** — they are `simulation_rejected` and counted separately, because the engine never ran. | One `simulation_failed` event |
| **Measured-impact user** | A participant who completed **both** the pre- and the post-assessment for the same instrument. | `HAVING COUNT(DISTINCT phase) >= 2` |
| **Return rate** | returning users ÷ unique users. Zero when there are no unique users. | derived |
| **Simulations per user** | completed runs ÷ unique users. | derived |
| **Failure rate** | failed runs ÷ (completed + failed). Excludes rejected runs. | derived |

### 1.1 Deliberate choices worth knowing when reading a report

* **A participant who signs in but never completes a simulation is not a unique
  user.** They exist in `participants` and produce `session_started`, but the
  headline count deliberately requires actually running something.
* **Unique and returning are not date-filtered; active is.** Filtering "unique"
  by date would make it a synonym for "active".
* **Demo records are excluded from every figure by default.** See §4.
* **Assessment gains are inflated by the design.** The same instrument is used
  before and after, so part of any improvement comes from having seen the
  questions. The dashboard says this next to the numbers.

---

## 2. Tables

### 2.1 `organisations`

One row per participating organisation.

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `code` | text, unique | Short code, e.g. `SCHOOL-A`. Upper-cased. |
| `name` | text | Display name in the dashboard. |
| `org_type` | text | `school`, `university`, `company`, `research_institute`, `other`. |
| `country` | text, null | Optional. Country only — never an address. |
| `is_demo` | bool | True if created outside production. |
| `active` | bool | |
| `created_at` | datetime | UTC, naive. |
| `notes` | text, null | Coordinator's own notes. Not participant data. |

`UNREGISTERED` / "Unregistered pilot code" is created automatically to group
participants whose code is not registered. It exists so those sessions stay
countable **without inventing an institutional relationship**.

### 2.2 `pilot_codes`

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `organisation_id` | int FK | |
| `code` | text, unique | What a participant types. Upper-cased. |
| `label` | text, null | e.g. "Year 12 physics". |
| `cohort` | text, null | Optional grouping within an organisation. |
| `active`, `is_demo` | bool | |
| `created_at`, `expires_at` | datetime | |

### 2.3 `participants` — anonymous

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `participant_hash` | text(64), unique | `HMAC-SHA256(salt, PILOT_CODE\|PARTICIPANT_CODE)`. **The participant code itself is never stored.** Irreversible; not comparable across deployments with different salts. |
| `participant_ref` | text(8) | First 8 hex characters, so a row can be referred to in the dashboard. |
| `pilot_code_id`, `organisation_id` | int FK, null | |
| `user_type` | text | `student`, `teacher`, `engineer_researcher`, `other`. Self-declared. |
| `first_seen_at` / `last_seen_at` | datetime | UTC. |
| `first_seen_date` / `last_seen_date` | date | |
| `distinct_active_days` | int | Separate dates seen. Maintained on sign-in. |
| `consent_version` | text | Which privacy notice was agreed to. |
| `consent_at` | datetime | |
| `is_demo` | bool | |

There is **no column** for a name, email address, age, date of birth, IP
address, browser or user agent. This is asserted by
`test_no_column_can_hold_an_email_or_a_name`.

### 2.4 `sessions`

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `session_token` | text, unique | Random per visit. Not derived from identity; used only to key the rate limiter. |
| `participant_id` | int FK | |
| `pilot_code_id`, `organisation_id` | int FK, null | |
| `track` | text, null | `guided`, `professional`, `free`. |
| `started_at` / `started_date` / `last_active_at` | datetime/date | |
| `app_version` | text | Which build produced the record. |
| `is_return_visit` | bool | True when this session began on a **later date** than the participant's previous activity. |
| `is_demo` | bool | |

### 2.5 `events` — the analytics record

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `event_type` | text | See §3. Values outside that list are refused. |
| `occurred_at` / `occurred_date` | datetime/date | UTC. |
| `session_id`, `participant_id`, `pilot_code_id`, `organisation_id` | int FK, null | |
| `user_type`, `track` | text, null | Denormalised so reports do not need joins. |
| `scenario_category` | text, null | e.g. `relative_motion`. |
| `scenario_id` | text, null | e.g. `pd_rendezvous`. |
| `feature_category` | text, null | e.g. `gain_sweep`, `download`, `assessment`. |
| `duration_ms` | int, null | Measured execution time. |
| `success` | bool, null | |
| `error_class` | text, null | Exception **class name only** — never the message, which could echo input. |
| `detail_json` | text, null | See §3.1. |
| `app_version` | text | |
| `is_demo` | bool | |

### 2.6 `assessments`

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `participant_id` | int FK | |
| `session_id`, `organisation_id`, `pilot_code_id` | int FK, null | |
| `instrument_id` / `instrument_version` | text | So revising an instrument does not pool incomparable data. |
| `phase` | text | `pre` or `post`. |
| `score` / `max_score` | int | |
| `confidence_rating` | int, null | Self-rated 1–5. |
| `item_responses_json` | text, null | Per item: `item_id`, `topic`, `chosen_index`, `correct`. **No free text.** |
| `completed_at` / `completed_date` | datetime/date | |
| `user_type` | text, null | |
| `is_demo` | bool | |

Unique on `(participant_id, instrument_id, phase)` — retaking overwrites rather
than creating a second row.

### 2.7 `feedback`

| Column | Type | Description |
| --- | --- | --- |
| `id` | int PK | |
| `participant_id` | int FK | |
| `session_id`, `organisation_id`, `pilot_code_id` | int FK, null | |
| `user_type`, `track` | text, null | |
| `usefulness_rating` | int, null | 1–5. For engineers/researchers the question asks about **real engineering or research work**; for others about understanding. Reported separately. |
| `usability_rating` | int, null | 1–5, ease of use. |
| `clarity_rating` | int, null | 1–5. Clarity of explanations, or of assumptions and limitations for professionals. |
| `would_recommend` | int, null | 0–10. |
| `most_useful`, `improvement`, `bug_report`, `feature_request` | text, null | **Free text, stored verbatim, capped at 2000 characters.** The form warns participants not to enter personal information. This is the one place where personal data could reach the database, and only if a participant types it. |
| `submitted_at` / `submitted_date` | datetime/date | |
| `is_demo` | bool | |

### 2.8 `schema_migrations` and `admin_audit`

`schema_migrations` records `version`, `name`, `applied_at`.
`admin_audit` records administrative actions with no participant data.

---

## 3. Event vocabulary

Only these values may be written; `record_event` refuses anything else.

| Event | When |
| --- | --- |
| `pilot_accessed` | A participant completed sign-in. |
| `session_started` | A session row was created. |
| `return_visit` | Session began on a later date than previous activity. |
| `track_selected` | Guided / professional / free chosen. |
| `simulation_started` | Validation and rate limiting passed; the engine is about to run. |
| `simulation_completed` | Finished successfully and displayed results. |
| `simulation_failed` | The engine raised. |
| `simulation_rejected` | Blocked **before** running, by validation or rate limiting. Not a failure. |
| `scenario_category_used` | Emitted with each completion, for category reporting. |
| `feature_category_used` | Emitted with each completion and for non-simulation features. |
| `results_downloaded` | A download button was used. |
| `activity_completed` | A guided activity was finished. |
| `assessment_completed` | A pre or post assessment was submitted. |
| `feedback_submitted` | The feedback form was submitted. |
| `consent_given` | Reserved; consent is currently recorded on the participant row. |

### 3.1 `detail_json` — what may appear

Filtered against an allow-list. Anything not on it is dropped before the write.

| Key | Example | Note |
| --- | --- | --- |
| `model` | `"comparison"` | Cost model of the scenario. |
| `force_model` | `"two-body + J2"` | Which perturbations were on. |
| `controlled` | `true` | |
| `steps_bucket` | `"2k-10k"` | Band, not the number. |
| `altitude_bucket_km` | `"300-500km"` | Band. |
| `separation_bucket_m` | `"100m-1km"` | Band. |
| `warning_count` | `0` | |
| `instrument_id`, `phase` | `"orbital_basics"`, `"pre"` | |
| `activity_id`, `step_index` | | |
| `download_kind` | `"scenario_config"` | |
| `rejected_reason` | `"validation"` or `"rate_limit"` | |

**Raw orbital parameters are never stored** unless
`PILOT_LOG_RAW_PARAMETERS=true`, which is off by default. With it on, values go
under a `raw_parameters` key, the consent page says so explicitly, and the admin
area shows a standing warning.

Buckets exist so that "students mostly worked at small separations" is
answerable while "what exactly did participant 3f9a1c02 simulate" is not.

---

## 4. Demo vs pilot data

Every observation table carries `is_demo`, set from `PILOT_ENV`:

* `PILOT_ENV=production` → `is_demo = false` — real pilot data.
* **anything else** (`development`, `demo`, `test`, unset) → `is_demo = true`.

Every reporting query filters `is_demo = false` by default. The admin sidebar
has an explicit switch to include demo rows; turning it on raises a standing
warning that the figures are not pilot results.

This is why a fresh development instance shows zeros: it is working correctly.

---

## 5. Retention and deletion

`PILOT_RETENTION_DAYS` (default 730) is the period stated on the consent page.
**Deletion is not automatic.** Run:

```bash
python scripts/purge_expired.py --dry-run
```

then `--confirm` to delete. It removes expired rows from `events`,
`assessments`, `feedback` and `sessions`, and removes participants whose last
activity is older than the cutoff, so nothing remains that could be correlated
over a longer window than the policy allows.

Because the data is anonymous, an individual's records cannot be identified for
deletion on request. The consent page says this and offers the alternative:
close the page, or run the simulator locally with no analytics.
