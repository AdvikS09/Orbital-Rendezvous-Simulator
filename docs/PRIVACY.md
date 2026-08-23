# Privacy and data retention

What this platform records, what it deliberately cannot record, and how long it
keeps anything.

This is the reference document. The participant-facing version is the consent
page in the app; the two are kept consistent, and the wording shown to
participants lives in `pilot/ui/common.py` so it cannot drift between pages.

---

## 1. Summary

The platform records **anonymous, aggregate usage information** so that a pilot
can report whether it was used and whether it helped. It does not record
anything that identifies a person, and for most of the categories people worry
about, there is nowhere in the database to put them.

## 2. What is recorded

| Category | Detail |
| --- | --- |
| Anonymous identifier | A one-way hash of the pilot code and the participant's self-chosen code |
| User type | Student, teacher, engineer/researcher, or other — self-declared |
| Organisation | Only the pilot code the participant typed, mapped to a registered organisation if there is one |
| Activity | Which scenarios were run, whether they succeeded, how long they took |
| Parameter *bands* | Coarse bands such as "separation between 100 m and 1 km" — never the values typed |
| Assessment results | Score, per-item correctness, self-rated confidence |
| Feedback | Ratings, and free text exactly as written |
| Timestamps | When each event happened (UTC) |
| Build | Which version of the platform produced the record |

## 3. What is never recorded

* Names
* Email addresses
* Exact ages or dates of birth
* IP addresses
* Browser, device or user-agent information
* Geolocation
* Confidential company data or operational mission data
* **The participant code itself**
* **The orbital parameters a participant types** (unless explicitly enabled — §6)

These are not merely "not collected". The database schema has **no column** that
could hold them, and an automated test
(`test_no_column_can_hold_an_email_or_a_name`) fails the build if such a column
is ever added.

## 4. How anonymity works

A participant enters a pilot code and an anonymous code of their own choosing.
The platform computes:

```
participant_hash = HMAC-SHA256(server_secret_salt, "PILOTCODE|PARTICIPANTCODE")
```

and stores only that. Properties:

* **Deterministic** — the same person typing the same code on a later date is
  recognised, which is what makes "returning user" measurable at all.
* **Irreversible** — the stored value cannot be turned back into the code.
* **Deployment-scoped** — a different salt produces different values, so records
  cannot be correlated between deployments.
* **Not guessable in bulk** — without the server secret, an attacker with the
  database cannot test candidate codes.

The first 8 characters (`participant_ref`) appear in the admin area purely so a
row can be referred to in conversation.

### 4.1 The limits of this, stated honestly

* A participant who chooses a code that is itself personal (their name, their
  email) has undermined their own anonymity. The entry form rejects anything
  containing `@` and anything with spaces or punctuation beyond `-` and `_`, and
  the page says why — but it cannot detect every such case.
* Free-text feedback is stored **verbatim**. If someone writes their name in it,
  it is stored. The form warns against this.
* An organisation running a pilot with a small cohort could, in principle,
  narrow down who a record belongs to from the combination of pilot code and
  timing. This is inherent to any pilot with few participants and is a reason to
  report at cohort level rather than row level.

## 5. Why anything is collected

To answer, for a pilot report: how many people used it, from how many
organisations, how often, which parts they used, whether simulations succeeded,
whether learners' understanding changed, and what they asked for next.

It is **not** used for advertising, is not sold, and is not shared with third
parties. Streamlit's own usage statistics are disabled in
`.streamlit/config.toml`, so nothing goes to a third party either.

## 6. The raw-parameter switch

`PILOT_LOG_RAW_PARAMETERS` is **false** by default: the numbers a participant
types are not stored, only bands.

Setting it to `true` records the exact values. It exists for a research pilot
that genuinely needs to know what people tried. If enabled:

* the consent page adds an explicit sentence telling participants their values
  are being recorded, and
* the administrator dashboard shows a standing warning.

Do not enable it without participants' informed consent.

## 7. Retention

Default 730 days, configurable with `PILOT_RETENTION_DAYS` and stated on the
consent page.

**Deletion is not automatic.** For the promise to be true, run on a schedule:

```bash
python scripts/purge_expired.py --dry-run
python scripts/purge_expired.py --confirm
```

It deletes expired events, assessments, feedback and sessions, and deletes
participants whose last activity predates the cutoff.

## 8. Withdrawal

Because the data is anonymous, individual records cannot be identified in order
to delete them on request — which is the trade-off that comes with genuine
anonymity. Participants who want nothing recorded are told to close the page
before signing in, or to run the simulator locally from source with no analytics
at all. Both options are stated on the consent page.

## 9. Security

* Secrets come only from environment variables or the host's secrets panel.
  There is no credential file and no default password.
* The administrator password is stored as a PBKDF2-HMAC-SHA256 hash
  (240 000 iterations). Comparison is constant-time. Five failed attempts lock
  the form for five minutes.
* Database passwords are stripped from every label the app displays and from
  every log line — `safe_database_label()` is the only way a connection is ever
  described, and a test asserts a password cannot appear in a health report.
* Logs contain event types, error **class names** and timing. They never contain
  participant codes, orbital parameters, or feedback text.
* CI fails if a `.env` file, a database file, or a credential appears in a
  tracked file.

## 10. For a coordinator running a real pilot

1. Set `PILOT_ENV=production` — otherwise everything is flagged as demo data and
   excluded from your reports.
2. Set a fixed `PILOT_PARTICIPANT_SALT` **before** the pilot starts and never
   change it. Changing it makes every returning participant look new.
3. Set `DATABASE_URL` to managed Postgres. The SQLite fallback is erased on
   every restart on most hosts.
4. Check whether your institution requires ethics approval. Anonymous aggregate
   usage data is usually lighter-touch than personal data, but **that is a
   judgement for your institution, not for this document.** In particular,
   pilots involving school-age children commonly need safeguarding and parental
   consent processes that this platform does not implement and cannot replace.
5. Schedule `scripts/purge_expired.py`.
