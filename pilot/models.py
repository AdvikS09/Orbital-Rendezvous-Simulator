"""Database schema for the pilot platform (SQLAlchemy Core).

Deliberately **anonymous by design**. The tables below cannot store a name, an
email address, an exact age, or an institution's confidential data, because
there is nowhere to put them.

Participant identity
--------------------
A participant types an organisation/pilot code and a self-chosen anonymous
participant code. Neither the participant code nor any combination of it is
stored. What is stored is ``participant_hash`` -- an HMAC-SHA256 of
``pilot_code || participant_code`` keyed with a server-side secret salt. That
gives exactly two properties:

* the same person typing the same code is recognised on a later date, so
  "returning user" is measurable; and
* the stored value cannot be reversed to the code the participant chose, and
  cannot be linked across deployments with different salts.

``participant_ref`` is the first 8 hex characters of that hash, shown in the
admin area purely so a row can be talked about ("participant 3f9a1c02").

Development data
----------------
Every table that holds pilot observations carries ``is_demo``. Rows created
while ``PILOT_ENV`` is not ``production`` are marked ``is_demo = 1``, and every
reporting query filters them out by default, so demonstration and test runs can
never be counted as real pilot activity.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()


# Pilot structure
organisations = Table(
    "organisations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("code", String(64), nullable=False, unique=True),
    Column("name", String(200), nullable=False),
    Column("org_type", String(40), nullable=False, server_default="other"),
    Column("country", String(80), nullable=True),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
    Column("active", Boolean, nullable=False, server_default="1"),
    Column("created_at", DateTime, nullable=False),
    Column("notes", Text, nullable=True),
)

pilot_codes = Table(
    "pilot_codes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=False),
    Column("code", String(64), nullable=False, unique=True),
    Column("label", String(200), nullable=True),
    Column("cohort", String(120), nullable=True),
    Column("active", Boolean, nullable=False, server_default="1"),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
    Column("created_at", DateTime, nullable=False),
    Column("expires_at", DateTime, nullable=True),
)

# Anonymous participants and sessions
participants = Table(
    "participants", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("participant_hash", String(64), nullable=False, unique=True),
    Column("participant_ref", String(8), nullable=False),
    Column("pilot_code_id", Integer, ForeignKey("pilot_codes.id"), nullable=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=True),
    # student | teacher | engineer_researcher | other
    Column("user_type", String(40), nullable=False, server_default="other"),
    Column("first_seen_at", DateTime, nullable=False),
    Column("last_seen_at", DateTime, nullable=False),
    Column("first_seen_date", Date, nullable=False),
    Column("last_seen_date", Date, nullable=False),
    Column("distinct_active_days", Integer, nullable=False, server_default="1"),
    Column("consent_version", String(20), nullable=True),
    Column("consent_at", DateTime, nullable=True),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
)

sessions = Table(
    "sessions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_token", String(64), nullable=False, unique=True),
    Column("participant_id", Integer, ForeignKey("participants.id"), nullable=False),
    Column("pilot_code_id", Integer, ForeignKey("pilot_codes.id"), nullable=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=True),
    # guided | professional | free
    Column("track", String(40), nullable=True),
    Column("started_at", DateTime, nullable=False),
    Column("started_date", Date, nullable=False),
    Column("last_active_at", DateTime, nullable=False),
    Column("app_version", String(40), nullable=True),
    Column("is_return_visit", Boolean, nullable=False, server_default="0"),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
)

# Analytics events
events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_type", String(60), nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Column("occurred_date", Date, nullable=False),
    Column("session_id", Integer, ForeignKey("sessions.id"), nullable=True),
    Column("participant_id", Integer, ForeignKey("participants.id"), nullable=True),
    Column("pilot_code_id", Integer, ForeignKey("pilot_codes.id"), nullable=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=True),
    Column("user_type", String(40), nullable=True),
    Column("track", String(40), nullable=True),
    # Categories only -- never free-form user input.
    Column("scenario_category", String(60), nullable=True),
    Column("scenario_id", String(80), nullable=True),
    Column("feature_category", String(60), nullable=True),
    Column("duration_ms", Integer, nullable=True),
    Column("success", Boolean, nullable=True),
    Column("error_class", String(80), nullable=True),
    # JSON of *derived, non-identifying* detail (bucketed values, counts).
    # Raw orbital parameters land here only when PILOT_LOG_RAW_PARAMETERS=true.
    Column("detail_json", Text, nullable=True),
    Column("app_version", String(40), nullable=True),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
)

Index("ix_events_type_date", events.c.event_type, events.c.occurred_date)
Index("ix_events_participant", events.c.participant_id)
Index("ix_events_org", events.c.organisation_id)

# Educational pre/post assessment
assessments = Table(
    "assessments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("participant_id", Integer, ForeignKey("participants.id"), nullable=False),
    Column("session_id", Integer, ForeignKey("sessions.id"), nullable=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=True),
    Column("pilot_code_id", Integer, ForeignKey("pilot_codes.id"), nullable=True),
    Column("instrument_id", String(60), nullable=False),
    Column("instrument_version", String(20), nullable=False),
    Column("phase", String(10), nullable=False),          # pre | post
    Column("score", Integer, nullable=False),
    Column("max_score", Integer, nullable=False),
    Column("confidence_rating", Integer, nullable=True),  # 1-5 self-report
    Column("item_responses_json", Text, nullable=True),   # per-item correctness only
    Column("completed_at", DateTime, nullable=False),
    Column("completed_date", Date, nullable=False),
    Column("user_type", String(40), nullable=True),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
    UniqueConstraint("participant_id", "instrument_id", "phase",
                     name="uq_assessment_participant_phase"),
)

# Feedback
feedback = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("participant_id", Integer, ForeignKey("participants.id"), nullable=False),
    Column("session_id", Integer, ForeignKey("sessions.id"), nullable=True),
    Column("organisation_id", Integer, ForeignKey("organisations.id"), nullable=True),
    Column("pilot_code_id", Integer, ForeignKey("pilot_codes.id"), nullable=True),
    Column("user_type", String(40), nullable=True),
    Column("track", String(40), nullable=True),
    Column("usefulness_rating", Integer, nullable=True),   # 1-5
    Column("usability_rating", Integer, nullable=True),    # 1-5
    Column("would_recommend", Integer, nullable=True),     # 0-10
    Column("clarity_rating", Integer, nullable=True),      # 1-5
    Column("most_useful", Text, nullable=True),
    Column("improvement", Text, nullable=True),
    Column("bug_report", Text, nullable=True),
    Column("feature_request", Text, nullable=True),
    Column("submitted_at", DateTime, nullable=False),
    Column("submitted_date", Date, nullable=False),
    Column("is_demo", Boolean, nullable=False, server_default="0"),
)

# Schema versioning
schema_migrations = Table(
    "schema_migrations", metadata,
    Column("version", Integer, primary_key=True),
    Column("name", String(120), nullable=False),
    Column("applied_at", DateTime, nullable=False),
)

# Kept so an admin can see how long a reporting query took without any
# participant data being involved.
admin_audit = Table(
    "admin_audit", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("action", String(60), nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Column("detail", String(200), nullable=True),
    Column("success", Boolean, nullable=False, server_default="1"),
)


ALL_TABLES = (
    organisations, pilot_codes, participants, sessions, events,
    assessments, feedback, schema_migrations, admin_audit,
)

#: Tables carrying participant observations -- these are what retention and the
#: demo/production split apply to.
PARTICIPANT_TABLES = (participants, sessions, events, assessments, feedback)

__all__ = [
    "metadata", "organisations", "pilot_codes", "participants", "sessions",
    "events", "assessments", "feedback", "schema_migrations", "admin_audit",
    "ALL_TABLES", "PARTICIPANT_TABLES",
]
