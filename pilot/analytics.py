"""Anonymous usage analytics: identity, event recording and metric queries.

Privacy rules enforced here, not merely documented:

* No name, email address, exact age, IP address or user agent is accepted by any
  function in this module -- there are no parameters for them.
* The participant code is never stored. Only ``HMAC-SHA256(salt, pilot||code)``
  is, so a returning participant is recognisable but not identifiable.
* Event detail is restricted to **categories and derived numbers**. Raw
  user-supplied orbital parameters are dropped unless
  ``PILOT_LOG_RAW_PARAMETERS`` is explicitly enabled.
* Every write carries ``is_demo``, set from ``PILOT_ENV``, and every reporting
  query excludes demo rows by default.

Metric definitions (used consistently across the admin area and exports):

===================== =========================================================
Unique user           an anonymous participant who has completed at least one
                      simulation
Active user           an anonymous participant who completed a simulation during
                      the selected reporting period
Returning user        an anonymous participant who used the platform on at least
                      two separate dates
Completed run         a simulation that finished successfully and displayed
                      results (one ``simulation_completed`` event)
Measured-impact user  a participant who completed both the pre- and the post-
                      assessment for the same instrument
===================== =========================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import date

from sqlalchemy import and_, distinct, func, select
from sqlalchemy.exc import SQLAlchemyError

from .config import Settings, get_settings
from .db import get_engine, utcnow
from .models import (
    assessments,
    events,
    feedback,
    organisations,
    participants,
    pilot_codes,
    sessions,
)

logger = logging.getLogger("pilot.analytics")


# --------------------------------------------------------------------------
# Event vocabulary -- the only event types that may be written
# --------------------------------------------------------------------------
class EventType:
    PILOT_ACCESSED = "pilot_accessed"
    CONSENT_GIVEN = "consent_given"
    SESSION_STARTED = "session_started"
    RETURN_VISIT = "return_visit"
    TRACK_SELECTED = "track_selected"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_REJECTED = "simulation_rejected"
    SCENARIO_CATEGORY_USED = "scenario_category_used"
    FEATURE_CATEGORY_USED = "feature_category_used"
    RESULTS_DOWNLOADED = "results_downloaded"
    ASSESSMENT_COMPLETED = "assessment_completed"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    ACTIVITY_COMPLETED = "activity_completed"


KNOWN_EVENT_TYPES = frozenset(
    value for name, value in vars(EventType).items() if not name.startswith("_")
)

USER_TYPES = ("student", "teacher", "engineer_researcher", "other")
USER_TYPE_LABELS = {
    "student": "Student",
    "teacher": "Teacher",
    "engineer_researcher": "Engineer / researcher",
    "other": "Other",
}
TRACKS = ("guided", "professional", "free")

#: Keys that may appear in ``detail_json`` without raw-parameter consent. These
#: are derived, bucketed or categorical -- never a value the participant typed.
SAFE_DETAIL_KEYS = frozenset({
    "model", "force_model", "controlled", "n_samples", "steps_bucket",
    "duration_orbits_bucket", "separation_bucket_m", "altitude_bucket_km",
    "converged", "saturated", "download_kind", "activity_id", "step_index",
    "instrument_id", "phase", "rejected_reason", "warning_count",
})


# --------------------------------------------------------------------------
# Anonymous identity
# --------------------------------------------------------------------------
def participant_hash(pilot_code: str, participant_code: str,
                     settings: Settings | None = None) -> str:
    """Irreversible, salt-keyed identifier for an anonymous participant.

    Deterministic for a given (pilot code, participant code, salt), so the same
    person is recognised on a later visit; not reversible to the code they
    chose, and not comparable across deployments with different salts.
    """
    settings = settings or get_settings()
    message = f"{pilot_code.strip().upper()}|{participant_code.strip().upper()}"
    return hmac.new(settings.participant_salt.encode("utf-8"),
                    message.encode("utf-8"), hashlib.sha256).hexdigest()


def new_session_token() -> str:
    return secrets.token_hex(16)


# --------------------------------------------------------------------------
# Identity context passed around the UI
# --------------------------------------------------------------------------
@dataclass
class ParticipantContext:
    """Everything the analytics layer needs to attribute an event."""

    participant_id: int
    participant_ref: str
    session_id: int
    session_token: str
    organisation_id: int | None
    pilot_code_id: int | None
    user_type: str
    track: str | None = None
    is_return_visit: bool = False
    organisation_name: str | None = None
    pilot_code: str | None = None


# --------------------------------------------------------------------------
# Organisation / pilot-code resolution
# --------------------------------------------------------------------------
def resolve_pilot_code(code: str, settings: Settings | None = None):
    """Look up a pilot code. Returns ``(pilot_code_row, organisation_row)``.

    Returns ``(None, None)`` when the code is unknown. Whether an unknown code
    is allowed is a policy decision made by the caller via
    ``settings.require_known_pilot_code``.
    """
    settings = settings or get_settings()
    normalised = code.strip().upper()
    if not normalised:
        return None, None
    engine = get_engine(settings)
    with engine.connect() as connection:
        row = connection.execute(
            select(pilot_codes, organisations.c.name.label("organisation_name"),
                   organisations.c.code.label("organisation_code"))
            .select_from(pilot_codes.join(
                organisations, pilot_codes.c.organisation_id == organisations.c.id))
            .where(func.upper(pilot_codes.c.code) == normalised)
        ).mappings().first()
    if row is None:
        return None, None
    return row, {"id": row["organisation_id"], "name": row["organisation_name"],
                 "code": row["organisation_code"]}


def ensure_unlisted_organisation(settings: Settings | None = None) -> int:
    """Organisation used for participants whose pilot code is not registered.

    Keeps such sessions countable without inventing an institutional
    relationship: the organisation is literally called "Unregistered pilot code".
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    code = "UNREGISTERED"
    with engine.begin() as connection:
        existing = connection.execute(
            select(organisations.c.id).where(organisations.c.code == code)
        ).scalar_one_or_none()
        if existing is not None:
            return int(existing)
        return int(connection.execute(organisations.insert().values(
            code=code, name="Unregistered pilot code", org_type="other",
            is_demo=settings.is_demo_data, active=True, created_at=utcnow(),
            notes="Auto-created for participants entering a code that is not registered.",
        ).returning(organisations.c.id)).scalar_one())


def ensure_unlisted_pilot_code(code: str, organisation_id: int,
                               settings: Settings | None = None) -> int:
    """Record an unregistered pilot code so cohorts stay distinguishable.

    Without this, every unregistered code collapses into the single
    "Unregistered pilot code" organisation and two different cohorts that both
    forgot to register become impossible to tell apart in reporting.

    The code itself is recorded, but the *organisation* stays "Unregistered
    pilot code" -- so nothing here invents an institution or an institutional
    relationship. A pilot code is a shared cohort identifier chosen by a
    coordinator, not personal data.
    """
    settings = settings or get_settings()
    normalised = code.strip().upper()
    engine = get_engine(settings)
    with engine.begin() as connection:
        existing = connection.execute(
            select(pilot_codes.c.id).where(func.upper(pilot_codes.c.code) == normalised)
        ).scalar_one_or_none()
        if existing is not None:
            return int(existing)
        return int(connection.execute(pilot_codes.insert().values(
            organisation_id=organisation_id, code=normalised,
            label="(entered by a participant; not registered by a coordinator)",
            active=True, is_demo=settings.is_demo_data, created_at=utcnow(),
        ).returning(pilot_codes.c.id)).scalar_one())


# --------------------------------------------------------------------------
# Session start
# --------------------------------------------------------------------------
def start_session(pilot_code: str, participant_code: str, user_type: str,
                  consent_version: str, settings: Settings | None = None
                  ) -> ParticipantContext:
    """Register (or recognise) an anonymous participant and open a session.

    Also maintains the two facts that "returning user" depends on:
    ``distinct_active_days`` and ``last_seen_date``.
    """
    settings = settings or get_settings()
    if user_type not in USER_TYPES:
        user_type = "other"

    digest = participant_hash(pilot_code, participant_code, settings)
    ref = digest[:8]
    now = utcnow()
    today = now.date()

    code_row, org = resolve_pilot_code(pilot_code, settings)
    if code_row is not None:
        pilot_code_id = int(code_row["id"])
        organisation_id = int(org["id"])
        organisation_name = org["name"]
    else:
        organisation_id = ensure_unlisted_organisation(settings)
        organisation_name = "Unregistered pilot code"
        # Record the code itself so two unregistered cohorts stay distinguishable.
        pilot_code_id = ensure_unlisted_pilot_code(
            pilot_code, organisation_id, settings) if pilot_code.strip() else None

    engine = get_engine(settings)
    with engine.begin() as connection:
        existing = connection.execute(
            select(participants).where(participants.c.participant_hash == digest)
        ).mappings().first()

        if existing is None:
            participant_id = int(connection.execute(participants.insert().values(
                participant_hash=digest, participant_ref=ref,
                pilot_code_id=pilot_code_id, organisation_id=organisation_id,
                user_type=user_type, first_seen_at=now, last_seen_at=now,
                first_seen_date=today, last_seen_date=today,
                distinct_active_days=1, consent_version=consent_version,
                consent_at=now, is_demo=settings.is_demo_data,
            ).returning(participants.c.id)).scalar_one())
            is_return_visit = False
        else:
            participant_id = int(existing["id"])
            new_day = existing["last_seen_date"] != today
            # "Returning" means a *separate date*, not merely a new session.
            is_return_visit = new_day
            connection.execute(participants.update()
                               .where(participants.c.id == participant_id)
                               .values(
                                   last_seen_at=now, last_seen_date=today,
                                   user_type=user_type,
                                   distinct_active_days=(
                                       existing["distinct_active_days"] + 1
                                       if new_day else existing["distinct_active_days"]),
                                   # Keep the newest known pilot/org attribution.
                                   pilot_code_id=pilot_code_id or existing["pilot_code_id"],
                                   organisation_id=organisation_id or existing["organisation_id"],
                               ))

        token = new_session_token()
        session_id = int(connection.execute(sessions.insert().values(
            session_token=token, participant_id=participant_id,
            pilot_code_id=pilot_code_id, organisation_id=organisation_id,
            started_at=now, started_date=today, last_active_at=now,
            app_version=settings.app_version, is_return_visit=is_return_visit,
            is_demo=settings.is_demo_data,
        ).returning(sessions.c.id)).scalar_one())

    context = ParticipantContext(
        participant_id=participant_id, participant_ref=ref, session_id=session_id,
        session_token=token, organisation_id=organisation_id,
        pilot_code_id=pilot_code_id, user_type=user_type,
        is_return_visit=is_return_visit, organisation_name=organisation_name,
        pilot_code=pilot_code.strip().upper() or None,
    )

    record_event(context, EventType.SESSION_STARTED, settings=settings)
    if is_return_visit:
        record_event(context, EventType.RETURN_VISIT, settings=settings)
    return context


# --------------------------------------------------------------------------
# Event recording
# --------------------------------------------------------------------------
def _sanitise_detail(detail: dict | None, settings: Settings) -> str | None:
    """Drop anything that is not an approved, non-identifying key."""
    if not detail:
        return None
    if settings.log_raw_parameters:
        safe = dict(detail)
    else:
        safe = {k: v for k, v in detail.items() if k in SAFE_DETAIL_KEYS}
    if not safe:
        return None
    try:
        return json.dumps(safe, default=str)[:4000]
    except (TypeError, ValueError):
        return None


def record_event(context: ParticipantContext | None, event_type: str, *,
                 scenario_category: str | None = None,
                 scenario_id: str | None = None,
                 feature_category: str | None = None,
                 duration_ms: int | None = None,
                 success: bool | None = None,
                 error_class: str | None = None,
                 detail: dict | None = None,
                 settings: Settings | None = None) -> bool:
    """Write one analytics event. Never raises.

    Analytics must not be able to break a simulation the participant is running,
    so a database failure is logged and swallowed. The return value says whether
    the write succeeded, which the tests assert on.
    """
    settings = settings or get_settings()
    if event_type not in KNOWN_EVENT_TYPES:
        logger.warning("refusing to record unknown event type %r", event_type)
        return False

    now = utcnow()
    values = {
        "event_type": event_type,
        "occurred_at": now,
        "occurred_date": now.date(),
        "session_id": context.session_id if context else None,
        "participant_id": context.participant_id if context else None,
        "pilot_code_id": context.pilot_code_id if context else None,
        "organisation_id": context.organisation_id if context else None,
        "user_type": context.user_type if context else None,
        "track": context.track if context else None,
        "scenario_category": scenario_category,
        "scenario_id": scenario_id,
        "feature_category": feature_category,
        "duration_ms": duration_ms,
        "success": success,
        "error_class": error_class,
        "detail_json": _sanitise_detail(detail, settings),
        "app_version": settings.app_version,
        "is_demo": settings.is_demo_data,
    }

    try:
        engine = get_engine(settings)
        with engine.begin() as connection:
            connection.execute(events.insert().values(**values))
            if context is not None:
                connection.execute(sessions.update()
                                   .where(sessions.c.id == context.session_id)
                                   .values(last_active_at=now))
        return True
    except SQLAlchemyError as exc:
        # Log the class and message, never the values (they may hold parameters).
        logger.error("failed to record %s event: %s", event_type, type(exc).__name__)
        return False


def set_track(context: ParticipantContext, track: str,
              settings: Settings | None = None) -> None:
    """Record the participant's chosen track on the session and in events."""
    settings = settings or get_settings()
    if track not in TRACKS:
        return
    context.track = track
    try:
        engine = get_engine(settings)
        with engine.begin() as connection:
            connection.execute(sessions.update()
                               .where(sessions.c.id == context.session_id)
                               .values(track=track))
    except SQLAlchemyError as exc:
        logger.error("failed to set track: %s", type(exc).__name__)
    record_event(context, EventType.TRACK_SELECTED, feature_category=track,
                 settings=settings)


# --------------------------------------------------------------------------
# Assessments and feedback
# --------------------------------------------------------------------------
def record_assessment(context: ParticipantContext, *, instrument_id: str,
                      instrument_version: str, phase: str, score: int,
                      max_score: int, item_responses: list | None = None,
                      confidence_rating: int | None = None,
                      settings: Settings | None = None) -> bool:
    """Store one pre- or post-assessment result.

    ``item_responses`` holds per-item correctness only (the item id and whether
    it was right), never the free text of an answer.
    """
    settings = settings or get_settings()
    if phase not in ("pre", "post"):
        return False
    now = utcnow()
    payload = json.dumps(item_responses)[:4000] if item_responses else None

    try:
        engine = get_engine(settings)
        with engine.begin() as connection:
            existing = connection.execute(
                select(assessments.c.id).where(and_(
                    assessments.c.participant_id == context.participant_id,
                    assessments.c.instrument_id == instrument_id,
                    assessments.c.phase == phase))
            ).scalar_one_or_none()
            values = {
                "session_id": context.session_id,
                "organisation_id": context.organisation_id,
                "pilot_code_id": context.pilot_code_id,
                "instrument_version": instrument_version,
                "score": int(score), "max_score": int(max_score),
                "confidence_rating": confidence_rating,
                "item_responses_json": payload,
                "completed_at": now, "completed_date": now.date(),
                "user_type": context.user_type, "is_demo": settings.is_demo_data,
            }
            if existing is not None:
                # Re-taking overwrites: one pre and one post per participant.
                connection.execute(assessments.update()
                                   .where(assessments.c.id == existing)
                                   .values(**values))
            else:
                connection.execute(assessments.insert().values(
                    participant_id=context.participant_id,
                    instrument_id=instrument_id, phase=phase, **values))
    except SQLAlchemyError as exc:
        logger.error("failed to record assessment: %s", type(exc).__name__)
        return False

    record_event(context, EventType.ASSESSMENT_COMPLETED,
                 feature_category="assessment",
                 detail={"instrument_id": instrument_id, "phase": phase},
                 settings=settings)
    return True


def record_feedback(context: ParticipantContext, *,
                    usefulness_rating: int | None = None,
                    usability_rating: int | None = None,
                    would_recommend: int | None = None,
                    clarity_rating: int | None = None,
                    most_useful: str | None = None,
                    improvement: str | None = None,
                    bug_report: str | None = None,
                    feature_request: str | None = None,
                    settings: Settings | None = None) -> bool:
    """Store one feedback submission."""
    settings = settings or get_settings()
    now = utcnow()

    def clip(text: str | None) -> str | None:
        if not text:
            return None
        return text.strip()[:2000] or None

    try:
        engine = get_engine(settings)
        with engine.begin() as connection:
            connection.execute(feedback.insert().values(
                participant_id=context.participant_id, session_id=context.session_id,
                organisation_id=context.organisation_id,
                pilot_code_id=context.pilot_code_id, user_type=context.user_type,
                track=context.track, usefulness_rating=usefulness_rating,
                usability_rating=usability_rating, would_recommend=would_recommend,
                clarity_rating=clarity_rating, most_useful=clip(most_useful),
                improvement=clip(improvement), bug_report=clip(bug_report),
                feature_request=clip(feature_request),
                submitted_at=now, submitted_date=now.date(),
                is_demo=settings.is_demo_data))
    except SQLAlchemyError as exc:
        logger.error("failed to record feedback: %s", type(exc).__name__)
        return False

    record_event(context, EventType.FEEDBACK_SUBMITTED, feature_category="feedback",
                 settings=settings)
    return True


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _demo_filter(table, include_demo: bool):
    return None if include_demo else (table.c.is_demo == False)  # noqa: E712


def _apply_filters(query, table, *, include_demo: bool,
                   start: date | None, end: date | None, date_column,
                   organisation_id: int | None = None,
                   user_type: str | None = None):
    conditions = []
    demo = _demo_filter(table, include_demo)
    if demo is not None:
        conditions.append(demo)
    if start is not None:
        conditions.append(date_column >= start)
    if end is not None:
        conditions.append(date_column <= end)
    if organisation_id is not None:
        conditions.append(table.c.organisation_id == organisation_id)
    if user_type is not None and "user_type" in table.c:
        conditions.append(table.c.user_type == user_type)
    return query.where(and_(*conditions)) if conditions else query


@dataclass
class ReportFilters:
    """Filter set shared by every admin query."""

    start: date | None = None
    end: date | None = None
    organisation_id: int | None = None
    user_type: str | None = None
    include_demo: bool = False


def summary_metrics(filters: ReportFilters, settings: Settings | None = None) -> dict:
    """Headline pilot metrics, using the definitions documented above."""
    settings = settings or get_settings()
    engine = get_engine(settings)
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    failed = events.c.event_type == EventType.SIMULATION_FAILED

    with engine.connect() as connection:
        def count_participants(condition):
            query = select(func.count(distinct(events.c.participant_id))).where(condition)
            query = _apply_filters(query, events, include_demo=filters.include_demo,
                                   start=filters.start, end=filters.end,
                                   date_column=events.c.occurred_date,
                                   organisation_id=filters.organisation_id,
                                   user_type=filters.user_type)
            return int(connection.execute(query).scalar() or 0)

        def count_events(condition):
            query = select(func.count()).select_from(events).where(condition)
            query = _apply_filters(query, events, include_demo=filters.include_demo,
                                   start=filters.start, end=filters.end,
                                   date_column=events.c.occurred_date,
                                   organisation_id=filters.organisation_id,
                                   user_type=filters.user_type)
            return int(connection.execute(query).scalar() or 0)

        active_users = count_participants(completed)
        completed_runs = count_events(completed)
        failed_runs = count_events(failed)

        # Unique users: at least one completed simulation *ever* within the
        # organisation/user-type filter -- deliberately not date-limited, since
        # "unique" is a property of the participant, not of the window.
        unique_query = select(func.count(distinct(events.c.participant_id))).where(completed)
        unique_query = _apply_filters(unique_query, events,
                                      include_demo=filters.include_demo,
                                      start=None, end=None,
                                      date_column=events.c.occurred_date,
                                      organisation_id=filters.organisation_id,
                                      user_type=filters.user_type)
        unique_users = int(connection.execute(unique_query).scalar() or 0)

        # Returning users: activity on >= 2 separate dates. Deliberately not
        # date-windowed -- "returning" is a property of the participant.
        returning_inner = (
            select(events.c.participant_id)
            .where(events.c.participant_id.isnot(None))
            .group_by(events.c.participant_id)
            .having(func.count(distinct(events.c.occurred_date)) >= 2))
        returning_inner = _apply_filters(returning_inner, events,
                                         include_demo=filters.include_demo,
                                         start=None, end=None,
                                         date_column=events.c.occurred_date,
                                         organisation_id=filters.organisation_id,
                                         user_type=filters.user_type)
        returning_users = int(connection.execute(
            select(func.count()).select_from(returning_inner.subquery())).scalar() or 0)

        # Measured-impact users: both pre and post for the same instrument.
        impact_inner = (
            select(assessments.c.participant_id)
            .group_by(assessments.c.participant_id, assessments.c.instrument_id)
            .having(func.count(distinct(assessments.c.phase)) >= 2))
        impact_inner = _apply_filters(impact_inner, assessments,
                                      include_demo=filters.include_demo,
                                      start=filters.start, end=filters.end,
                                      date_column=assessments.c.completed_date,
                                      organisation_id=filters.organisation_id,
                                      user_type=filters.user_type)
        measured_impact = int(connection.execute(
            select(func.count(distinct(impact_inner.subquery().c.participant_id)))
        ).scalar() or 0)

        # Organisations with any recorded activity.
        org_query = select(func.count(distinct(events.c.organisation_id)))
        org_query = _apply_filters(org_query, events, include_demo=filters.include_demo,
                                   start=filters.start, end=filters.end,
                                   date_column=events.c.occurred_date,
                                   organisation_id=filters.organisation_id,
                                   user_type=filters.user_type)
        org_count = int(connection.execute(org_query).scalar() or 0)

        feedback_query = select(func.count()).select_from(feedback)
        feedback_query = _apply_filters(feedback_query, feedback,
                                        include_demo=filters.include_demo,
                                        start=filters.start, end=filters.end,
                                        date_column=feedback.c.submitted_date,
                                        organisation_id=filters.organisation_id,
                                        user_type=filters.user_type)
        feedback_count = int(connection.execute(feedback_query).scalar() or 0)

    total_runs = completed_runs + failed_runs
    return {
        "organisations": org_count,
        "unique_users": unique_users,
        "active_users": active_users,
        "returning_users": returning_users,
        "return_rate": (returning_users / unique_users) if unique_users else 0.0,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "failure_rate": (failed_runs / total_runs) if total_runs else 0.0,
        "runs_per_user": (completed_runs / unique_users) if unique_users else 0.0,
        "measured_impact_users": measured_impact,
        "feedback_submissions": feedback_count,
    }


def _rows(connection, query) -> list[dict]:
    return [dict(row) for row in connection.execute(query).mappings().all()]


def usage_by_date(filters: ReportFilters, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    query = (select(events.c.occurred_date.label("date"),
                    func.count().filter(
                        events.c.event_type == EventType.SIMULATION_COMPLETED
                    ).label("completed_runs"),
                    func.count().filter(
                        events.c.event_type == EventType.SIMULATION_FAILED
                    ).label("failed_runs"),
                    func.count(distinct(events.c.participant_id)).label("participants"))
             .group_by(events.c.occurred_date)
             .order_by(events.c.occurred_date))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def usage_by_organisation(filters: ReportFilters,
                          settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    query = (select(organisations.c.code.label("organisation_code"),
                    organisations.c.name.label("organisation"),
                    organisations.c.org_type,
                    func.count(distinct(events.c.participant_id)).label("participants"),
                    func.count().filter(completed).label("completed_runs"),
                    func.count().filter(
                        events.c.event_type == EventType.SIMULATION_FAILED
                    ).label("failed_runs"),
                    func.count(distinct(events.c.occurred_date)).label("active_days"))
             .select_from(events.join(organisations,
                                      events.c.organisation_id == organisations.c.id))
             .group_by(organisations.c.code, organisations.c.name, organisations.c.org_type)
             .order_by(func.count().filter(completed).desc()))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def usage_by_pilot_code(filters: ReportFilters,
                        settings: Settings | None = None) -> list[dict]:
    """Activity per pilot code -- the cohort dimension.

    Separate from :func:`usage_by_organisation` because one organisation may run
    several cohorts, and because unregistered codes all share the single
    "Unregistered pilot code" organisation while remaining distinct cohorts.
    """
    settings = settings or get_settings()
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    query = (select(pilot_codes.c.code.label("pilot_code"),
                    pilot_codes.c.label,
                    organisations.c.name.label("organisation"),
                    func.count(distinct(events.c.participant_id)).label("participants"),
                    func.count().filter(completed).label("completed_runs"),
                    func.count(distinct(events.c.occurred_date)).label("active_days"))
             .select_from(events.join(pilot_codes,
                                      events.c.pilot_code_id == pilot_codes.c.id)
                          .outerjoin(organisations,
                                     organisations.c.id == pilot_codes.c.organisation_id))
             .group_by(pilot_codes.c.code, pilot_codes.c.label, organisations.c.name)
             .order_by(func.count().filter(completed).desc()))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def usage_by_scenario(filters: ReportFilters, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    query = (select(events.c.scenario_category, events.c.scenario_id,
                    func.count().filter(completed).label("completed_runs"),
                    func.count().filter(
                        events.c.event_type == EventType.SIMULATION_FAILED
                    ).label("failed_runs"),
                    func.count(distinct(events.c.participant_id)).label("participants"),
                    func.avg(events.c.duration_ms).filter(completed).label("mean_duration_ms"))
             .where(events.c.scenario_id.isnot(None))
             .group_by(events.c.scenario_category, events.c.scenario_id)
             .order_by(func.count().filter(completed).desc()))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def usage_by_user_type(filters: ReportFilters, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    query = (select(events.c.user_type,
                    func.count(distinct(events.c.participant_id)).label("participants"),
                    func.count().filter(completed).label("completed_runs"))
             .group_by(events.c.user_type)
             .order_by(func.count().filter(completed).desc()))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def usage_by_feature(filters: ReportFilters, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    query = (select(events.c.feature_category,
                    func.count().label("uses"),
                    func.count(distinct(events.c.participant_id)).label("participants"))
             .where(events.c.feature_category.isnot(None))
             .group_by(events.c.feature_category)
             .order_by(func.count().desc()))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def simulations_per_user(filters: ReportFilters,
                         settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    completed = events.c.event_type == EventType.SIMULATION_COMPLETED
    inner = (select(events.c.participant_id.label("participant_id"),
                    func.count().label("completed_runs"),
                    func.count(distinct(events.c.occurred_date)).label("active_days"))
             .where(completed)
             .group_by(events.c.participant_id))
    inner = _apply_filters(inner, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    sub = inner.subquery()
    query = (select(participants.c.participant_ref, participants.c.user_type,
                    organisations.c.name.label("organisation"),
                    sub.c.completed_runs, sub.c.active_days)
             .select_from(sub.join(participants, participants.c.id == sub.c.participant_id)
                          .outerjoin(organisations,
                                     organisations.c.id == participants.c.organisation_id))
             .order_by(sub.c.completed_runs.desc()))
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def assessment_changes(filters: ReportFilters,
                       settings: Settings | None = None) -> list[dict]:
    """Per-participant pre/post scores and the change between them."""
    settings = settings or get_settings()
    pre = assessments.alias("pre")
    post = assessments.alias("post")
    query = (select(participants.c.participant_ref,
                    participants.c.user_type,
                    organisations.c.name.label("organisation"),
                    pre.c.instrument_id,
                    pre.c.score.label("pre_score"),
                    post.c.score.label("post_score"),
                    pre.c.max_score,
                    (post.c.score - pre.c.score).label("change"),
                    pre.c.confidence_rating.label("pre_confidence"),
                    post.c.confidence_rating.label("post_confidence"),
                    post.c.completed_date.label("post_date"))
             .select_from(
                 pre.join(post, and_(pre.c.participant_id == post.c.participant_id,
                                     pre.c.instrument_id == post.c.instrument_id))
                    .join(participants, participants.c.id == pre.c.participant_id)
                    .outerjoin(organisations,
                               organisations.c.id == pre.c.organisation_id))
             .where(and_(pre.c.phase == "pre", post.c.phase == "post")))

    conditions = []
    if not filters.include_demo:
        conditions.append(pre.c.is_demo == False)  # noqa: E712
    if filters.start is not None:
        conditions.append(post.c.completed_date >= filters.start)
    if filters.end is not None:
        conditions.append(post.c.completed_date <= filters.end)
    if filters.organisation_id is not None:
        conditions.append(pre.c.organisation_id == filters.organisation_id)
    if filters.user_type is not None:
        conditions.append(pre.c.user_type == filters.user_type)
    if conditions:
        query = query.where(and_(*conditions))

    with get_engine(settings).connect() as connection:
        return _rows(connection, query.order_by(post.c.completed_date.desc()))


def feedback_rows(filters: ReportFilters, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    query = (select(feedback.c.submitted_date, feedback.c.user_type, feedback.c.track,
                    organisations.c.name.label("organisation"),
                    participants.c.participant_ref,
                    feedback.c.usefulness_rating, feedback.c.usability_rating,
                    feedback.c.clarity_rating, feedback.c.would_recommend,
                    feedback.c.most_useful, feedback.c.improvement,
                    feedback.c.bug_report, feedback.c.feature_request)
             .select_from(feedback
                          .outerjoin(participants,
                                     participants.c.id == feedback.c.participant_id)
                          .outerjoin(organisations,
                                     organisations.c.id == feedback.c.organisation_id))
             .order_by(feedback.c.submitted_at.desc()))
    query = _apply_filters(query, feedback, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=feedback.c.submitted_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def feedback_ratings_summary(filters: ReportFilters,
                             settings: Settings | None = None) -> dict:
    """Mean ratings, split so professional usefulness can be read separately."""
    settings = settings or get_settings()
    professional = feedback.c.user_type == "engineer_researcher"
    query = select(
        func.count().label("responses"),
        func.avg(feedback.c.usefulness_rating).label("mean_usefulness"),
        func.avg(feedback.c.usability_rating).label("mean_usability"),
        func.avg(feedback.c.clarity_rating).label("mean_clarity"),
        func.avg(feedback.c.would_recommend).label("mean_recommend"),
        func.count().filter(professional).label("professional_responses"),
        func.avg(feedback.c.usefulness_rating).filter(professional)
            .label("professional_mean_usefulness"),
        func.avg(feedback.c.usability_rating).filter(professional)
            .label("professional_mean_usability"),
    ).select_from(feedback)
    query = _apply_filters(query, feedback, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=feedback.c.submitted_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        row = connection.execute(query).mappings().first()
    return dict(row) if row else {}


def bug_and_feature_rows(filters: ReportFilters,
                         settings: Settings | None = None) -> list[dict]:
    """Feedback rows that actually reported a bug or requested a feature."""
    rows = feedback_rows(filters, settings)
    return [row for row in rows if row.get("bug_report") or row.get("feature_request")]


def list_organisations(include_demo: bool = False,
                       settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    query = select(organisations).order_by(organisations.c.name)
    if not include_demo:
        query = query.where(organisations.c.is_demo == False)  # noqa: E712
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


def recent_events(filters: ReportFilters, limit: int = 200,
                  settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    query = (select(events.c.occurred_at, events.c.event_type, events.c.user_type,
                    events.c.track, events.c.scenario_category, events.c.scenario_id,
                    events.c.feature_category, events.c.duration_ms, events.c.success,
                    events.c.error_class, participants.c.participant_ref,
                    organisations.c.name.label("organisation"))
             .select_from(events
                          .outerjoin(participants,
                                     participants.c.id == events.c.participant_id)
                          .outerjoin(organisations,
                                     organisations.c.id == events.c.organisation_id))
             .order_by(events.c.occurred_at.desc()).limit(limit))
    query = _apply_filters(query, events, include_demo=filters.include_demo,
                           start=filters.start, end=filters.end,
                           date_column=events.c.occurred_date,
                           organisation_id=filters.organisation_id,
                           user_type=filters.user_type)
    with get_engine(settings).connect() as connection:
        return _rows(connection, query)


__all__ = [
    "EventType", "KNOWN_EVENT_TYPES", "USER_TYPES", "USER_TYPE_LABELS", "TRACKS",
    "ParticipantContext", "ReportFilters", "participant_hash", "start_session",
    "record_event", "set_track", "record_assessment", "record_feedback",
    "resolve_pilot_code", "summary_metrics", "usage_by_date",
    "usage_by_organisation", "usage_by_pilot_code", "usage_by_scenario",
    "usage_by_user_type", "ensure_unlisted_pilot_code",
    "usage_by_feature", "simulations_per_user", "assessment_changes",
    "feedback_rows", "feedback_ratings_summary", "bug_and_feature_rows",
    "list_organisations", "recent_events",
]
