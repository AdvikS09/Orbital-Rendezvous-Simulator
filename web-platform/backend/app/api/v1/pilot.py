from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app import bootstrap  # noqa: F401

from pilot import analytics  # noqa: E402
from pilot.analytics import EventType  # noqa: E402
from pilot.config import get_settings  # noqa: E402
from pilot.content.activities import ACTIVITIES, get_activity  # noqa: E402
from pilot.content.assessments import (  # noqa: E402
    get_instrument,
    score_responses,
)
from pilot.ratelimit import get_limiter  # noqa: E402
from pilot.validation import (  # noqa: E402
    validate_participant_code,
    validate_pilot_code,
)

from app.api.deps import optional_participant  # noqa: E402
from app.schemas.pilot import (  # noqa: E402
    Acknowledgement,
    ActivityModel,
    ActivityStepModel,
    AssessmentInstrument,
    AssessmentItem,
    AssessmentResult,
    AssessmentSubmission,
    FeedbackRequest,
    SessionRequest,
    SessionResponse,
    TrackSelection,
)

router = APIRouter(tags=["pilot"])

IMPACT_CAVEAT = (
    "One person's before-and-after is not evidence of much on its own - you have "
    "seen the questions twice, which helps by itself. Pooled across a whole group "
    "it becomes more informative."
)


def _require_context(context):
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "No anonymous session. Call POST /api/v1/sessions first and send "
                "the returned token in the X-Session-Token header."
            ),
        )
    return context


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start an anonymous session",
)
def start_session(payload: Annotated[SessionRequest, Body()]) -> SessionResponse:
    """Register or recognise an anonymous participant.

    The participant code is never stored. What is stored is an HMAC of the
    pilot and participant codes keyed with a server secret, which recognises a
    returning participant without being reversible to the code they chose.
    """
    settings = get_settings()

    pilot_code, pilot_error = validate_pilot_code(payload.pilot_code)
    participant_code, participant_error = validate_participant_code(
        payload.participant_code
    )
    if pilot_error or participant_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {"field": field, "message": message}
                for field, message in (
                    ("pilot_code", pilot_error),
                    ("participant_code", participant_error),
                )
                if message
            ],
        )

    code_row, _ = analytics.resolve_pilot_code(pilot_code, settings)
    if code_row is None and settings.require_known_pilot_code:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{
                "field": "pilot_code",
                "message": ("That pilot code is not registered on this deployment. "
                            "Check it with your coordinator."),
            }],
        )

    try:
        context = analytics.start_session(
            pilot_code=pilot_code,
            participant_code=participant_code,
            user_type=payload.user_type,
            consent_version=payload.consent_version,
            settings=settings,
        )
    except Exception:  # noqa: BLE001 - never surface a database traceback
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The usage database is not reachable, so a tracked session could "
                "not be created. Simulations still work without a session token."
            ),
        )

    analytics.record_event(context, EventType.PILOT_ACCESSED, settings=settings)
    decision = get_limiter().check(context.session_token)

    return SessionResponse(
        session_token=context.session_token,
        participant_ref=context.participant_ref,
        user_type=context.user_type,  # type: ignore[arg-type]
        organisation_name=context.organisation_name or "Unregistered pilot code",
        pilot_code=context.pilot_code,
        pilot_code_registered=code_row is not None,
        is_return_visit=context.is_return_visit,
        consent_version=payload.consent_version,
        runs_remaining=decision.runs_remaining,
    )


@router.post("/sessions/track", response_model=Acknowledgement, summary="Record the chosen track")
def select_track(
    payload: Annotated[TrackSelection, Body()],
    context=Depends(optional_participant),
) -> Acknowledgement:
    context = _require_context(context)
    analytics.set_track(context, payload.track)
    return Acknowledgement(recorded=True, message=f"Track set to {payload.track}.")


@router.post("/feedback", response_model=Acknowledgement, summary="Submit feedback")
def submit_feedback(
    payload: Annotated[FeedbackRequest, Body()],
    context=Depends(optional_participant),
) -> Acknowledgement:
    context = _require_context(context)
    recorded = analytics.record_feedback(
        context,
        usefulness_rating=payload.usefulness_rating,
        usability_rating=payload.usability_rating,
        clarity_rating=payload.clarity_rating,
        would_recommend=payload.would_recommend,
        most_useful=payload.most_useful,
        improvement=payload.improvement,
        bug_report=payload.bug_report,
        feature_request=payload.feature_request,
    )
    return Acknowledgement(
        recorded=recorded,
        message=(
            "Thank you - your feedback has been recorded anonymously."
            if recorded
            else "Your feedback could not be saved because the database is not reachable."
        ),
    )


def _to_item(item, reveal: bool) -> AssessmentItem:
    return AssessmentItem(
        id=item.id,
        question=item.question,
        options=list(item.options),
        topic=item.topic,
        correct_index=item.correct_index if reveal else None,
        explanation=item.explanation if reveal else None,
    )


@router.get(
    "/assessments/{instrument_id}",
    response_model=AssessmentInstrument,
    summary="Fetch an assessment instrument",
)
def get_assessment(instrument_id: str) -> AssessmentInstrument:
    """Return the questions.

    Correct answers are withheld here so the pre-assessment cannot be answered
    by reading the API; they come back with the post-assessment result.
    """
    instrument = get_instrument(instrument_id)
    return AssessmentInstrument(
        id=instrument.id,
        version=instrument.version,
        title=instrument.title,
        description=instrument.description,
        items=[_to_item(item, reveal=False) for item in instrument.items],
        confidence_question=instrument.confidence_question,
        confidence_labels=list(instrument.confidence_labels),
        max_score=instrument.max_score,
        review_status=instrument.review_status,
    )


@router.post(
    "/assessments/{instrument_id}/submit",
    response_model=AssessmentResult,
    summary="Submit assessment answers",
)
def submit_assessment(
    instrument_id: str,
    payload: Annotated[AssessmentSubmission, Body()],
    context=Depends(optional_participant),
) -> AssessmentResult:
    context = _require_context(context)
    instrument = get_instrument(instrument_id)
    score, per_item = score_responses(instrument, payload.responses)

    recorded = analytics.record_assessment(
        context,
        instrument_id=instrument.id,
        instrument_version=instrument.version,
        phase=payload.phase,
        score=score,
        max_score=instrument.max_score,
        item_responses=per_item,
        confidence_rating=payload.confidence_rating,
    )

    previous: int | None = None
    change: int | None = None
    if payload.phase == "post":
        rows = analytics.assessment_changes(
            analytics.ReportFilters(include_demo=True)
        )
        mine = [
            row for row in rows
            if row["participant_ref"] == context.participant_ref
            and row["instrument_id"] == instrument.id
        ]
        if mine:
            previous = int(mine[0]["pre_score"])
            change = int(mine[0]["change"])

    return AssessmentResult(
        recorded=recorded,
        instrument_id=instrument.id,
        phase=payload.phase,
        score=score,
        max_score=instrument.max_score,
        previous_score=previous,
        change=change,
        items=(
            [_to_item(item, reveal=True) for item in instrument.items]
            if payload.phase == "post"
            else []
        ),
        caveat=IMPACT_CAVEAT if payload.phase == "post" else "",
    )


def _to_activity(activity) -> ActivityModel:
    return ActivityModel(
        id=activity.id,
        title=activity.title,
        level=activity.level,
        duration_minutes=activity.duration_minutes,
        summary=activity.summary,
        learning_points=list(activity.learning_points),
        prerequisites=activity.prerequisites,
        review_status=activity.review_status,
        steps=[
            ActivityStepModel(
                id=step.id,
                title=step.title,
                scenario_id=step.scenario_id,
                brief=step.brief,
                predict_prompt=step.predict_prompt,
                predict_options=list(step.predict_options),
                params=dict(step.params),
                look_for=step.look_for,
                takeaway=step.takeaway,
            )
            for step in activity.steps
        ],
    )


@router.get(
    "/activities",
    response_model=list[ActivityModel],
    summary="List the guided educational activities",
)
def list_activities() -> list[ActivityModel]:
    return [_to_activity(activity) for activity in ACTIVITIES]


@router.get(
    "/activities/{activity_id}",
    response_model=ActivityModel,
    summary="One guided activity with all of its steps",
)
def read_activity(activity_id: str) -> ActivityModel:
    activity = get_activity(activity_id)
    if activity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No activity with id {activity_id!r}.",
        )
    return _to_activity(activity)


@router.post(
    "/activities/{activity_id}/complete",
    response_model=Acknowledgement,
    summary="Record that an activity was finished",
)
def complete_activity(
    activity_id: str,
    context=Depends(optional_participant),
) -> Acknowledgement:
    context = _require_context(context)
    if get_activity(activity_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No activity with id {activity_id!r}.",
        )
    recorded = analytics.record_event(
        context,
        EventType.ACTIVITY_COMPLETED,
        feature_category="guided_activity",
        detail={"activity_id": activity_id},
    )
    return Acknowledgement(recorded=recorded)
