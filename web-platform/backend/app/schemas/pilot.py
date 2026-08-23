"""Sessions, feedback, assessments and administrator reporting.

These mirror the anonymity guarantees of the Streamlit platform exactly. In
particular there is no field anywhere in this module for a name, an email
address, an age, an IP address or a user agent -- the request models simply do
not accept them, so a future caller cannot start sending them by accident.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from .common import StrictModel

UserType = Literal["student", "teacher", "engineer_researcher", "other"]
Track = Literal["guided", "professional", "free"]
Phase = Literal["pre", "post"]


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
class SessionRequest(StrictModel):
    """Start an anonymous session.

    ``participant_code`` is chosen by the participant and is never stored: the
    server keeps only an HMAC of it. It is rejected if it looks like personal
    data (an email address, or anything with spaces).
    """

    pilot_code: str = Field(min_length=1, max_length=40)
    participant_code: str = Field(min_length=3, max_length=40)
    user_type: UserType = "other"
    consent_given: bool
    consent_version: str = "1.0"

    @field_validator("consent_given")
    @classmethod
    def _must_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "Consent is required before an anonymous session can be created."
            )
        return value


class SessionResponse(StrictModel):
    session_token: str
    participant_ref: str
    user_type: UserType
    organisation_name: str
    pilot_code: str | None = None
    pilot_code_registered: bool
    is_return_visit: bool
    consent_version: str
    #: Runs still available in this session, for the interface to show.
    runs_remaining: int | None = None


class TrackSelection(StrictModel):
    track: Track


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------
class FeedbackRequest(StrictModel):
    """One feedback submission.

    Free text is stored verbatim and capped; the form warns participants not to
    put personal information in it. That is the single place where personal
    data could enter the database, and only if someone types it.
    """

    usefulness_rating: int | None = Field(default=None, ge=1, le=5)
    usability_rating: int | None = Field(default=None, ge=1, le=5)
    clarity_rating: int | None = Field(default=None, ge=1, le=5)
    would_recommend: int | None = Field(default=None, ge=0, le=10)
    most_useful: str | None = Field(default=None, max_length=2000)
    improvement: str | None = Field(default=None, max_length=2000)
    bug_report: str | None = Field(default=None, max_length=2000)
    feature_request: str | None = Field(default=None, max_length=2000)


class Acknowledgement(StrictModel):
    status: Literal["ok"] = "ok"
    recorded: bool
    message: str = ""


# --------------------------------------------------------------------------
# Assessments
# --------------------------------------------------------------------------
class AssessmentItem(StrictModel):
    id: str
    question: str
    options: list[str]
    topic: str
    #: The correct answer is withheld until the post-assessment is submitted, so
    #: the pre-assessment cannot be answered by reading the API.
    correct_index: int | None = None
    explanation: str | None = None


class AssessmentInstrument(StrictModel):
    id: str
    version: str
    title: str
    description: str
    items: list[AssessmentItem]
    confidence_question: str
    confidence_labels: list[str]
    max_score: int
    review_status: str


class AssessmentSubmission(StrictModel):
    instrument_id: str = "orbital_basics"
    phase: Phase
    #: item id -> chosen option index. Unanswered items may be omitted and are
    #: scored as incorrect, matching the Streamlit behaviour.
    responses: dict[str, int] = Field(default_factory=dict)
    confidence_rating: int | None = Field(default=None, ge=1, le=5)


class AssessmentResult(StrictModel):
    status: Literal["ok"] = "ok"
    recorded: bool
    instrument_id: str
    phase: Phase
    score: int
    max_score: int
    #: Populated only after the post assessment.
    previous_score: int | None = None
    change: int | None = None
    items: list[AssessmentItem] = Field(default_factory=list)
    caveat: str = ""


# --------------------------------------------------------------------------
# Educational activities
# --------------------------------------------------------------------------
class ActivityStepModel(StrictModel):
    id: str
    title: str
    scenario_id: str
    brief: str
    predict_prompt: str
    predict_options: list[str]
    params: dict[str, Any]
    look_for: str
    takeaway: str


class ActivityModel(StrictModel):
    id: str
    title: str
    level: str
    duration_minutes: int
    summary: str
    learning_points: list[str]
    prerequisites: str
    review_status: str
    steps: list[ActivityStepModel]


# --------------------------------------------------------------------------
# Administration
# --------------------------------------------------------------------------
class ReportQuery(StrictModel):
    start: str | None = None
    end: str | None = None
    organisation_id: int | None = None
    user_type: UserType | None = None
    include_demo: bool = False


class SummaryMetrics(StrictModel):
    organisations: int
    unique_users: int
    active_users: int
    returning_users: int
    return_rate: float
    completed_runs: int
    failed_runs: int
    failure_rate: float
    runs_per_user: float
    measured_impact_users: int
    feedback_submissions: int


class AdminReport(StrictModel):
    summary: SummaryMetrics
    usage_by_date: list[dict[str, Any]]
    usage_by_organisation: list[dict[str, Any]]
    usage_by_pilot_code: list[dict[str, Any]]
    usage_by_scenario: list[dict[str, Any]]
    usage_by_user_type: list[dict[str, Any]]
    usage_by_feature: list[dict[str, Any]]
    simulations_per_user: list[dict[str, Any]]
    assessment_changes: list[dict[str, Any]]
    feedback: list[dict[str, Any]]
    feedback_ratings: dict[str, Any]
    bugs_and_requests: list[dict[str, Any]]
    environment: str
    is_demo_environment: bool
    include_demo: bool
    metric_definitions: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
