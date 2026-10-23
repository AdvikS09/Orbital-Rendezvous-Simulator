from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from .common import StrictModel

UserType = Literal["student", "teacher", "engineer_researcher", "other"]
Track = Literal["guided", "professional", "free"]
Phase = Literal["pre", "post"]


class SessionRequest(StrictModel):
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
    runs_remaining: int | None = None


class TrackSelection(StrictModel):
    track: Track


class FeedbackRequest(StrictModel):
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


class AssessmentItem(StrictModel):
    id: str
    question: str
    options: list[str]
    topic: str
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
    responses: dict[str, int] = Field(default_factory=dict)
    confidence_rating: int | None = Field(default=None, ge=1, le=5)


class AssessmentResult(StrictModel):
    status: Literal["ok"] = "ok"
    recorded: bool
    instrument_id: str
    phase: Phase
    score: int
    max_score: int
    previous_score: int | None = None
    change: int | None = None
    items: list[AssessmentItem] = Field(default_factory=list)
    caveat: str = ""


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
