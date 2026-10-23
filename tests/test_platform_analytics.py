import json
from datetime import date, timedelta

import pytest
from sqlalchemy import func, select, update

from pilot import analytics, db
from pilot.analytics import EventType, ReportFilters
from pilot.models import events, feedback, participants, sessions


def test_participant_hash_is_stable_for_the_same_codes(pilot_env):
    first = analytics.participant_hash("ORG1", "ALICE01", pilot_env)
    second = analytics.participant_hash("ORG1", "ALICE01", pilot_env)
    assert first == second


def test_participant_hash_is_case_insensitive(pilot_env):
    assert (analytics.participant_hash("org1", "alice01", pilot_env)
            == analytics.participant_hash("ORG1", "ALICE01", pilot_env))


def test_participant_hash_differs_between_codes(pilot_env):
    assert (analytics.participant_hash("ORG1", "ALICE01", pilot_env)
            != analytics.participant_hash("ORG1", "ALICE02", pilot_env))


def test_participant_hash_differs_between_pilots(pilot_env):
    assert (analytics.participant_hash("ORG1", "ALICE01", pilot_env)
            != analytics.participant_hash("ORG2", "ALICE01", pilot_env))


def test_participant_hash_depends_on_the_salt(monkeypatch, pilot_env):
    from pilot import config

    baseline = analytics.participant_hash("ORG1", "ALICE01", pilot_env)
    monkeypatch.setenv("PILOT_PARTICIPANT_SALT", "a-completely-different-salt")
    other = config.get_settings(refresh=True)
    assert analytics.participant_hash("ORG1", "ALICE01", other) != baseline


def test_the_raw_participant_code_is_never_stored(pilot_env):
    analytics.start_session("SCHOOL-A", "SECRETCODE99", "student", "1.0", pilot_env)
    engine = db.get_engine(pilot_env)
    with engine.connect() as connection:
        rows = connection.execute(select(participants)).mappings().all()
    blob = json.dumps([dict(row) for row in rows], default=str)
    assert "SECRETCODE99" not in blob
    assert len(rows) == 1
    assert len(rows[0]["participant_hash"]) == 64
    assert rows[0]["participant_ref"] == rows[0]["participant_hash"][:8]


def test_returning_on_the_same_day_is_not_a_return_visit(pilot_env):
    first = analytics.start_session("ORG", "USER1", "student", "1.0", pilot_env)
    second = analytics.start_session("ORG", "USER1", "student", "1.0", pilot_env)
    assert first.participant_id == second.participant_id
    assert second.is_return_visit is False


def test_returning_on_a_later_date_is_a_return_visit(pilot_env):
    first = analytics.start_session("ORG", "USER1", "student", "1.0", pilot_env)

    engine = db.get_engine(pilot_env)
    yesterday = date.today() - timedelta(days=1)
    with engine.begin() as connection:
        connection.execute(update(participants)
                           .where(participants.c.id == first.participant_id)
                           .values(last_seen_date=yesterday, first_seen_date=yesterday))

    second = analytics.start_session("ORG", "USER1", "student", "1.0", pilot_env)
    assert second.is_return_visit is True
    assert second.participant_id == first.participant_id

    with engine.connect() as connection:
        row = connection.execute(
            select(participants).where(participants.c.id == first.participant_id)
        ).mappings().one()
    assert row["distinct_active_days"] == 2

    with engine.connect() as connection:
        types = [r[0] for r in connection.execute(select(events.c.event_type)).all()]
    assert EventType.RETURN_VISIT in types


def test_a_different_participant_gets_a_separate_record(pilot_env):
    first = analytics.start_session("ORG", "USER1", "student", "1.0", pilot_env)
    second = analytics.start_session("ORG", "USER2", "teacher", "1.0", pilot_env)
    assert first.participant_id != second.participant_id


def test_unknown_pilot_codes_get_the_unregistered_organisation(pilot_env):
    context = analytics.start_session("NOT-REGISTERED", "USER1", "other", "1.0", pilot_env)
    assert context.organisation_id is not None
    assert context.organisation_name == "Unregistered pilot code"
    assert context.pilot_code_id is not None


def test_two_unregistered_cohorts_stay_distinguishable(pilot_env):
    from pilot.models import pilot_codes

    first = analytics.start_session("SCHOOL-X", "P1", "student", "1.0", pilot_env)
    second = analytics.start_session("SCHOOL-Y", "P2", "student", "1.0", pilot_env)

    assert first.organisation_id == second.organisation_id, (
        "unregistered codes should share the 'Unregistered' organisation")
    assert first.pilot_code_id != second.pilot_code_id, (
        "different cohorts must not collapse into one pilot code")

    with db.get_engine(pilot_env).connect() as connection:
        codes = {row[0] for row in connection.execute(select(pilot_codes.c.code)).all()}
    assert {"SCHOOL-X", "SCHOOL-Y"}.issubset(codes)


def test_the_same_unregistered_code_is_reused_not_duplicated(pilot_env):
    from pilot.models import pilot_codes

    first = analytics.start_session("SCHOOL-X", "P1", "student", "1.0", pilot_env)
    second = analytics.start_session("school-x", "P2", "student", "1.0", pilot_env)
    assert first.pilot_code_id == second.pilot_code_id

    with db.get_engine(pilot_env).connect() as connection:
        count = connection.execute(
            select(func.count()).select_from(pilot_codes)).scalar()
    assert count == 1


def test_usage_by_pilot_code_separates_cohorts(pilot_env):
    alpha = analytics.start_session("COHORT-A", "P1", "student", "1.0", pilot_env)
    beta = analytics.start_session("COHORT-B", "P2", "student", "1.0", pilot_env)
    _complete_runs(alpha, 3, pilot_env)
    _complete_runs(beta, 1, pilot_env)

    rows = analytics.usage_by_pilot_code(ReportFilters(include_demo=True), pilot_env)
    by_code = {row["pilot_code"]: row for row in rows}
    assert by_code["COHORT-A"]["completed_runs"] == 3
    assert by_code["COHORT-B"]["completed_runs"] == 1


def test_registered_codes_keep_their_real_organisation(pilot_env):
    from pilot.db import utcnow
    from pilot.models import organisations, pilot_codes

    with db.get_engine(pilot_env).begin() as connection:
        org_id = int(connection.execute(organisations.insert().values(
            code="REALORG", name="Example High School", org_type="school",
            is_demo=True, active=True, created_at=utcnow(),
        ).returning(organisations.c.id)).scalar_one())
        connection.execute(pilot_codes.insert().values(
            organisation_id=org_id, code="REALORG-Y12", label="Year 12",
            active=True, is_demo=True, created_at=utcnow()))

    context = analytics.start_session("REALORG-Y12", "P1", "student", "1.0", pilot_env)
    assert context.organisation_id == org_id
    assert context.organisation_name == "Example High School"
    assert context.pilot_code_id is not None


def test_record_event_writes_a_row(participant, pilot_env):
    assert analytics.record_event(participant, EventType.SIMULATION_COMPLETED,
                                  scenario_category="relative_motion",
                                  scenario_id="pd_rendezvous", duration_ms=120,
                                  success=True, settings=pilot_env)
    with db.get_engine(pilot_env).connect() as connection:
        rows = connection.execute(
            select(events).where(events.c.event_type == EventType.SIMULATION_COMPLETED)
        ).mappings().all()
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "pd_rendezvous"
    assert rows[0]["success"] is True


def test_unknown_event_types_are_refused(participant, pilot_env):
    assert analytics.record_event(participant, "definitely_not_an_event",
                                  settings=pilot_env) is False


def test_raw_orbital_parameters_are_dropped_by_default(participant, pilot_env):
    analytics.record_event(
        participant, EventType.SIMULATION_COMPLETED,
        detail={"altitude_bucket_km": "300-500km", "rel_x": 1234.5,
                "raw_parameters": {"rel_y": -987.6}, "kp": 4.6e-5},
        settings=pilot_env)
    with db.get_engine(pilot_env).connect() as connection:
        detail = connection.execute(
            select(events.c.detail_json)
            .where(events.c.event_type == EventType.SIMULATION_COMPLETED)
        ).scalar_one()
    payload = json.loads(detail)
    assert payload == {"altitude_bucket_km": "300-500km"}
    assert "1234.5" not in detail
    assert "raw_parameters" not in detail


def test_raw_parameters_are_kept_only_when_explicitly_enabled(monkeypatch, participant):
    from pilot import config

    monkeypatch.setenv("PILOT_LOG_RAW_PARAMETERS", "true")
    settings = config.get_settings(refresh=True)
    analytics.record_event(participant, EventType.SIMULATION_COMPLETED,
                           detail={"rel_x": 1234.5}, settings=settings)
    with db.get_engine(settings).connect() as connection:
        detail = connection.execute(
            select(events.c.detail_json)
            .where(events.c.event_type == EventType.SIMULATION_COMPLETED)
        ).scalar_one()
    assert "1234.5" in detail
    config.get_settings(refresh=True)


def test_events_are_marked_as_demo_outside_production(participant, pilot_env):
    assert pilot_env.is_demo_data is True
    analytics.record_event(participant, EventType.SIMULATION_COMPLETED, settings=pilot_env)
    with db.get_engine(pilot_env).connect() as connection:
        flags = [row[0] for row in connection.execute(select(events.c.is_demo)).all()]
    assert all(flags), "test-environment rows must be flagged as demo data"


def test_analytics_failure_does_not_raise(participant, pilot_env):
    db.reset_engine()
    from pilot import config

    original = pilot_env.database_url
    broken = config.Settings(**{**pilot_env.__dict__,
                                "database_url": "sqlite:///?bad?path?/nope.db"})
    result = analytics.record_event(participant, EventType.SIMULATION_COMPLETED,
                                    settings=broken)
    assert result in (True, False)
    db.reset_engine()
    assert original


def test_no_column_can_hold_an_email_or_a_name():
    forbidden = {"email", "name", "full_name", "first_name", "surname", "age",
                 "date_of_birth", "dob", "ip", "ip_address", "user_agent", "phone"}
    for table in (participants, sessions, events, feedback):
        columns = {column.name.lower() for column in table.columns}
        assert not (columns & forbidden), f"{table.name} exposes {columns & forbidden}"


def test_set_track_updates_the_session(participant, pilot_env):
    analytics.set_track(participant, "guided", pilot_env)
    assert participant.track == "guided"
    with db.get_engine(pilot_env).connect() as connection:
        track = connection.execute(
            select(sessions.c.track).where(sessions.c.id == participant.session_id)
        ).scalar_one()
    assert track == "guided"


def test_record_assessment_and_retake_overwrites(participant, pilot_env):
    assert analytics.record_assessment(
        participant, instrument_id="orbital_basics", instrument_version="1.0",
        phase="pre", score=3, max_score=8, settings=pilot_env)
    assert analytics.record_assessment(
        participant, instrument_id="orbital_basics", instrument_version="1.0",
        phase="pre", score=5, max_score=8, settings=pilot_env)

    from pilot.models import assessments
    with db.get_engine(pilot_env).connect() as connection:
        rows = connection.execute(select(assessments)).mappings().all()
    assert len(rows) == 1, "re-taking must overwrite, not duplicate"
    assert rows[0]["score"] == 5


def test_record_assessment_rejects_a_bad_phase(participant, pilot_env):
    assert analytics.record_assessment(
        participant, instrument_id="x", instrument_version="1.0", phase="middle",
        score=1, max_score=2, settings=pilot_env) is False


def test_record_feedback_writes_a_row(participant, pilot_env):
    assert analytics.record_feedback(
        participant, usefulness_rating=4, usability_rating=5, would_recommend=9,
        most_useful="The gain sweep", bug_report="Nothing broke",
        settings=pilot_env)
    with db.get_engine(pilot_env).connect() as connection:
        rows = connection.execute(select(feedback)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["usefulness_rating"] == 4


def _complete_runs(context, count, settings):
    for _ in range(count):
        analytics.record_event(context, EventType.SIMULATION_COMPLETED,
                               scenario_category="relative_motion",
                               scenario_id="pd_rendezvous", success=True,
                               settings=settings)


def test_summary_metric_definitions(pilot_env):
    filters = ReportFilters(include_demo=True)

    alice = analytics.start_session("ORG", "ALICE", "student", "1.0", pilot_env)
    bob = analytics.start_session("ORG", "BOB", "engineer_researcher", "1.0", pilot_env)
    analytics.start_session("ORG", "CAROL", "teacher", "1.0", pilot_env)

    _complete_runs(alice, 3, pilot_env)
    _complete_runs(bob, 1, pilot_env)
    analytics.record_event(bob, EventType.SIMULATION_FAILED, success=False,
                           error_class="ValueError", settings=pilot_env)

    metrics = analytics.summary_metrics(filters, pilot_env)

    assert metrics["unique_users"] == 2
    assert metrics["active_users"] == 2
    assert metrics["completed_runs"] == 4
    assert metrics["failed_runs"] == 1
    assert metrics["runs_per_user"] == pytest.approx(2.0)
    assert metrics["returning_users"] == 0
    assert metrics["failure_rate"] == pytest.approx(1 / 5)


def test_returning_user_requires_two_separate_dates(pilot_env):
    context = analytics.start_session("ORG", "DAVE", "student", "1.0", pilot_env)
    _complete_runs(context, 2, pilot_env)

    filters = ReportFilters(include_demo=True)
    assert analytics.summary_metrics(filters, pilot_env)["returning_users"] == 0

    engine = db.get_engine(pilot_env)
    yesterday = date.today() - timedelta(days=1)
    with engine.begin() as connection:
        first_id = connection.execute(select(events.c.id).limit(1)).scalar_one()
        connection.execute(update(events).where(events.c.id == first_id)
                           .values(occurred_date=yesterday))

    metrics = analytics.summary_metrics(filters, pilot_env)
    assert metrics["returning_users"] == 1
    assert metrics["return_rate"] == pytest.approx(1.0)


def test_measured_impact_requires_both_pre_and_post(pilot_env):
    context = analytics.start_session("ORG", "ERIN", "student", "1.0", pilot_env)
    _complete_runs(context, 1, pilot_env)
    filters = ReportFilters(include_demo=True)

    analytics.record_assessment(context, instrument_id="orbital_basics",
                                instrument_version="1.0", phase="pre",
                                score=2, max_score=8, settings=pilot_env)
    assert analytics.summary_metrics(filters, pilot_env)["measured_impact_users"] == 0

    analytics.record_assessment(context, instrument_id="orbital_basics",
                                instrument_version="1.0", phase="post",
                                score=6, max_score=8, settings=pilot_env)
    assert analytics.summary_metrics(filters, pilot_env)["measured_impact_users"] == 1

    changes = analytics.assessment_changes(filters, pilot_env)
    assert len(changes) == 1
    assert changes[0]["pre_score"] == 2
    assert changes[0]["post_score"] == 6
    assert changes[0]["change"] == 4


def test_demo_rows_are_excluded_by_default(pilot_env):
    context = analytics.start_session("ORG", "FRANK", "student", "1.0", pilot_env)
    _complete_runs(context, 2, pilot_env)

    with_demo = analytics.summary_metrics(ReportFilters(include_demo=True), pilot_env)
    without_demo = analytics.summary_metrics(ReportFilters(include_demo=False), pilot_env)

    assert with_demo["completed_runs"] == 2
    assert without_demo["completed_runs"] == 0, (
        "test-environment rows must never appear in real pilot reporting")


def test_breakdowns_return_rows(pilot_env):
    context = analytics.start_session("ORG", "GINA", "student", "1.0", pilot_env)
    analytics.set_track(context, "guided", pilot_env)
    _complete_runs(context, 2, pilot_env)
    analytics.record_event(context, EventType.FEATURE_CATEGORY_USED,
                           feature_category="pd_rendezvous", settings=pilot_env)
    filters = ReportFilters(include_demo=True)

    assert analytics.usage_by_date(filters, pilot_env)
    assert analytics.usage_by_organisation(filters, pilot_env)
    assert analytics.usage_by_scenario(filters, pilot_env)
    assert analytics.usage_by_user_type(filters, pilot_env)
    assert analytics.usage_by_feature(filters, pilot_env)
    assert analytics.simulations_per_user(filters, pilot_env)
    assert analytics.recent_events(filters, settings=pilot_env)


def test_organisation_filter_isolates_data(pilot_env):
    from pilot.models import organisations

    alpha = analytics.start_session("ALPHA", "U1", "student", "1.0", pilot_env)
    _complete_runs(alpha, 3, pilot_env)

    with db.get_engine(pilot_env).connect() as connection:
        org_id = connection.execute(select(organisations.c.id)).scalar_one()

    matching = analytics.summary_metrics(
        ReportFilters(include_demo=True, organisation_id=org_id), pilot_env)
    other = analytics.summary_metrics(
        ReportFilters(include_demo=True, organisation_id=org_id + 999), pilot_env)

    assert matching["completed_runs"] == 3
    assert other["completed_runs"] == 0


def test_user_type_filter(pilot_env):
    student = analytics.start_session("ORG", "S1", "student", "1.0", pilot_env)
    engineer = analytics.start_session("ORG", "E1", "engineer_researcher", "1.0", pilot_env)
    _complete_runs(student, 2, pilot_env)
    _complete_runs(engineer, 5, pilot_env)

    students = analytics.summary_metrics(
        ReportFilters(include_demo=True, user_type="student"), pilot_env)
    engineers = analytics.summary_metrics(
        ReportFilters(include_demo=True, user_type="engineer_researcher"), pilot_env)

    assert students["completed_runs"] == 2
    assert engineers["completed_runs"] == 5


def test_date_filter_windows_active_users(pilot_env):
    context = analytics.start_session("ORG", "H1", "student", "1.0", pilot_env)
    _complete_runs(context, 2, pilot_env)

    future = date.today() + timedelta(days=5)
    windowed = analytics.summary_metrics(
        ReportFilters(include_demo=True, start=future, end=future), pilot_env)
    assert windowed["active_users"] == 0
    assert windowed["unique_users"] == 1


def test_feedback_summary_separates_professionals(pilot_env):
    student = analytics.start_session("ORG", "S9", "student", "1.0", pilot_env)
    engineer = analytics.start_session("ORG", "E9", "engineer_researcher", "1.0", pilot_env)
    analytics.record_feedback(student, usefulness_rating=5, usability_rating=5,
                              settings=pilot_env)
    analytics.record_feedback(engineer, usefulness_rating=3, usability_rating=4,
                              feature_request="Add LQR guidance", settings=pilot_env)

    filters = ReportFilters(include_demo=True)
    summary = analytics.feedback_ratings_summary(filters, pilot_env)
    assert summary["responses"] == 2
    assert summary["professional_responses"] == 1
    assert summary["professional_mean_usefulness"] == pytest.approx(3.0)

    requests = analytics.bug_and_feature_rows(filters, pilot_env)
    assert len(requests) == 1
    assert requests[0]["feature_request"] == "Add LQR guidance"
