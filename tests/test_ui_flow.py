"""End-to-end tests of the participant journey, driven headlessly.

Uses Streamlit's own ``AppTest`` harness, which executes the real
``pilot_app.py`` script and gives programmatic access to the widgets and to
session state. That makes these genuine user-journey tests rather than unit
tests of the callbacks -- if the consent gate stops working, or a simulation
stops recording its analytics event, these fail.
"""

from __future__ import annotations

import pytest
import streamlit as st
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from pilot import db
from pilot.analytics import USER_TYPES, EventType
from pilot.models import events, feedback as feedback_table, participants

APP = "pilot_app.py"
TIMEOUT = 120

#: The admin area is reached by a query parameter, which this version of
#: ``AppTest`` cannot set. Driving the dashboard function directly exercises the
#: same code; the query-parameter routing itself is covered by the manual
#: checklist rather than here.
ADMIN_SCRIPT = """
import os, sys
sys.path.insert(0, os.getcwd())
from pilot.ui import admin
from pilot.ui.common import init_state
init_state()
admin.admin_dashboard()
"""


def _admin_app():
    st.cache_resource.clear()
    instance = AppTest.from_string(ADMIN_SCRIPT, default_timeout=TIMEOUT)
    instance.run()
    return instance


@pytest.fixture
def app(pilot_env):
    """A fresh app instance wired to the temporary test database."""
    # The app memoises its migration check with st.cache_resource, which is
    # process-global; clear it so each test starts from the fixture's database.
    st.cache_resource.clear()
    instance = AppTest.from_file(APP, default_timeout=TIMEOUT)
    instance.run()
    return instance


def _button(app_instance, label_startswith: str):
    for button in app_instance.button:
        if button.label.strip().startswith(label_startswith):
            return button
    raise AssertionError(
        f"no button starting with {label_startswith!r}; found "
        f"{[b.label for b in app_instance.button]}")


def _has_button(app_instance, label_startswith: str) -> bool:
    return any(b.label.strip().startswith(label_startswith)
               for b in app_instance.button)


def _all_text(app_instance) -> str:
    parts = []
    for collection in (app_instance.markdown, app_instance.title, app_instance.caption,
                       app_instance.warning, app_instance.error, app_instance.info,
                       app_instance.success, app_instance.subheader):
        for element in collection:
            parts.append(str(element.value))
    return "\n".join(parts)


def _sign_in(app_instance, pilot_code="TESTORG", participant_code="TESTUSER",
             user_type_index=0):
    """Walk landing -> consent -> entry and return the app at track selection."""
    _button(app_instance, "Start").click().run()
    app_instance.checkbox[0].check().run()
    _button(app_instance, "Continue").click().run()

    app_instance.text_input[0].set_value(pilot_code)
    app_instance.text_input[1].set_value(participant_code)
    # set_value takes the *raw* option, not the label format_func produces.
    app_instance.radio[0].set_value(USER_TYPES[user_type_index])
    _button(app_instance, "Continue").click().run()
    return app_instance


def _event_types(settings) -> list[str]:
    with db.get_engine(settings).connect() as connection:
        return [row[0] for row in connection.execute(select(events.c.event_type)).all()]


# 1. Landing
def test_landing_renders_with_the_required_disclaimers(app):
    assert not app.exception
    text = _all_text(app)
    assert "Orbital Rendezvous Simulator" in text
    # The three claims the platform must never make are explicitly denied.
    assert "not NASA-approved" in text or "not\nNASA-approved" in text
    assert "flight-certified" in text
    assert "GMAT" in text
    assert _has_button(app, "Start")


def test_landing_states_the_limitations(app):
    text = _all_text(app)
    assert "linear" in text.lower()
    assert "atmosphere" in text.lower() or "drag" in text.lower()


# 2. Consent
def test_consent_is_required_before_continuing(app):
    _button(app, "Start").click().run()
    assert "privacy" in _all_text(app).lower() or "consent" in _all_text(app).lower()

    # The Continue button exists but is disabled until the box is ticked.
    continue_button = _button(app, "Continue")
    assert continue_button.disabled is True

    app.checkbox[0].check().run()
    assert _button(app, "Continue").disabled is False


def test_consent_explains_what_is_and_is_not_collected(app):
    _button(app, "Start").click().run()
    text = _all_text(app)
    assert "never recorded" in text.lower()
    assert "email" in text.lower()
    assert "hash" in text.lower()


def test_cannot_skip_straight_into_the_flow_without_consent(app):
    """Deep-linking into a mid-flow stage must bounce back to the landing page."""
    app.session_state["stage"] = "feedback"
    app.run()
    assert app.session_state["stage"] == "landing"


# 3. Entry
def test_entry_rejects_an_email_shaped_participant_code(app, pilot_env):
    _button(app, "Start").click().run()
    app.checkbox[0].check().run()
    _button(app, "Continue").click().run()

    app.text_input[0].set_value("TESTORG")
    app.text_input[1].set_value("alice@example.org")
    _button(app, "Continue").click().run()

    assert any("email" in str(e.value).lower() for e in app.error)
    with db.get_engine(pilot_env).connect() as connection:
        assert connection.execute(select(participants)).first() is None


def test_entry_rejects_a_too_short_code(app, pilot_env):
    _button(app, "Start").click().run()
    app.checkbox[0].check().run()
    _button(app, "Continue").click().run()
    app.text_input[0].set_value("TESTORG")
    app.text_input[1].set_value("ab")
    _button(app, "Continue").click().run()
    assert any("at least" in str(e.value) for e in app.error)


def test_successful_sign_in_creates_an_anonymous_participant(app, pilot_env):
    _sign_in(app, participant_code="BLUE-FALCON-42")

    assert app.session_state["context"] is not None
    assert app.session_state["stage"] == "track"

    with db.get_engine(pilot_env).connect() as connection:
        rows = connection.execute(select(participants)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["user_type"] == "student"
    # The raw code must not appear anywhere in the row.
    assert "BLUE-FALCON-42" not in str(dict(rows[0]))

    assert EventType.PILOT_ACCESSED in _event_types(pilot_env)
    assert EventType.SESSION_STARTED in _event_types(pilot_env)


def test_unregistered_pilot_code_is_grouped_not_rejected(app, pilot_env):
    _sign_in(app, pilot_code="NOBODY-KNOWS-ME")
    assert app.session_state["context"] is not None
    assert app.session_state["context"].organisation_name == "Unregistered pilot code"


# 4. Track selection
def test_choosing_the_free_track_records_it(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    assert app.session_state["track"] == "free"
    assert app.session_state["stage"] == "workspace"
    assert EventType.TRACK_SELECTED in _event_types(pilot_env)


def test_guided_track_offers_the_pre_assessment_first(app):
    _sign_in(app)
    _button(app, "Choose guided").click().run()
    assert app.session_state["stage"] == "assessment_pre"


# 5-7. Running a simulation and downloading results
def test_running_a_simulation_records_a_completed_event(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()

    assert _has_button(app, "Run simulation")
    _button(app, "Run simulation").click().run()

    assert not app.exception
    outcome = app.session_state["last_outcome"]
    assert outcome is not None and outcome.ok, getattr(outcome, "errors", None)
    assert app.session_state["completed_runs"] == 1

    types = _event_types(pilot_env)
    assert types.count(EventType.SIMULATION_STARTED) == 1
    assert types.count(EventType.SIMULATION_COMPLETED) == 1
    assert EventType.SCENARIO_CATEGORY_USED in types
    assert EventType.FEATURE_CATEGORY_USED in types


def test_results_show_assumptions_and_limitations(app):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Run simulation").click().run()

    text = _all_text(app)
    assert "Assumptions behind this result" in text
    assert "Limitations you should know about" in text


def test_results_offer_downloads_including_a_reproducible_config(app):
    """AppTest does not expose st.download_button, so check the section renders
    and that the underlying export actually produces the three artefacts."""
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Run simulation").click().run()

    assert "Download" in _all_text(app)
    assert "nothing that identifies you" in _all_text(app)

    from pilot.content.scenarios_catalog import get_scenario
    from pilot.export import config_json, result_tables, results_readme

    outcome = app.session_state["last_outcome"]
    scenario = get_scenario(outcome.scenario_id)
    params = app.session_state["params"]

    tables = result_tables(scenario.result_kind, outcome.result)
    assert tables, "no result tables to download"
    assert config_json(scenario, params)
    assert results_readme(scenario, params, outcome.warnings)


def test_two_runs_count_as_two(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Run simulation").click().run()
    _button(app, "Run simulation").click().run()

    assert app.session_state["completed_runs"] == 2
    assert _event_types(pilot_env).count(EventType.SIMULATION_COMPLETED) == 2


# 8. Feedback
def test_feedback_submission_is_recorded(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Run simulation").click().run()
    _button(app, "Give feedback and finish").click().run()

    assert app.session_state["stage"] == "feedback"
    app.slider[0].set_value(5)
    app.slider[1].set_value(4)
    app.text_area[0].set_value("The gain sweep made the trade-off obvious.")
    app.text_area[2].set_value("Nothing broke.")
    _button(app, "Submit feedback").click().run()

    assert app.session_state["feedback_done"] is True
    assert app.session_state["stage"] == "confirmation"

    with db.get_engine(pilot_env).connect() as connection:
        rows = connection.execute(select(feedback_table)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["usefulness_rating"] == 5
    assert "gain sweep" in rows[0]["most_useful"]
    assert EventType.FEEDBACK_SUBMITTED in _event_types(pilot_env)


def test_feedback_can_be_skipped(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Give feedback and finish").click().run()
    _button(app, "Skip feedback").click().run()
    assert app.session_state["stage"] == "confirmation"


# 10. Confirmation
def test_confirmation_explains_what_happens_to_feedback(app):
    _sign_in(app)
    _button(app, "Choose free simulation").click().run()
    _button(app, "Run simulation").click().run()
    _button(app, "Give feedback and finish").click().run()
    _button(app, "Skip feedback").click().run()

    text = _all_text(app)
    assert "may" in text.lower() and "influence" in text.lower()
    assert "same participant code" in text.lower()
    assert "1" in text  # the completed-run count


# Assessments
def test_pre_and_post_assessment_are_recorded_and_measured(app, pilot_env):
    from pilot.content.assessments import ORBITAL_BASICS

    _sign_in(app)
    _button(app, "Choose guided").click().run()
    assert app.session_state["stage"] == "assessment_pre"

    # Answer everything wrong first.
    for index, item in enumerate(ORBITAL_BASICS.items):
        wrong = (item.correct_index + 1) % len(item.options)
        app.radio[index].set_value(item.options[wrong])
    _button(app, "Submit answers").click().run()
    assert app.session_state["assessment_pre_done"] is True
    assert app.session_state["assessment_pre_score"] == 0

    _button(app, "Continue").click().run()
    assert app.session_state["stage"] == "workspace"

    # Jump to the post assessment and answer everything correctly.
    app.session_state["stage"] = "assessment_post"
    app.run()
    for index, item in enumerate(ORBITAL_BASICS.items):
        app.radio[index].set_value(item.options[item.correct_index])
    _button(app, "Submit answers").click().run()
    assert app.session_state["assessment_post_done"] is True

    from pilot import analytics

    changes = analytics.assessment_changes(
        analytics.ReportFilters(include_demo=True), pilot_env)
    assert len(changes) == 1
    assert changes[0]["pre_score"] == 0
    assert changes[0]["post_score"] == ORBITAL_BASICS.max_score
    assert changes[0]["change"] == ORBITAL_BASICS.max_score

    metrics = analytics.summary_metrics(
        analytics.ReportFilters(include_demo=True), pilot_env)
    assert metrics["measured_impact_users"] == 1


def test_assessment_can_be_skipped(app):
    _sign_in(app)
    _button(app, "Choose guided").click().run()
    _button(app, "Skip this").click().run()
    assert app.session_state["assessment_pre_done"] is True
    assert app.session_state["stage"] == "workspace"


# Guided activity
def test_guided_step_requires_a_prediction_before_running(app):
    _sign_in(app)
    _button(app, "Choose guided").click().run()
    _button(app, "Skip this").click().run()
    _button(app, "Start: What keeps").click().run()

    run_button = _button(app, "Run this step")
    assert run_button.disabled is True, "Run must be locked until a prediction is made"

    app.radio[0].set_value(app.radio[0].options[0]).run()
    assert _button(app, "Run this step").disabled is False


def test_guided_step_runs_and_shows_the_takeaway(app, pilot_env):
    _sign_in(app)
    _button(app, "Choose guided").click().run()
    _button(app, "Skip this").click().run()
    _button(app, "Start: What keeps").click().run()
    app.radio[0].set_value(app.radio[0].options[0]).run()
    _button(app, "Run this step").click().run()

    assert not app.exception
    assert app.session_state["last_outcome"].ok
    text = _all_text(app)
    assert "What to look for" in text
    assert "Takeaway" in text
    assert _event_types(pilot_env).count(EventType.SIMULATION_COMPLETED) == 1


# Administrator area
def test_admin_area_requires_a_password(pilot_env):
    instance = _admin_app()

    text = _all_text(instance)
    assert "Administrator sign-in" in text
    # With no credentials configured the area refuses outright.
    assert any("No administrator credentials" in str(e.value) for e in instance.error)
    assert instance.session_state["admin_authenticated"] is False
    # And it must not render any report.
    assert "Pilot administration" not in text


def test_admin_rejects_wrong_credentials(monkeypatch, pilot_env):
    from pilot import auth, config

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD_HASH",
                       auth.hash_password("right-password", iterations=1000))
    config.get_settings(refresh=True)
    auth.reset_authenticator()

    instance = _admin_app()
    instance.text_input[0].set_value("coordinator")
    instance.text_input[1].set_value("wrong-password")
    _button(instance, "Sign in").click().run()

    assert instance.session_state["admin_authenticated"] is False
    assert any("Incorrect" in str(e.value) for e in instance.error)
    assert "Pilot administration" not in _all_text(instance)


def test_admin_dashboard_reports_and_separates_demo_data(monkeypatch, pilot_env):
    """The whole admin path: sign in, see the metrics, confirm demo separation."""
    from pilot import analytics, auth, config

    # Seed activity through the analytics layer.
    context = analytics.start_session("SCHOOL-A", "PUPIL-1", "student", "1.0", pilot_env)
    for _ in range(3):
        analytics.record_event(context, EventType.SIMULATION_COMPLETED,
                               scenario_category="relative_motion",
                               scenario_id="pd_rendezvous", success=True,
                               settings=pilot_env)
    analytics.record_feedback(context, usefulness_rating=4, usability_rating=5,
                              settings=pilot_env)

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD_HASH",
                       auth.hash_password("right-password", iterations=1000))
    config.get_settings(refresh=True)
    auth.reset_authenticator()

    instance = _admin_app()
    instance.text_input[0].set_value("coordinator")
    instance.text_input[1].set_value("right-password")
    _button(instance, "Sign in").click().run()

    assert instance.session_state["admin_authenticated"] is True
    assert not instance.exception

    metrics = {m.label: m.value for m in instance.metric}
    # Demo records are excluded by default, and this environment is "test".
    assert metrics["Completed runs"] == "0", (
        "development records must not be counted as pilot activity")

    # Turning demo records on reveals them, with a visible warning.
    demo_checkbox = next(c for c in instance.checkbox
                         if "demo" in c.label.lower())
    demo_checkbox.check().run()

    metrics = {m.label: m.value for m in instance.metric}
    assert metrics["Completed runs"] == "3"
    assert metrics["Unique users"] == "1"
    assert any("not" in str(w.value).lower() and "pilot results" in str(w.value).lower()
               for w in instance.warning)


def test_admin_export_offers_csv_and_bundle(monkeypatch, pilot_env):
    from pilot import analytics, auth, config

    context = analytics.start_session("SCHOOL-A", "PUPIL-1", "student", "1.0", pilot_env)
    analytics.record_event(context, EventType.SIMULATION_COMPLETED,
                           scenario_category="relative_motion",
                           scenario_id="pd_rendezvous", success=True,
                           settings=pilot_env)

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD_HASH",
                       auth.hash_password("pw", iterations=1000))
    config.get_settings(refresh=True)
    auth.reset_authenticator()

    instance = _admin_app()
    instance.text_input[0].set_value("coordinator")
    instance.text_input[1].set_value("pw")
    _button(instance, "Sign in").click().run()
    next(c for c in instance.checkbox if "demo" in c.label.lower()).check().run()

    # AppTest does not expose st.download_button, so verify the export section
    # renders and that the aggregate tables behind it actually hold data.
    text = _all_text(instance)
    assert "Export everything" in text
    assert "anonymous" in text.lower()

    from pilot import analytics

    filters = analytics.ReportFilters(include_demo=True)
    assert analytics.usage_by_organisation(filters, pilot_env)
    assert analytics.usage_by_scenario(filters, pilot_env)
    assert analytics.simulations_per_user(filters, pilot_env)


def test_admin_page_never_shows_a_secret(monkeypatch, pilot_env):
    from pilot import auth, config

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD", "super-secret-password")
    monkeypatch.setenv("DATABASE_URL", pilot_env.database_url)
    config.get_settings(refresh=True)
    auth.reset_authenticator()

    instance = _admin_app()
    instance.text_input[0].set_value("coordinator")
    instance.text_input[1].set_value("super-secret-password")
    _button(instance, "Sign in").click().run()

    rendered = _all_text(instance) + str([
        (m.label, m.value) for m in instance.metric])
    assert "super-secret-password" not in rendered
    assert "pbkdf2_sha256$" not in rendered
