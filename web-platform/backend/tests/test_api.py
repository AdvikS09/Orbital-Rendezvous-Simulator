"""API behaviour: validation, errors, sessions, feedback, assessments, admin.

Parity (does the API return the right numbers?) lives in
``test_engine_parity.py``. This file covers everything else: does it reject the
right things, protect what it should, and record what it claims to record.
"""

from __future__ import annotations

import base64
import io
import zipfile

import pytest

from pilot.content.scenarios_catalog import SCENARIOS

SCENARIO_IDS = [scenario.id for scenario in SCENARIOS]


def _auth_header(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# --------------------------------------------------------------------------
# Packaging: the backend package must not be shadowed
# --------------------------------------------------------------------------
def test_app_resolves_to_the_backend_package_not_the_legacy_module():
    """The repository root holds app.py, the legacy Streamlit MVP.

    This package is also called `app`. If the root ever precedes the backend
    directory on sys.path, `import app` silently resolves to that single module
    and every `from app.services import ...` fails. bootstrap.py appends rather
    than inserts to prevent it; this locks the behaviour down.
    """
    import app

    assert hasattr(app, "__path__"), (
        "`app` resolved to a module, not a package - the repository-root "
        "app.py is shadowing the backend package"
    )
    from app.services import catalogue, simulation  # noqa: F401


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health_reports_engine_and_database(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()

    import orbital_core

    assert body["status"] == "ok"
    assert body["engine_version"] == orbital_core.__version__
    assert body["database_ok"] is True
    assert body["scenario_count"] == len(SCENARIOS)
    assert body["environment"] == "test"


def test_health_never_exposes_the_database_password(monkeypatch, client):
    from pilot import config, db

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://admin:sup3rs3cret@db.example.com:5432/pilot"
    )
    config.get_settings(refresh=True)
    db.reset_engine()
    try:
        body = client.get("/api/v1/health").json()
        assert "sup3rs3cret" not in str(body)
        assert "admin" not in body["database"]
    finally:
        db.reset_engine()
        config.get_settings(refresh=True)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
def test_list_scenarios(client):
    body = client.get("/api/v1/scenarios").json()
    assert len(body) == len(SCENARIOS)
    assert {s["id"] for s in body} == {s.id for s in SCENARIOS}


@pytest.mark.parametrize("track,expected", [("guided", 8), ("professional", 11), ("free", 11)])
def test_track_filter_preserves_the_gating(client, track, expected):
    body = client.get(f"/api/v1/scenarios?track={track}").json()
    assert len(body) == expected


def test_unknown_track_is_rejected(client):
    assert client.get("/api/v1/scenarios?track=nonsense").status_code == 400


def test_unknown_scenario_is_404(client):
    assert client.get("/api/v1/scenarios/does-not-exist").status_code == 404
    response = client.post(
        "/api/v1/simulations/does-not-exist", json={"params": {}}
    )
    assert response.status_code == 404


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_detail_carries_its_full_content(client, scenario_id):
    """Explanations, assumptions and limitations must survive the migration."""
    body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
    assert len(body["summary"]) > 40
    assert len(body["explanation"]) > 250
    assert body["assumptions"], "assumptions must never be empty"
    assert body["limitations"], "limitations must never be empty"
    assert body["reading_guide"], "every scenario has a reading guide"
    assert body["params"]


# --------------------------------------------------------------------------
# Input validation - the silent-wrong-answer cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "params,expect",
    [
        ({"altitude_km": -7000.0}, "Altitude"),
        ({"altitude_km": 400.0, "dt_s": 0.0}, "Timestep"),
        ({"altitude_km": 400.0, "duration_s": 1e7, "dt_s": 1.0}, "steps"),
        ({"altitude_km": 400.0, "duration_s": -500.0}, "Duration"),
    ],
)
def test_invalid_inputs_are_rejected_before_the_engine_runs(client, params, expect):
    """Each of these previously produced a wrong answer, not an error."""
    response = client.post("/api/v1/simulations/research_workbench", json={"params": params})
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["status"] == "error"
    assert body["rejected"] is True
    joined = body["message"] + " ".join(d["message"] for d in body["detail"])
    assert expect.lower() in joined.lower()


def test_validation_errors_are_attributed_to_fields(client):
    response = client.post(
        "/api/v1/simulations/research_workbench",
        json={"params": {"altitude_km": -7000.0}},
    )
    body = response.json()
    fields = {d["field"] for d in body["detail"]}
    assert "altitude_km" in fields, body


def test_nan_input_is_rejected(client):
    """JSON has no NaN, so a client sends the string; it must not become 0."""
    response = client.post(
        "/api/v1/simulations/cw_natural_motion",
        json={"params": {"rel_x": "not-a-number"}},
    )
    assert response.status_code == 422


def test_zero_offset_with_zero_velocity_is_rejected(client):
    response = client.post(
        "/api/v1/simulations/cw_natural_motion",
        json={
            "params": {
                "rel_x": 0.0, "rel_y": 0.0, "rel_z": 0.0,
                "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
            }
        },
    )
    assert response.status_code == 422
    assert "nothing to simulate" in response.text


def test_warnings_are_returned_with_a_successful_run(client):
    """A large separation runs, but must carry the CW-validity warning."""
    response = client.post(
        "/api/v1/simulations/cw_natural_motion",
        json={"params": {"rel_x": 0.0, "rel_y": 80_000.0, "rel_z": 0.0}},
    )
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("linearisation" in w for w in warnings), warnings


def test_preflight_predicts_cost_without_running(client):
    body = client.post(
        "/api/v1/scenarios/pd_rendezvous/preflight", json={"params": {}}
    ).json()
    assert body["ok"] is True
    assert body["n_steps"] > 0
    assert body["estimated_seconds"] > 0


def test_preflight_reports_failure_without_running(client):
    body = client.post(
        "/api/v1/scenarios/pd_rendezvous/preflight",
        json={"params": {"altitude_km": -100.0}},
    ).json()
    assert body["ok"] is False
    assert body["errors"]


# --------------------------------------------------------------------------
# Sessions and analytics
# --------------------------------------------------------------------------
def test_session_requires_consent(client):
    response = client.post(
        "/api/v1/sessions",
        json={
            "pilot_code": "ORG", "participant_code": "USER1",
            "user_type": "student", "consent_given": False,
        },
    )
    assert response.status_code == 422


def test_session_rejects_an_email_shaped_participant_code(client):
    response = client.post(
        "/api/v1/sessions",
        json={
            "pilot_code": "ORG", "participant_code": "alice@example.org",
            "user_type": "student", "consent_given": True,
        },
    )
    assert response.status_code == 422
    assert "email" in response.text.lower()


def test_session_never_returns_the_participant_code(client):
    response = client.post(
        "/api/v1/sessions",
        json={
            "pilot_code": "ORG", "participant_code": "BLUE-FALCON-42",
            "user_type": "student", "consent_given": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert "BLUE-FALCON-42" not in str(body)
    assert len(body["participant_ref"]) == 8


def test_completing_a_simulation_records_the_analytics_events(session_client, api_env):
    from sqlalchemy import select

    from pilot import db
    from pilot.analytics import EventType
    from pilot.models import events

    response = session_client.post(
        "/api/v1/simulations/two_body_orbit", json={"params": {}}
    )
    assert response.status_code == 200

    with db.get_engine(api_env).connect() as connection:
        types = [row[0] for row in connection.execute(select(events.c.event_type)).all()]

    assert types.count(EventType.SIMULATION_STARTED) == 1
    assert types.count(EventType.SIMULATION_COMPLETED) == 1
    assert EventType.SCENARIO_CATEGORY_USED in types
    assert EventType.FEATURE_CATEGORY_USED in types


def test_a_rejected_run_is_not_recorded_as_a_failure(session_client, api_env):
    from sqlalchemy import select

    from pilot import db
    from pilot.analytics import EventType
    from pilot.models import events

    session_client.post(
        "/api/v1/simulations/research_workbench",
        json={"params": {"altitude_km": -5000.0}},
    )

    with db.get_engine(api_env).connect() as connection:
        types = [row[0] for row in connection.execute(select(events.c.event_type)).all()]

    assert EventType.SIMULATION_REJECTED in types
    assert EventType.SIMULATION_FAILED not in types
    assert EventType.SIMULATION_STARTED not in types


def test_raw_parameters_are_not_logged(session_client, api_env):
    import json

    from sqlalchemy import select

    from pilot import db
    from pilot.analytics import EventType
    from pilot.models import events

    session_client.post(
        "/api/v1/simulations/cw_natural_motion",
        json={"params": {"rel_y": 1234.5}},
    )
    with db.get_engine(api_env).connect() as connection:
        detail = connection.execute(
            select(events.c.detail_json).where(
                events.c.event_type == EventType.SIMULATION_COMPLETED
            )
        ).scalar_one()

    assert "1234.5" not in detail
    payload = json.loads(detail)
    assert payload["separation_bucket_m"] in (
        "<100m", "100m-1km", "1-10km", "10-50km", ">50km"
    )


def test_simulation_works_without_a_session(client):
    """Analytics is optional; the simulator must not require it."""
    response = client.post("/api/v1/simulations/two_body_orbit", json={"params": {}})
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
def test_rate_limit_returns_429(session_client, monkeypatch):
    from pilot import ratelimit
    from pilot.ratelimit import SessionRateLimiter

    monkeypatch.setattr(
        ratelimit, "_limiter", SessionRateLimiter(max_per_minute=1, max_per_session=1)
    )
    assert session_client.post(
        "/api/v1/simulations/two_body_orbit", json={"params": {}}
    ).status_code == 200

    blocked = session_client.post(
        "/api/v1/simulations/two_body_orbit", json={"params": {}}
    )
    assert blocked.status_code == 429
    body = blocked.json()
    assert body["error_class"] == "RateLimited"
    assert body["rejected"] is True
    ratelimit.reset_limiter()


# --------------------------------------------------------------------------
# Feedback and assessments
# --------------------------------------------------------------------------
def test_feedback_requires_a_session(client):
    assert client.post("/api/v1/feedback", json={"usefulness_rating": 4}).status_code == 401


def test_feedback_is_recorded(session_client, api_env):
    from sqlalchemy import select

    from pilot import db
    from pilot.models import feedback as feedback_table

    response = session_client.post(
        "/api/v1/feedback",
        json={
            "usefulness_rating": 5, "usability_rating": 4, "clarity_rating": 5,
            "would_recommend": 9, "most_useful": "The gain sweep.",
        },
    )
    assert response.status_code == 200
    assert response.json()["recorded"] is True

    with db.get_engine(api_env).connect() as connection:
        rows = connection.execute(select(feedback_table)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["usefulness_rating"] == 5


@pytest.mark.parametrize("field,value", [("usefulness_rating", 9), ("would_recommend", 20)])
def test_feedback_rejects_out_of_range_ratings(session_client, field, value):
    assert session_client.post("/api/v1/feedback", json={field: value}).status_code == 422


def test_assessment_withholds_answers_until_the_post_phase(session_client):
    body = session_client.get("/api/v1/assessments/orbital_basics").json()
    assert len(body["items"]) == 10
    assert all(item["correct_index"] is None for item in body["items"]), (
        "the pre-assessment must not be answerable by reading the API"
    )
    assert all(item["explanation"] is None for item in body["items"])
    assert "review" in body["review_status"].lower()


def test_assessment_pre_and_post_measure_change(session_client):
    from pilot.content.assessments import ORBITAL_BASICS

    wrong = {
        item.id: (item.correct_index + 1) % len(item.options)
        for item in ORBITAL_BASICS.items
    }
    right = {item.id: item.correct_index for item in ORBITAL_BASICS.items}

    pre = session_client.post(
        "/api/v1/assessments/orbital_basics/submit",
        json={"phase": "pre", "responses": wrong, "confidence_rating": 2},
    ).json()
    assert pre["score"] == 0
    assert pre["items"] == []

    post = session_client.post(
        "/api/v1/assessments/orbital_basics/submit",
        json={"phase": "post", "responses": right, "confidence_rating": 4},
    ).json()
    assert post["score"] == ORBITAL_BASICS.max_score
    assert post["previous_score"] == 0
    assert post["change"] == ORBITAL_BASICS.max_score
    assert len(post["items"]) == 10
    assert all(item["correct_index"] is not None for item in post["items"])
    assert "seen the questions twice" in post["caveat"]


def test_activities_are_served_with_their_steps(client):
    activities = client.get("/api/v1/activities").json()
    assert len(activities) == 3
    assert sum(len(a["steps"]) for a in activities) == 11
    for activity in activities:
        assert "review" in activity["review_status"].lower()
        for step in activity["steps"]:
            assert step["brief"] and step["look_for"] and step["takeaway"]
            assert len(step["predict_options"]) >= 2


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
def test_config_export_is_reproducible_and_clean(client):
    import json

    response = client.post(
        "/api/v1/exports/pd_rendezvous/config", json={"params": {}}
    )
    assert response.status_code == 200
    config = json.loads(response.text)
    assert config["numerics"]["deterministic"] is True
    assert config["numerics"]["random_seed"] is None
    assert "not" in config["provenance"]["external_validation"].lower()
    assert "gmat" in config["provenance"]["external_validation"].lower()
    for forbidden in ("participant", "session_token", "pilot_code"):
        assert forbidden not in response.text.lower()


def test_notes_export_carries_assumptions_and_warnings(client):
    response = client.post(
        "/api/v1/exports/cw_natural_motion/notes",
        json={"params": {}, "warnings": ["Initial separation is 80.0 km."]},
    )
    assert response.status_code == 200
    assert "ASSUMPTIONS" in response.text
    assert "LIMITATIONS" in response.text
    assert "80.0 km" in response.text
    assert "NASA" in response.text


# --------------------------------------------------------------------------
# Administrator area
# --------------------------------------------------------------------------
def test_admin_refuses_when_no_credentials_are_configured(client):
    response = client.get("/api/v1/admin/report")
    assert response.status_code == 503
    assert "No administrator credentials" in response.text


def test_admin_requires_credentials(client, admin_auth):
    assert client.get("/api/v1/admin/report").status_code == 401


def test_admin_rejects_a_wrong_password(client, admin_auth):
    username, _ = admin_auth
    response = client.get(
        "/api/v1/admin/report", headers=_auth_header(username, "wrong-password")
    )
    assert response.status_code == 401


def test_admin_report_separates_demo_data(session_client, admin_auth):
    """PILOT_ENV=test means every row is demo data and must be excluded."""
    username, password = admin_auth
    session_client.post("/api/v1/simulations/two_body_orbit", json={"params": {}})

    default = session_client.get(
        "/api/v1/admin/report", headers=_auth_header(username, password)
    ).json()
    assert default["summary"]["completed_runs"] == 0, (
        "development records must not be counted as pilot activity"
    )

    with_demo = session_client.get(
        "/api/v1/admin/report?include_demo=true",
        headers=_auth_header(username, password),
    ).json()
    assert with_demo["summary"]["completed_runs"] == 1
    assert with_demo["summary"]["unique_users"] == 1
    assert with_demo["is_demo_environment"] is True


def test_admin_report_defines_its_metrics(client, admin_auth):
    username, password = admin_auth
    body = client.get(
        "/api/v1/admin/report", headers=_auth_header(username, password)
    ).json()
    definitions = body["metric_definitions"]
    assert "unique_user" in definitions
    assert "returning_user" in definitions
    assert "two separate dates" in definitions["returning_user"]
    assert "page_view" in definitions
    assert "NOT counted as a user" in definitions["page_view"]


def test_admin_export_is_a_zip_with_definitions(session_client, admin_auth):
    username, password = admin_auth
    session_client.post("/api/v1/simulations/two_body_orbit", json={"params": {}})

    response = session_client.get(
        "/api/v1/admin/export?include_demo=true",
        headers=_auth_header(username, password),
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert "METRIC_DEFINITIONS.txt" in names
        assert any(name.endswith(".csv") for name in names)
        blob = b"".join(archive.read(name) for name in names).decode("utf-8", "replace")

    assert "TESTUSER1" not in blob, "a participant code must never reach an export"


def test_admin_report_never_contains_a_secret(client, admin_auth):
    username, password = admin_auth
    body = client.get(
        "/api/v1/admin/report", headers=_auth_header(username, password)
    ).text
    assert password not in body
    assert "pbkdf2_sha256$" not in body


# --------------------------------------------------------------------------
# Structured errors from raised HTTPExceptions
# --------------------------------------------------------------------------
def test_http_exceptions_carry_an_error_class(client):
    """A 404 must be as machine-readable as any other failure.

    Without the StarletteHTTPException handler these came back as FastAPI's
    bare {"detail": ...}, which the frontend's ApiError cannot interpret.
    """
    body = client.get("/api/v1/scenarios/nope").json()
    assert body["status"] == "error"
    assert body["error_class"] == "NotFound"
    assert body["rejected"] is True
    assert "nope" in body["message"]


def test_session_field_errors_survive_as_field_attribution(client):
    """The per-field messages must reach the client, not be flattened."""
    response = client.post(
        "/api/v1/sessions",
        json={
            "pilot_code": "has spaces", "participant_code": "me@example.org",
            "user_type": "student", "consent_given": True,
        },
    )
    assert response.status_code == 422
    body = response.json()
    fields = {d["field"]: d["message"] for d in body["detail"]}
    assert "pilot_code" in fields
    assert "participant_code" in fields
    assert "email" in fields["participant_code"].lower()


def test_unauthorized_admin_keeps_the_www_authenticate_header(client, admin_auth):
    """Dropping the header would stop a browser offering its credential prompt."""
    response = client.get("/api/v1/admin/report")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.json()["error_class"] == "Unauthorized"


def test_unknown_api_route_returns_json_not_html(client):
    body = client.get("/api/v1/not-a-real-endpoint").json()
    assert body["error_class"] == "NotFound"
