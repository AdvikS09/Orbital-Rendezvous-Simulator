import numpy as np
import pytest
from sqlalchemy import select

from pilot import analytics, auth, db
from pilot.analytics import EventType, ReportFilters
from pilot.content.scenarios_catalog import (
    SCENARIOS,
    default_params,
    get_scenario,
    scenarios_for,
)
from pilot.models import events
from pilot.ratelimit import SessionRateLimiter
from pilot.runner import run_scenario, validate_params


def _event_types(settings):
    with db.get_engine(settings).connect() as connection:
        return [row[0] for row in connection.execute(select(events.c.event_type)).all()]


def test_every_scenario_has_unique_id_and_required_content():
    ids = [scenario.id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids))
    for scenario in SCENARIOS:
        assert scenario.title and scenario.summary and scenario.explanation
        assert scenario.assumptions, f"{scenario.id} has no assumptions"
        assert scenario.limitations, f"{scenario.id} has no limitations"
        assert scenario.params, f"{scenario.id} has no parameters"
        assert callable(scenario.runner)
        assert scenario.levels


def test_scenario_defaults_validate():
    for scenario in SCENARIOS:
        result = validate_params(scenario, default_params(scenario))
        assert result.ok, f"{scenario.id} default parameters rejected: {result.errors}"


def test_tracks_offer_scenarios():
    for track in ("guided", "professional", "free"):
        assert scenarios_for(track), f"no scenarios for track {track}"


def test_guided_track_excludes_the_expert_only_scenarios():
    guided = {scenario.id for scenario in scenarios_for("guided")}
    assert "research_workbench" not in guided
    assert "integrator_convergence" not in guided


def test_no_scenario_claims_external_validation_as_a_result():
    forbidden = ("nasa-approved", "nasa approved", "nasa-validated", "nasa validated",
                 "flight-certified", "flight certified", "operational mission software")
    for scenario in SCENARIOS:
        blob = " ".join([scenario.summary, scenario.explanation,
                         *scenario.assumptions, *scenario.limitations]).lower()
        for phrase in forbidden:
            assert phrase not in blob or "not " in blob, (
                f"{scenario.id} may be claiming {phrase!r}")

    external = get_scenario("external_validation_export")
    blob = (external.explanation + " ".join(external.limitations)).lower()
    assert "no such comparison has been run" in blob or "no benchmark result" in blob


def test_running_a_scenario_records_the_full_event_sequence(participant, pilot_env):
    scenario = get_scenario("pd_rendezvous")
    outcome = run_scenario(scenario, default_params(scenario), participant, pilot_env)

    assert outcome.ok, outcome.errors
    assert outcome.result is not None
    assert outcome.duration_ms >= 0

    types = _event_types(pilot_env)
    assert types.count(EventType.SIMULATION_STARTED) == 1
    assert types.count(EventType.SIMULATION_COMPLETED) == 1
    assert types.count(EventType.SCENARIO_CATEGORY_USED) == 1
    assert types.count(EventType.FEATURE_CATEGORY_USED) == 1
    assert EventType.SIMULATION_FAILED not in types


def test_a_completed_simulation_makes_the_participant_a_unique_user(participant, pilot_env):
    before = analytics.summary_metrics(ReportFilters(include_demo=True), pilot_env)
    assert before["unique_users"] == 0

    scenario = get_scenario("two_body_orbit")
    assert run_scenario(scenario, default_params(scenario), participant, pilot_env).ok

    after = analytics.summary_metrics(ReportFilters(include_demo=True), pilot_env)
    assert after["unique_users"] == 1
    assert after["active_users"] == 1
    assert after["completed_runs"] == 1


def test_the_recorded_scenario_matches_what_was_run(participant, pilot_env):
    scenario = get_scenario("cw_natural_motion")
    run_scenario(scenario, default_params(scenario), participant, pilot_env)
    with db.get_engine(pilot_env).connect() as connection:
        row = connection.execute(
            select(events.c.scenario_id, events.c.scenario_category,
                   events.c.feature_category)
            .where(events.c.event_type == EventType.SIMULATION_COMPLETED)
        ).mappings().one()
    assert row["scenario_id"] == "cw_natural_motion"
    assert row["scenario_category"] == scenario.category
    assert row["feature_category"] == scenario.feature_category


def test_invalid_parameters_are_rejected_without_running(participant, pilot_env):
    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    params["altitude_km"] = -5000.0

    outcome = run_scenario(scenario, params, participant, pilot_env)
    assert not outcome.ok
    assert outcome.rejected
    assert outcome.error_class == "ValidationError"
    assert outcome.result is None

    types = _event_types(pilot_env)
    assert EventType.SIMULATION_REJECTED in types
    assert EventType.SIMULATION_STARTED not in types
    assert EventType.SIMULATION_COMPLETED not in types


def test_an_engine_exception_is_recorded_as_a_failure_not_raised(participant, pilot_env):
    import dataclasses

    scenario = get_scenario("pd_rendezvous")

    def explode(_params):
        raise RuntimeError("simulated engine fault")

    patched = dataclasses.replace(scenario, runner=explode)

    outcome = run_scenario(patched, default_params(scenario), participant, pilot_env)
    assert not outcome.ok
    assert outcome.error_class == "RuntimeError"
    assert not outcome.rejected
    assert "Traceback" not in " ".join(outcome.errors)

    types = _event_types(pilot_env)
    assert EventType.SIMULATION_STARTED in types
    assert EventType.SIMULATION_FAILED in types
    assert EventType.SIMULATION_COMPLETED not in types


def test_warnings_are_carried_through_to_the_outcome(participant, pilot_env):
    scenario = get_scenario("cw_natural_motion")
    params = default_params(scenario)
    params["rel_y"] = 40_000.0
    outcome = run_scenario(scenario, params, participant, pilot_env)
    assert outcome.ok, outcome.errors
    assert outcome.warnings


def test_running_without_a_context_still_works(pilot_env):
    scenario = get_scenario("two_body_orbit")
    outcome = run_scenario(scenario, default_params(scenario), None, pilot_env)
    assert outcome.ok
    assert _event_types(pilot_env) == []


def test_analytics_detail_holds_only_buckets_not_raw_values(participant, pilot_env):
    import json

    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    params["rel_x"] = 1234.5
    run_scenario(scenario, params, participant, pilot_env)

    with db.get_engine(pilot_env).connect() as connection:
        detail = connection.execute(
            select(events.c.detail_json)
            .where(events.c.event_type == EventType.SIMULATION_COMPLETED)
        ).scalar_one()
    payload = json.loads(detail)
    assert "1234.5" not in detail
    assert payload["separation_bucket_m"] in ("<100m", "100m-1km", "1-10km",
                                              "10-50km", ">50km")
    assert payload["steps_bucket"]


@pytest.mark.parametrize("scenario_id", [scenario.id for scenario in SCENARIOS])
def test_every_scenario_runs_with_its_defaults(scenario_id, participant, pilot_env):
    scenario = get_scenario(scenario_id)
    outcome = run_scenario(scenario, default_params(scenario), participant, pilot_env)
    assert outcome.ok, f"{scenario_id} failed: {outcome.errors}"
    assert outcome.result is not None


def test_scenario_results_are_deterministic(pilot_env):
    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    first = run_scenario(scenario, params, None, pilot_env).result
    second = run_scenario(scenario, params, None, pilot_env).result
    np.testing.assert_array_equal(first["states"], second["states"])


def test_rate_limiter_allows_up_to_the_burst_limit():
    limiter = SessionRateLimiter(max_per_minute=3, max_per_session=100)
    for _ in range(3):
        assert limiter.consume("token").allowed
    blocked = limiter.consume("token")
    assert not blocked.allowed
    assert "wait" in blocked.reason


def test_rate_limiter_window_expires():
    limiter = SessionRateLimiter(max_per_minute=2, max_per_session=100,
                                 window_seconds=60.0)
    assert limiter.consume("token", now=0.0).allowed
    assert limiter.consume("token", now=1.0).allowed
    assert not limiter.consume("token", now=2.0).allowed
    assert limiter.consume("token", now=70.0).allowed


def test_rate_limiter_enforces_a_session_total():
    limiter = SessionRateLimiter(max_per_minute=1000, max_per_session=3)
    for _ in range(3):
        assert limiter.consume("token").allowed
    blocked = limiter.consume("token")
    assert not blocked.allowed
    assert "per-session limit" in blocked.reason


def test_rate_limiter_isolates_sessions():
    limiter = SessionRateLimiter(max_per_minute=1, max_per_session=10)
    assert limiter.consume("alice").allowed
    assert not limiter.consume("alice").allowed
    assert limiter.consume("bob").allowed, "one session must not limit another"


def test_check_does_not_consume():
    limiter = SessionRateLimiter(max_per_minute=1, max_per_session=10)
    assert limiter.check("token").allowed
    assert limiter.check("token").allowed
    assert limiter.consume("token").allowed


def test_rate_limited_runs_are_recorded_and_blocked(participant, pilot_env, monkeypatch):
    from pilot import ratelimit

    ratelimit.reset_limiter()
    monkeypatch.setattr(ratelimit, "_limiter",
                        SessionRateLimiter(max_per_minute=1, max_per_session=1))

    scenario = get_scenario("two_body_orbit")
    params = default_params(scenario)
    assert run_scenario(scenario, params, participant, pilot_env).ok

    blocked = run_scenario(scenario, params, participant, pilot_env)
    assert not blocked.ok
    assert blocked.error_class == "RateLimited"
    assert blocked.rejected

    types = _event_types(pilot_env)
    assert types.count(EventType.SIMULATION_COMPLETED) == 1
    assert EventType.SIMULATION_REJECTED in types
    ratelimit.reset_limiter()


def test_password_hash_roundtrip():
    encoded = auth.hash_password("correct horse battery staple", iterations=1000)
    assert auth.verify_password("correct horse battery staple", encoded)
    assert not auth.verify_password("wrong password", encoded)


def test_hash_is_salted_so_two_hashes_differ():
    first = auth.hash_password("same", iterations=1000)
    second = auth.hash_password("same", iterations=1000)
    assert first != second
    assert auth.verify_password("same", first)
    assert auth.verify_password("same", second)


def test_verify_password_rejects_malformed_hashes():
    assert not auth.verify_password("x", "not-a-hash")
    assert not auth.verify_password("x", "")
    assert not auth.verify_password("x", "md5$1$aa$bb")


def test_admin_refuses_when_nothing_is_configured(pilot_env):
    authenticator = auth.AdminAuthenticator(pilot_env)
    assert not authenticator.configured
    result = authenticator.authenticate("admin", "anything")
    assert not result.ok
    assert "No administrator credentials" in result.message


def test_admin_accepts_correct_hashed_credentials(monkeypatch, pilot_env):
    from pilot import config

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD_HASH",
                       auth.hash_password("a-strong-password", iterations=1000))
    settings = config.get_settings(refresh=True)

    authenticator = auth.AdminAuthenticator(settings)
    assert authenticator.authenticate("coordinator", "a-strong-password").ok
    assert not authenticator.authenticate("coordinator", "wrong").ok
    assert not authenticator.authenticate("someone-else", "a-strong-password").ok
    config.get_settings(refresh=True)


def test_admin_locks_out_after_repeated_failures(monkeypatch, pilot_env):
    from pilot import config

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD", "letmein")
    monkeypatch.setenv("PILOT_ADMIN_MAX_ATTEMPTS", "3")
    settings = config.get_settings(refresh=True)

    authenticator = auth.AdminAuthenticator(settings)
    for _ in range(3):
        assert not authenticator.authenticate("coordinator", "nope").ok
    locked = authenticator.authenticate("coordinator", "letmein")
    assert not locked.ok, "correct credentials must be refused while locked out"
    assert "too many failed attempts" in locked.message.lower()
    assert locked.locked_until > 0.0
    config.get_settings(refresh=True)


def test_plaintext_admin_password_produces_a_warning(monkeypatch):
    from pilot import config

    monkeypatch.setenv("PILOT_ADMIN_USERNAME", "coordinator")
    monkeypatch.setenv("PILOT_ADMIN_PASSWORD", "letmein")
    monkeypatch.delenv("PILOT_ADMIN_PASSWORD_HASH", raising=False)
    settings = config.get_settings(refresh=True)
    assert any("plain text" in warning for warning in settings.deployment_warnings())
    config.get_settings(refresh=True)
