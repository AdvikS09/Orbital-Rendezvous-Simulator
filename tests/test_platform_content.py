"""Tests for educational content: activities, assessments and exports.

Two things matter here beyond "does it run".

First, **the numbers quoted to learners must be true**. Activity text tells
students what to expect ("about 1595 s", "roughly 3770 m", "about 5.8 m"), and a
wrong expected value teaches the wrong lesson more effectively than no value at
all. Those claims are checked against the engine here.

Second, **every activity step must actually be runnable** through the same
validated path as any other simulation.
"""

import numpy as np
import pytest

from pilot.content.activities import ACTIVITIES, ACTIVITIES_BY_ID, get_activity
from pilot.content.assessments import (
    INSTRUMENTS,
    ORBITAL_BASICS,
    get_instrument,
    score_responses,
)
from pilot.content.scenarios_catalog import default_params, get_scenario
from pilot.runner import run_scenario, validate_params


# --------------------------------------------------------------------------
# Activity structure
# --------------------------------------------------------------------------
def test_activities_have_unique_ids_and_content():
    ids = [activity.id for activity in ACTIVITIES]
    assert len(ids) == len(set(ids))
    for activity in ACTIVITIES:
        assert activity.title and activity.summary
        assert activity.learning_points
        assert activity.steps
        assert activity.duration_minutes > 0


def test_every_activity_step_points_at_a_real_scenario():
    for activity in ACTIVITIES:
        for step in activity.steps:
            assert get_scenario(step.scenario_id) is not None, (
                f"{activity.id}/{step.id} references unknown scenario "
                f"{step.scenario_id!r}")


def test_every_activity_step_has_a_prediction_and_a_takeaway():
    """Predict-then-check is the point; a step without a prediction is a demo."""
    for activity in ACTIVITIES:
        for step in activity.steps:
            assert step.predict_prompt, f"{activity.id}/{step.id} has no prediction"
            assert len(step.predict_options) >= 2, f"{activity.id}/{step.id}"
            assert step.look_for, f"{activity.id}/{step.id} has no 'look for'"
            assert step.takeaway, f"{activity.id}/{step.id} has no takeaway"


def test_activity_step_parameters_validate():
    for activity in ACTIVITIES:
        for step in activity.steps:
            scenario = get_scenario(step.scenario_id)
            params = {**default_params(scenario), **step.params}
            result = validate_params(scenario, params)
            assert result.ok, f"{activity.id}/{step.id} rejected: {result.errors}"


@pytest.mark.parametrize("activity_id", [activity.id for activity in ACTIVITIES])
def test_every_activity_step_runs(activity_id, pilot_env):
    activity = get_activity(activity_id)
    for step in activity.steps:
        scenario = get_scenario(step.scenario_id)
        params = {**default_params(scenario), **step.params}
        outcome = run_scenario(scenario, params, None, pilot_env)
        assert outcome.ok, f"{activity_id}/{step.id} failed: {outcome.errors}"


def test_activities_are_flagged_as_needing_teacher_review():
    """The draft status must stay visible until a teacher has actually reviewed."""
    for activity in ACTIVITIES:
        assert "review" in activity.review_status.lower()


# --------------------------------------------------------------------------
# The numbers quoted to learners must be true
# --------------------------------------------------------------------------
def test_period_at_1200km_is_about_109_minutes():
    """Activity 1 step 2 tells learners to expect ~109 minutes."""
    from src import orbit

    period = orbit.orbital_period(orbit.R_EARTH + 1_200_000.0)
    assert period / 60 == pytest.approx(109.0, abs=1.0)


def test_period_at_400km_is_about_92_6_minutes():
    from src import orbit

    period = orbit.orbital_period(orbit.R_EARTH + 400_000.0)
    assert period / 60 == pytest.approx(92.6, abs=0.1)


def test_radial_offset_drift_matches_the_quoted_3770m(pilot_env):
    """Activity 2 step 2 tells learners to expect about -3770 m along-track."""
    activity = get_activity("relative_motion")
    step = next(s for s in activity.steps if s.id == "step2_radial")
    scenario = get_scenario(step.scenario_id)
    outcome = run_scenario(scenario, {**default_params(scenario), **step.params},
                           None, pilot_env)
    assert outcome.ok
    final_along_track = float(outcome.result["states"][-1, 1])
    assert final_along_track == pytest.approx(-3770.0, rel=0.02)
    # And it is the textbook -12*pi*x0 result.
    assert final_along_track == pytest.approx(-12 * np.pi * 100.0, rel=0.02)


def test_alongtrack_parking_really_stays_put(pilot_env):
    """Activity 2 step 1 claims the chaser 'barely moves'."""
    activity = get_activity("relative_motion")
    step = next(s for s in activity.steps if s.id == "step1_alongtrack")
    scenario = get_scenario(step.scenario_id)
    outcome = run_scenario(scenario, {**default_params(scenario), **step.params},
                           None, pilot_env)
    assert outcome.ok
    positions = outcome.result["states"][:, :3]
    drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    assert drift < 1.0, f"'barely moves' is false: drifted {drift:.1f} m"


def test_cross_track_returns_to_where_it_started(pilot_env):
    """Activity 2 step 3 claims a full sine cycle returning to the start."""
    activity = get_activity("relative_motion")
    step = next(s for s in activity.steps if s.id == "step3_crosstrack")
    scenario = get_scenario(step.scenario_id)
    outcome = run_scenario(scenario, {**default_params(scenario), **step.params},
                           None, pilot_env)
    assert outcome.ok
    states = outcome.result["states"]
    assert float(states[-1, 2]) == pytest.approx(200.0, abs=1.0)
    # In-plane components must stay at zero -- the decoupling claim.
    assert float(np.max(np.abs(states[:, 0]))) < 1e-6
    assert float(np.max(np.abs(states[:, 1]))) < 1e-6


def test_rendezvous_numbers_quoted_in_activity_three(pilot_env):
    """Activity 3 step 1 quotes ~1595 s, ~2.33 m/s and ~0.017 m/s^2."""
    activity = get_activity("control_and_cost")
    step = next(s for s in activity.steps if s.id == "step1_rendezvous")
    scenario = get_scenario(step.scenario_id)
    outcome = run_scenario(scenario, {**default_params(scenario), **step.params},
                           None, pilot_env)
    assert outcome.ok
    metrics = outcome.result["metrics"].iloc[0]
    assert float(metrics["convergence_time_s"]) == pytest.approx(1595.0, abs=25.0)
    assert float(metrics["delta_v_mps"]) == pytest.approx(2.33, abs=0.05)
    assert float(metrics["max_control_accel_mps2"]) == pytest.approx(0.017, abs=0.001)
    # And the claim that it is "comfortably under the 0.05 limit".
    assert float(metrics["max_control_accel_mps2"]) < 0.05


def test_higher_bandwidth_is_faster_and_costlier(pilot_env):
    """Activity 3 step 2 claims faster but more expensive. Verify the direction."""
    scenario = get_scenario("pd_rendezvous")
    base = {**default_params(scenario), "rel_x": 200.0, "rel_y": -300.0, "rel_z": 80.0,
            "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 5.0, "max_accel": 0.05}

    slow = run_scenario(scenario, {**base, "omega_n_factor": 6.0}, None, pilot_env)
    fast = run_scenario(scenario, {**base, "omega_n_factor": 12.0}, None, pilot_env)
    assert slow.ok and fast.ok

    slow_metrics = slow.result["metrics"].iloc[0]
    fast_metrics = fast.result["metrics"].iloc[0]
    assert fast_metrics["convergence_time_s"] < slow_metrics["convergence_time_s"]
    assert fast_metrics["delta_v_mps"] > slow_metrics["delta_v_mps"]
    assert fast_metrics["max_control_accel_mps2"] > slow_metrics["max_control_accel_mps2"]


def test_j2_roughly_doubles_the_model_gap_as_the_activity_claims(pilot_env):
    """Activity 3 step 4 quotes ~2.8 m without J2 and ~5.8 m with it."""
    scenario = get_scenario("cw_vs_reference")
    base = {**default_params(scenario), "rel_x": 0.0, "rel_y": 1000.0, "rel_z": 0.0,
            "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0, "altitude_km": 400.0,
            "inclination_deg": 51.6, "duration_s": 5553.62, "dt_s": 10.0,
            "controlled": False, "use_drag": False}

    without = run_scenario(scenario, {**base, "use_j2": False}, None, pilot_env)
    with_j2 = run_scenario(scenario, {**base, "use_j2": True}, None, pilot_env)
    assert without.ok and with_j2.ok

    plain = float(without.result["comparison"]["summary"].iloc[0]["max_pos_error_m"])
    perturbed = float(with_j2.result["comparison"]["summary"].iloc[0]["max_pos_error_m"])
    assert plain == pytest.approx(2.8, abs=0.15)
    assert perturbed == pytest.approx(5.8, abs=0.15)


def test_radial_offset_is_much_worse_than_along_track(pilot_env):
    """Activity 3 step 4's closing claim, and a scenario limitation elsewhere."""
    scenario = get_scenario("cw_vs_reference")
    base = {**default_params(scenario), "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
            "altitude_km": 400.0, "inclination_deg": 51.6, "duration_s": 5553.62,
            "dt_s": 10.0, "controlled": False, "use_j2": False, "use_drag": False}

    along = run_scenario(scenario, {**base, "rel_x": 0.0, "rel_y": 1000.0, "rel_z": 0.0},
                         None, pilot_env)
    radial = run_scenario(scenario, {**base, "rel_x": 1000.0, "rel_y": 0.0, "rel_z": 0.0},
                          None, pilot_env)
    assert along.ok and radial.ok

    along_error = float(along.result["comparison"]["summary"].iloc[0]["max_pos_error_m"])
    radial_error = float(radial.result["comparison"]["summary"].iloc[0]["max_pos_error_m"])
    assert radial_error > 10 * along_error, (
        f"radial {radial_error:.1f} m vs along-track {along_error:.1f} m")


# --------------------------------------------------------------------------
# Assessment instrument
# --------------------------------------------------------------------------
def test_instrument_structure():
    for instrument in INSTRUMENTS.values():
        assert instrument.items
        assert instrument.max_score == len(instrument.items)
        ids = [item.id for item in instrument.items]
        assert len(ids) == len(set(ids)), "duplicate item id"
        for item in instrument.items:
            assert len(item.options) >= 2
            assert 0 <= item.correct_index < len(item.options)
            assert item.question and item.topic
            assert item.explanation, f"{item.id} has no explanation"


def test_instrument_is_flagged_as_needing_teacher_review():
    for instrument in INSTRUMENTS.values():
        assert "review" in instrument.review_status.lower()


def test_get_instrument_falls_back_to_the_default():
    assert get_instrument(None).id == ORBITAL_BASICS.id
    assert get_instrument("does-not-exist").id == ORBITAL_BASICS.id


def test_scoring_all_correct():
    responses = {item.id: item.correct_index for item in ORBITAL_BASICS.items}
    score, per_item = score_responses(ORBITAL_BASICS, responses)
    assert score == ORBITAL_BASICS.max_score
    assert all(entry["correct"] for entry in per_item)


def test_scoring_all_wrong():
    responses = {item.id: (item.correct_index + 1) % len(item.options)
                 for item in ORBITAL_BASICS.items}
    score, per_item = score_responses(ORBITAL_BASICS, responses)
    assert score == 0
    assert not any(entry["correct"] for entry in per_item)


def test_scoring_handles_unanswered_items():
    score, per_item = score_responses(ORBITAL_BASICS, {})
    assert score == 0
    assert all(entry["chosen_index"] is None for entry in per_item)


def test_scored_responses_store_no_free_text():
    """Only correctness is persisted -- never a written answer."""
    responses = {item.id: item.correct_index for item in ORBITAL_BASICS.items}
    _, per_item = score_responses(ORBITAL_BASICS, responses)
    for entry in per_item:
        assert set(entry) == {"item_id", "topic", "chosen_index", "correct"}


def test_assessment_items_cover_the_activity_topics():
    """The instrument should measure what the activities actually teach."""
    topics = {item.topic for item in ORBITAL_BASICS.items}
    assert {"orbits", "perturbations", "relative_motion", "control",
            "modelling"}.issubset(topics)


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
def test_scenario_config_is_reproducible_and_clean():
    from pilot.export import config_json, scenario_config

    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    config = scenario_config(scenario, params)

    assert config["scenario"]["id"] == "pd_rendezvous"
    assert config["parameters"] == {k: v for k, v in params.items()}
    assert config["numerics"]["deterministic"] is True
    assert config["numerics"]["random_seed"] is None
    assert config["engine_version"]
    assert config["environment"]["numpy"]
    assert config["assumptions"] and config["limitations"]
    # It must be JSON-serialisable.
    assert config_json(scenario, params)


def test_exports_carry_no_participant_information():
    from pilot.export import config_json, results_readme

    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    blob = config_json(scenario, params) + results_readme(scenario, params, [])
    for forbidden in ("participant", "session_token", "pilot_code", "organisation"):
        assert forbidden not in blob.lower(), f"export leaks {forbidden}"


def test_export_states_the_validation_status_honestly():
    from pilot.export import config_json, results_readme

    scenario = get_scenario("pd_rendezvous")
    params = default_params(scenario)
    config = config_json(scenario, params).lower()
    readme = results_readme(scenario, params, []).lower()
    for blob in (config, readme):
        assert "not" in blob and "gmat" in blob
        assert "nasa-approved" in blob or "nasa approved" in blob


@pytest.mark.parametrize("scenario_id", [
    "two_body_orbit", "j2_effect", "drag_effect", "cw_natural_motion",
    "pd_rendezvous", "gain_tuning", "cw_vs_reference", "proximity_safety",
    "integrator_convergence", "external_validation_export", "research_workbench",
])
def test_every_scenario_produces_downloadable_tables(scenario_id, pilot_env):
    from pilot.export import result_tables, tables_to_csv

    scenario = get_scenario(scenario_id)
    outcome = run_scenario(scenario, default_params(scenario), None, pilot_env)
    assert outcome.ok, outcome.errors

    tables = result_tables(scenario.result_kind, outcome.result)
    assert tables, f"{scenario_id} produced no downloadable tables"
    for name, frame in tables.items():
        assert len(frame) > 0, f"{scenario_id}/{name} is empty"

    files = tables_to_csv(tables)
    assert all(name.endswith(".csv") for name in files)
    assert all(len(content) > 0 for content in files.values())


def test_result_tables_degrade_gracefully_on_a_bad_bundle():
    from pilot.export import result_tables

    assert result_tables("relative", None) == {}
    assert result_tables("relative", {"nonsense": 1}) == {}
    assert result_tables("unknown_kind", {"anything": 1}) == {}


def test_results_readme_includes_run_warnings():
    from pilot.export import results_readme

    scenario = get_scenario("cw_natural_motion")
    text = results_readme(scenario, default_params(scenario),
                          ["Initial separation is 80.0 km."])
    assert "WARNINGS RAISED FOR THIS PARTICULAR RUN" in text
    assert "80.0 km" in text
