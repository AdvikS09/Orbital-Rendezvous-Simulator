"""Tests for input validation and the resource safeguards.

Every "silent wrong answer" recorded in CURRENT_STATE_AUDIT.md §4.5 gets a test
here proving it is now caught before the engine runs.
"""

import pytest

from pilot.validation import (
    MAX_DURATION_ORBITS,
    ValidationError,
    check_number,
    validate_participant_code,
    validate_pilot_code,
    validate_simulation_request,
)


# --------------------------------------------------------------------------
# The audit's silent-wrong-answer cases
# --------------------------------------------------------------------------
def test_negative_altitude_is_rejected():
    """Audit §4.5: this produced an all-NaN trajectory with no error at all."""
    result = validate_simulation_request(altitude_km=-7000.0)
    assert not result.ok
    assert any("Altitude" in error for error in result.errors)


def test_nan_input_is_rejected():
    """Audit §4.5: NaN propagated silently all the way to NaN metrics."""
    result = validate_simulation_request(rel_pos=[float("nan"), 0.0, 0.0])
    assert not result.ok
    assert any("NaN" in error for error in result.errors)


def test_infinite_input_is_rejected():
    result = validate_simulation_request(altitude_km=float("inf"))
    assert not result.ok
    assert any("finite" in error for error in result.errors)


def test_overlong_horizon_is_rejected_rather_than_silently_truncated():
    """Audit §4.5: 1e7 s was quietly clamped to 2e5 s, a 20x shorter run."""
    result = validate_simulation_request(altitude_km=400.0, duration_s=1e7, dt_s=1.0)
    assert not result.ok
    joined = " ".join(result.errors)
    assert "orbits" in joined or "steps" in joined


def test_negative_duration_is_rejected():
    result = validate_simulation_request(altitude_km=400.0, duration_s=-500.0, dt_s=5.0)
    assert not result.ok


def test_zero_timestep_is_rejected_before_it_can_divide_by_zero():
    result = validate_simulation_request(altitude_km=400.0, duration_s=5000.0, dt_s=0.0)
    assert not result.ok
    assert any("Timestep" in error for error in result.errors)


def test_zero_mass_is_rejected_before_it_can_divide_by_zero():
    result = validate_simulation_request(mass=0.0)
    assert not result.ok
    assert any("Mass" in error for error in result.errors)


def test_chaser_exactly_on_target_is_rejected():
    result = validate_simulation_request(rel_pos=[0.0, 0.0, 0.0], rel_vel=[0.0, 0.0, 0.0])
    assert not result.ok
    assert any("nothing to simulate" in error for error in result.errors)


# --------------------------------------------------------------------------
# Valid input passes
# --------------------------------------------------------------------------
def test_the_published_reference_case_is_accepted():
    result = validate_simulation_request(
        altitude_km=400.0, inclination_deg=51.6, duration_s=5553.62, dt_s=5.0,
        rel_pos=[200.0, -300.0, 80.0], rel_vel=[0.0, 0.0, 0.0],
        max_accel=0.05, model="cw")
    assert result.ok, result.errors
    assert result.warnings == []
    assert result.n_steps == pytest.approx(1111, abs=2)


def test_valid_request_predicts_a_cost():
    result = validate_simulation_request(altitude_km=400.0, duration_s=5553.62,
                                         dt_s=5.0, model="comparison")
    assert result.ok
    assert result.n_steps > 0
    assert result.estimated_seconds > 0.0


# --------------------------------------------------------------------------
# Warnings (run proceeds, but the result must be read carefully)
# --------------------------------------------------------------------------
def test_large_separation_warns_about_cw_validity():
    result = validate_simulation_request(
        altitude_km=400.0, rel_pos=[0.0, 80_000.0, 0.0], duration_s=5553.62, dt_s=5.0)
    assert result.ok, result.errors
    assert any("linearisation" in warning for warning in result.warnings)


def test_moderate_separation_warns_more_gently():
    result = validate_simulation_request(
        altitude_km=400.0, rel_pos=[0.0, 5000.0, 0.0], duration_s=5553.62, dt_s=5.0)
    assert result.ok
    assert any("1 km" in warning for warning in result.warnings)


def test_small_separation_produces_no_separation_warning():
    result = validate_simulation_request(
        altitude_km=400.0, rel_pos=[0.0, 500.0, 0.0], duration_s=5553.62, dt_s=5.0)
    assert result.ok
    assert not any("separation" in warning.lower() for warning in result.warnings)


def test_coarse_timestep_warns_about_under_resolution():
    result = validate_simulation_request(altitude_km=400.0, duration_s=5553.62, dt_s=100.0)
    assert result.ok, result.errors
    assert any("steps per orbit" in warning for warning in result.warnings)


def test_very_low_altitude_warns_about_the_atmosphere_model():
    result = validate_simulation_request(altitude_km=180.0)
    assert result.ok, result.errors
    assert any("atmosphere" in warning for warning in result.warnings)


# --------------------------------------------------------------------------
# Resource limits
# --------------------------------------------------------------------------
def test_step_budget_is_enforced():
    result = validate_simulation_request(altitude_km=400.0, duration_s=5553.62,
                                         dt_s=0.05, model="cw", max_steps=40_000)
    assert not result.ok
    assert any("integration steps" in error for error in result.errors)


def test_time_budget_is_enforced():
    result = validate_simulation_request(altitude_km=400.0, duration_s=5553.62,
                                         dt_s=0.2, model="comparison",
                                         max_steps=10_000_000, max_seconds=2.0)
    assert not result.ok
    assert any("estimated to take" in error for error in result.errors)


def test_duration_limit_is_expressed_in_orbits():
    """The same limit must mean the same thing at every altitude."""
    low = validate_simulation_request(altitude_km=400.0, duration_s=1e6, dt_s=100.0)
    high = validate_simulation_request(altitude_km=1200.0, duration_s=1e6, dt_s=100.0)
    # 1e6 s is ~180 orbits at 400 km but ~137 at 1200 km; both exceed the limit.
    assert not low.ok and not high.ok
    assert any(str(int(MAX_DURATION_ORBITS)) in error for error in low.errors)


def test_raise_if_invalid():
    result = validate_simulation_request(altitude_km=-1.0)
    with pytest.raises(ValidationError):
        result.raise_if_invalid()
    validate_simulation_request(altitude_km=400.0).raise_if_invalid()  # must not raise


# --------------------------------------------------------------------------
# check_number
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["abc", None, [1, 2]])
def test_check_number_rejects_non_numbers(value):
    number, error = check_number(value, "Thing", 0.0, 10.0)
    assert number is None and error


def test_check_number_reports_the_unit():
    _, error = check_number(5000.0, "Altitude", 0.0, 1200.0, "km")
    assert "km" in error


# --------------------------------------------------------------------------
# Identity codes
# --------------------------------------------------------------------------
def test_participant_code_is_normalised_to_upper_case():
    code, error = validate_participant_code("  alice-01 ")
    assert error is None
    assert code == "ALICE-01"


@pytest.mark.parametrize("bad,expected", [
    ("", "Enter"),
    ("ab", "at least"),
    ("x" * 41, "at most"),
    ("alice@example.org", "email"),
    ("Alice Smith", "letters, digits"),
])
def test_participant_code_rejects_bad_input(bad, expected):
    code, error = validate_participant_code(bad)
    assert code == ""
    assert expected in error


def test_participant_code_rejects_anything_email_shaped():
    """An anonymous code must not be able to be personal data."""
    _, error = validate_participant_code("me@example.com")
    assert "email" in error


def test_pilot_code_validation():
    code, error = validate_pilot_code(" school-a ")
    assert error is None and code == "SCHOOL-A"
    _, error = validate_pilot_code("")
    assert error
    _, error = validate_pilot_code("has spaces")
    assert error
