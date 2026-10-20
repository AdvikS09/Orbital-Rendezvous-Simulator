"""Numerical parity: the API must return exactly what the engine computes.

This is the test that decides whether the migration is safe. For every scenario
it runs the engine directly, runs the same inputs through the HTTP endpoint, and
compares the numbers.

**Tolerance.** ``RTOL = 1e-9`` with ``atol = 0``, taken from
``tests/test_regression_baseline.py:31`` — the repository's only existing
numerical tolerance. It is not invented for this migration.

In principle the comparison should be exact: the API calls the same function
objects in the same process, so the only transformation is Python float ->
JSON -> Python float, which is lossless for IEEE-754 doubles under
``repr``-based serialisation. The 1e-9 tolerance is therefore slack, kept only
so a future change of JSON encoder cannot fail the suite for a reason that has
nothing to do with physics. Any *real* difference would be orders of magnitude
larger.

``None`` in a JSON payload maps to ``NaN`` from the engine: JSON has no NaN, and
a non-converging rendezvous legitimately produces one for convergence time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pilot.content.scenarios_catalog import SCENARIOS, default_params, get_scenario
from pilot.runner import run_scenario

#: The repository's regression tolerance. See the module docstring.
RTOL = 1e-9
ATOL = 0.0

SCENARIO_IDS = [scenario.id for scenario in SCENARIOS]


def _close(actual: Any, expected: Any) -> bool:
    """Compare one value, treating None and NaN as the same absence."""
    expected_missing = expected is None or (
        isinstance(expected, float) and (math.isnan(expected) or math.isinf(expected))
    )
    actual_missing = actual is None or (
        isinstance(actual, float) and (math.isnan(actual) or math.isinf(actual))
    )
    if expected_missing or actual_missing:
        return expected_missing and actual_missing
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) == bool(expected)
    if isinstance(expected, str) or isinstance(actual, str):
        return str(actual) == str(expected)
    return bool(np.isclose(float(actual), float(expected), rtol=RTOL, atol=ATOL))


def _max_relative_difference(actual: list, expected: np.ndarray) -> tuple[float, float]:
    """Return ``(max_abs_diff, max_rel_diff)`` over a pair of arrays."""
    a = np.array([np.nan if v is None else float(v) for v in actual], dtype=float)
    e = np.asarray(expected, dtype=float).ravel()
    assert a.shape == e.shape, f"length differs: {a.shape} vs {e.shape}"

    both_finite = np.isfinite(a) & np.isfinite(e)
    # Missing values must be missing in both.
    assert np.array_equal(np.isfinite(a), np.isfinite(e)), (
        "finite/non-finite pattern differs between API and engine"
    )
    if not both_finite.any():
        return 0.0, 0.0

    diff = np.abs(a[both_finite] - e[both_finite])
    scale = np.abs(e[both_finite])
    max_abs = float(diff.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, 0.0)
    return max_abs, float(np.nanmax(rel))


def _engine_result(scenario_id: str, params: dict | None = None):
    """Run the engine directly, with no HTTP layer involved."""
    scenario = get_scenario(scenario_id)
    resolved = {**default_params(scenario), **(params or {})}
    outcome = run_scenario(scenario, resolved, None)
    assert outcome.ok, f"engine run failed: {outcome.errors}"
    return scenario, resolved, outcome


def _api_result(client, scenario_id: str, params: dict) -> dict:
    response = client.post(f"/api/v1/simulations/{scenario_id}", json={"params": params})
    assert response.status_code == 200, response.text
    return response.json()


# Every scenario, at its documented defaults
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scalar_metric_parity(client, scenario_id):
    """Every scalar the API reports equals the engine's value."""
    scenario, params, outcome = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    assert payload["scenario_id"] == scenario_id
    assert payload["result_kind"] == scenario.result_kind
    assert payload["metrics"], f"{scenario_id} reported no metrics"

    # Rebuild the expected metrics from the same adapter the API used, then
    # confirm the serialised payload preserved them.
    from app.services.adapters import build_result

    expected = build_result(
        scenario=scenario, params=params, result=outcome.result,
        runtime_ms=outcome.duration_ms, n_steps=outcome.n_steps,
        warnings=list(outcome.warnings),
    )
    expected_metrics = {m.key: m.value for m in expected.metrics}

    assert {m["key"] for m in payload["metrics"]} == set(expected_metrics)
    for metric in payload["metrics"]:
        assert _close(metric["value"], expected_metrics[metric["key"]]), (
            f"{scenario_id}.{metric['key']}: API {metric['value']!r} "
            f"vs engine {expected_metrics[metric['key']]!r}"
        )


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_metric_keys_are_unique(client, scenario_id):
    """Two different quantities must never share a key.

    The comparison scenarios previously emitted ``final_pos_error_m`` twice:
    once as the chaser's distance from the target and once as the gap between
    the two models. Same name, unrelated meanings.
    """
    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)
    keys = [metric["key"] for metric in payload["metrics"]]
    duplicates = {key for key in keys if keys.count(key) > 1}
    assert not duplicates, f"{scenario_id}: duplicate metric keys {duplicates}"

    series_keys = list(payload["series"])
    assert len(series_keys) == len(set(series_keys))

    plot_ids = [plot["id"] for plot in payload["plots"]]
    assert len(plot_ids) == len(set(plot_ids)), f"{scenario_id}: duplicate plot ids"

    table_ids = [table["id"] for table in payload["tables"]]
    assert len(table_ids) == len(set(table_ids)), f"{scenario_id}: duplicate table ids"


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_series_array_parity(client, scenario_id):
    """Every numeric array matches element-for-element."""
    scenario, params, outcome = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    from app.services.adapters import build_result

    expected = build_result(
        scenario=scenario, params=params, result=outcome.result,
        runtime_ms=outcome.duration_ms, n_steps=outcome.n_steps,
        warnings=list(outcome.warnings),
    )

    assert set(payload["series"]) == set(expected.series), (
        f"{scenario_id}: series keys differ"
    )
    assert payload["series"], f"{scenario_id} returned no series"

    worst_abs = worst_rel = 0.0
    for key, values in payload["series"].items():
        reference = np.array(
            [np.nan if v is None else float(v) for v in expected.series[key]],
            dtype=float,
        )
        max_abs, max_rel = _max_relative_difference(values, reference)
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        assert max_rel <= RTOL, (
            f"{scenario_id}.{key}: max relative difference {max_rel:.3e} "
            f"exceeds {RTOL:.0e}"
        )
    # Recorded for docs/ENGINE_PARITY_REPORT.md.
    print(f"PARITY {scenario_id}: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}")


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_table_parity(client, scenario_id):
    """Downloadable tables carry the engine's values, not rounded ones."""
    scenario, params, outcome = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    from pilot.export import result_tables

    engine_tables = result_tables(scenario.result_kind, outcome.result)
    api_tables = {table["id"]: table for table in payload["tables"]}

    for name, frame in engine_tables.items():
        assert name in api_tables, f"{scenario_id}: table {name} missing from API"
        rows = api_tables[name]["rows"]
        assert len(rows) == len(frame), f"{scenario_id}.{name}: row count differs"

        for column in frame.columns:
            expected_column = frame[column].to_numpy()
            if not np.issubdtype(expected_column.dtype, np.number):
                continue
            actual_column = [row.get(str(column)) for row in rows]
            _, max_rel = _max_relative_difference(actual_column, expected_column)
            assert max_rel <= RTOL, (
                f"{scenario_id}.{name}.{column}: relative difference "
                f"{max_rel:.3e} exceeds {RTOL:.0e}"
            )


# Units, defaults and derived quantities
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_defaults_and_units_match_the_catalogue(client, scenario_id):
    """The API must not silently re-default or relabel a parameter."""
    scenario = get_scenario(scenario_id)
    response = client.get(f"/api/v1/scenarios/{scenario_id}")
    assert response.status_code == 200, response.text
    detail = response.json()

    catalogue = {p.name: p for p in scenario.params}
    served = {p["name"]: p for p in detail["params"]}
    assert set(served) == set(catalogue), f"{scenario_id}: parameter set differs"

    for name, param in catalogue.items():
        api_param = served[name]
        assert api_param["label"] == param.label
        assert api_param["kind"] == param.kind
        assert api_param["unit"] == param.unit
        assert api_param["help"] == param.help
        assert api_param["advanced"] == param.advanced
        assert list(api_param["options"]) == list(param.options)
        assert _close(api_param["minimum"], param.minimum)
        assert _close(api_param["maximum"], param.maximum)
        assert _close(api_param["step"], param.step)
        assert _close(api_param["default"], param.default)

    assert detail["defaults"] == {
        name: param.default for name, param in catalogue.items()
    }


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_result_echoes_inputs_with_units(client, scenario_id):
    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    echoed = {item["name"]: item for item in payload["inputs"]}
    catalogue = {p.name: p for p in scenario.params}
    for name, param in catalogue.items():
        assert name in echoed, f"{scenario_id}: input {name} not echoed"
        assert echoed[name]["unit"] == param.unit
        assert echoed[name]["label"] == param.label
        assert _close(echoed[name]["value"], params[name])


# Conservation metrics, specifically
def test_conservation_metrics_match_the_regression_baseline(client):
    """The two-body validation numbers the research publishes still hold."""
    params = {"altitude_km": 400.0, "inclination_deg": 51.6, "n_orbits": 2.0}
    payload = _api_result(client, "two_body_orbit", params)
    metrics = {m["key"]: m["value"] for m in payload["metrics"]}

    from orbital_core import orbit

    expected_period = orbit.orbital_period(orbit.R_EARTH + 400_000.0)
    assert _close(metrics["period_s"], expected_period)

    # Independent of the baseline: the integrator must stay excellent.
    assert metrics["max_energy_drift_rel"] < 1e-11
    assert metrics["period_rel_err"] < 1e-9
    assert metrics["max_ang_mom_drift_rel"] < 1e-11


def test_cw_vs_reference_reproduces_the_published_j2_figures(client):
    """2.78 m two-body, 5.81 m with J2 - the headline published result."""
    base = {
        "rel_x": 0.0, "rel_y": 1000.0, "rel_z": 0.0,
        "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
        "altitude_km": 400.0, "inclination_deg": 51.6,
        "duration_s": 5553.62, "dt_s": 10.0,
        "controlled": False, "use_drag": False,
    }

    plain = _api_result(client, "cw_vs_reference", {**base, "use_j2": False})
    with_j2 = _api_result(client, "cw_vs_reference", {**base, "use_j2": True})

    def gap(payload):
        return next(
            m["value"] for m in payload["metrics"] if m["key"] == "model_gap_max_m"
        )

    assert gap(plain) == pytest.approx(2.78, abs=0.02)
    assert gap(with_j2) == pytest.approx(5.81, abs=0.02)
    # And the relationship, not just the numbers.
    assert 1.8 < gap(with_j2) / gap(plain) < 2.5


def test_pd_rendezvous_reproduces_the_published_metrics(client):
    """Convergence about 1595 s, delta-v 2.33 m/s, peak 0.017 m/s^2."""
    payload = _api_result(client, "pd_rendezvous", {
        "rel_x": 200.0, "rel_y": -300.0, "rel_z": 80.0,
        "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
        "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 5.0,
        "omega_n_factor": 6.0, "max_accel": 0.05,
    })
    metrics = {m["key"]: m["value"] for m in payload["metrics"]}

    assert metrics["convergence_time_s"] == pytest.approx(1595.0, abs=25.0)
    assert metrics["delta_v_mps"] == pytest.approx(2.33, abs=0.05)
    assert metrics["max_control_accel_mps2"] == pytest.approx(0.017, abs=0.001)
    assert metrics["initial_pos_error_m"] == pytest.approx(369.32, abs=0.01)


def test_integrator_convergence_reports_fourth_order(client):
    payload = _api_result(client, "integrator_convergence", {
        "rel_x": 1000.0, "rel_y": 0.0, "rel_z": 500.0, "altitude_km": 400.0,
    })
    metrics = {m["key"]: m["value"] for m in payload["metrics"]}
    assert 3.6 <= metrics["mean_observed_order"] <= 4.4


# Solver metadata travels with the result
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_solver_information_is_reported(client, scenario_id):
    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    solver = payload["solver"]
    assert solver["integrator"] == "fixed-step classical Runge-Kutta 4"
    assert solver["deterministic"] is True
    assert solver["random_seed"] is None
    assert solver["model"] == scenario.model

    import orbital_core

    assert solver["engine_version"] == orbital_core.__version__
    assert payload["runtime_ms"] >= 0
    assert payload["completed_at"]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_assumptions_and_limitations_travel_with_every_result(client, scenario_id):
    """A number without its qualifications is how a wrong belief forms."""
    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    assert payload["assumptions"] == list(scenario.assumptions)
    assert payload["limitations"] == list(scenario.limitations)
    assert payload["reading_guide"] == (scenario.reading_guide or "")


# Determinism
@pytest.mark.parametrize("scenario_id", ["pd_rendezvous", "cw_vs_reference"])
def test_identical_requests_give_identical_numbers(client, scenario_id):
    """The reproducibility claim in every exported configuration."""
    scenario = get_scenario(scenario_id)
    params = default_params(scenario)
    first = _api_result(client, scenario_id, params)
    second = _api_result(client, scenario_id, params)
    assert first["series"] == second["series"]
    assert [m["value"] for m in first["metrics"]] == [
        m["value"] for m in second["metrics"]
    ]


# Presentation: no machine notation may reach a participant
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_interpretation_contains_machine_notation(client, scenario_id):
    """The defect this migration exists to fix.

    The Streamlit interface rendered numbers with Python's %g, so an energy
    drift appeared on screen as `2e-12`. Metric values are formatted by the
    frontend, but interpretation *sentences* are composed here, and an
    f-string would put machine notation straight back into the interface.
    """
    import re

    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    machine = re.compile(r"\d+(\.\d+)?[eE][+-]?\d+")
    for interpretation in payload["interpretations"]:
        assert not machine.search(interpretation["text"]), (
            f"{scenario_id}: machine notation in an interpretation shown to a "
            f"participant: {interpretation['text']!r}"
        )


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_user_facing_text_contains_an_emoji(client, scenario_id):
    """Emoji were removed from the interface; nothing may reintroduce them."""
    import re

    scenario, params, _ = _engine_result(scenario_id)
    payload = _api_result(client, scenario_id, params)

    # Ranges are built from code points rather than written as literal glyphs,
    # so this file -- the emoji detector -- does not itself contain an emoji and
    # trip a repository-wide scan.
    ranges = [
        (0x1F000, 0x1FAFF),   # emoticons, symbols, pictographs, transport, flags
        (0x2600, 0x27BF),     # miscellaneous symbols and dingbats
        (0x2B00, 0x2BFF),     # arrows and geometric shapes
        (0xFE0F, 0xFE0F),     # variation selector 16
    ]
    pictographic = re.compile(
        "[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in ranges) + "]"
    )
    blobs = (
        [i["text"] for i in payload["interpretations"]]
        + [i.get("title") or "" for i in payload["interpretations"]]
        + payload["warnings"]
        + payload["assumptions"]
        + payload["limitations"]
        + [payload["reading_guide"], payload["scenario_title"]]
        + [m["label"] for m in payload["metrics"]]
        + [m.get("help") or "" for m in payload["metrics"]]
        + [p["title"] for p in payload["plots"]]
        + [p.get("caption") or "" for p in payload["plots"]]
    )
    for blob in blobs:
        found = pictographic.findall(blob)
        assert not found, f"{scenario_id}: emoji {found} in {blob[:80]!r}"
