"""Downloadable results and reproducible scenario configurations.

Two kinds of download:

* **Results** -- the numerical time history or summary table as CSV.
* **Scenario configuration** -- a JSON file that fully determines the run, so
  the same result can be reproduced later, by a colleague, or in a script
  without this app.

Reproducibility is only a real claim if the file carries everything that
affects the answer. The exported configuration therefore records the engine
version, the scenario id, every parameter, the integration step, and the
library versions the run was made with. Because the engine is deterministic
fixed-step RK4 with no RNG, that set is sufficient.

Nothing participant-identifying is written into any export: no participant
code, no pilot code, no organisation, no session token.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import orbital_core
from .content.scenarios_catalog import Scenario

CONFIG_FORMAT_VERSION = "1.0"


def _library_versions() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__}
    try:
        import plotly
        versions["plotly"] = plotly.__version__
    except Exception:  # noqa: BLE001 - a missing optional import must not break export
        pass
    return versions


def scenario_config(scenario: Scenario, params: dict,
                    app_version: str = "1.0.0") -> dict:
    """A complete, re-runnable description of one simulation."""
    return {
        "format": "orbital-pilot-scenario",
        "format_version": CONFIG_FORMAT_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": orbital_core.__version__,
        "app_version": app_version,
        "scenario": {
            "id": scenario.id,
            "title": scenario.title,
            "category": scenario.category,
            "model": scenario.model,
        },
        "parameters": {key: _jsonable(value) for key, value in params.items()},
        "numerics": {
            "integrator": "fixed-step classical RK4",
            "deterministic": True,
            "random_seed": None,
            "units": "SI (m, m/s, s, kg, rad)",
            "frames": "ECI for absolute states; LVLH (radial, along-track, "
                      "cross-track) for relative states",
        },
        "assumptions": list(scenario.assumptions),
        "limitations": list(scenario.limitations),
        "environment": _library_versions(),
        "reproduce": {
            "note": "Re-run with the same parameters to reproduce this result exactly; "
                    "the engine is deterministic and uses no random numbers.",
            "python_snippet": _python_snippet(scenario, params),
        },
        "provenance": {
            "repository": "Orbital Rendezvous Research Engine",
            "external_validation": "This engine has NOT been benchmarked against "
                                   "GMAT, STK or any other external tool. It is not "
                                   "NASA-approved, NASA-validated, flight-certified "
                                   "or operational mission software.",
        },
    }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _python_snippet(scenario: Scenario, params: dict) -> str:
    """A runnable snippet reproducing the run without this app."""
    pretty = json.dumps({k: _jsonable(v) for k, v in params.items()}, indent=4)
    return (
        "from pilot.content.scenarios_catalog import get_scenario\n\n"
        f"scenario = get_scenario({scenario.id!r})\n"
        f"params = {pretty}\n"
        "result = scenario.runner(params)\n"
    )


def config_json(scenario: Scenario, params: dict, app_version: str = "1.0.0") -> str:
    return json.dumps(scenario_config(scenario, params, app_version), indent=2)


# --------------------------------------------------------------------------
# Result tables
# --------------------------------------------------------------------------
def result_tables(result_kind: str, result) -> dict[str, pd.DataFrame]:
    """Extract the downloadable tables from a scenario result bundle.

    Returns ``{filename_stem: DataFrame}``. Unknown shapes give an empty dict
    rather than raising, so a new scenario cannot break the download button.
    """
    tables: dict[str, pd.DataFrame] = {}
    if result is None:
        return tables

    try:
        if result_kind == "orbit":
            tables["orbit_validation"] = result["validation"]
            tables["orbit_timeseries"] = pd.DataFrame({
                "t_s": result["times"],
                "x_m": result["states"][:, 0], "y_m": result["states"][:, 1],
                "z_m": result["states"][:, 2],
                "vx_mps": result["states"][:, 3], "vy_mps": result["states"][:, 4],
                "vz_mps": result["states"][:, 5],
                "altitude_km": result["altitude_km"],
            })
        elif result_kind == "orbit_comparison":
            tables["perturbation_effect"] = pd.DataFrame({
                "t_s": result["times"],
                "separation_m": result["separation_m"],
                "baseline_altitude_km": result["baseline"]["altitude_km"],
                "perturbed_altitude_km": result["perturbed"]["altitude_km"],
            })
        elif result_kind == "relative":
            tables["relative_timeseries"] = result["history"]
            tables["metrics"] = result["metrics"]
        elif result_kind == "gain_sweep":
            tables["gain_sweep"] = result["table"]
        elif result_kind == "comparison":
            primary = result.get("primary")
            if primary is not None:
                tables["relative_timeseries"] = primary["history"]
                tables["metrics"] = primary["metrics"]
            if "comparison" in result:
                tables["model_divergence_summary"] = result["comparison"]["summary"]
                tables["model_divergence_timeseries"] = pd.DataFrame({
                    "t_s": result["times"],
                    "pos_error_m": result["comparison"]["pos_error"],
                    "vel_error_mps": result["comparison"]["vel_error"],
                })
        elif result_kind == "collision":
            analysis = result["analysis"]
            tables["proximity_summary"] = analysis["summary"]
            columns = {"t_s": result["times"],
                       "dist_to_target_m": analysis["dist_to_target"]}
            if "dist_to_debris" in analysis:
                columns["dist_to_debris_m"] = analysis["dist_to_debris"]
            tables["proximity_timeseries"] = pd.DataFrame(columns)
            tables["chaser_timeseries"] = result["chaser"]["history"]
        elif result_kind == "convergence":
            tables["convergence_study"] = pd.DataFrame(result["rows"])
        elif result_kind == "external_export":
            tables["ephemeris"] = result["ephemeris"]
    except (KeyError, TypeError, IndexError):
        # A malformed bundle should degrade to "no downloads", not an error page.
        return {}
    return tables


def tables_to_csv(tables: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    return {f"{name}.csv": table.to_csv(index=False).encode("utf-8")
            for name, table in tables.items()}


def results_readme(scenario: Scenario, params: dict, warnings: list[str],
                   app_version: str = "1.0.0") -> str:
    """A plain-text header shipped with downloads, so a stray CSV stays honest."""
    lines = [
        f"{scenario.title}",
        "=" * len(scenario.title),
        "",
        f"Exported     : {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        f"Engine       : Orbital Rendezvous Research Engine v{orbital_core.__version__}",
        f"Platform     : v{app_version}",
        f"Scenario id  : {scenario.id}",
        "",
        "WHAT THIS IS",
        "------------",
        scenario.summary,
        "",
        "ASSUMPTIONS",
        "-----------",
    ]
    lines += [f"  * {item}" for item in scenario.assumptions]
    lines += ["", "LIMITATIONS", "-----------"]
    lines += [f"  * {item}" for item in scenario.limitations]
    if warnings:
        lines += ["", "WARNINGS RAISED FOR THIS PARTICULAR RUN", "-" * 39]
        lines += [f"  ! {item}" for item in warnings]
    lines += [
        "",
        "PARAMETERS",
        "----------",
    ]
    lines += [f"  {key} = {value}" for key, value in sorted(params.items())]
    lines += [
        "",
        "STATUS OF THIS SOFTWARE",
        "-----------------------",
        "  Research and educational software. It has NOT been benchmarked against",
        "  GMAT, STK or any other external tool, and it is not NASA-approved,",
        "  NASA-validated, flight-certified or operational mission software.",
        "  Do not use it for operational flight decisions.",
        "",
        "  Licensed under the MIT License.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["scenario_config", "config_json", "result_tables", "tables_to_csv",
           "results_readme", "CONFIG_FORMAT_VERSION"]
