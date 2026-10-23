from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

handle, db_path = tempfile.mkstemp(suffix=".db", prefix="parity_")
os.close(handle)
os.environ["DATABASE_URL"] = f"sqlite:///{db_path.replace(os.sep, '/')}"
os.environ["PILOT_ENV"] = "test"
os.environ["PILOT_PARTICIPANT_SALT"] = "parity-report-salt"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pilot import config, db  # noqa: E402
from pilot.content.scenarios_catalog import (  # noqa: E402
    SCENARIOS,
    default_params,
)
from pilot.export import result_tables  # noqa: E402
from pilot.runner import run_scenario  # noqa: E402

settings = config.get_settings(refresh=True)
db.run_migrations(db.get_engine(settings))

from app.main import app  # noqa: E402
from app.services.adapters import build_result  # noqa: E402

RTOL = 1e-9


def worst(actual, expected) -> tuple[float, float, int]:
    a = np.array([np.nan if v is None else float(v) for v in actual], dtype=float)
    e = np.array([np.nan if v is None else float(v) for v in expected], dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0
    finite = np.isfinite(a) & np.isfinite(e)
    if not finite.any():
        return 0.0, 0.0, int(a.size)
    diff = np.abs(a[finite] - e[finite])
    scale = np.abs(e[finite])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, 0.0)
    return float(diff.max()), float(np.nanmax(rel)), int(a.size)


rows = []
with TestClient(app) as client:
    for scenario in SCENARIOS:
        params = default_params(scenario)
        outcome = run_scenario(scenario, params, None)
        assert outcome.ok, outcome.errors

        expected = build_result(
            scenario=scenario, params=params, result=outcome.result,
            runtime_ms=outcome.duration_ms, n_steps=outcome.n_steps,
            warnings=list(outcome.warnings),
        )
        response = client.post(
            f"/api/v1/simulations/{scenario.id}", json={"params": params}
        )
        assert response.status_code == 200, response.text
        payload = response.json()

        max_abs = max_rel = 0.0
        n_values = 0

        for key, values in payload["series"].items():
            a, r, n = worst(values, expected.series[key])
            max_abs, max_rel, n_values = max(max_abs, a), max(max_rel, r), n_values + n

        api_metrics = {m["key"]: m["value"] for m in payload["metrics"]}
        engine_metrics = {m.key: m.value for m in expected.metrics}
        numeric_keys = [
            k for k in engine_metrics
            if isinstance(engine_metrics[k], (int, float)) or engine_metrics[k] is None
        ]
        a, r, n = worst(
            [api_metrics[k] for k in numeric_keys],
            [engine_metrics[k] for k in numeric_keys],
        )
        max_abs, max_rel, n_values = max(max_abs, a), max(max_rel, r), n_values + n

        api_tables = {t["id"]: t for t in payload["tables"]}
        for name, frame in result_tables(scenario.result_kind, outcome.result).items():
            for column in frame.columns:
                col = frame[column].to_numpy()
                if not np.issubdtype(col.dtype, np.number):
                    continue
                a, r, n = worst(
                    [row.get(str(column)) for row in api_tables[name]["rows"]], col
                )
                max_abs = max(max_abs, a)
                max_rel = max(max_rel, r)
                n_values += n

        rows.append({
            "scenario": scenario.id,
            "result_kind": scenario.result_kind,
            "values_compared": n_values,
            "n_series": len(payload["series"]),
            "n_metrics": len(payload["metrics"]),
            "n_tables": len(payload["tables"]),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "tolerance": RTOL,
            "pass": max_rel <= RTOL,
            "runtime_ms": payload["runtime_ms"],
            "n_steps": payload["n_steps"],
        })

print(json.dumps(rows, indent=2))
print()
header = f"{'scenario':30}{'values':>9}{'max_abs':>13}{'max_rel':>13}  result"
print(header)
print("-" * len(header))
for row in rows:
    print(
        f"{row['scenario']:30}{row['values_compared']:>9,}"
        f"{row['max_abs_diff']:>13.3e}{row['max_rel_diff']:>13.3e}"
        f"  {'PASS' if row['pass'] else 'FAIL'}"
    )
print()
print(f"total values compared: {sum(r['values_compared'] for r in rows):,}")
print(f"scenarios passing    : {sum(r['pass'] for r in rows)} / {len(rows)}")

db.reset_engine()
try:
    os.unlink(db_path)
except OSError:
    pass
