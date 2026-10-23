from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture
def api_env(monkeypatch):
    from pilot import auth, config, db, ratelimit

    handle, path = tempfile.mkstemp(suffix=".db", prefix="api_test_")
    os.close(handle)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path.replace(os.sep, '/')}")
    monkeypatch.setenv("PILOT_ENV", "test")
    monkeypatch.setenv("PILOT_PARTICIPANT_SALT", "api-test-salt-not-a-real-secret")
    monkeypatch.setenv("PILOT_APP_VERSION", "test")
    for name in (
        "PILOT_LOG_RAW_PARAMETERS",
        "PILOT_ADMIN_USERNAME",
        "PILOT_ADMIN_PASSWORD",
        "PILOT_ADMIN_PASSWORD_HASH",
        "PILOT_REQUIRE_KNOWN_CODE",
    ):
        monkeypatch.delenv(name, raising=False)

    db.reset_engine()
    ratelimit.reset_limiter()
    auth.reset_authenticator()
    settings = config.get_settings(refresh=True)
    db.run_migrations(db.get_engine(settings))

    yield settings

    db.reset_engine()
    ratelimit.reset_limiter()
    auth.reset_authenticator()
    config.get_settings(refresh=True)
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def client(api_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_client(client):
    response = client.post(
        "/api/v1/sessions",
        json={
            "pilot_code": "TESTORG",
            "participant_code": "TESTUSER1",
            "user_type": "student",
            "consent_given": True,
        },
    )
    assert response.status_code == 201, response.text
    token = response.json()["session_token"]
    client.headers.update({"X-Session-Token": token})
    return client


@pytest.fixture
def admin_auth(monkeypatch, api_env):
    from pilot import auth, config

    username, password = "coordinator", "test-password-1234"
    monkeypatch.setenv("PILOT_ADMIN_USERNAME", username)
    monkeypatch.setenv(
        "PILOT_ADMIN_PASSWORD_HASH", auth.hash_password(password, iterations=1000)
    )
    config.get_settings(refresh=True)
    auth.reset_authenticator()
    return (username, password)
