"""Shared pytest fixtures.

The platform tests must never touch a real deployment database, so a fixture
points every one of them at a temporary SQLite file and resets the module-level
caches (engine, rate limiter, authenticator, settings) around each test.
"""

import os
import tempfile

import pytest


@pytest.fixture
def pilot_env(monkeypatch):
    """Isolated platform environment backed by a temporary SQLite database.

    Yields the resolved ``Settings``. ``PILOT_ENV`` is set to ``test``, so every
    row written carries ``is_demo = 1`` -- which is itself part of what the
    tests check.
    """
    from pilot import auth, config, db, ratelimit

    handle, path = tempfile.mkstemp(suffix=".db", prefix="pilot_test_")
    os.close(handle)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path.replace(os.sep, '/')}")
    monkeypatch.setenv("PILOT_ENV", "test")
    monkeypatch.setenv("PILOT_PARTICIPANT_SALT", "test-salt-not-a-real-secret")
    monkeypatch.setenv("PILOT_APP_VERSION", "test")
    monkeypatch.delenv("PILOT_LOG_RAW_PARAMETERS", raising=False)
    monkeypatch.delenv("PILOT_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("PILOT_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("PILOT_ADMIN_PASSWORD_HASH", raising=False)

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
        # Windows may still hold the file; a stale temp file is harmless.
        pass


@pytest.fixture
def participant(pilot_env):
    """A registered anonymous participant with an open session."""
    from pilot import analytics

    return analytics.start_session(
        pilot_code="TESTORG", participant_code="TESTUSER1",
        user_type="student", consent_version="1.0", settings=pilot_env)
