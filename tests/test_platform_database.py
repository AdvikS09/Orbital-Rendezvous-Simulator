"""Tests for the database layer: migrations, health, and secret handling."""

from sqlalchemy import inspect, select

from pilot import db
from pilot.models import metadata, schema_migrations


def test_migrations_create_every_table(pilot_env):
    engine = db.get_engine(pilot_env)
    with engine.connect() as connection:
        tables = set(inspect(connection).get_table_names())
    expected = {table.name for table in metadata.tables.values()}
    assert expected.issubset(tables), f"missing: {sorted(expected - tables)}"


def test_migrations_are_idempotent(pilot_env):
    engine = db.get_engine(pilot_env)
    # The fixture already migrated; a second run must be a no-op.
    assert db.run_migrations(engine) == []
    assert db.run_migrations(engine) == []
    with engine.connect() as connection:
        rows = connection.execute(select(schema_migrations)).fetchall()
    assert len(rows) == len(db.MIGRATIONS)


def test_schema_version_reports_latest(pilot_env):
    assert db.schema_version(db.get_engine(pilot_env)) == max(
        version for version, _, _ in db.MIGRATIONS)


def test_migration_versions_are_unique_and_ordered():
    versions = [version for version, _, _ in db.MIGRATIONS]
    assert versions == sorted(versions), "migrations must be in ascending order"
    assert len(versions) == len(set(versions)), "duplicate migration version"


def test_healthcheck_reports_ok(pilot_env):
    health = db.healthcheck(db.get_engine(pilot_env))
    assert health["ok"] is True
    assert health["error"] is None
    assert health["missing_tables"] == []


def test_healthcheck_does_not_raise_on_a_bad_url(monkeypatch):
    from pilot import config

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@nonexistent.invalid/db")
    db.reset_engine()
    config.get_settings(refresh=True)
    try:
        health = db.healthcheck()
        assert health["ok"] is False
        assert health["error"]
        # The failure message must not leak the password.
        assert "hunter2" not in str(health)
    finally:
        db.reset_engine()
        config.get_settings(refresh=True)


def test_safe_database_label_hides_the_password(monkeypatch):
    from pilot import config

    monkeypatch.setenv("DATABASE_URL", "postgresql://admin:sup3rs3cret@db.example.com:5432/pilot")
    settings = config.get_settings(refresh=True)
    label = db.safe_database_label(settings)
    assert "sup3rs3cret" not in label
    assert "admin" not in label
    assert "db.example.com" in label
    config.get_settings(refresh=True)


def test_sqlite_fallback_is_used_when_database_url_is_unset(monkeypatch):
    from pilot import config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = config.get_settings(refresh=True)
    assert settings.database_url.startswith("sqlite:///")
    assert settings.database_url_is_default is True
    config.get_settings(refresh=True)


def test_production_without_database_url_warns(monkeypatch):
    from pilot import config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PILOT_ENV", "production")
    settings = config.get_settings(refresh=True)
    warnings = " ".join(settings.deployment_warnings())
    assert "DATABASE_URL" in warnings
    assert "SQLite" in warnings
    config.get_settings(refresh=True)


def test_production_without_a_fixed_salt_warns(monkeypatch):
    from pilot import config

    monkeypatch.setenv("PILOT_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///ignored.db")
    monkeypatch.delenv("PILOT_PARTICIPANT_SALT", raising=False)
    settings = config.get_settings(refresh=True)
    warnings = " ".join(settings.deployment_warnings())
    assert "PILOT_PARTICIPANT_SALT" in warnings
    config.get_settings(refresh=True)


def test_utcnow_is_naive_utc():
    now = db.utcnow()
    assert now.tzinfo is None
