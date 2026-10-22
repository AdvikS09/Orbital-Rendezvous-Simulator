"""Database engine, connection handling and versioned migrations.

Works against SQLite (zero-setup local development) and PostgreSQL (hosted
pilots) through the same SQLAlchemy Core API, so nothing above this module has
to care which one is in use.

Migrations
----------
A deliberately small runner rather than Alembic: each migration is a numbered
entry in :data:`MIGRATIONS` with a callable that receives a live connection.
Applied versions are recorded in ``schema_migrations``, so
:func:`run_migrations` is idempotent and safe to call on every application
start. Adding a column later means appending a new numbered entry -- never
editing an existing one.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from .config import Settings, get_settings
from .models import metadata, schema_migrations

logger = logging.getLogger("pilot.db")

_engine: Engine | None = None
_engine_url: str | None = None
_lock = threading.Lock()


# Engine
def _create_engine(settings: Settings) -> Engine:
    url = make_url(settings.database_url)
    kwargs: dict = {"future": True, "pool_pre_ping": True}

    if url.get_backend_name() == "sqlite":
        # Streamlit serves each session on its own thread; SQLite objects are
        # otherwise pinned to their creating thread.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
        kwargs["pool_recycle"] = 900

    engine = create_engine(url, **kwargs)

    if url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            # WAL keeps concurrent readers working while a write is in flight.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


def get_engine(settings: Settings | None = None, refresh: bool = False) -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine, _engine_url
    settings = settings or get_settings()
    with _lock:
        if refresh and _engine is not None:
            _engine.dispose()
            _engine = None
        if _engine is None or _engine_url != settings.database_url:
            _engine = _create_engine(settings)
            _engine_url = settings.database_url
            logger.info("database engine created (%s)", safe_database_label(settings))
        return _engine


def reset_engine() -> None:
    """Dispose the cached engine. Used by tests and by config changes."""
    global _engine, _engine_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _engine_url = None


def safe_database_label(settings: Settings | None = None) -> str:
    """A description of the database with any password removed.

    Never log ``settings.database_url`` directly -- a Postgres URL contains the
    password.
    """
    settings = settings or get_settings()
    try:
        url = make_url(settings.database_url)
    except Exception:  # noqa: BLE001 - a malformed URL must not leak either
        return "<unparseable database url>"
    backend = url.get_backend_name()
    if backend == "sqlite":
        return f"sqlite ({url.database or ':memory:'})"
    host = url.host or "?"
    database = url.database or "?"
    return f"{backend} ({host}/{database})"


# Migrations
def _migration_001_initial(connection) -> None:
    """Create the full initial schema."""
    metadata.create_all(connection, checkfirst=True)


#: ``(version, name, callable)``. Append only; never edit an applied entry.
MIGRATIONS: list[tuple[int, str, callable]] = [
    (1, "initial_schema", _migration_001_initial),
]


def _ensure_migration_table(connection) -> None:
    schema_migrations.create(connection, checkfirst=True)


def applied_versions(connection) -> set[int]:
    _ensure_migration_table(connection)
    rows = connection.execute(select(schema_migrations.c.version)).fetchall()
    return {row[0] for row in rows}


def run_migrations(engine: Engine | None = None) -> list[int]:
    """Apply any outstanding migrations. Idempotent.

    Returns the versions applied by this call (empty when already up to date).
    """
    engine = engine or get_engine()
    applied: list[int] = []
    with engine.begin() as connection:
        already = applied_versions(connection)
        for version, name, migrate in MIGRATIONS:
            if version in already:
                continue
            logger.info("applying migration %03d %s", version, name)
            migrate(connection)
            connection.execute(schema_migrations.insert().values(
                version=version, name=name,
                applied_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            applied.append(version)
    return applied


def schema_version(engine: Engine | None = None) -> int:
    """Highest applied migration version, or 0 if the schema is empty."""
    engine = engine or get_engine()
    try:
        with engine.connect() as connection:
            versions = applied_versions(connection)
        return max(versions) if versions else 0
    except SQLAlchemyError:
        return 0


def healthcheck(engine: Engine | None = None) -> dict:
    """Report whether the database is reachable and migrated.

    Returns a plain dict rather than raising, so callers (the app, the admin
    area, ``scripts/init_db.py``) can render the problem instead of crashing.
    """
    settings = get_settings()
    try:
        engine = engine or get_engine(settings)
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
        expected = {table.name for table in metadata.tables.values()}
        return {
            "ok": expected.issubset(tables),
            "database": safe_database_label(settings),
            "schema_version": schema_version(engine),
            "missing_tables": sorted(expected - tables),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - health checks must not raise
        return {
            "ok": False,
            "database": safe_database_label(settings),
            "schema_version": 0,
            "missing_tables": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def utcnow() -> datetime:
    """Naive UTC timestamp.

    Stored naive because SQLite has no timezone-aware type; everything written
    here is UTC by construction, and nothing reads local time.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


__all__ = [
    "get_engine", "reset_engine", "run_migrations", "schema_version",
    "healthcheck", "safe_database_label", "utcnow", "MIGRATIONS",
]
