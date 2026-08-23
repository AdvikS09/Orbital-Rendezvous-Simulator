"""Environment-driven configuration for the pilot platform.

Every secret and every deployment-specific value is read from the environment
(or from Streamlit secrets, which Streamlit itself surfaces as environment
variables when configured that way). Nothing sensitive is ever written to the
repository -- see ``.env.example`` for the placeholder set.

The package is named ``pilot`` rather than ``platform`` deliberately: a
top-level package called ``platform`` would shadow the standard-library module
of that name and break NumPy, Streamlit and pytest on import.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "pilot_data" / "pilot.db"

#: Marker written onto every row created while ``PILOT_ENV`` is not "production".
#: Used to keep development and demonstration records out of real pilot reporting.
DEV_ENVIRONMENTS = ("development", "demo", "test")


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, trimmed. Empty string means "unset"."""
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _default_database_url() -> str:
    """Local-development fallback: a SQLite file beside the project.

    Deployments must set ``DATABASE_URL`` to a managed Postgres instance. The
    SQLite fallback exists so a fresh clone runs with zero setup; it is not
    suitable for a hosted pilot, because most hosts (including Streamlit
    Community Cloud) give the app an ephemeral filesystem that is wiped on every
    restart or redeploy.
    """
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


@dataclass(frozen=True)
class Settings:
    """Resolved platform configuration."""

    # --- environment ---
    environment: str = "development"
    app_version: str = "1.0.0"

    # --- database ---
    database_url: str = ""
    database_url_is_default: bool = True

    # --- privacy / identity ---
    participant_salt: str = ""
    participant_salt_is_ephemeral: bool = True
    log_raw_parameters: bool = False
    retention_days: int = 730

    # --- admin ---
    admin_username: str = ""
    admin_password: str = ""
    admin_password_hash: str = ""
    admin_max_attempts: int = 5

    # --- safeguards ---
    max_runs_per_session: int = 300
    max_runs_per_minute: int = 20
    max_simulation_steps: int = 40_000
    max_estimated_seconds: float = 20.0

    # --- pilot behaviour ---
    require_known_pilot_code: bool = False
    open_pilot_codes: tuple = field(default_factory=tuple)

    # ---------------------------------------------------------------- helpers
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_demo_data(self) -> bool:
        """True when records created now must be marked as non-pilot data."""
        return self.environment in DEV_ENVIRONMENTS

    @property
    def admin_configured(self) -> bool:
        return bool(self.admin_username) and bool(
            self.admin_password or self.admin_password_hash)

    def deployment_warnings(self) -> list[str]:
        """Configuration problems that matter for a real deployment.

        Surfaced in the administrator area rather than raised, so that a
        misconfigured deployment is visible instead of silently degraded.
        """
        warnings: list[str] = []
        if self.is_production and self.database_url_is_default:
            warnings.append(
                "DATABASE_URL is not set, so the app is using the local SQLite "
                "fallback. On most hosts this file is erased on every restart "
                "and all pilot data will be lost. Set DATABASE_URL to a managed "
                "Postgres database.")
        if self.is_production and self.participant_salt_is_ephemeral:
            warnings.append(
                "PILOT_PARTICIPANT_SALT is not set. A random salt was generated "
                "for this process, so returning participants will NOT be "
                "recognised after a restart and return-rate figures will be "
                "wrong. Set a fixed secret salt.")
        if self.is_production and not self.admin_configured:
            warnings.append(
                "No administrator credentials are configured. Set "
                "PILOT_ADMIN_USERNAME and PILOT_ADMIN_PASSWORD_HASH.")
        if self.admin_password and not self.admin_password_hash:
            warnings.append(
                "PILOT_ADMIN_PASSWORD is set as plain text. Prefer "
                "PILOT_ADMIN_PASSWORD_HASH (generate it with "
                "`python scripts/hash_password.py`).")
        if self.log_raw_parameters:
            warnings.append(
                "PILOT_LOG_RAW_PARAMETERS is enabled. Raw user-supplied orbital "
                "parameters are being stored. Only do this with participants' "
                "informed consent.")
        return warnings


def load_settings() -> Settings:
    """Build :class:`Settings` from the current environment."""
    database_url = _env("DATABASE_URL")
    salt = _env("PILOT_PARTICIPANT_SALT")

    return Settings(
        environment=_env("PILOT_ENV", "development").lower(),
        app_version=_env("PILOT_APP_VERSION", "1.0.0"),
        database_url=database_url or _default_database_url(),
        database_url_is_default=not database_url,
        # An ephemeral salt keeps a fresh clone working without any setup, at the
        # cost of not recognising returning participants across restarts.
        participant_salt=salt or secrets.token_hex(32),
        participant_salt_is_ephemeral=not salt,
        log_raw_parameters=_env_bool("PILOT_LOG_RAW_PARAMETERS", False),
        retention_days=_env_int("PILOT_RETENTION_DAYS", 730),
        admin_username=_env("PILOT_ADMIN_USERNAME"),
        admin_password=_env("PILOT_ADMIN_PASSWORD"),
        admin_password_hash=_env("PILOT_ADMIN_PASSWORD_HASH"),
        admin_max_attempts=_env_int("PILOT_ADMIN_MAX_ATTEMPTS", 5),
        max_runs_per_session=_env_int("PILOT_MAX_RUNS_PER_SESSION", 300),
        max_runs_per_minute=_env_int("PILOT_MAX_RUNS_PER_MINUTE", 20),
        max_simulation_steps=_env_int("PILOT_MAX_SIMULATION_STEPS", 40_000),
        max_estimated_seconds=_env_float("PILOT_MAX_ESTIMATED_SECONDS", 20.0),
        require_known_pilot_code=_env_bool("PILOT_REQUIRE_KNOWN_CODE", False),
        open_pilot_codes=tuple(
            c.strip().upper() for c in _env("PILOT_OPEN_CODES").split(",") if c.strip()),
    )


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide cached settings."""
    global _settings
    if _settings is None or refresh:
        _settings = load_settings()
    return _settings


__all__ = ["Settings", "get_settings", "load_settings", "PROJECT_ROOT", "DEV_ENVIRONMENTS"]
