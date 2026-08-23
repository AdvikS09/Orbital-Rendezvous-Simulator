"""Administrator authentication.

Credentials come only from the environment -- there is no default password, no
credential file and nothing checked into the repository. If nothing is
configured, the admin area refuses to unlock at all rather than falling back to
something guessable.

Two credential forms are supported:

``PILOT_ADMIN_PASSWORD_HASH``
    Preferred. A PBKDF2-HMAC-SHA256 hash in the format
    ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``, produced by
    ``scripts/hash_password.py``. The plaintext never exists in the deployment
    environment.

``PILOT_ADMIN_PASSWORD``
    Plain text in an environment secret. Simpler, and acceptable for a small
    pilot, but the hash form is better and the admin area says so.

Both comparisons are constant-time. Failed attempts are counted per process and
lock the form after a threshold, which makes online guessing impractical
without needing any storage.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass

from .config import Settings, get_settings

logger = logging.getLogger("pilot.auth")

PBKDF2_ITERATIONS = 240_000
_HASH_PREFIX = "pbkdf2_sha256"

#: Lock-out duration after too many failures, in seconds.
LOCKOUT_SECONDS = 300.0


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS,
                  salt: bytes | None = None) -> str:
    """Produce a storable password hash."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_HASH_PREFIX}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Check a password against an encoded hash, in constant time."""
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$")
        if algorithm != _HASH_PREFIX:
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(actual, expected)


@dataclass
class AuthResult:
    ok: bool
    message: str = ""
    locked_until: float = 0.0


class AdminAuthenticator:
    """Credential check with per-process failure counting and lock-out."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._failures = 0
        self._locked_until = 0.0
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self.settings.admin_configured

    def authenticate(self, username: str, password: str,
                     now: float | None = None) -> AuthResult:
        now = time.monotonic() if now is None else now

        if not self.configured:
            return AuthResult(False, (
                "No administrator credentials are configured on this deployment. "
                "Set PILOT_ADMIN_USERNAME and PILOT_ADMIN_PASSWORD_HASH in the "
                "environment, then restart."))

        with self._lock:
            if now < self._locked_until:
                remaining = self._locked_until - now
                return AuthResult(False,
                                  f"Too many failed attempts. Try again in "
                                  f"{remaining / 60:.0f} minutes.",
                                  locked_until=self._locked_until)

        username_ok = hmac.compare_digest(
            (username or "").strip(), self.settings.admin_username)

        if self.settings.admin_password_hash:
            password_ok = verify_password(password or "", self.settings.admin_password_hash)
        else:
            password_ok = hmac.compare_digest(password or "", self.settings.admin_password)

        # Evaluate both before branching so the response time does not reveal
        # which half was wrong.
        if username_ok and password_ok:
            with self._lock:
                self._failures = 0
            logger.info("administrator authenticated")
            return AuthResult(True, "Signed in.")

        with self._lock:
            self._failures += 1
            remaining_attempts = self.settings.admin_max_attempts - self._failures
            if remaining_attempts <= 0:
                self._locked_until = now + LOCKOUT_SECONDS
                self._failures = 0
                logger.warning("administrator sign-in locked out after repeated failures")
                return AuthResult(False, (
                    "Too many failed attempts. The sign-in form is locked for "
                    f"{LOCKOUT_SECONDS / 60:.0f} minutes."),
                    locked_until=self._locked_until)

        logger.warning("failed administrator sign-in attempt")
        return AuthResult(False,
                          f"Incorrect username or password. "
                          f"{remaining_attempts} attempt(s) remaining.")

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._locked_until = 0.0


_authenticator: AdminAuthenticator | None = None
_authenticator_lock = threading.Lock()


def get_authenticator(settings: Settings | None = None) -> AdminAuthenticator:
    global _authenticator
    with _authenticator_lock:
        if _authenticator is None:
            _authenticator = AdminAuthenticator(settings)
        return _authenticator


def reset_authenticator() -> None:
    """Drop the process-wide authenticator. Used by tests."""
    global _authenticator
    with _authenticator_lock:
        _authenticator = None


__all__ = ["hash_password", "verify_password", "AdminAuthenticator", "AuthResult",
           "get_authenticator", "reset_authenticator", "PBKDF2_ITERATIONS"]
