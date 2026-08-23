"""Per-session safeguards against accidental resource exhaustion.

The threat model here is *accident*, not attack: a class of thirty students
holding down a Run button, or someone leaving a wide parameter sweep looping.
The engine is pure-Python and a single high-fidelity run can occupy a Streamlit
worker for seconds (see ``CURRENT_STATE_AUDIT.md`` §4.6), so an unbounded run
rate will make the app unusable for everyone else on the instance.

Deliberately in-process rather than in the database: the check runs on every Run
click and must not add a database round-trip, and losing the counters on restart
is harmless. Note the consequence -- with multiple app processes each keeps its
own counters, so the effective limit scales with the process count. That is
documented rather than solved, because a hosted Streamlit pilot runs a single
process.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: str = ""
    retry_after_s: float = 0.0
    runs_used: int = 0
    runs_remaining: int = 0


class SessionRateLimiter:
    """Sliding-window run limiter, keyed by session token.

    Two independent limits:

    * a short window (default 20 runs/minute) that absorbs a stuck button; and
    * a session total (default 300 runs) that bounds one participant's whole
      visit.
    """

    def __init__(self, max_per_minute: int = 20, max_per_session: int = 300,
                 window_seconds: float = 60.0):
        self.max_per_minute = max_per_minute
        self.max_per_session = max_per_session
        self.window_seconds = window_seconds
        self._recent: dict[str, deque] = defaultdict(deque)
        self._totals: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def check(self, session_token: str, now: float | None = None) -> RateLimitDecision:
        """Test whether a run may start, without consuming an allowance."""
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = self._recent[session_token]
            self._expire(recent, now)
            total = self._totals[session_token]

            if total >= self.max_per_session:
                return RateLimitDecision(
                    allowed=False, runs_used=total, runs_remaining=0,
                    reason=(f"This session has run {total} simulations, which is the "
                            f"per-session limit of {self.max_per_session}. Reload the "
                            "page to start a new session, or run the engine locally "
                            "for heavy studies."))

            if len(recent) >= self.max_per_minute:
                retry_after = max(0.0, self.window_seconds - (now - recent[0]))
                return RateLimitDecision(
                    allowed=False, runs_used=total,
                    runs_remaining=self.max_per_session - total,
                    retry_after_s=retry_after,
                    reason=(f"That is {self.max_per_minute} simulations in under a "
                            f"minute. Please wait {retry_after:.0f} s -- the pilot "
                            "server is shared with other participants."))

            return RateLimitDecision(
                allowed=True, runs_used=total,
                runs_remaining=self.max_per_session - total)

    def consume(self, session_token: str, now: float | None = None) -> RateLimitDecision:
        """Check and, if allowed, record one run against the allowance."""
        now = time.monotonic() if now is None else now
        decision = self.check(session_token, now)
        if not decision.allowed:
            return decision
        with self._lock:
            self._recent[session_token].append(now)
            self._totals[session_token] += 1
            total = self._totals[session_token]
        return RateLimitDecision(allowed=True, runs_used=total,
                                 runs_remaining=self.max_per_session - total)

    def reset(self, session_token: str | None = None) -> None:
        with self._lock:
            if session_token is None:
                self._recent.clear()
                self._totals.clear()
            else:
                self._recent.pop(session_token, None)
                self._totals.pop(session_token, None)

    def _expire(self, window: deque, now: float) -> None:
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()


_limiter: SessionRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter(max_per_minute: int | None = None,
                max_per_session: int | None = None) -> SessionRateLimiter:
    """Process-wide limiter, configured from settings on first use."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            from .config import get_settings
            settings = get_settings()
            _limiter = SessionRateLimiter(
                max_per_minute=max_per_minute or settings.max_runs_per_minute,
                max_per_session=max_per_session or settings.max_runs_per_session)
        return _limiter


def reset_limiter() -> None:
    """Drop the process-wide limiter. Used by tests."""
    global _limiter
    with _limiter_lock:
        _limiter = None


__all__ = ["SessionRateLimiter", "RateLimitDecision", "get_limiter", "reset_limiter"]
