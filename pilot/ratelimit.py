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
    def __init__(self, max_per_minute: int = 20, max_per_session: int = 300,
                 window_seconds: float = 60.0):
        self.max_per_minute = max_per_minute
        self.max_per_session = max_per_session
        self.window_seconds = window_seconds
        self._recent: dict[str, deque] = defaultdict(deque)
        self._totals: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def check(self, session_token: str, now: float | None = None) -> RateLimitDecision:
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
    global _limiter
    with _limiter_lock:
        _limiter = None


__all__ = ["SessionRateLimiter", "RateLimitDecision", "get_limiter", "reset_limiter"]
