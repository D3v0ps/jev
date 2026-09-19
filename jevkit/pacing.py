"""A token bucket for the account's published ceilings.

Jev allows 250,000 tokens per second and 1,200 requests per minute. A tight loop
or a wide fan-out will hit one of those before it hits anything else, and the
answer to a 429 is to not send it. Time is injectable so the limiter is testable
without sleeping.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .limits import REQUESTS_PER_MINUTE, TOKENS_PER_SECOND


class RateLimiter:
    """Two buckets, requests and tokens, refilled continuously.

    Not thread-safe: use one limiter per event loop or per worker thread.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        tokens_per_second: int = TOKENS_PER_SECOND,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0 or tokens_per_second <= 0:
            raise ValueError("both limits must be positive")
        self.request_rate = requests_per_minute / 60.0
        self.token_rate = float(tokens_per_second)
        self._requests = self.request_rate
        self._tokens = self.token_rate
        self._now = monotonic
        self._last = monotonic()

    def _refill(self) -> None:
        now = self._now()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._requests = min(self.request_rate, self._requests + elapsed * self.request_rate)
        self._tokens = min(self.token_rate, self._tokens + elapsed * self.token_rate)

    def delay_for(self, tokens: int) -> float:
        """Seconds to wait before a request of `tokens` fits both buckets. 0.0 when it fits now."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        self._refill()
        need_request = max(0.0, 1 - self._requests) / self.request_rate
        need_tokens = max(0.0, min(tokens, self.token_rate) - self._tokens) / self.token_rate
        return max(need_request, need_tokens)

    def consume(self, tokens: int) -> None:
        """Charge one request of `tokens` to both buckets. Call after delay_for returns 0."""
        self._refill()
        self._requests -= 1
        self._tokens -= tokens

    def acquire(self, tokens: int, sleep: Callable[[float], None] = time.sleep) -> float:
        """Block until a request of `tokens` fits, charge it, and return the seconds waited."""
        waited = 0.0
        while (delay := self.delay_for(tokens)) > 0:
            sleep(delay)
            waited += delay
        self.consume(tokens)
        return waited

    async def acquire_async(self, tokens: int) -> float:
        """Await until a request of `tokens` fits, charge it, and return the seconds waited."""
        waited = 0.0
        while (delay := self.delay_for(tokens)) > 0:
            await asyncio.sleep(delay)
            waited += delay
        self.consume(tokens)
        return waited
