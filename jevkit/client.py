"""A thin wrapper over the TypeSafe SDK that measures every request.

The SDK already handles auth, retries, and typed answers. `Jev` adds the three
things every pattern in this repo needs: a local size check so an oversized
request fails without a round trip, a latency and cost measurement per call, and
optional pacing against the account's published rate limits.

Nothing here reinterprets an answer. Reading answers lives in jevkit.answers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient

from .answers import Reply
from .ledger import Ledger
from .limits import check_request, estimate_tokens
from .pacing import RateLimiter

#: One (state, questions) pair: the unit both clients send.
Request = tuple[Any, Mapping[str, Any]]


class Jev:
    """Synchronous client. One per process is enough; it is safe to share.

    `check` runs the local context-budget check before sending. `limiter` paces
    requests against the account ceilings; pass one when a loop or a fan-out
    could outrun them. `ledger` accumulates tokens, dollars, and latency.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        transport: Any | None = None,
        http_client: Any | None = None,
        check: bool = True,
        limiter: RateLimiter | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.client = TypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            retry=retry,
            timeout=timeout,
            transport=transport,
            http_client=http_client,
        )
        self.check = check
        self.limiter = limiter
        self.ledger = ledger if ledger is not None else Ledger()

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Reply:
        """Evaluate `state` against `questions` in one request.

        Latency covers the whole call including any retry the SDK performed, which
        is what a caller waited for.
        """
        if not questions:
            raise ValueError("ask needs at least one question")
        if self.check:
            estimate = check_request(state, questions)
        else:
            estimate = estimate_tokens(state) + sum(estimate_tokens(q) for q in questions.values())
        if self.limiter is not None:
            self.limiter.acquire(estimate)
        started = time.perf_counter()
        response = self.client.system_one(state, questions, model=model, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        reply = Reply(response=response, latency_ms=latency_ms, questions=questions)
        self.ledger.record(reply)
        return reply

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Jev:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncJev:
    """Asynchronous client, plus bounded-concurrency fan-out over many requests."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        transport: Any | None = None,
        http_client: Any | None = None,
        check: bool = True,
        limiter: RateLimiter | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            retry=retry,
            timeout=timeout,
            transport=transport,
            http_client=http_client,
        )
        self.check = check
        self.limiter = limiter
        self.ledger = ledger if ledger is not None else Ledger()

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Reply:
        """Evaluate `state` against `questions` in one request."""
        if not questions:
            raise ValueError("ask needs at least one question")
        if self.check:
            estimate = check_request(state, questions)
        else:
            estimate = estimate_tokens(state) + sum(estimate_tokens(q) for q in questions.values())
        if self.limiter is not None:
            await self.limiter.acquire_async(estimate)
        started = time.perf_counter()
        response = await self.client.system_one(state, questions, model=model, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        reply = Reply(response=response, latency_ms=latency_ms, questions=questions)
        self.ledger.record(reply)
        return reply

    async def map(
        self,
        requests: Sequence[Request] | Iterable[Request],
        *,
        concurrency: int = 16,
        model: str | None = None,
    ) -> list[Reply]:
        """Send many requests with at most `concurrency` in flight, preserving input order.

        One failed request fails the call: a partial fan-out is rarely safe to act
        on, and the caller can catch and retry the batch it chose.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        items = list(requests)
        gate = asyncio.Semaphore(concurrency)

        async def one(state: Any, questions: Mapping[str, Any]) -> Reply:
            async with gate:
                return await self.ask(state, questions, model=model)

        return list(await asyncio.gather(*(one(state, questions) for state, questions in items)))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> AsyncJev:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def measured(call: Callable[[], Any]) -> tuple[Any, float]:
    """Run `call` and return its result with the milliseconds it took."""
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1000


async def measured_async(call: Callable[[], Awaitable[Any]]) -> tuple[Any, float]:
    """Await `call` and return its result with the milliseconds it took."""
    started = time.perf_counter()
    result = await call()
    return result, (time.perf_counter() - started) * 1000
