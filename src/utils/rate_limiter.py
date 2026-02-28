"""Async per-provider rate limiting and retry helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import random
import time
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class RateLimitConfig:
    """Simple RPM-based rate limiter configuration."""

    rpm: int


class AsyncRateLimiter:
    """Token bucket style limiter using a 60-second rolling window."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, key: str, rpm: int) -> None:
        """Wait until a call token is available for key."""
        if rpm <= 0:
            return

        queue = self._queues.setdefault(key, deque())
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            while True:
                now = time.monotonic()
                while queue and now - queue[0] > 60.0:
                    queue.popleft()

                if len(queue) < rpm:
                    queue.append(now)
                    return

                wait_seconds = max(0.01, 60.0 - (now - queue[0]))
                await asyncio.sleep(wait_seconds)


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Retry awaitable function using exponential backoff with jitter."""
    attempt = 0
    while True:
        try:
            return await fn()
        except retryable_exceptions:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_seconds = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 0.25 * sleep_seconds)
            await asyncio.sleep(sleep_seconds + jitter)