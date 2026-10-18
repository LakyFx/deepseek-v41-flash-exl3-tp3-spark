"""Cancellation-safe fair priority admission gate."""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


class QueueFullError(RuntimeError):
    """Raised when the bounded HTTP waiting room is full."""


class AdmissionWeightError(ValueError):
    """Raised when one request cannot fit inside the admission token budget."""


@dataclass(order=True, slots=True)
class _Waiter:
    priority: int
    sequence: int
    enqueued_at: float = field(compare=False)
    future: asyncio.Future[None] = field(compare=False)
    weight: int = field(default=1, compare=False)
    granted: bool = field(default=False, compare=False)
    cancelled: bool = field(default=False, compare=False)


class FairPriorityGate:
    """Bound concurrent upstream work while keeping excess HTTP calls waiting.

    Lower numeric priority is served first; equal priorities are strict FIFO.
    Cancellation removes queued callers and cannot leak an active slot.
    """

    def __init__(
        self,
        capacity: int,
        max_queue: int,
        max_weight: int | None = None,
        on_change: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.max_queue = max(1, int(max_queue))
        self.max_weight = max(
            self.capacity,
            int(max_weight) if max_weight is not None else self.capacity,
        )
        self._active = 0
        self._active_weight = 0
        self._sequence = 0
        self._waiters: list[_Waiter] = []
        self._lock = asyncio.Lock()
        self._on_change = on_change

    @property
    def active(self) -> int:
        return self._active

    @property
    def active_weight(self) -> int:
        return self._active_weight

    @property
    def waiting(self) -> int:
        return sum(1 for waiter in self._waiters if not waiter.cancelled)

    def _can_grant_locked(self, weight: int) -> bool:
        return (
            self._active < self.capacity
            and self._active_weight + weight <= self.max_weight
        )

    def _notify_locked(self) -> None:
        if self._on_change is not None:
            self._on_change(self._active, self.waiting, self._active_weight)

    def _promote_locked(self) -> None:
        while self._active < self.capacity and self._waiters:
            waiter = self._waiters[0]
            if waiter.cancelled or waiter.future.cancelled():
                heapq.heappop(self._waiters)
                continue
            # Preserve strict priority/FIFO semantics. A lower-priority light
            # request must not bypass an older or higher-priority heavy one.
            if not self._can_grant_locked(waiter.weight):
                break
            heapq.heappop(self._waiters)
            waiter.granted = True
            self._active += 1
            self._active_weight += waiter.weight
            if not waiter.future.done():
                waiter.future.set_result(None)
        self._notify_locked()

    async def acquire(self, priority: int = 0, weight: int = 1) -> float:
        resolved_weight = int(weight)
        if resolved_weight < 1:
            raise AdmissionWeightError("admission weight must be positive")
        if resolved_weight > self.max_weight:
            raise AdmissionWeightError(
                f"request needs {resolved_weight} admission tokens; "
                f"gateway limit is {self.max_weight}"
            )
        enqueued_at = time.monotonic()
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._can_grant_locked(resolved_weight) and not self._waiters:
                self._active += 1
                self._active_weight += resolved_weight
                self._notify_locked()
                return 0.0
            if self.waiting >= self.max_queue:
                raise QueueFullError("Laguna gateway waiting room is full")
            self._sequence += 1
            waiter = _Waiter(
                priority=int(priority),
                sequence=self._sequence,
                enqueued_at=enqueued_at,
                future=loop.create_future(),
                weight=resolved_weight,
            )
            heapq.heappush(self._waiters, waiter)
            self._notify_locked()

        try:
            await waiter.future
        except asyncio.CancelledError:
            async with self._lock:
                waiter.cancelled = True
                if waiter.granted:
                    self._active = max(0, self._active - 1)
                    self._active_weight = max(0, self._active_weight - waiter.weight)
                self._promote_locked()
            raise
        return max(0.0, time.monotonic() - enqueued_at)

    async def release(self, weight: int = 1) -> None:
        resolved_weight = int(weight)
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("admission gate released without active slot")
            if resolved_weight < 1 or resolved_weight > self._active_weight:
                raise RuntimeError("invalid admission weight release")
            self._active -= 1
            self._active_weight -= resolved_weight
            self._promote_locked()

    @asynccontextmanager
    async def slot(
        self,
        priority: int = 0,
        weight: int = 1,
    ) -> AsyncIterator[float]:
        wait_seconds = await self.acquire(priority=priority, weight=weight)
        try:
            yield wait_seconds
        finally:
            await self.release(weight=weight)
