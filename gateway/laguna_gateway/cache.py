"""Small bounded TTL caches used only for transient inference metadata."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """Single-process LRU+TTL cache.

    Gateway deployment uses one event loop and one worker process, so cache
    operations intentionally remain synchronous and tiny.  Raw model reasoning
    never touches disk.
    """

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry.value

    def put(self, key: K, value: V) -> None:
        now = time.monotonic()
        self._entries[key] = _Entry(
            value=value,
            expires_at=now + self.ttl_seconds,
        )
        self._entries.move_to_end(key)
        self.prune(now)

    def prune(self, now: float | None = None) -> None:
        cutoff = time.monotonic() if now is None else now
        expired = [
            key for key, entry in self._entries.items() if entry.expires_at <= cutoff
        ]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        self.prune()
        return len(self._entries)
