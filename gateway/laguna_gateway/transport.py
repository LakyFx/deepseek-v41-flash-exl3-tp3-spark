"""Retry only failures that prove no request reached the model server."""

import asyncio
import time

import httpx


class ConnectRetryTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        retries: int = 2,
        backoff: float = 0.25,
        recovery_seconds: float = 0,
    ) -> None:
        self.inner = inner
        self.retries = retries
        self.backoff = backoff
        self.recovery_seconds = recovery_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        recover = (
            self.recovery_seconds > 0
            and request.method == "POST"
            and request.url.path
            in {"/v1/chat/completions", "/v1/responses", "/tokenize", "/apply-template"}
        )
        deadline = time.monotonic() + self.recovery_seconds
        attempt = 0
        while True:
            try:
                return await self.inner.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                delay = self.backoff * 2 ** min(attempt, 3)
                if recover:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    delay = min(delay, remaining)
                elif attempt >= self.retries:
                    raise
                # Cancellation interrupts this wait immediately. Health/metrics
                # retain fast failure and ambiguous reads/writes never retry.
                await asyncio.sleep(delay)
                attempt += 1

    async def aclose(self) -> None:
        await self.inner.aclose()
