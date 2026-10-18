import asyncio

import httpx
import pytest

from laguna_gateway.stream_validation import (
    ChatStreamValidator,
    ResponsesStreamValidator,
)
from laguna_gateway.transport import ConnectRetryTransport


@pytest.mark.parametrize("kind", [httpx.ConnectError, httpx.ConnectTimeout])
def test_connection_failure_can_retry_before_submission(kind):
    calls = []

    async def handler(request):
        calls.append(request.content)
        if len(calls) < 3:
            raise kind("not connected", request=request)
        return httpx.Response(200, json={"ok": True})

    async def run():
        transport = ConnectRetryTransport(httpx.MockTransport(handler), backoff=0)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.post(
                "http://model/v1/chat/completions", json={"request_id": "same-request"}
            )
        assert response.status_code == 200

    asyncio.run(run())
    assert len(calls) == 3 and len(set(calls)) == 1


@pytest.mark.parametrize(
    "kind",
    [httpx.ReadTimeout, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError],
)
def test_ambiguous_transport_failure_is_never_replayed(kind):
    calls = []

    async def handler(request):
        calls.append(request)
        raise kind("submission may have reached the model", request=request)

    async def run():
        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(httpx.MockTransport(handler), backoff=0)
        ) as client:
            with pytest.raises(kind):
                await client.post("http://model/v1/chat/completions", json={})

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/tokenize"])
def test_model_startup_wait_can_outlast_short_connect_retry_count(path):
    bodies = []

    async def handler(request):
        bodies.append(request.content)
        if len(bodies) < 6:
            raise httpx.ConnectError("model still loading", request=request)
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(
                httpx.MockTransport(handler), backoff=0, recovery_seconds=1
            )
        ) as client:
            response = await client.post(
                "http://model" + path, json={"request_id": "unchanged"}
            )
            assert response.status_code == 200

    asyncio.run(run())
    assert len(bodies) == 6
    assert len(set(bodies)) == 1


def test_model_startup_recovery_has_deadline_and_keeps_health_fast(monkeypatch):
    clock = [0.0]
    calls = []

    async def handler(request):
        calls.append(request)
        clock[0] += 0.02
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr("laguna_gateway.transport.time.monotonic", lambda: clock[0])

    async def run():
        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(
                httpx.MockTransport(handler), backoff=0, recovery_seconds=0.05
            )
        ) as client:
            with pytest.raises(httpx.ConnectError):
                await client.post("http://model/v1/responses", json={})
            assert len(calls) == 3
        calls.clear()
        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(
                httpx.MockTransport(handler), backoff=0, recovery_seconds=900
            )
        ) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("http://model/health")
            assert len(calls) == 3

    asyncio.run(run())


def test_model_startup_recovery_is_cancelled_without_another_attempt():
    async def run():
        attempted = asyncio.Event()
        calls = []

        async def handler(request):
            calls.append(request)
            attempted.set()
            raise httpx.ConnectError("loading", request=request)

        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(
                httpx.MockTransport(handler), backoff=10, recovery_seconds=900
            )
        ) as client:
            task = asyncio.create_task(
                client.post("http://model/v1/responses", json={})
            )
            await attempted.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["read", "write", "http503"])
def test_long_startup_grace_never_replays_ambiguous_or_accepted_requests(failure):
    calls = []

    async def handler(request):
        calls.append(request)
        if failure == "http503":
            return httpx.Response(503)
        kind = httpx.ReadTimeout if failure == "read" else httpx.WriteError
        raise kind("request may have been submitted", request=request)

    async def run():
        async with httpx.AsyncClient(
            transport=ConnectRetryTransport(
                httpx.MockTransport(handler), backoff=0, recovery_seconds=900
            )
        ) as client:
            if failure == "http503":
                assert (
                    await client.post("http://model/v1/responses", json={})
                ).status_code == 503
            else:
                with pytest.raises(httpx.RequestError):
                    await client.post("http://model/v1/responses", json={})

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("width", [1, 3, 17, 10000])
def test_stream_validator_handles_arbitrary_network_fragmentation(width):
    raw = b': heartbeat\r\n\r\ndata: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\r\n\r\ndata: {"choices":[{"finish_reason":"length"}]}\r\n\r\ndata: {"choices":[],"usage":{"completion_tokens":600}}\r\n\r\ndata: [DONE]\r\n\r\n'
    validator = ChatStreamValidator()
    for start in range(0, len(raw), width):
        validator.feed(raw[start : start + width])
    validator.finish()


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        b"data: [DONE]\n\n",
        b'data: {"choices":[{"finish_reason":"stop"}]}\n\n',
        b'data: {"error":{"message":"failed"}}\n\n',
        b'data: {"choices":[{"finish_reason":"error"}]}\n\n',
        b"data: not-json\n\n",
    ],
)
def test_incomplete_or_failed_sse_cannot_be_silent_success(raw):
    validator = ChatStreamValidator()
    with pytest.raises(ValueError):
        validator.feed(raw)
        validator.finish()


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_responses_requires_one_consistent_terminal_state(status):
    import json

    validator = ResponsesStreamValidator()
    frame = (
        b"data: "
        + json.dumps(
            {"type": f"response.{status}", "response": {"status": status}}
        ).encode()
        + b"\n\n"
    )
    for byte in frame:
        validator.feed(bytes([byte]))
    validator.finish()
    with pytest.raises(ValueError):
        validator.feed(frame)


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n',
        b"data: [DONE]\n\n",
        b'data: {"type":"response.completed","response":{"status":"incomplete"}}\n\n',
    ],
)
def test_responses_does_not_invent_success_on_eof_or_conflicting_status(raw):
    validator = ResponsesStreamValidator()
    with pytest.raises(ValueError):
        validator.feed(raw)
        validator.finish()
