"""Cancellation transport checks. No GPU, inference service or credentials."""
import asyncio
from dataclasses import replace

import httpx
import pytest
import hashlib
import json
from pathlib import Path

from laguna_gateway.config import Settings
from laguna_gateway.engine import GatewayEngine
from laguna_gateway.metrics import GatewayMetrics


def settings(mode="tensorfold_disconnect"):
    return Settings(backend_family="qwen38_vllm", qwen_native_vllm=True,
                    upstream_abort_mode=mode, enforce_czech=False)


def engine(client, mode="tensorfold_disconnect"):
    return GatewayEngine(settings=settings(mode), client=client, vision_client=None,
                         metrics=GatewayMetrics())




def test_abort_mode_is_explicit_and_rejects_incompatible_backends(monkeypatch):
    assert Settings().upstream_abort_mode == "legacy"
    settings().validate()
    with pytest.raises(ValueError, match="Unsupported upstream_abort_mode"):
        replace(settings(), upstream_abort_mode="guess").validate()
    with pytest.raises(ValueError, match="TensorFold disconnect mode"):
        replace(Settings(), upstream_abort_mode="tensorfold_disconnect").validate()
    monkeypatch.setenv("LAGUNA_GATEWAY_UPSTREAM_ABORT_MODE", "tensorfold_disconnect")
    monkeypatch.setenv("LAGUNA_GATEWAY_BACKEND_FAMILY", "qwen38_vllm")
    monkeypatch.setenv("LAGUNA_GATEWAY_QWEN_NATIVE_VLLM", "1")
    assert Settings.from_env().upstream_abort_mode == "tensorfold_disconnect"


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_native_mode_never_calls_vllm_abort_or_claims_a_receipt(api):
    async def run():
        calls = []
        async def handler(req):
            calls.append(req.url.path)
            return httpx.Response(404)
        async with httpx.AsyncClient(base_url="http://offline", transport=httpx.MockTransport(handler)) as client:
            e = engine(client)
            await e._abort_upstream_stream(client=client, request_id="owned", api=api)
            assert not calls
            samples = e.metrics.upstream_aborts.collect()[0].samples
            assert any(s.labels.get("outcome") == "disconnect_delegated" and s.value == 1 for s in samples)
            assert not any(s.labels.get("outcome") == "success" for s in samples)
    asyncio.run(run())


def test_legacy_vllm_keeps_precise_owned_id_and_no_abort_all():
    async def run():
        calls = []
        async def handler(req):
            import json
            calls.append((req.url.path, json.loads(req.content)))
            return httpx.Response(200, json={"success": True})
        async with httpx.AsyncClient(base_url="http://offline", transport=httpx.MockTransport(handler)) as client:
            e = engine(client, "legacy")
            await e._abort_upstream_stream(client=client, request_id=None, api="chat_completions")
            assert not calls
            await e._abort_upstream_stream(client=client, request_id="owned", api="chat_completions")
            assert calls == [("/abort_requests", {"request_ids": ["chatcmpl-owned"]})]
    asyncio.run(run())


@pytest.mark.parametrize("effort", ["medium", "max"])
def test_successful_request_policy_is_identical(effort):
    async def run():
        async with httpx.AsyncClient(base_url="http://offline") as client:
            body = {"model": settings().served_model, "max_tokens": 3000,
                    "reasoning_effort": effort,
                    "messages": [{"role": "user", "content": "Describe the result"}]}
            assert engine(client).normalise_request(body) == engine(client, "legacy").normalise_request(body)
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_real_http_socket_cancellation_does_not_close_another_request(stream):
    """Use real httpx/TCP cancellation, rather than a mocked stream close."""
    async def run():
        seen, gone = {}, {}
        async def serve(reader, writer):
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                name = head.split(b" ")[1].decode()
                length = next((int(x.split(b":", 1)[1]) for x in head.split(b"\r\n")
                               if x.lower().startswith(b"content-length:")), 0)
                await reader.readexactly(length)
                seen.setdefault(name, asyncio.Event()).set()
                if stream:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n")
                    await writer.drain()
                await reader.read()
                gone.setdefault(name, asyncio.Event()).set()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server, httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            e = engine(client)
            if stream:
                handles = [await client.send(client.build_request("POST", path, json={}), stream=True)
                           for path in ("/owned", "/other")]
                await handles[0].aclose()
                await e._abort_upstream_stream(client=client, request_id="owned", api="chat_completions")
                await asyncio.wait_for(gone.setdefault("/owned", asyncio.Event()).wait(), 1)
                assert not gone.setdefault("/other", asyncio.Event()).is_set()
                await handles[1].aclose()
            else:
                tasks = [asyncio.create_task(e._post_owned_request(client, path, {}, api="chat_completions"))
                         for path in ("/owned", "/other")]
                await asyncio.wait_for(seen.setdefault("/owned", asyncio.Event()).wait(), 1)
                await asyncio.wait_for(seen.setdefault("/other", asyncio.Event()).wait(), 1)
                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                await asyncio.wait_for(gone.setdefault("/owned", asyncio.Event()).wait(), 1)
                assert not gone.setdefault("/other", asyncio.Event()).is_set()
                tasks[1].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[1]
            await asyncio.wait_for(gone.setdefault("/other", asyncio.Event()).wait(), 1)
    asyncio.run(run())
