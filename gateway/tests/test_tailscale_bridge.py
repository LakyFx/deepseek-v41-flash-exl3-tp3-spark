from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from laguna_gateway.tailscale_bridge import create_bridge_app


class BridgeUpstream:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "authorization": request.headers.get("Authorization"),
                "tailscale_header": request.headers.get("Tailscale-User-Login"),
                "body": body,
            }
        )
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"models": [{"slug": "qwen3.8-27b-q8-dspark"}]},
            )
        if request.url.path == "/v1/responses":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b": dgx-gateway-heartbeat\n\n"
                    b'data: {"type":"response.output_text.delta",'
                    b'"delta":"BRIDGE_OK"}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b": dgx-gateway-heartbeat\n\n"
                    b'data: {"choices":[{"delta":{"content":"CHAT_OK"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        return httpx.Response(404)


def bridge_client() -> tuple[TestClient, BridgeUpstream]:
    upstream = BridgeUpstream()
    app = create_bridge_app(
        gateway_base_url="http://gateway.internal:18002",
        gateway_api_key="internal-secret",
        allowed_users=("owner@example.com",),
        upstream_transport=httpx.MockTransport(upstream),
    )
    return TestClient(app), upstream


def identity(user: str = "owner@example.com") -> dict[str, str]:
    return {"Tailscale-User-Login": user}


def test_bridge_requires_allowed_tailscale_identity_and_route() -> None:
    client, upstream = bridge_client()
    with client:
        missing = client.get("/health")
        denied = client.get("/health", headers=identity("intruder@example.com"))
        hidden = client.get("/metrics", headers=identity())

    assert missing.status_code == 401
    assert denied.status_code == 403
    assert hidden.status_code == 404
    assert not upstream.requests


def test_bridge_injects_internal_bearer_and_strips_identity_header() -> None:
    client, upstream = bridge_client()
    with client:
        health = client.get("/health", headers=identity())
        models = client.get("/v1/models", headers=identity())
        response = client.post(
            "/v1/responses",
            headers={
                **identity(),
                "Authorization": "Bearer must-not-pass",
            },
            json={"model": "qwen3.8-27b-q8-dspark", "input": "hello"},
        )

    assert health.status_code == 200
    assert models.status_code == 200
    assert response.status_code == 200
    assert "BRIDGE_OK" in response.text
    assert len(upstream.requests) == 3
    assert all(
        request["authorization"] == "Bearer internal-secret"
        for request in upstream.requests
    )
    assert all(request["tailscale_header"] is None for request in upstream.requests)


def test_bridge_exposes_chat_completions_without_leaking_client_bearer() -> None:
    client, upstream = bridge_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                **identity(),
                "Authorization": "Bearer external-client-key",
            },
            json={
                "model": "qwen3.8-27b-q8-dspark",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "dgx-gateway-heartbeat" in response.text
    assert "CHAT_OK" in response.text
    assert upstream.requests == [
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "authorization": "Bearer internal-secret",
            "tailscale_header": None,
            "body": {
                "model": "qwen3.8-27b-q8-dspark",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        }
    ]
