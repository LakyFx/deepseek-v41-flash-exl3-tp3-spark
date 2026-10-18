from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from laguna_gateway.lan_bridge import create_lan_bridge_app


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
                json={"data": [{"id": "qwen3.8-27b-ud-q4-k-xl-mtp"}]},
            )
        if request.url.path == "/v1/responses":
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404)


def key_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def lan_client() -> tuple[TestClient, BridgeUpstream]:
    upstream = BridgeUpstream()
    app = create_lan_bridge_app(
        gateway_base_url="http://gateway.internal:18002",
        gateway_api_key="internal-secret",
        client_key_hashes=(key_hash("external-secret"),),
        upstream_transport=httpx.MockTransport(upstream),
    )
    return TestClient(app), upstream


def bearer(value: str = "external-secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def test_lan_bridge_requires_valid_bearer() -> None:
    client, upstream = lan_client()
    with client:
        missing = client.get("/health")
        denied = client.get("/health", headers=bearer("wrong-secret"))

    assert missing.status_code == 401
    assert denied.status_code == 403
    assert not upstream.requests


def test_lan_bridge_injects_internal_bearer_and_strips_external_identity() -> None:
    client, upstream = lan_client()
    with client:
        response = client.post(
            "/v1/responses",
            headers={
                **bearer(),
                "Tailscale-User-Login": "spoofed@example.com",
            },
            json={"model": "qwen3.8-27b-ud-q4-k-xl-mtp", "input": "hello"},
        )

    assert response.status_code == 200
    assert upstream.requests == [
        {
            "method": "POST",
            "path": "/v1/responses",
            "authorization": "Bearer internal-secret",
            "tailscale_header": None,
            "body": {
                "model": "qwen3.8-27b-ud-q4-k-xl-mtp",
                "input": "hello",
            },
        }
    ]


def test_lan_bridge_keeps_route_allowlist() -> None:
    client, upstream = lan_client()
    with client:
        hidden = client.get("/metrics", headers=bearer())

    assert hidden.status_code == 404
    assert not upstream.requests


def test_lan_bridge_rejects_malformed_configured_hash() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        create_lan_bridge_app(
            gateway_base_url="http://gateway.internal:18002",
            gateway_api_key="internal-secret",
            client_key_hashes=("not-a-hash",),
        )
