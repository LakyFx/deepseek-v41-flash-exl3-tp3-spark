"""Bearer-authenticated HTTPS ingress for trusted DGX LAN clients."""

from __future__ import annotations

import hashlib
import hmac
import os
import string
from collections.abc import Awaitable, Callable

import httpx
from fastapi.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from laguna_gateway.tailscale_bridge import create_bridge_app

_LAN_IDENTITY = "dgx-lan-bearer"


def _error(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": code,
                "param": None,
                "code": code,
            }
        },
    )


def _normalise_hashes(values: tuple[str, ...]) -> tuple[str, ...]:
    hashes: list[str] = []
    for value in values:
        candidate = value.strip().lower()
        if len(candidate) != 64 or any(ch not in string.hexdigits for ch in candidate):
            raise ValueError(
                "DGX_LAN_CLIENT_KEY_SHA256 must contain SHA-256 hex digests"
            )
        hashes.append(candidate)
    if not hashes:
        raise ValueError("DGX_LAN_CLIENT_KEY_SHA256 must not be empty")
    return tuple(hashes)


class _BearerAuthMiddleware:
    def __init__(self, app, allowed_hashes: tuple[str, ...]) -> None:
        self.app = app
        self.allowed_hashes = allowed_hashes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            response = _error(401, "LAN API bearer is required", "missing_lan_bearer")
            await response(scope, receive, send)
            return

        digest = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
        if not any(
            hmac.compare_digest(digest, expected) for expected in self.allowed_hashes
        ):
            response = _error(403, "LAN API bearer is not allowed", "lan_bearer_denied")
            await response(scope, receive, send)
            return

        forwarded_scope = dict(scope)
        forwarded_headers = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() != b"authorization"
            and not name.lower().startswith(b"tailscale-")
        ]
        forwarded_headers.append(
            (b"tailscale-user-login", _LAN_IDENTITY.encode("ascii"))
        )
        forwarded_scope["headers"] = forwarded_headers
        await self.app(forwarded_scope, receive, send)


def create_lan_bridge_app(
    *,
    gateway_base_url: str | None = None,
    gateway_api_key: str | None = None,
    client_key_hashes: tuple[str, ...] | None = None,
    max_request_bytes: int | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    if client_key_hashes is None:
        client_key_hashes = tuple(
            part.strip()
            for part in os.getenv("DGX_LAN_CLIENT_KEY_SHA256", "").split(",")
            if part.strip()
        )
    allowed_hashes = _normalise_hashes(client_key_hashes)
    inner = create_bridge_app(
        gateway_base_url=gateway_base_url
        or os.getenv("DGX_LAN_GATEWAY_UPSTREAM", "http://127.0.0.1:18002"),
        gateway_api_key=gateway_api_key,
        allowed_users=(_LAN_IDENTITY,),
        max_request_bytes=max_request_bytes
        or int(os.getenv("DGX_LAN_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))),
        upstream_transport=upstream_transport,
    )
    return _BearerAuthMiddleware(inner, allowed_hashes)
