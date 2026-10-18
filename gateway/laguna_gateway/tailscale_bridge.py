"""Loopback-only Tailscale Serve bridge for OpenAI-compatible traffic."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .body_limit import BodyTooLarge, read_limited_body

_ALLOWED_ROUTES = {
    ("GET", "health"),
    ("GET", "v1/models"),
    ("POST", "v1/chat/completions"),
    ("POST", "v1/responses"),
    ("POST", "v1/vision/jobs"),
}
_HOP_BY_HOP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


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


def create_bridge_app(
    *,
    gateway_base_url: str | None = None,
    gateway_api_key: str | None = None,
    allowed_users: tuple[str, ...] | None = None,
    max_request_bytes: int | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    base_url = (
        gateway_base_url
        or os.getenv("DGX_CODEX_GATEWAY_UPSTREAM", "http://127.0.0.1:18002")
    ).rstrip("/")
    api_key = gateway_api_key
    if api_key is None:
        api_key = os.getenv("LAGUNA_GATEWAY_API_KEY", "")
    if allowed_users is None:
        allowed_users = tuple(
            part.strip().lower()
            for part in os.getenv("DGX_CODEX_ALLOWED_TAILSCALE_USERS", "").split(",")
            if part.strip()
        )
    resolved_users = frozenset(user.lower() for user in allowed_users)
    request_limit = max_request_bytes or int(
        os.getenv("DGX_CODEX_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))
    )
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("DGX_CODEX_GATEWAY_UPSTREAM must be an HTTP(S) URL")
    if not api_key:
        raise ValueError("LAGUNA_GATEWAY_API_KEY must be configured")
    if not resolved_users:
        raise ValueError("DGX_CODEX_ALLOWED_TAILSCALE_USERS must not be empty")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(connect=10.0, read=3_600.0, write=120.0, pool=30.0)
        app.state.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=upstream_transport,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="DGX Codex Tailnet Bridge",
        version="1.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def _stream_and_close(response: httpx.Response) -> AsyncIterator[bytes]:
        try:
            if response.is_stream_consumed:
                yield response.content
                return
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(path: str, request: Request):
        identity = request.headers.get("Tailscale-User-Login", "").strip().lower()
        if not identity:
            return _error(
                401,
                "Tailscale Serve identity is required",
                "missing_tailnet_identity",
            )
        if identity not in resolved_users:
            return _error(
                403,
                "Tailscale identity is not allowed",
                "tailnet_identity_denied",
            )

        normalised_path = path.strip("/")
        vision_status = request.method == 'GET' and re.fullmatch(r'v1/vision/jobs/vision_[a-f0-9]{32}(?:[a-f0-9]{32})?',normalised_path)
        if (request.method, normalised_path) not in _ALLOWED_ROUTES and not vision_status:
            return _error(404, "route is not exposed", "route_not_exposed")

        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > request_limit:
                    return _error(
                        413,
                        "request body exceeds bridge limit",
                        "request_too_large",
                    )
            except ValueError:
                pass
        try:
            body = await read_limited_body(request, request_limit)
        except BodyTooLarge:
            return _error(
                413,
                "request body exceeds bridge limit",
                "request_too_large",
            )

        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP
            and name.lower() != "authorization"
            and not name.lower().startswith("tailscale-")
        }
        headers["Authorization"] = f"Bearer {api_key}"
        client: httpx.AsyncClient = request.app.state.client
        upstream_request = client.build_request(
            request.method,
            f"/{normalised_path}",
            params=request.query_params,
            headers=headers,
            content=body,
        )
        try:
            response = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            return _error(504, "gateway request timed out", "gateway_timeout")
        except httpx.RequestError:
            return _error(502, "gateway is unavailable", "gateway_unavailable")

        response_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        return StreamingResponse(
            _stream_and_close(response),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    return app
