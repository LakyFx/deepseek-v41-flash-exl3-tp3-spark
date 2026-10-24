"""FastAPI surface for the DGX Botio model gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .body_limit import BodyTooLarge, read_limited_body
from .engine import GatewayEngine, GatewayError
from .metrics import GatewayMetrics
from .transport import ConnectRetryTransport
from .version import GATEWAY_VERSION
from .auxiliary_vision import (
    AuxiliaryVisionError,
    add_auxiliary_vision_catalog,
    apply_auxiliary_vision_policy,
)
from .vision_jobs import Jobs, AdmissionError, standalone, contaminated, owner, error, job_headers, stream_result

logger = logging.getLogger(__name__)

_SSE_HEARTBEAT = b": dgx-gateway-heartbeat\n\n"


def _stream_error_event(
    *,
    api: str,
    message: str,
    code: str,
    status_code: int | None = None,
) -> bytes:
    error = {
        "message": message,
        "type": code,
        "param": None,
        "code": code,
    }
    if status_code is not None:
        error["status_code"] = status_code
    if api == "responses":
        payload: dict[str, Any] = {
            "type": "response.failed",
            "sequence_number": 0,
            "response": {
                "id": f"resp_gateway_error_{secrets.token_hex(8)}",
                "object": "response",
                "status": "failed",
                "error": error,
                "output": [],
            },
        }
        return (
            "event: response.failed\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            "data: [DONE]\n\n"
        ).encode()
    return (
        f"data: {json.dumps({'error': error}, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    ).encode()


async def _cancel_and_drain(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _nonstream_until_disconnect(
    request: Request, start: Callable[[], Awaitable[Any]]
) -> Any:
    """Stop queued/compacting/generating work when its HTTP caller leaves.

    The route has already consumed the complete JSON body, so this is the only
    remaining receive consumer. Waiting for ASGI disconnect needs no polling.
    """

    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    operation = asyncio.create_task(start())
    watcher = asyncio.create_task(disconnected())
    try:
        await asyncio.wait({operation, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if operation.done():
            return await operation
        # Propagate a receive failure instead of silently treating it as success.
        await watcher
        raise GatewayError(499, "client disconnected", code="request_cancelled")
    finally:
        await _cancel_and_drain(watcher)
        await _cancel_and_drain(operation)


async def _heartbeat_stream(
    *,
    api: str,
    start: Callable[[], Awaitable[Any]],
    relay: Callable[[Any], AsyncIterator[bytes]],
    close: Callable[..., Awaitable[None]],
    metrics: GatewayMetrics,
    heartbeat_seconds: float,
) -> AsyncIterator[bytes]:
    """Open SSE immediately and pulse while queue/upstream reads are idle."""

    start_task = asyncio.create_task(start())
    handle: Any | None = None
    iterator: AsyncIterator[bytes] | None = None
    next_task: asyncio.Task[bytes] | None = None
    close_outcome = "cancelled"
    try:
        metrics.stream_heartbeats.labels(api=api, phase="pending").inc()
        yield _SSE_HEARTBEAT
        while not start_task.done():
            done, _pending = await asyncio.wait(
                {start_task},
                timeout=heartbeat_seconds,
            )
            if done:
                break
            metrics.stream_heartbeats.labels(api=api, phase="pending").inc()
            yield _SSE_HEARTBEAT

        handle = start_task.result()
        if handle.response.status_code >= 400:
            raw = await handle.response.aread()
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            upstream_error = payload.get("error") if isinstance(payload, dict) else None
            message = (
                str(
                    upstream_error.get("message") or "upstream streaming request failed"
                )
                if isinstance(upstream_error, dict)
                else raw.decode("utf-8", errors="replace")[:2_000]
            )
            code = (
                str(upstream_error.get("code") or "upstream_error")
                if isinstance(upstream_error, dict)
                else "upstream_error"
            )
            close_outcome = f"http_{handle.response.status_code}"
            logger.warning(
                "%s upstream rejected streaming request: status=%d code=%s message=%s",
                api,
                handle.response.status_code,
                code,
                message,
            )
            yield _stream_error_event(
                api=api,
                message=message,
                code=code,
                status_code=handle.response.status_code,
            )
            return

        iterator = relay(handle)
        while True:
            next_task = asyncio.create_task(anext(iterator))
            while not next_task.done():
                done, _pending = await asyncio.wait(
                    {next_task},
                    timeout=heartbeat_seconds,
                )
                if done:
                    break
                metrics.stream_heartbeats.labels(api=api, phase="upstream").inc()
                yield _SSE_HEARTBEAT
            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                close_outcome = "success"
                return
            finally:
                next_task = None
            yield chunk
    except GatewayError as exc:
        close_outcome = exc.code
        yield _stream_error_event(
            api=api,
            message=str(exc),
            code=exc.code,
            status_code=exc.status_code,
        )
    except asyncio.CancelledError:
        if handle is None:
            metrics.stream_pending_cancellations.labels(api=api).inc()
        raise
    except Exception as exc:
        close_outcome = "gateway_stream_error"
        logger.exception("%s stream failed after SSE opened", api)
        yield _stream_error_event(
            api=api,
            message=str(exc),
            code="gateway_stream_error",
        )
    finally:
        # Starlette cancels at every await after a disconnect. Protect the
        # entire ownership handoff, including a handle completed while the
        # initial heartbeat was being sent, not only engine.close_stream().
        with anyio.CancelScope(shield=True):
            await _cancel_and_drain(next_task)
            try:
                if iterator is not None:
                    await iterator.aclose()
            finally:
                await _cancel_and_drain(start_task)
                if (
                    handle is None
                    and start_task.done()
                    and not start_task.cancelled()
                    and start_task.exception() is None
                ):
                    handle = start_task.result()
                if handle is not None:
                    await close(handle, outcome=close_outcome)


def _stream_headers(
    settings: Settings,
    *,
    api: str,
    profile: str,
    effective_max_tokens: int,
    route: str | None = None,
) -> dict[str, str]:
    """Headers known before a queued streaming request is admitted."""

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "X-DGX-API": api,
        "X-DGX-Backend": settings.backend_family,
        "X-DGX-Model": settings.served_model,
        "X-Laguna-Profile": profile,
        "X-Laguna-Effective-Max-Tokens": str(effective_max_tokens),
        "X-DGX-Stream-Heartbeat-Seconds": f"{settings.stream_heartbeat_seconds:g}",
    }
    if route is not None:
        headers["X-DGX-Route"] = route
    return headers


def _local_model_catalog(settings: Settings) -> dict[str, Any]:
    model = settings.served_model
    if (
        settings.backend_family == "deepseek_pulsar"
        and settings.qwen_reasoning_policy == "fixed_medium"
    ):
        supported_reasoning_levels = [
            {
                "effort": "medium",
                "description": "Server-enforced balanced reasoning",
            }
        ]
        default_reasoning_level = "medium"
        description = "Private DeepSeek V4 Flash endpoint on DGX Spark"
    elif settings.backend_family == "deepseek_pulsar":
        display_name = "DeepSeek V4 Flash 0731 NVFP4 Adaptive SSD (DGX)"
    elif settings.deepseek_v41_vllm:
        display_name = "DeepSeek V4.1 Flash EXL3 TP3 (DGX)"
    elif "fp8-dspark-k7" in model:
        display_name = "Qwen3.8 27B FP8 DSpark K7 (DGX)"
    elif "fp8-dspark-k14" in model:
        display_name = "Qwen3.8 27B FP8 DSpark K14 (DGX)"
    elif "ud-q4-k-xl-mtp" in model:
        display_name = "Qwen3.8 27B UD-Q4_K_XL Native MTP3 (DGX)"
    elif settings.backend_family == "qwen38_vllm":
        display_name = "Qwen3.8 27B NVFP4 MTP3 (DGX)"
    else:
        display_name = "Qwen3.8 27B Q8 DSpark (DGX)"
    if settings.backend_family == "deepseek_pulsar":
        supported_reasoning_levels = [
            {
                "effort": "high",
                "description": "Native DeepSeek reasoning",
            },
            {
                "effort": "max",
                "description": "Maximum native DeepSeek reasoning",
            },
        ]
        default_reasoning_level = "high"
        description = "Private DeepSeek V4 Flash endpoint on DGX Spark"
    elif settings.deepseek_v41_vllm:
        supported_reasoning_levels = [
            {
                "effort": "none",
                "description": "Disable hidden reasoning",
            },
            {
                "effort": "low",
                "description": "Fast, lighter reasoning (native 25/100)",
            },
            {
                "effort": "medium",
                "description": "Balanced reasoning (native high 50/100)",
            },
            {
                "effort": "max",
                "description": "Maximum native reasoning (100/100); default when omitted",
            },
        ]
        default_reasoning_level = "max"
        description = "Private DeepSeek V4.1 Flash endpoint on three DGX Sparks"
    elif settings.qwen_reasoning_policy in {"fixed_medium", "fixed_xhigh"}:
        forced_effort = settings.qwen_reasoning_policy.removeprefix("fixed_")
        supported_reasoning_levels = [
            {
                "effort": forced_effort,
                "description": (
                    "Server-enforced maximum qualified Qwen reasoning"
                    if forced_effort == "xhigh"
                    else "Server-enforced balanced reasoning"
                ),
            }
        ]
        default_reasoning_level = forced_effort
        description = "Private Qwen3.8 endpoint on DGX Spark"
    else:
        supported_reasoning_levels = [
            {
                "effort": "none",
                "description": "Disable hidden reasoning",
            },
            {
                "effort": "low",
                "description": "Fast, lighter reasoning",
            },
            {
                "effort": "medium",
                "description": "Balanced reasoning depth",
            },
            {
                "effort": "xhigh",
                "description": "Maximum qualified Qwen reasoning",
            },
        ]
        default_reasoning_level = settings.qwen_reasoning_default
        description = "Private Qwen3.8 endpoint on DGX Spark"
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "dgx-spark",
            }
        ],
        # Codex local clients also understand this richer catalog.  Supplying
        # it avoids treating llama-server's unrelated `models` metadata as a
        # Codex catalog and pins the capabilities used by the model picker.
        "models": [
            {
                "slug": model,
                "display_name": display_name,
                "description": description,
                "default_reasoning_level": default_reasoning_level,
                "supported_reasoning_levels": supported_reasoning_levels,
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 0,
                "base_instructions": "",
                "support_verbosity": False,
                "default_verbosity": "low",
                "supports_parallel_tool_calls": True,
                "supports_reasoning_summaries": True,
                "experimental_supported_tools": [],
                # Codex currently understands text/image here. Video is
                # exposed through Chat Completions and advertised by gateway
                # status without adding an unknown Codex catalog enum.
                "input_modalities": (
                    ["text", "image"] if settings.vision_enabled else ["text"]
                ),
                # This richer catalog is consumed by Codex Responses clients.
                # OpenAI Chat clients consume the simple data[] model list.
                "context_window": settings.responses_context_limit,
                "max_context_window": settings.responses_context_limit,
                "effective_context_window_percent": 100
                if settings.responses_context_window
                else max(
                    1,
                    min(
                        100,
                        100
                        * min(
                            settings.compact_trigger_tokens,
                            settings.safe_input_limit(settings.max_profile_max_tokens),
                        )
                        // settings.max_model_len,
                    ),
                ),
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": 10_000,
                },
            }
        ],
    }


def _openai_error(
    status_code: int,
    message: str,
    code: str,
    *,
    retry_after: int | None = None,
) -> JSONResponse:
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": code,
                "param": None,
                "code": code,
            }
        },
    )


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    vision_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=resolved_settings.connect_timeout_seconds,
            read=resolved_settings.read_timeout_seconds,
            write=resolved_settings.write_timeout_seconds,
            pool=resolved_settings.pool_timeout_seconds,
        )
        limits = httpx.Limits(
            max_connections=max(16, resolved_settings.max_concurrency * 4),
            max_keepalive_connections=max(8, resolved_settings.max_concurrency * 2),
            keepalive_expiry=30.0,
        )
        client = httpx.AsyncClient(
            base_url=resolved_settings.upstream_base_url,
            timeout=timeout,
            limits=limits,
            transport=ConnectRetryTransport(
                upstream_transport or httpx.AsyncHTTPTransport(limits=limits),
                recovery_seconds=resolved_settings.upstream_connect_recovery_seconds,
            ),
            headers={"Content-Type": "application/json"},
        )
        vision_client = (
            httpx.AsyncClient(
                base_url=resolved_settings.vision_upstream_base_url,
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=2,
                    max_keepalive_connections=1,
                    keepalive_expiry=30.0,
                ),
                transport=ConnectRetryTransport(
                    vision_transport
                    or httpx.AsyncHTTPTransport(
                        limits=httpx.Limits(
                            max_connections=2, max_keepalive_connections=1
                        )
                    ),
                    recovery_seconds=resolved_settings.upstream_connect_recovery_seconds,
                ),
                headers={"Content-Type": "application/json"},
            )
            if resolved_settings.vision_enabled
            else None
        )
        metrics = GatewayMetrics()
        app.state.settings = resolved_settings
        app.state.metrics = metrics
        app.state.engine = GatewayEngine(
            settings=resolved_settings,
            client=client,
            vision_client=vision_client,
            metrics=metrics,
        )
        app.state.vision_jobs = None
        control_client = None
        if os.getenv('DGX_VISION_JOBS_DB'):
            control = None
            if os.getenv('DGX_VISION_CONTROL_ENABLED') == '1':
                from .vision_control import NativeControl
                # Separate pool: both streaming image connections may be busy
                # when cancellation must reach the native backend.
                control_client = httpx.AsyncClient(
                    base_url=resolved_settings.vision_upstream_base_url,
                    timeout=resolved_settings.health_timeout_seconds,
                    transport=vision_transport or httpx.AsyncHTTPTransport(
                        limits=httpx.Limits(max_connections=2,max_keepalive_connections=1)),
                )
                control = NativeControl(control_client)
            app.state.vision_jobs = Jobs(os.environ['DGX_VISION_JOBS_DB'], app.state.engine, control=control)
            await app.state.vision_jobs.start()
        try:
            yield
        finally:
            if app.state.vision_jobs is not None:
                await app.state.vision_jobs.close()
            if control_client is not None:
                await control_client.aclose()
            await app.state.engine.close_background_tasks()
            if vision_client is not None:
                await vision_client.aclose()
            await client.aclose()

    app = FastAPI(
        title="DGX Botio Model Gateway",
        version=GATEWAY_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def _authorised(request: Request) -> bool:
        expected = resolved_settings.api_key
        if not expected:
            return True
        header = request.headers.get("Authorization", "")
        return secrets.compare_digest(header, f"Bearer {expected}")

    async def _json_body(request: Request) -> dict[str, Any]:
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > resolved_settings.max_request_bytes:
                    raise GatewayError(
                        413,
                        "request body exceeds gateway limit",
                        code="request_too_large",
                    )
            except ValueError:
                pass
        try:
            raw = await read_limited_body(request, resolved_settings.max_request_bytes)
        except BodyTooLarge:
            raise GatewayError(
                413,
                "request body exceeds gateway limit",
                code="request_too_large",
            )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GatewayError(
                400,
                "request body is not valid UTF-8 JSON",
                code="invalid_json",
            ) from exc
        if not isinstance(parsed, dict):
            raise GatewayError(
                400,
                "request JSON must be an object",
                code="invalid_request",
            )
        # This fence is generated by the durable dispatcher, never by callers.
        parsed.pop('dgx_vision_control', None)
        try:
            return apply_auxiliary_vision_policy(
                parsed,
                model=resolved_settings.served_model,
                enabled=resolved_settings.deepseek_v41_vllm,
            )
        except AuxiliaryVisionError as exc:
            raise GatewayError(400, str(exc), code="invalid_auxiliary_vision") from exc

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        _request: Request,
        exc: GatewayError,
    ) -> JSONResponse:
        return _openai_error(
            exc.status_code,
            str(exc),
            exc.code,
            retry_after=exc.retry_after,
        )

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        engine: GatewayEngine = request.app.state.engine
        ok, payload = await engine.health()
        jobs = request.app.state.vision_jobs
        if jobs is not None:
            states = jobs.ledger.counts()
            payload['vision_jobs'] = {'enabled':True,'worker_alive':not jobs.runner.done(),
                'states':states,'automatic_replay':False,
                'ready':not jobs.runner.done() and not jobs.ledger.blocked(),
                'blocked':jobs.ledger.blocked(),
                'degraded':bool(states.get('unknown',0)),
                'termination_pending':states.get('unknown',0),
                'automatic_reconciliation':jobs.control is not None,
                'queue_timeout_seconds':jobs.ledger.queue_timeout,
                'analysis_timeout_seconds':jobs.ledger.job_timeout,
                'output_budget':jobs.output_budget(),
                # Compatibility alias for existing monitor clients.
                'medium_output_budget':jobs.output_budget()}
            ok = ok and not jobs.runner.done()
        return JSONResponse(status_code=200 if ok else 503, content=payload)

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        return await health(request)

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        rendered = request.app.state.metrics.render()
        return Response(
            rendered,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/gateway/status")
    async def gateway_status(request: Request) -> JSONResponse:
        if not _authorised(request):
            return _openai_error(401, "invalid API key", "invalid_api_key")
        engine: GatewayEngine = request.app.state.engine
        _ok, payload = await engine.health()
        return JSONResponse(content=payload)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401, "invalid API key", "invalid_api_key")
        if resolved_settings.backend_family in {
            "deepseek_pulsar",
            "qwen38_llamacpp",
            "qwen38_vllm",
        }:
            catalog = _local_model_catalog(resolved_settings)
            if resolved_settings.deepseek_v41_vllm:
                catalog = add_auxiliary_vision_catalog(
                    catalog, model=resolved_settings.served_model
                )
            return JSONResponse(content=catalog)
        engine: GatewayEngine = request.app.state.engine
        try:
            upstream = await engine.client.get("/v1/models")
        except httpx.HTTPError as exc:
            return _openai_error(502, str(exc), "upstream_unavailable")
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.post("/tokenize")
    async def tokenize(request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401, "invalid API key", "invalid_api_key")
        body = await _json_body(request)
        engine: GatewayEngine = request.app.state.engine
        # Match the active model alias and Czech/reasoning template controls
        # when a chat-shaped payload is supplied.
        if isinstance(body.get("messages"), list):
            (
                normalised,
                _max_tokens,
                _priority,
                _effort_profile,
                _thinking_enabled,
            ) = engine.normalise_request(body)
            token_body: dict[str, Any] = {
                "model": normalised["model"],
                "messages": normalised["messages"],
                "add_generation_prompt": True,
                "return_token_strs": bool(body.get("return_token_strs", False)),
            }
            if normalised.get("tools"):
                token_body["tools"] = normalised["tools"]
            if normalised.get("chat_template_kwargs"):
                token_body["chat_template_kwargs"] = normalised["chat_template_kwargs"]
            if resolved_settings.backend_family in {
                "deepseek_ds4",
                "deepseek_pulsar",
                "qwen38_llamacpp",
                "qwen38_vllm",
            }:
                count = await engine.token_count(normalised)
                measurement = (
                    "local_tokenizer_with_template_margin"
                    if resolved_settings.backend_family
                    in {"deepseek_ds4", "deepseek_pulsar"}
                    else "upstream_visible_plus_hidden_reasoning_guard"
                    if resolved_settings.backend_family == "qwen38_vllm"
                    else "llamacpp_apply_template_tokenize"
                )
                return JSONResponse(
                    content={
                        "count": count,
                        "model": resolved_settings.served_model,
                        "measurement": measurement,
                    }
                )
        else:
            token_body = body
            token_body["model"] = engine._normalise_model(body.get("model"))
            if (
                resolved_settings.backend_family in {"deepseek_ds4", "deepseek_pulsar"}
                and engine.local_tokenizer is not None
            ):
                text = str(body.get("prompt") or body.get("content") or "")
                count = len(
                    engine.local_tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    ).ids
                )
                return JSONResponse(
                    content={
                        "count": count,
                        "model": resolved_settings.served_model,
                        "measurement": "local_tokenizer",
                    }
                )
        try:
            upstream = await engine.client.post("/tokenize", json=token_body)
        except httpx.HTTPError as exc:
            return _openai_error(502, str(exc), "upstream_unavailable")
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    def _accept_vision(request, body):
        jobs = request.app.state.vision_jobs
        key = request.headers.get('Idempotency-Key') or request.headers.get('X-Client-Request-Id')
        if key and len(key) > 256:
            raise AdmissionError('Idempotency key exceeds 256 characters')
        return jobs.accept(owner(request),body,key)

    @app.post('/v1/vision/jobs')
    async def vision_job_submit(request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401,'invalid API key','invalid_api_key')
        jobs = request.app.state.vision_jobs
        if jobs is None:
            return _openai_error(404,'Vision jobs are not enabled','not_found')
        body = await _json_body(request)
        if not standalone(body):
            return JSONResponse(status_code=422,content=error('standalone_image_required',
                'This endpoint accepts one image question without agent history or tools.'))
        request.app.state.engine.normalise_request(body)
        try:
            identity,created = _accept_vision(request,body)
        except AdmissionError as exc:
            return JSONResponse(status_code=409,headers={'X-Should-Retry':'false'},content=error('job_not_accepted',str(exc)))
        return JSONResponse(status_code=202,headers=job_headers(identity),
            content={**jobs.ledger.status(identity,owner(request)),'created':created})

    @app.get('/v1/vision/jobs/{identity}')
    async def vision_job_status(identity: str, request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401,'invalid API key','invalid_api_key')
        jobs = request.app.state.vision_jobs
        row = jobs.ledger.status(identity,owner(request)) if jobs else None
        if row is None:
            return _openai_error(404,'Job not found for this API credential','not_found')
        return JSONResponse(content=row,headers=job_headers(identity))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401, "invalid API key", "invalid_api_key")
        body = await _json_body(request)
        engine: GatewayEngine = request.app.state.engine
        jobs = request.app.state.vision_jobs
        if jobs is not None and contaminated(body):
            return JSONResponse(status_code=422,headers={'X-Should-Retry':'false'},
                content=error('repetitive_reasoning_history',
                'Assistant reasoning history contains a repetition loop. Stop retrying this context; start a clean conversation with a factual handoff.'))
        if jobs is not None and standalone(body):
            # Validate model/options before durable admission. Reasoning effort,
            # image bytes and requested budgets retain the existing policy.
            engine.normalise_request(body)
            try:
                identity,_ = _accept_vision(request,body)
            except AdmissionError as exc:
                return JSONResponse(status_code=409,headers={'X-Should-Retry':'false'},content=error('job_not_accepted',str(exc)))
            if bool(body.get('stream',False)):
                return StreamingResponse(stream_result(jobs,identity,owner(request),resolved_settings.stream_heartbeat_seconds),
                    media_type='text/event-stream',headers={**job_headers(identity),'X-Accel-Buffering':'no'})
            result = await _nonstream_until_disconnect(request,lambda:jobs.wait(identity,owner(request)))
            return JSONResponse(status_code=result['status_code'],content=result['data'],
                headers={**result['headers'],**job_headers(identity)})
        if not bool(body.get("stream", False)):
            result = await _nonstream_until_disconnect(
                request, lambda: engine.complete(body)
            )
            return JSONResponse(
                status_code=result.status_code,
                content=result.data,
                headers=result.headers,
            )
        (
            normalised,
            effective_max_tokens,
            _priority,
            effort_profile,
            _thinking_enabled,
        ) = engine.normalise_request(body)
        return StreamingResponse(
            _heartbeat_stream(
                api="chat_completions",
                start=lambda: engine.begin_stream(body),
                relay=engine.iter_stream,
                close=engine.close_stream,
                metrics=engine.metrics,
                heartbeat_seconds=resolved_settings.stream_heartbeat_seconds,
            ),
            status_code=200,
            headers=_stream_headers(
                resolved_settings,
                api="chat_completions",
                profile=effort_profile,
                effective_max_tokens=effective_max_tokens,
                route=engine._upstream_profile(normalised),
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        if not _authorised(request):
            return _openai_error(401, "invalid API key", "invalid_api_key")
        body = await _json_body(request)
        engine: GatewayEngine = request.app.state.engine
        if not bool(body.get("stream", False)):
            result = await _nonstream_until_disconnect(
                request, lambda: engine.complete_responses(body)
            )
            return JSONResponse(
                status_code=result.status_code,
                content=result.data,
                headers=result.headers,
            )
        (
            normalised,
            effective_max_tokens,
            _priority,
            effort_profile,
        ) = engine.normalise_responses_request(body)
        return StreamingResponse(
            _heartbeat_stream(
                api="responses",
                start=lambda: engine.begin_responses_stream(body),
                relay=engine.iter_responses_stream,
                close=engine.close_responses_stream,
                metrics=engine.metrics,
                heartbeat_seconds=resolved_settings.stream_heartbeat_seconds,
            ),
            status_code=200,
            headers=_stream_headers(
                resolved_settings,
                api="responses",
                profile=effort_profile,
                effective_max_tokens=effective_max_tokens,
                route=engine._responses_upstream_profile(normalised),
            ),
            media_type="text/event-stream",
        )

    return app


app = create_app()
