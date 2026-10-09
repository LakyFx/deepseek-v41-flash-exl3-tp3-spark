"""Small authenticated Chat Completions gateway with main MAX / vision MEDIUM."""
import hmac
import json
import os

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def has_images(messages):
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if isinstance(content, list) and any(isinstance(item, dict) and
                item.get('type') in ('image_url', 'input_image') for item in content):
            return True
    return False


def policy(body):
    # Both fields must agree: the encoder merges top-level effort over kwargs.
    native = 'high' if has_images(body.get('messages', [])) else 'max'
    body['reasoning_effort'] = native
    kwargs = body.setdefault('chat_template_kwargs', {})
    kwargs.update(thinking=True, enable_thinking=True, reasoning_effort=native)
    return body


def authorized(request):
    token = os.environ.get('RECIPE_API_KEY', '')
    return bool(token) and hmac.compare_digest(
        request.headers.get('authorization', ''), 'Bearer ' + token)


async def health(request):
    if not authorized(request):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(os.environ.get('RECIPE_UPSTREAM', 'http://127.0.0.1:8003') + '/health')
        return JSONResponse({'upstream_status': response.status_code}, status_code=response.status_code)


async def chat(request: Request):
    if not authorized(request):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 16 * 1024**2:
            return JSONResponse({'error': 'request too large'}, status_code=413)
    try:
        body = json.loads(raw)
        if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
            raise ValueError('messages required')
        policy(body)
    except (ValueError, TypeError, AttributeError):
        return JSONResponse({'error': 'invalid chat body'}, status_code=400)
    base = os.environ.get('RECIPE_UPSTREAM', 'http://127.0.0.1:8003').rstrip('/')
    timeout = httpx.Timeout(connect=10, read=float(os.environ.get('RECIPE_READ_TIMEOUT', '1200')),
                            write=60, pool=10)
    client = httpx.AsyncClient(timeout=timeout)
    headers = {'content-type': 'application/json'}
    if os.environ.get('RECIPE_UPSTREAM_KEY'):
        headers['authorization'] = 'Bearer ' + os.environ['RECIPE_UPSTREAM_KEY']
    try:
        upstream = await client.send(client.build_request('POST', base + '/v1/chat/completions',
                                        json=body, headers=headers), stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return JSONResponse({'error': 'upstream unavailable'}, status_code=502)
    if upstream.status_code != 200 or not body.get('stream'):
        try:
            payload = await upstream.aread()
            from starlette.responses import Response
            return Response(payload, status_code=upstream.status_code,
                            media_type=upstream.headers.get('content-type', 'application/json'))
        finally:
            await upstream.aclose()
            await client.aclose()
    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
    return StreamingResponse(chunks(), media_type='text/event-stream')


app = Starlette(routes=[Route('/health', health), Route('/v1/chat/completions', chat, methods=['POST'])])


if __name__ == '__main__':
    if not os.environ.get('RECIPE_API_KEY'):
        raise SystemExit('set RECIPE_API_KEY before exposing the forwarding lane')
    import uvicorn
    uvicorn.run(app, host=os.environ.get('RECIPE_BIND', '127.0.0.1'),
                port=int(os.environ.get('RECIPE_PORT', '8000')))
