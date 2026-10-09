"""Narrow cancellation endpoint for gateway-owned vLLM chat requests."""
import re
from starlette.requests import Request
from starlette.responses import JSONResponse

class OwnedAbortMiddleware:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or scope['path']!='/abort_requests':
            return await self.app(scope,receive,send)
        response=None
        if scope['method']!='POST':
            response=JSONResponse({'error':'POST required'},status_code=405)
        else:
            request=Request(scope,receive)
            try:
                body=await request.body()
                if len(body)>4096:raise ValueError('body too large')
                import json
                ids=json.loads(body).get('request_ids')
                if not isinstance(ids,list) or not 1<=len(ids)<=8:raise ValueError('non-empty owned IDs required')
                if any(not isinstance(x,str) or not re.fullmatch(r'chatcmpl-dgx-gateway-[0-9a-f]{32}',x) for x in ids):
                    raise ValueError('only gateway-owned chat IDs accepted')
            except (ValueError,AttributeError):
                response=JSONResponse({'error':'invalid owned request_ids'},status_code=400)
            else:
                await scope['app'].state.engine_client.abort(ids)
                response=JSONResponse({'status':'aborted','aborted':len(ids)})
        await response(scope,receive,send)
