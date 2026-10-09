"""Narrow benchmark operations; no general vLLM developer API enabled.

Inactive without the owned closed-window lease. All generation traffic passes
straight through ASGI, without buffering or changing sampling/scheduling.
"""
import json
import os
from pathlib import Path
import re

LEASE=Path('/opt/dgx_campaign/lease/lease.json')
PATHS={'/reset_prefix_cache','/collective_rpc','/abort_requests'}

def authorize(scope):
    source=scope.get('client') or ()
    peer=os.environ.get('DGX_CAMPAIGN_CONTROL_PEER_ADDRESS','')
    if (not peer or scope.get('method')!='POST' or len(source)!=2 or source[0]!=peer
        or not 20000<=source[1]<=20255 or LEASE.is_symlink() or not LEASE.is_file()):
        return None
    owner=json.loads(LEASE.read_bytes())
    if owner.get('window_closed') is not True:return None
    return owner

class CampaignControl:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path') not in PATHS:
            return await self.app(scope,receive,send)
        try:owner=authorize(scope)
        except (OSError,ValueError):owner=None
        if owner is None:return await self.app(scope,receive,send)
        from starlette.responses import JSONResponse
        try:
            raw=b''
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                raw+=message.get('body',b'')
                if len(raw)>131072:raise ValueError('bounded control body required')
                if not message.get('more_body'):break
            body=json.loads(raw) if raw else None
            engine=scope['app'].state.engine_client
            if scope['path']=='/reset_prefix_cache':
                if body is not None:raise ValueError('reset body must be empty')
                result={'success':bool(await engine.reset_prefix_cache(False,False))}
            elif scope['path']=='/abort_requests':
                ids=body.get('request_ids') if isinstance(body,dict) else None
                if (not isinstance(ids,list) or not 1<=len(ids)<=64 or any(not isinstance(x,str)
                    or not re.fullmatch(r'chatcmpl-dgx-gateway-[a-f0-9]{32}',x) for x in ids)):
                    raise ValueError('only explicit benchmark request IDs allowed')
                await engine.abort(ids)
                result={'status':'aborted','aborted':len(ids)}
            else:
                if (not isinstance(body,dict) or set(body)!={'method','args','kwargs','timeout'}
                    or body['method']!='dgx_campaign_qualify' or body['kwargs']!={} or body['timeout']!=300
                    or not isinstance(body['args'],list) or len(body['args'])!=1 or not isinstance(body['args'][0],str)):
                    raise ValueError('only prepared qualification RPC allowed')
                request=json.loads(body['args'][0])
                if any(request.get(k)!=owner.get(k) for k in ('campaign_id','plan_sha256')):
                    raise ValueError('RPC identity differs')
                result={'results':await engine.collective_rpc(method=body['method'],timeout=300,args=tuple(body['args']),kwargs={})}
            response=JSONResponse(result)
        except (ValueError,KeyError,TypeError) as exc:
            response=JSONResponse({'error':str(exc)},status_code=400)
        await response(scope,receive,send)
