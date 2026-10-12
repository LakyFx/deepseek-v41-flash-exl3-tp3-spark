"""Historical Owen measurements over the same quarantined, bound transport.

Extract only the pinned upstream streaming and aggregate-rate functions. Host
management from the original client is never imported or executed.
"""
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import types
import urllib.request
import uuid

SOURCE_SHA256 = 'b130f172ec4e039532e9fb0b7d4d6256ddc5f16918c88d0689841146631ba46a'


def functions(source, urlopen):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:
        raise ValueError('historical client source drift')
    wanted={'stream_request','decode_window_tps'}
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(nodes)!=2:raise ValueError('historical function layout drift')
    env={'threading':threading,'time':time,'hashlib':hashlib,'json':json,'BENCH_TIMEOUT_SECONDS':600,
         'urllib':types.SimpleNamespace(request=types.SimpleNamespace(Request=urllib.request.Request,urlopen=urlopen))}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    return env


def run(harness, fixture_path, source):
    fixture=json.loads(fixture_path.read_text())
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest()!='40427145003f73d3db195de5e98810fcc6ed294d68872df8028f9a3181fcb367':
        raise ValueError('frozen Linux fixture drift')
    h=harness
    deadline=time.monotonic()+1200

    class OwnedResponse:
        def __init__(self, connection, response, sock, server_id):
            self.connection,self.response,self.sock,self.server_id=connection,response,sock,server_id
            self.status=response.status
        def __enter__(self):return self
        def __exit__(self,*exc):
            self.response.close();self.connection.close()
            h.save('owen-owned-closed',{'server_id':self.server_id,'cancellation':'ASGI disconnect; require engine-idle proof before next wave'})
        def __iter__(self):
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Owen stream deadline')
                self.sock.settimeout(min(600,remaining))
                line=self.response.readline()
                if not line:return
                yield line

    def bound_open(request, timeout):
        if not request.full_url.startswith(h.net.BASE+'/v1/'):
            raise ValueError('unreviewed inference endpoint')
        body=json.loads(request.data)
        if request.full_url.endswith('/v1/chat/completions'):
            from runner.profiles import body_for_profile
            body = body_for_profile(body, 'max')
        body['request_id']='dgx-gateway-'+uuid.uuid4().hex
        chat=request.full_url.endswith('/v1/chat/completions')
        server_id=('chatcmpl-' if chat else 'cmpl-')+body['request_id']
        h.save('owen-owned',{'server_id':server_id,'path':request.full_url})
        # The existing campaign abort API only accepts chat IDs. Completions
        # use their actual cmpl ID and the server's verified with_cancellation
        # disconnect path, followed by the same mandatory engine-idle gate.
        if chat:
            with h.client.lock:
                h.client.pending_aborts.add(server_id)
                h.client._checkpoint()
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Owen benchmark deadline')
        c=h.net.connection(h.net.BASE,timeout=min(timeout,remaining))
        c.connect();sock=c.sock
        c.request('POST',request.full_url[len(h.net.BASE):],json.dumps(body).encode(),
                  {'Content-Type':'application/json','Connection':'close'})
        try:return OwnedResponse(c,c.getresponse(),sock,server_id)
        except BaseException:
            c.close()
            raise

    upstream=functions(source,bound_open)

    def wave_inner(payloads):
        h.guard()
        if time.monotonic()>=deadline:raise TimeoutError('Owen wave deadline')
        stop=threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            futures=[pool.submit(upstream['stream_request'],h.net.BASE+path,body,stop) for path,body in payloads]
            rows=[f.result() for f in futures]
        h.save('owen-requests',rows)
        h.drain(deadline=time.monotonic()+15)
        for row in rows:
            if not row['ok'] or not isinstance(row['prompt_tokens'],int) or not isinstance(row['completion_tokens'],int) or row['finish_reason'] not in ('length','stop'):
                raise RuntimeError('incomplete Owen request; preserve failed evidence')
            row['first_s']=row.pop('first_at');row['last_s']=row.pop('last_at')
        return rows

    def wave(payloads):
        if not hasattr(h, 'monitor'):
            return wave_inner(payloads)
        result = h.monitor('owen-wave', len(payloads), deadline,
                           lambda: {'rows': wave_inner(payloads)})
        if result['native_metrics'].get('coverage_verified') is not True:
            raise RuntimeError('Owen wave counter coverage unknown or foreign traffic detected')
        return result['rows']

    results=[]
    h.reset()
    for cell in fixture['decode']:
        rows=wave(cell['payloads'])
        value={'point':cell['point'],'sample':cell['sample'],'requests':rows,
               'aggregate_decode_tps':upstream['decode_window_tps'](rows),
               'single_stream_decode_tps':rows[0]['decode_tps'] if len(rows)==1 else None}
        h.save('owen-decode',value);results.append(value)
    for cell in fixture['prefill']:
        body={**cell['body'],'cache_salt':uuid.uuid4().hex}
        rows=wave([(cell['path'],body)])
        row=rows[0]
        value={'nominal_tokens':cell['nominal_tokens'],'repeat':cell['repeat'],'request':row,
               'prompt_tokens_per_ttft_second':row['prompt_tokens']/row['ttft_s'],
               'prompt_sha256':cell['prompt_sha256'],'cache_salt':body['cache_salt']}
        h.save('owen-prefill',value);results.append(value)
    result={'complete':True,'rows':results,'reasoning':'production MAX; author effort unverified',
            'limits':fixture['limits'],'wall_seconds':1200-(deadline-time.monotonic())}
    h.save('OWEN-RESULT',result)
    return result
