import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from laguna_gateway.app import create_app
from laguna_gateway.config import Settings
from laguna_gateway.tailscale_bridge import create_bridge_app
from laguna_gateway.vision_jobs import AdmissionError, Jobs, Ledger, contaminated, standalone
from laguna_gateway.vision_result import Collector
from laguna_gateway.vision_jobs_operator import reconcile

MODEL = 'deepseek-v4.1-flash'
IMAGE = {'type':'image_url','image_url':{'url':'data:image/png;base64,OFFLINE_IMAGE'}}


def body(**kwargs):
    return {'model':MODEL,'messages':[{'role':'user','content':[{'type':'text','text':'Describe'},deepcopy(IMAGE)]}],
            'reasoning_effort':'medium','max_tokens':6144,**kwargs}


def event(content='', finish=None, reasoning=''):
    return ('data: '+json.dumps({'id':'chat-test','model':MODEL,'choices':[{'index':0,
        'delta':{'content':content,'reasoning_content':reasoning},'finish_reason':finish}]})+'\n\n').encode()


class Fake:
    def __init__(self):
        self.calls = []
        self.aborts = []
        self.content = 'Professor with a pointer.'
        self.finish = 'stop'
        self.reasoning = 'Private reasoning must not reach the client.'

    async def __call__(self, req):
        data = json.loads(req.content) if req.content else {}
        if req.url.path == '/tokenize':
            return httpx.Response(200,json={'count':100})
        if req.url.path == '/health':
            return httpx.Response(200,json={'status':'ok'})
        if req.url.path == '/abort_requests':
            self.aborts.append(data)
            return httpx.Response(200,json={'success':True})
        assert req.url.path == '/v1/chat/completions', req.url.path
        self.calls.append(data)
        if data.get('stream'):
            class Stream(httpx.AsyncByteStream):
                async def __aiter__(stream):
                    yield event(self.content,self.finish,self.reasoning)+b'data: [DONE]\n\n'
            return httpx.Response(200,headers={'content-type':'text/event-stream'},
                stream=Stream())
        return httpx.Response(200,json={'id':'normal','model':MODEL,'choices':[{'index':0,
            'message':{'role':'assistant','content':'Normal result'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':2,'total_tokens':102}})


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_VISION_JOBS_DB',str(tmp_path/'jobs.sqlite3'))
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS','1')
    fake = Fake()
    settings = Settings(upstream_base_url='http://fake',vision_upstream_base_url='http://fake',
        served_model=MODEL,backend_family='qwen38_vllm',deepseek_v41_vllm=True,qwen_native_vllm=True,
        qwen_reasoning_default='xhigh',qwen_reasoning_policy='client_or_default',default_max_tokens=32768,
        max_max_tokens=131072,xhigh_default_max_tokens=131072,xhigh_max_tokens=131072,
        max_profile_default_max_tokens=131072,max_profile_max_tokens=131072,max_model_len=401408,
        vision_max_model_len=401408,vision_max_images_per_prompt=32,chat_image_overflow_mode='off',
        enforce_czech=False,max_concurrency=12,max_admitted_tokens=5336064)
    app = create_app(settings,upstream_transport=httpx.MockTransport(fake),vision_transport=httpx.MockTransport(fake))
    with TestClient(app) as client:
        yield client,fake,app


def test_transparent_nonstream_reuses_same_inference_and_preserves_budgets(api):
    client,fake,_ = api
    first = client.post('/v1/chat/completions',json=body())
    second = client.post('/v1/chat/completions',json=body())
    assert first.status_code == second.status_code == 200, first.text
    assert first.headers['x-dgx-vision-job-id'] == second.headers['x-dgx-vision-job-id']
    assert len(fake.calls)==1
    assert fake.calls[0]['max_tokens']==6144 and fake.calls[0]['reasoning_effort']=='high'
    assert fake.calls[0]['messages'][-1]['content'][-1]==IMAGE
    assert 'reasoning' not in first.text and 'Private' not in first.text
    assert first.json()['choices'][0]['message']['content']=='Professor with a pointer.'


def test_stream_is_openai_compatible_with_job_progress_and_no_reasoning(api):
    client,fake,_ = api
    response = client.post('/v1/chat/completions',json=body(stream=True))
    assert response.status_code==200 and ': vision-job ' in response.text
    chunks=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
    assert chunks[0]['choices'][0]['delta']['content']=='Professor with a pointer.'
    assert chunks[-1]['choices'][0]['finish_reason']=='stop'
    assert 'Private reasoning' not in response.text and response.text.endswith('data: [DONE]\n\n')
    assert len(fake.calls)==1


def test_explicit_acceptance_and_status_survive_client_detach(api):
    client,fake,app = api
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses=list(pool.map(lambda _:client.post('/v1/vision/jobs',json=body()),range(8)))
    assert all(r.status_code==202 for r in responses)
    ids={r.json()['job_id'] for r in responses}
    assert len(ids)==1 and sum(r.json()['created'] for r in responses)==1
    identity=ids.pop()
    for _ in range(100):
        status=client.get('/v1/vision/jobs/'+identity)
        if status.json()['status']=='succeeded':break
        time.sleep(.01)
    assert status.json()['status']=='succeeded' and len(fake.calls)==1
    assert status.json()['backend_request_id'].startswith('dgx-gateway-')
    assert client.get('/v1/vision/jobs/'+identity,headers={'Authorization':'Bearer another-owner'}).status_code==404


@pytest.mark.parametrize('content,finish',[('', 'stop'),('Partial','length'),('<think>Private</think>','stop')])
def test_incomplete_answers_are_not_success_or_retried(api,content,finish):
    client,fake,_ = api
    fake.content,fake.finish=content,finish
    for _ in range(2):
        response=client.post('/v1/chat/completions',json=body())
        assert response.status_code==422, response.text
        assert response.json()['error']['retryable'] is False
        assert response.headers['x-should-retry']=='false'
        assert 'Private' not in response.text
    assert len(fake.calls)==1


def test_repetition_is_buffered_and_blocks_unknown_replay(api):
    client,fake,_ = api
    fake.reasoning='Let me check this again. '*100
    response=client.post('/v1/chat/completions',json=body())
    assert response.status_code==409 and 'Let me' not in response.text
    assert len(fake.calls)==1
    assert client.post('/v1/chat/completions',json=body()).status_code==409
    assert client.post('/v1/chat/completions',json=body(max_tokens=6143)).status_code==409
    assert len(fake.calls)==1
    assert fake.aborts and fake.aborts[0]['request_ids']==['chatcmpl-'+fake.calls[0]['request_id']]


def test_contaminated_history_stops_before_gpu_but_clean_main_remains_max(api):
    client,fake,_=api
    request={'model':MODEL,'messages':[{'role':'user','content':'Continue'},
        {'role':'assistant','content':'Prior answer','reasoning_content':'Let me check it again. '*100},
        {'role':'user','content':'Next'}]}
    response=client.post('/v1/chat/completions',json=request)
    assert response.status_code==422 and response.json()['error']['code']=='repetitive_reasoning_history'
    assert not fake.calls
    request['messages'][1]['reasoning_content']='I checked the image and found a pointer.'
    response=client.post('/v1/chat/completions',json=request)
    assert response.status_code==200
    assert fake.calls[-1]['reasoning_effort']=='max' and not fake.calls[-1].get('stream')


def test_agent_tools_and_image_history_bypass_job_path(api):
    client,fake,_=api
    request=body(tools=[{'type':'function','function':{'name':'inspect','parameters':{'type':'object'}}}])
    request.pop('reasoning_effort')
    response=client.post('/v1/chat/completions',json=request)
    assert response.status_code==200 and 'x-dgx-vision-job-id' not in response.headers
    assert fake.calls[-1]['reasoning_effort']=='max'
    request=body()
    request['messages'] += [{'role':'assistant','content':'Prior'},{'role':'user','content':'Continue'}]
    assert not standalone(request)


def test_idempotency_key_cannot_be_reused_for_changed_input(api):
    client,fake,_=api
    first=client.post('/v1/chat/completions',json=body(),headers={'Idempotency-Key':'test-key'})
    second=client.post('/v1/chat/completions',json=body(max_tokens=6143),headers={'Idempotency-Key':'test-key'})
    assert first.status_code==200 and second.status_code==409 and len(fake.calls)==1


def test_ledger_restart_preserves_unknown_without_replay(tmp_path):
    store=Ledger(tmp_path/'db.sqlite3')
    identity,_=store.accept('owner',body())
    assert store.claim()['id']==identity
    store.backend(identity,'owned-backend')
    restarted=Ledger(store.path)
    restarted.recover()
    assert restarted.read(identity)['state']=='unknown' and restarted.claim() is None
    assert restarted.accept('owner',body())==(identity,False)
    with pytest.raises(AdmissionError):restarted.accept('owner',body(max_tokens=1))
    assert restarted.status(identity,'another-owner') is None
    with pytest.raises(ValueError):
        reconcile(restarted,identity,{'job_id':identity,'termination_confirmed':True})
    reconcile(restarted,identity,{'job_id':identity,'backend_request_id':'owned-backend',
        'termination_confirmed':True,'evidence':'Verified only this owned backend request has terminated; retained receipt.'})
    assert restarted.accept('owner',body())==(identity,False)
    assert restarted.accept('owner',body(max_tokens=1))[1] is True


def test_worker_shutdown_is_bounded_and_never_redispatches(tmp_path):
    async def scenario():
        calls=[]
        async def sender(row):
            calls.append(row['id'])
            await asyncio.Event().wait()
        jobs=Jobs(tmp_path/'jobs.sqlite3',None,sender=sender)
        await jobs.start()
        identity,_=jobs.accept('owner',body())
        await asyncio.sleep(.01)
        await asyncio.wait_for(jobs.close(),1)
        assert jobs.ledger.read(identity)['state']=='unknown'
        restarted=Jobs(jobs.ledger.path,None,sender=sender)
        await restarted.start()
        await asyncio.sleep(.01)
        assert calls==[identity]
        await restarted.close()
    asyncio.run(scenario())


def test_queue_expiry_and_input_dedup_are_transactional(tmp_path):
    clock=[1000.0]
    store=Ledger(tmp_path/'db',clock=lambda:clock[0])
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted=list(pool.map(lambda _:store.accept('owner',body()),range(8)))
    assert len({r[0] for r in accepted})==1 and sum(r[1] for r in accepted)==1
    clock[0]=1901
    assert store.claim() is None and store.read(accepted[0][0])['state']=='failed'


def test_collector_handles_chunk_boundaries_and_discards_unclosed_thought():
    collector=Collector()
    raw=event('<think>Never final','length')+b'data: [DONE]\n\n'
    for byte in raw:collector.feed(bytes([byte]))
    result=collector.result('fixture')
    assert result['status_code']==422 and 'Never final' not in json.dumps(result)
