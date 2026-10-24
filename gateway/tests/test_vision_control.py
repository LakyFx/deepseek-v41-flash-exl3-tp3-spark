"""Offline fault injection for deadlines, native receipts and restart recovery."""
import asyncio
import json
import time
from pathlib import Path
import runpy

import httpx
import pytest

from laguna_gateway.vision_control import NativeControl, PATH, SCHEMA
from laguna_gateway.vision_jobs import Jobs, Ledger
from test_vision_jobs import Fake, api, body, event




def eventually(predicate, timeout=3):
    end = time.monotonic()+timeout
    while not predicate() and time.monotonic()<end:
        time.sleep(.005)
    assert predicate()


class Native:
    def __init__(self):
        self.instance = 'native-test-instance'
        self.confirm = True
        self.available = True
        self.wrong_id = False
        self.bad_instance = False
        self.non_object = False
        self.cancels = []
        self.posts = []
        self.fail_headers = False

    def control(self, req):
        if not self.available:
            return httpx.Response(404,json={})
        if req.method == 'GET':
            return httpx.Response(200,json={'schema':SCHEMA,'instance':self.instance})
        data = json.loads(req.content)
        self.cancels.append(data)
        if self.non_object:
            return httpx.Response(200,json=[])
        same = data['instance']==self.instance
        return httpx.Response(200,json={'schema':SCHEMA,'instance':None if self.bad_instance else self.instance,
            'requested_instance':data['instance'],
            'request_id':'wrong-request' if self.wrong_id else data['request_id'],
            'termination_confirmed':self.confirm,
            'state':'instance_replaced' if not same else 'terminated' if self.confirm else 'cancelling'})


@pytest.fixture
def controlled(monkeypatch, request):
    monkeypatch.setenv('DGX_VISION_CONTROL_ENABLED','1')
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY','2')
    monkeypatch.setenv('DGX_VISION_CANCEL_GRACE_SECONDS','.05')
    native = Native()
    original = Fake.__call__

    async def serve(fake, req):
        if req.url.path==PATH:
            return native.control(req)
        if req.url.path=='/v1/chat/completions':
            native.posts.append(json.loads(req.content))
            if native.fail_headers:
                raise httpx.ReadTimeout('lost first response headers',request=req)
        return await original(fake,req)

    monkeypatch.setattr(Fake,'__call__',serve)
    client,fake,app = request.getfixturevalue('api')
    return client,fake,app,native


def shorten_analysis(monkeypatch, ledger):
    claim = ledger.claim
    shortened = False
    def short():
        nonlocal shortened
        row = claim()
        if row and not shortened:
            shortened = True
            row['deadline'] = time.time()+.08
            with ledger.connect() as db:
                db.execute('UPDATE jobs SET deadline=? WHERE id=?',(row['deadline'],row['id']))
        return row
    monkeypatch.setattr(ledger,'claim',short)


def test_queue_wait_does_not_spend_the_analysis_budget(tmp_path):
    clock=[1000.0]
    ledger=Ledger(tmp_path/'jobs.db',clock=lambda:clock[0])
    identity,_=ledger.accept('owner',body())
    clock[0]+=1790
    row=ledger.claim()
    assert row['id']==identity and row['deadline']==clock[0]+1800
    assert ledger.status(identity,'owner')['elapsed_seconds']==1790


def test_queued_status_expires_even_if_the_dispatcher_is_stopped(tmp_path):
    clock=[1000.0]
    ledger=Ledger(tmp_path/'db',clock=lambda:clock[0])
    identity,_=ledger.accept('owner',body())
    clock[0]+=ledger.queue_timeout+1
    assert ledger.status(identity,'owner')['status']=='failed'
    assert ledger.claim() is None


def test_all_unconfirmed_slots_are_bounded_and_queued_work_expires(tmp_path,monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY','2')
    clock=[1000.0]
    ledger=Ledger(tmp_path/'jobs.db',clock=lambda:clock[0],controlled=True)
    for budget in [6144,6143]:
        identity,_=ledger.accept('owner',body(max_tokens=budget))
        assert ledger.claim()['id']==identity
        ledger.dispatch(identity,{'request_id':identity,'instance':'owned','expires':2800})
        ledger.finish(identity,'unknown',{'status_code':409,'data':{},'headers':{}})
    queued,_=ledger.accept('owner',body(max_tokens=6142))
    assert ledger.claim() is None and ledger.blocked()
    clock[0]+=ledger.queue_timeout+1
    assert ledger.claim() is None and ledger.read(queued)['state']=='failed'
    assert ledger.status(queued,'owner')['result']['error']['code']=='queue_deadline'


def test_restart_before_dispatch_is_provably_safe_and_does_not_leave_a_stale_lock(tmp_path):
    async def run():
        ledger=Ledger(tmp_path/'db',controlled=True)
        identity,_=ledger.accept('owner',body())
        row=ledger.claim()
        assert ledger.dispatch_for(identity)=={'pending':True}
        class Control:
            async def cancel(self,dispatch):
                raise AssertionError('A pre-dispatch job has no native generation to cancel')
        jobs=Jobs(ledger.path,None,control=Control())
        await jobs.start()
        try:
            end=asyncio.get_running_loop().time()+2
            while jobs.ledger.read(identity)['state']!='failed':
                assert asyncio.get_running_loop().time()<end
                await asyncio.sleep(.005)
            assert not jobs.ledger.blocked()
            jobs.ledger.not_dispatched(identity,'vision_interrupted')
            jobs.ledger.not_dispatched(identity,'vision_interrupted')
            # A delayed capability lookup in the old sender cannot turn this
            # locally reconciled job back into a new native inference.
            with pytest.raises(ValueError,match='no longer owned'):
                ledger.dispatch(row['id'],{'instance':'native','request_id':'late','expires':9999})
            assert jobs.accept('owner',body())==(identity,False)
        finally:
            await jobs.close()
    asyncio.run(run())


@pytest.mark.parametrize('name', ['DGX_VISION_QUEUE_TIMEOUT_SECONDS','DGX_VISION_JOB_TIMEOUT_SECONDS'])
@pytest.mark.parametrize('value', ['nan','inf','-1','0','3601'])
def test_invalid_timeout_rejected_at_start(tmp_path,monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError):
        Ledger(tmp_path/'db')


def test_control_preserves_quality_and_main_agent_behavior(controlled):
    client,fake,app,native=controlled
    result=client.post('/v1/chat/completions',json=body())
    assert result.status_code==200
    sent=native.posts[0]
    assert sent['reasoning_effort']=='high' and sent['max_tokens']==6144
    assert sent['request_id']==sent['dgx_vision_control']['request_id']
    identity=result.headers['x-dgx-vision-job-id']
    assert app.state.vision_jobs.ledger.dispatch_for(identity)['request_id']==sent['request_id']
    # An untrusted body cannot inject a tracked native request on the text path.
    response=client.post('/v1/chat/completions',json={'model':'deepseek-v4.1-flash',
        'messages':[{'role':'user','content':'Text task'}],
        'dgx_vision_control':{'request_id':'forged','instance':'forged'}})
    assert response.status_code==200 and native.posts[-1]['reasoning_effort']=='max'
    assert 'dgx_vision_control' not in native.posts[-1]
    assert not native.cancels


def test_confirmed_timeout_frees_queue_for_next_image(controlled,monkeypatch):
    client,fake,app,native=controlled
    fake.abort_status=404
    fake.frames=[event(reasoning='Still considering the image')]
    fake.block_after_frames=True
    ledger=app.state.vision_jobs.ledger
    shorten_analysis(monkeypatch,ledger)
    accepted=client.post('/v1/vision/jobs',json=body()).json()
    identity=accepted['job_id']
    eventually(lambda:ledger.read(identity)['state']=='failed')
    status=client.get('/v1/vision/jobs/'+identity).json()
    assert status['result']['data']['error']['code']=='vision_deadline'
    assert status['result']['status_code']==504 and native.cancels
    assert len(native.posts)==1  # Cancellation never regenerates the image.
    fake.frames=None
    fake.block_after_frames=False
    assert client.post('/v1/chat/completions',json=body(max_tokens=6143)).status_code==200
    assert len(native.posts)==2 and not ledger.counts().get('unknown')


def test_lost_headers_are_reconciled_using_id_persisted_before_dispatch(controlled):
    client,fake,app,native=controlled
    native.fail_headers=True
    fake.abort_status=404
    accepted=client.post('/v1/vision/jobs',json=body()).json()
    identity=accepted['job_id']
    ledger=app.state.vision_jobs.ledger
    eventually(lambda:ledger.read(identity)['state']=='failed')
    dispatch=ledger.dispatch_for(identity)
    assert dispatch['request_id']==native.posts[0]['request_id']==native.cancels[0]['request_id']
    assert len(native.posts)==1


def test_missing_control_fails_before_inference_and_does_not_block_text(controlled):
    client,fake,app,native=controlled
    native.available=False
    result=client.post('/v1/chat/completions',json=body())
    assert result.status_code==503 and not native.posts
    assert app.state.vision_jobs.ledger.counts()=={'failed':1}
    assert client.post('/v1/chat/completions',json={'model':'deepseek-v4.1-flash',
        'messages':[{'role':'user','content':'Normal task'}]}).status_code==200


def test_unconfirmed_cancel_reserves_its_slot_but_does_not_freeze_another_lane(controlled,monkeypatch):
    client,fake,app,native=controlled
    native.confirm=False
    fake.abort_status=404
    fake.frames=[event('partial')]
    fake.block_after_frames=True
    ledger=app.state.vision_jobs.ledger
    shorten_analysis(monkeypatch,ledger)
    accepted=client.post('/v1/vision/jobs',json=body()).json()
    eventually(lambda:ledger.read(accepted['job_id'])['state']=='unknown')
    health=client.get('/health').json()['vision_jobs']
    assert health['termination_pending']==1 and health['ready'] is True and health['degraded'] is True
    fake.frames=None
    fake.block_after_frames=False
    assert client.post('/v1/chat/completions',json=body(max_tokens=6143)).status_code==200
    assert len(native.posts)==2 and ledger.counts()=={'unknown':1,'succeeded':1}


def test_background_reconciliation_survives_restart_and_never_replays(tmp_path):
    async def run():
        native=Native()
        async def serve(req):
            return native.control(req)
        async with httpx.AsyncClient(base_url='http://fake',transport=httpx.MockTransport(serve)) as client:
            ledger=Ledger(tmp_path/'db')
            identity,_=ledger.accept('owner',body())
            row=ledger.claim()
            control=NativeControl(client)
            dispatch=await control.dispatch(row,ledger)
            # Simulate a new native instance after the gateway interruption.
            native.instance='replacement-native-instance'
            calls=[]
            async def sender(row):
                calls.append(row['id'])
                return {'status_code':200,'data':{},'headers':{}}
            jobs=Jobs(ledger.path,None,sender=sender,control=control)
            await jobs.start()
            try:
                end=asyncio.get_running_loop().time()+2
                while jobs.ledger.read(identity)['state']!='failed':
                    assert asyncio.get_running_loop().time()<end
                    await asyncio.sleep(.005)
                assert not calls and native.cancels[0]['request_id']==dispatch['request_id']
                assert jobs.accept('owner',body())==(identity,False)
                new,_=jobs.accept('owner',body(max_tokens=6143))
                while jobs.ledger.read(new)['state']!='succeeded':
                    assert asyncio.get_running_loop().time()<end
                    await asyncio.sleep(.005)
                assert calls==[new]
            finally:
                await jobs.close()
    asyncio.run(run())


@pytest.mark.parametrize('mode',['404','wrong_id','unconfirmed','bad_instance','non_object'])
def test_no_receipt_is_invented_when_control_fails(mode,monkeypatch):
    monkeypatch.setenv('DGX_VISION_CANCEL_GRACE_SECONDS','.02')
    async def run():
        native=Native()
        native.available=mode!='404'
        native.wrong_id=mode=='wrong_id'
        native.bad_instance=mode=='bad_instance'
        native.non_object=mode=='non_object'
        native.confirm=mode!='unconfirmed'
        async def serve(req):
            return native.control(req)
        async with httpx.AsyncClient(base_url='http://fake',transport=httpx.MockTransport(serve)) as client:
            control=NativeControl(client)
            dispatch={'instance':native.instance,'request_id':'owned','expires':int(time.time())+1800}
            assert await control.terminate(dispatch) is None
    asyncio.run(run())
