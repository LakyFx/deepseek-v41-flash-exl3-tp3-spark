"""Offline concurrency proofs: no real images, network or GPU inference."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from laguna_gateway.app import create_app
from laguna_gateway.config import Settings
from laguna_gateway.media_admission import media_close, media_concurrency, media_request
from laguna_gateway.vision_jobs import AdmissionError, Jobs, Ledger
from test_vision_jobs import Fake, body, event


def image(name):
    request = body(thinking_budget=3072)
    request['messages'][0]['content'][0]['text'] = name
    return request


async def reached(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 3)


def reached_sync(predicate):
    end = time.monotonic() + 3
    while not predicate() and time.monotonic() < end:
        time.sleep(.005)
    assert predicate()


@pytest.mark.parametrize('invalid', ['', '0', '-1', '3', 'unlimited'])
def test_invalid_concurrency_is_rejected(monkeypatch, invalid):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', invalid)
    with pytest.raises(ValueError, match='must be 1 or 2'):
        media_concurrency()


def test_default_stays_serial(monkeypatch):
    monkeypatch.delenv('DGX_MEDIA_MAX_CONCURRENCY', raising=False)
    assert media_concurrency() == 1


def test_two_claims_are_atomic_across_competing_threads(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    store = Ledger(tmp_path/'jobs.db')
    ids = [store.accept('owner', image(str(i)))[0] for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(pool.map(lambda _: store.claim(), range(8)))
    assert {r['id'] for r in claimed if r} == set(ids[:2])
    assert store.counts() == {'running': 2, 'queued': 6}
    store.finish(ids[0], 'unknown', {'unconfirmed': True})
    store.finish(ids[1], 'succeeded', {'status_code': 200})
    assert store.claim() is None
    assert store.accept('owner', image('0')) == (ids[0], False)
    with pytest.raises(AdmissionError):
        store.accept('owner', image('new'))


@pytest.mark.parametrize('limit', [1, 2])
def test_dispatch_capacity_refill_and_dedup(tmp_path, monkeypatch, limit):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', str(limit))
    async def scenario():
        started = []
        release = {}
        async def sender(row):
            started.append(row['id'])
            release[row['id']] = asyncio.Event()
            await release[row['id']].wait()
            return {'status_code': 200}
        jobs = Jobs(tmp_path/'jobs.db', None, sender=sender)
        await jobs.start()
        try:
            ids = [jobs.accept('owner', image(str(i)))[0] for i in range(4)]
            await reached(lambda: len(started) == limit)
            assert started == ids[:limit]
            assert jobs.ledger.counts() == {'running': limit, 'queued': 4-limit}
            assert jobs.accept('owner', image('0')) == (ids[0], False)
            release[ids[0]].set()
            await reached(lambda: len(started) == limit+1)
            assert started[-1] == ids[limit]
            if limit == 2:
                assert jobs.ledger.read(ids[1])['state'] == 'running'
            for identity in ids[1:]:
                await reached(lambda: identity in release)
                release[identity].set()
            await reached(lambda: jobs.ledger.counts() == {'succeeded': 4})
            assert started == ids
        finally:
            await jobs.close()
    asyncio.run(scenario())


def test_unknown_blocks_new_dispatch_but_peer_can_finish(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    async def scenario():
        started = []
        fail, finish = asyncio.Event(), asyncio.Event()
        async def sender(row):
            started.append(row['id'])
            if len(started) == 1:
                await fail.wait()
                raise OSError('Unconfirmed backend outcome')
            await finish.wait()
            return {'status_code': 200}
        jobs = Jobs(tmp_path/'jobs.db', None, sender=sender)
        await jobs.start()
        ids = [jobs.accept('owner', image(str(i)))[0] for i in range(3)]
        try:
            await reached(lambda: len(started) == 2)
            fail.set()
            await reached(lambda: jobs.ledger.read(ids[0])['state'] == 'unknown')
            assert jobs.ledger.read(ids[1])['state'] == 'running'
            assert jobs.ledger.read(ids[2])['state'] == 'queued'
            finish.set()
            await reached(lambda: jobs.ledger.read(ids[1])['state'] == 'succeeded')
            assert jobs.ledger.claim() is None and started == ids[:2]
            with pytest.raises(AdmissionError):
                jobs.accept('owner', image('new'))
        finally:
            await jobs.close()
    asyncio.run(scenario())


def test_shutdown_and_recovery_never_replay_either_active_job(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    async def scenario():
        calls = []
        async def sender(row):
            calls.append(row['id'])
            await asyncio.Event().wait()
        jobs = Jobs(tmp_path/'jobs.db', None, sender=sender)
        await jobs.start()
        ids = [jobs.accept('owner', image(str(i)))[0] for i in range(3)]
        await reached(lambda: len(calls) == 2)
        await asyncio.wait_for(jobs.close(), 1)
        assert [jobs.ledger.read(i)['state'] for i in ids] == ['unknown', 'unknown', 'queued']
        restarted = Jobs(jobs.ledger.path, None, sender=sender)
        await restarted.start()
        try:
            await asyncio.sleep(.01)
            assert calls == ids[:2] and restarted.ledger.claim() is None
        finally:
            await restarted.close()
    asyncio.run(scenario())


def test_deadline_includes_queue_and_unknown_never_replays(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    async def scenario():
        calls = []
        async def sender(row):
            calls.append(row['id'])
            await asyncio.Event().wait()
        jobs = Jobs(tmp_path/'jobs.db', None, sender=sender)
        now = [1000.0]
        jobs.ledger.clock = lambda: now[0]
        await jobs.start()
        ids = [jobs.accept('owner', image(str(i)))[0] for i in range(3)]
        assert all(jobs.ledger.read(i)['deadline']-jobs.ledger.read(i)['created'] == 900 for i in ids)
        # Advance only this test clock; production deadline remains 900 seconds.
        now[0] = 1899.98
        try:
            await reached(lambda: jobs.ledger.counts().get('unknown') == 2)
            assert calls == ids[:2]
            assert jobs.ledger.read(ids[2])['state'] == 'queued'
            now[0] = 1901
            assert jobs.ledger.claim() is None
            assert jobs.ledger.read(ids[2])['state'] == 'failed'
        finally:
            await jobs.close()
    asyncio.run(scenario())


class MediaEngine:
    def __init__(self):
        self.started = []
        self.release = {}

    @media_request()
    async def complete(self, request, nested=False):
        if nested:
            return await self.complete(request)
        if not request:
            return 'text'
        name = request['messages'][0]['content'][0]['text']
        self.started.append(name)
        self.release[name] = asyncio.Event()
        await self.release[name].wait()
        return name

    @media_request(stream=True, close_method='close')
    async def begin(self, request):
        return SimpleNamespace(media_release=None)

    @media_close
    async def close(self, handle, **kwargs):
        pass


def test_cancelled_request_keeps_one_of_two_slots_until_cleanup(monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS', '1')
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    async def scenario():
        engine = MediaEngine()
        tasks = [asyncio.create_task(engine.complete(image(str(i)), nested=True)) for i in range(3)]
        try:
            await reached(lambda: len(engine.started) == 2)
            tasks[0].cancel()
            await asyncio.sleep(.01)
            assert engine.started == ['0', '1']
            assert await engine.complete({}) == 'text'
            engine.release['1'].set()
            await reached(lambda: '2' in engine.started)
            assert not tasks[0].done()
            engine.release['0'].set()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            engine.release['2'].set()
            assert await tasks[1] == '1' and await tasks[2] == '2'
            assert engine._media_request_gate._value == 2
        finally:
            for release in engine.release.values():
                release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(scenario())


def test_two_streams_keep_slots_until_each_cleanup(monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS', '1')
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    async def scenario():
        engine = MediaEngine()
        first = await engine.begin(image('0'))
        second = await engine.begin(image('1'))
        third = asyncio.create_task(engine.begin(image('2')))
        await asyncio.sleep(.01)
        assert not third.done()
        await engine.close(first)
        await engine.close(first)
        third_handle = await asyncio.wait_for(third, 1)
        assert engine._media_request_gate._value == 0
        await engine.close(second)
        await engine.close(third_handle)
        assert engine._media_request_gate._value == 2
    asyncio.run(scenario())


class ControlledFake(Fake):
    def __init__(self):
        super().__init__()
        self.hold_images = self.hold_text = True
        self.images_started = self.text_started = 0

    async def __call__(self, request):
        if request.url.path != '/v1/chat/completions':
            return await super().__call__(request)
        data = json.loads(request.content)
        self.calls.append(data)
        if data.get('stream'):
            label = data['messages'][-1]['content'][0]['text']
            class Stream(httpx.AsyncByteStream):
                async def __aiter__(stream):
                    self.images_started += 1
                    while self.hold_images:
                        await asyncio.sleep(.002)
                    yield event('Image '+label, None, self.reasoning)
                    await asyncio.sleep(0)
                    yield event('.', 'stop')+b'data: [DONE]\n\n'
            return httpx.Response(200, stream=Stream(), headers={'content-type': 'text/event-stream'})
        self.text_started += 1
        while self.hold_text:
            await asyncio.sleep(.002)
        return httpx.Response(200, json={'id': 'text', 'model': data['model'], 'choices': [
            {'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})


def test_real_gateway_two_images_share_c12_with_ten_text_requests(tmp_path, monkeypatch):
    monkeypatch.setenv('DGX_VISION_JOBS_DB', str(tmp_path/'jobs.db'))
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS', '1')
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY', '2')
    fake = ControlledFake()
    settings = Settings(upstream_base_url='http://fake', vision_upstream_base_url='http://fake',
        served_model='deepseek-v4.1-flash', backend_family='qwen38_vllm', deepseek_v41_vllm=True,
        qwen_native_vllm=True, qwen_reasoning_default='xhigh', qwen_reasoning_policy='client_or_default',
        default_max_tokens=32768, max_max_tokens=131072, xhigh_default_max_tokens=131072,
        xhigh_max_tokens=131072, max_profile_default_max_tokens=131072, max_profile_max_tokens=131072,
        max_model_len=401408, vision_max_model_len=401408, vision_max_images_per_prompt=32,
        chat_image_overflow_mode='off', enforce_czech=False, max_concurrency=12, max_admitted_tokens=5336064)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake), vision_transport=httpx.MockTransport(fake))
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=11) as pool:
        try:
            jobs = [client.post('/v1/vision/jobs', json=image(str(i))) for i in range(3)]
            assert all(j.status_code == 202 for j in jobs)
            reached_sync(lambda: fake.images_started == 2)
            assert app.state.engine.gate.active == 2
            assert app.state.vision_jobs.ledger.counts() == {'running': 2, 'queued': 1}
            duplicate = client.post('/v1/vision/jobs', json=image('0')).json()
            assert duplicate['job_id'] == jobs[0].json()['job_id'] and not duplicate['created']
            text = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': 'Plain text'}]}
            futures = [pool.submit(client.post, '/v1/chat/completions', json=text) for _ in range(11)]
            reached_sync(lambda: fake.text_started == 10 and app.state.engine.gate.waiting == 1)
            assert app.state.engine.gate.active == 12 and len(fake.calls) == 12
            fake.hold_text = False
            assert all(f.result(timeout=3).status_code == 200 for f in futures)
            fake.hold_images = False
            reached_sync(lambda: app.state.vision_jobs.ledger.counts() == {'succeeded': 3})
            images = [c for c in fake.calls if c.get('stream')]
            assert len(images) == 3
            assert all(c['max_tokens'] == 6144 and c['thinking_budget'] == 3072
                       and c['reasoning_effort'] == 'high' for c in images)
            assert all(c['reasoning_effort'] == 'max' for c in fake.calls if not c.get('stream'))
            assert app.state.engine.gate.capacity == 12 and app.state.engine.gate.active == 0
            for index, accepted in enumerate(jobs):
                result = client.get('/v1/vision/jobs/'+accepted.json()['job_id']).json()['result']
                assert result['data']['choices'][0]['message']['content'] == 'Image '+str(index)+'.'
        finally:
            fake.hold_images = fake.hold_text = False
