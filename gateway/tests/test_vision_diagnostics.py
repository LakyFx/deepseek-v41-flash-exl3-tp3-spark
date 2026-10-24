import hashlib
import json
import os
from pathlib import Path
import stat
import time

import pytest

from laguna_gateway.vision_diagnostics import Diagnostics, MAX_FINAL_BYTES
from laguna_gateway.vision_jobs import repeated
from laguna_gateway.vision_result import Collector
from test_vision_jobs import api, body

A = 'vision_' + 'a' * 64
B = 'vision_' + 'b' * 64


def event(content='', reasoning='', finish=None):
    return ('data: '+json.dumps({'choices':[{'delta':{'content':content,
        'reasoning_content':reasoning},'finish_reason':finish}]})+'\n\n').encode()


@pytest.fixture
def armed(tmp_path, monkeypatch):
    root = tmp_path/'private'; root.mkdir(mode=0o700)
    allow = root/'ALLOWLIST.json'
    allow.write_text(json.dumps({'job_ids':[A,B], 'expires_epoch':time.time()+3600}))
    allow.chmod(0o600)
    monkeypatch.setenv('DGX_VISION_DIAGNOSTICS_DIR', str(root))
    return root


def finish(diag, collector, response):
    diag.finish(done=collector.done, finish_reason=collector.finish,
        repetitive=collector.repetitive, usage=collector.usage,
        result_status=response['status_code'], outcome='success')


@pytest.mark.parametrize('channel', ['content', 'reasoning'])
def test_trigger_and_later_final_are_private_and_public_rejection_unchanged(armed, channel):
    diag = Diagnostics.open(A, 'native-owned', {'instance':'one','request_id':'owned','secret':'never-record'})
    assert diag is not None
    c = Collector(diagnostics=diag)
    loop = 'Private repeated evidence 42.5. '*30
    assert repeated(loop)
    wire = event(**{channel:loop})
    # Diagnostics must not depend on HTTP or UTF-8 chunk boundaries.
    for i in range(0, len(wire), 7): c.feed(wire[i:i+7])
    triggered = json.loads((armed/(A+'.json')).read_bytes())
    assert triggered['status'] == 'triggered_draining_same_generation'
    assert triggered['trigger']['channels'][0]['channel'] == channel
    assert triggered['trigger']['channels'][0]['matched_words'] >= 60
    c.feed(event(content='{"recovered_final":true}', finish='stop') + b'data: [DONE]\n\n')
    result = c.result(A); finish(diag, c, result)
    assert result['status_code'] == 422
    assert result['data']['error']['code'] == 'vision_output_repetitive'
    assert 'Private' not in json.dumps(result) and 'recovered_final' not in json.dumps(result)
    stored = json.loads((armed/(A+'.json')).read_bytes())
    assert stored['stream_done'] and stored['finish_reason'] == 'stop'
    assert stored['private_final'].endswith('{"recovered_final":true}')
    assert 'never-record' not in json.dumps(stored)
    assert stored['native_request_id'] == 'native-owned'
    if os.name == 'posix': assert stat.S_IMODE((armed/(A+'.json')).stat().st_mode) == 0o600


def test_unlisted_expired_and_disabled_jobs_do_not_create_files(armed, monkeypatch):
    assert Diagnostics.open('vision_'+'c'*64, 'native') is None
    p = armed/'ALLOWLIST.json'; p.write_text(json.dumps({'job_ids':[A], 'expires_epoch':time.time()-1}))
    assert Diagnostics.open(A, 'native') is None
    monkeypatch.delenv('DGX_VISION_DIAGNOSTICS_DIR')
    assert Diagnostics.open(A, 'native') is None
    assert list(armed.glob('vision_*.json')) == []


def test_evidence_is_not_overwritten_and_final_capture_is_bounded(armed):
    d = Diagnostics.open(A, 'first'); assert d is not None
    assert Diagnostics.open(A, 'second') is None
    d.observe({'content':'x'*(MAX_FINAL_BYTES+500)})
    d.finish(done=True, finish_reason='length', repetitive=False, usage=None,
        result_status=422, outcome='success')
    p = armed/(A+'.json'); record=json.loads(p.read_bytes())
    assert record['native_request_id'] == 'first' and record['final_capture_truncated']
    assert record['final_bytes_retained'] == MAX_FINAL_BYTES
    assert p.stat().st_size < 1024*1024


def test_separate_jobs_never_mix_content(armed):
    ds = [Diagnostics.open(A, 'native-a'), Diagnostics.open(B, 'native-b')]
    cs = [Collector(diagnostics=d) for d in ds]
    for j in range(20):
        for i,c in enumerate(cs):c.feed(event(content=f'job{i}-item{j};'))
    for ident,d,c in zip([A,B],ds,cs):
        c.feed(event(finish='stop')+b'data: [DONE]\n\n');r=c.result(ident);finish(d,c,r)
        assert r['status_code']==200
    a,b=[json.loads((armed/(i+'.json')).read_bytes()) for i in [A,B]]
    assert 'job1-' not in a['private_final'] and 'job0-' not in b['private_final']


def test_observer_failures_do_not_change_collector_result(caplog):
    class Broken:
        def observe(self, _): raise OSError('private-secret-never-log')
        def trigger(self, *_): raise OSError('private-secret-never-log')
    c=Collector(diagnostics=Broken());c.feed(event(reasoning='loop again '*100))
    c.feed(event(content='Later',finish='stop')+b'data: [DONE]\n\n')
    assert c.result(A)['data']['error']['code']=='vision_output_repetitive'
    assert 'private-secret-never-log' not in caplog.text


@pytest.mark.skipif(os.name!='posix', reason='POSIX private ownership/mode enforcement')
def test_world_readable_and_symlink_paths_are_refused(armed, monkeypatch):
    armed.chmod(0o755); assert Diagnostics.open(A,'native') is None
    armed.chmod(0o700); (armed/'ALLOWLIST.json').chmod(0o644)
    assert Diagnostics.open(A,'native') is None
    (armed/'ALLOWLIST.json').chmod(0o600)
    alias=armed.parent/'alias';alias.symlink_to(armed,target_is_directory=True)
    monkeypatch.setenv('DGX_VISION_DIAGNOSTICS_DIR',str(alias))
    assert Diagnostics.open(A,'native') is None


def test_numeric_rows_remain_distinct_with_diagnostics(armed):
    d=Diagnostics.open(A,'native');c=Collector(diagnostics=d)
    text=' '.join(f'row {i} value {i+10}.5 result {i*3}' for i in range(40))
    c.feed(event(content=text,finish='stop')+b'data: [DONE]\n\n')
    result=c.result(A);finish(d,c,result)
    assert result['status_code']==200 and not c.repetitive
    assert json.loads((armed/(A+'.json')).read_bytes())['trigger'] is None


@pytest.mark.parametrize('transport_error', [False, True])
def test_canonical_collect_hook_traces_the_same_generation(api, armed, transport_error):
    import httpx

    client, fake, app = api
    key = 'diagnostic-canonical'
    owner = hashlib.sha256(b'').hexdigest()
    ident = 'vision_' + hashlib.sha256((owner+'\0'+key).encode()).hexdigest()
    (armed/'ALLOWLIST.json').write_text(json.dumps({
        'job_ids':[ident], 'expires_epoch':time.time()+3600}))
    fake.abort_status = 404
    fake.frames = [event(reasoning='Private trigger loop 42.5. '*30)]
    if transport_error:
        fake.frames.append(httpx.ReadError('Disconnected after the trigger'))
    else:
        fake.frames += [event(content='Private recovered answer', finish='stop'),
                       b'data: [DONE]\n\n']
    response = client.post('/v1/chat/completions', json=body(),
                           headers={'Idempotency-Key':key})
    assert len(fake.calls) == 1
    assert 'Private' not in response.text
    stored = json.loads((armed/(ident+'.json')).read_bytes())
    assert stored['job_id'] == ident
    assert stored['native_request_id'].startswith('dgx-gateway-')
    assert stored['trigger']['channels'][0]['channel'] == 'reasoning'
    assert stored['status'] == 'stream_ended'
    if transport_error:
        assert response.status_code == 409
        assert not stored['stream_done'] and stored['result_status'] is None
        assert stored['finish_reason'] is None
        assert stored['outcome'] == 'vision_job_failed'
        assert app.state.vision_jobs.ledger.counts() == {'unknown':1}
    else:
        assert response.status_code == 422
        assert fake.delivered == fake.frames and not fake.aborts
        assert stored['stream_done'] and stored['result_status'] == 422
        assert stored['finish_reason'] == 'stop'
        assert stored['private_final'] == 'Private recovered answer'
        assert app.state.vision_jobs.ledger.counts() == {'failed':1}
        fake.frames = [event(content='Healthy next image',finish='stop')+b'data: [DONE]\n\n']
        healthy = client.post('/v1/chat/completions', json=body(),
                              headers={'Idempotency-Key':'different-diagnostic-job'})
        assert healthy.status_code == 200 and len(fake.calls) == 2
        assert len(list(armed.glob('vision_*.json'))) == 1
