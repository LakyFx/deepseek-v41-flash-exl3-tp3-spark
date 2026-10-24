"""Configured MAX/8K standalone images preserve receipts and agent contracts."""
from copy import deepcopy
import json

import pytest

from test_vision_jobs import api, body
from laguna_gateway.vision_jobs import Jobs, Ledger


@pytest.fixture(autouse=True)
def maximum_profile(monkeypatch):
    monkeypatch.setenv('DGX_VISION_REASONING_EFFORT', 'max')
    monkeypatch.setenv('DGX_VISION_THINKING_BUDGET', '8192')
    monkeypatch.setenv('DGX_VISION_FINAL_ANSWER_TOKENS', '32768')


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('effort', ['medium', 'max'])
def test_standalone_image_uses_operator_max_8k_profile(api, stream, effort):
    client, fake, app = api
    request = body(stream=stream, reasoning_effort=effort,
                   thinking_budget=3072, max_completion_tokens=6144)
    original = deepcopy(request)
    response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200
    sent = fake.calls[0]
    assert sent['reasoning_effort'] == 'max'
    assert sent['thinking_budget'] == 8192
    assert sent['max_tokens'] == 40960
    assert 'max_completion_tokens' not in sent
    assert sent['messages'][-1]['content'] == original['messages'][-1]['content']
    identity = response.headers['x-dgx-vision-job-id']
    stored = json.loads(app.state.vision_jobs.ledger.read(identity)['body'])
    assert stored['reasoning_effort'] == 'max' and stored['max_tokens'] == 40960
    assert stored['thinking_budget'] == 8192
    assert request == original
    health = client.get('/health').json()['vision_jobs']
    assert health['output_budget'] == health['medium_output_budget'] == {
        'reasoning_effort':'max', 'thinking_budget':8192,
        'final_answer_tokens':32768, 'max_tokens':40960}


def test_explicit_jobs_endpoint_persists_effective_profile_before_dispatch(api):
    client, _, app = api
    response = client.post('/v1/vision/jobs', json=body())
    assert response.status_code == 202
    stored = json.loads(app.state.vision_jobs.ledger.read(response.json()['job_id'])['body'])
    assert (stored['reasoning_effort'], stored['thinking_budget'], stored['max_tokens']) == ('max',8192,40960)


@pytest.mark.parametrize('kind', ['text', 'image_history', 'ordinary_image'])
def test_main_task_keeps_its_explicit_reasoning_and_output_budget(api, kind):
    client, fake, _ = api
    request = body(reasoning_effort='max', thinking_budget=16384, max_tokens=131072)
    if kind == 'text':
        request['messages'] = [{'role':'user','content':'Continue the project'}]
    elif kind == 'image_history':
        request['messages'] += [{'role':'assistant','content':'Earlier observation'},
                                {'role':'user','content':'Evaluate the next step'}]
    else:
        request['tool_choice'] = 'none'  # Same ordinary route as the live A/B.
    response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200
    assert 'x-dgx-vision-job-id' not in response.headers
    assert fake.calls[0]['reasoning_effort'] == 'max'
    assert fake.calls[0]['thinking_budget'] == 16384
    assert fake.calls[0]['max_tokens'] == 131072


def test_profile_update_does_not_replay_old_accepted_input(tmp_path):
    ledger = Ledger(tmp_path/'jobs.sqlite3')
    request = body()
    identity, created = ledger.accept('owner', request, key='stable')
    assert created
    configured = Jobs(tmp_path/'jobs.sqlite3', None)
    upgraded = configured.generation_body(request)
    assert upgraded['reasoning_effort'] == 'max' and upgraded['thinking_budget'] == 8192
    assert ledger.accept('owner', request, key='stable', generation_body=upgraded) == (identity,False)
    assert ledger.accept('owner', request, generation_body=upgraded) == (identity,False)
    assert json.loads(ledger.read(identity)['body'])['max_tokens'] == 6144


@pytest.mark.parametrize('name,value', [
    ('DGX_VISION_THINKING_BUDGET','0'), ('DGX_VISION_THINKING_BUDGET','65537'),
    ('DGX_VISION_THINKING_BUDGET','nan'), ('DGX_VISION_REASONING_EFFORT','unknown')])
def test_bad_operator_profile_fails_at_start(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError):
        Jobs(tmp_path/'jobs.sqlite3', None)
