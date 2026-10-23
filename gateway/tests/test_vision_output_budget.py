"""Configured standalone vision headroom reaches the existing native transport."""
from copy import deepcopy
import json

import pytest

from test_vision_jobs import api, body
from laguna_gateway.vision_jobs import Jobs, Ledger


@pytest.fixture(autouse=True)
def configured_budget(monkeypatch):
    monkeypatch.setenv('DGX_VISION_FINAL_ANSWER_TOKENS', '32768')


@pytest.mark.parametrize('stream', [False, True])
def test_medium_job_receives_32k_final_headroom_without_extra_reasoning(api, stream):
    client, fake, app = api
    request = body(stream=stream, thinking_budget=3072, max_completion_tokens=6144)
    original = deepcopy(request)
    result = client.post('/v1/chat/completions', json=request)
    assert result.status_code == 200
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent['max_tokens'] == 35840 and sent['thinking_budget'] == 3072
    assert sent['reasoning_effort'] == 'high'  # Native wire value for public medium.
    assert 'max_completion_tokens' not in sent
    assert sent['messages'][-1]['content'] == original['messages'][-1]['content']
    identity = result.headers['x-dgx-vision-job-id']
    stored = json.loads(app.state.vision_jobs.ledger.read(identity)['body'])
    assert stored['max_tokens'] == 35840 and stored['thinking_budget'] == 3072
    assert request == original
    policy = client.get('/health').json()['vision_jobs']['medium_output_budget']
    assert policy == {'reasoning_effort':'medium', 'thinking_budget':3072,
                      'final_answer_tokens':32768, 'max_tokens':35840}


def test_main_max_text_and_image_history_keep_their_own_budgets(api):
    client, fake, _ = api
    request = body(reasoning_effort='max', max_tokens=8192)
    assert client.post('/v1/chat/completions', json=request).status_code == 200
    assert fake.calls[-1]['max_tokens'] == 8192
    assert fake.calls[-1]['reasoning_effort'] == 'max'
    request['messages'] = [{'role':'user', 'content':'Text task'}]
    assert client.post('/v1/chat/completions', json=request).status_code == 200
    assert fake.calls[-1]['max_tokens'] == 8192
    assert fake.calls[-1]['reasoning_effort'] == 'max'


def test_policy_changes_never_resubmit_previously_accepted_input(tmp_path):
    ledger = Ledger(tmp_path/'jobs.sqlite3')
    request = body()
    identity, created = ledger.accept('owner', request, key='stable')
    assert created
    promoted = {**request, 'max_tokens':35840, 'thinking_budget':3072}
    assert ledger.accept('owner', request, key='stable', generation_body=promoted) == (identity, False)
    assert ledger.accept('owner', request, generation_body=promoted) == (identity, False)
    assert json.loads(ledger.read(identity)['body'])['max_tokens'] == 6144
    new, created = ledger.accept('owner', body(max_tokens=6143), generation_body=promoted)
    assert created and new != identity
    assert json.loads(ledger.read(new)['body'])['max_tokens'] == 35840


def test_medium_image_history_does_not_receive_standalone_override(api):
    client, fake, _ = api
    request = body(thinking_budget=3072)
    request['messages'] += [{'role':'assistant', 'content':'Prior observation'},
                            {'role':'user', 'content':'Continue the conversation'}]
    result = client.post('/v1/chat/completions', json=request)
    assert result.status_code == 200
    assert 'x-dgx-vision-job-id' not in result.headers
    assert fake.calls[-1]['max_tokens'] == 6144
    assert fake.calls[-1]['thinking_budget'] == 3072


@pytest.mark.parametrize('value', ['-1', '65537', 'nan'])
def test_invalid_budget_fails_at_start(tmp_path, monkeypatch, value):
    monkeypatch.setenv('DGX_VISION_FINAL_ANSWER_TOKENS', value)
    with pytest.raises(ValueError):
        Jobs(tmp_path/'jobs.sqlite3', None)
