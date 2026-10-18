"""Real API/engine tests with an isolated fake upstream; no DGX requests."""
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from laguna_gateway.app import create_app
from laguna_gateway.auxiliary_vision import apply_auxiliary_vision_policy
from laguna_gateway.config import Settings

MODEL = 'deepseek-v4.1-flash'
IMAGE = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,OFFLINE_IMAGE', 'detail': 'high'}}


def image_body(model=MODEL, **kw):
    return {'model': model, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Popiš obrázek.'}, deepcopy(IMAGE)]}], **kw}


class Upstream:
    def __init__(self):
        self.generations = []

    async def __call__(self, req):
        body = json.loads(req.content) if req.content else {}
        if req.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 100})
        if req.url.path == '/v1/chat/completions':
            self.generations.append(body)
            return httpx.Response(200, json={'id': 'chatcmpl-offline', 'object': 'chat.completion', 'model': MODEL,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Ověřený výsledek.'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 2, 'total_tokens': 102}})
        raise AssertionError(f'Unexpected upstream path {req.url.path}')


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS', '1')
    upstream = Upstream()
    settings = Settings(upstream_base_url='http://offline.invalid', vision_upstream_base_url='http://offline.invalid',
        served_model=MODEL, backend_family='qwen38_vllm', deepseek_v41_vllm=True, qwen_native_vllm=True,
        qwen_reasoning_default='xhigh', qwen_reasoning_policy='client_or_default', default_max_tokens=32768,
        max_max_tokens=131072, xhigh_default_max_tokens=131072, xhigh_max_tokens=131072,
        max_profile_default_max_tokens=131072, max_profile_max_tokens=131072,
        max_model_len=401408, vision_max_model_len=401408, vision_max_images_per_prompt=32,
        chat_image_overflow_mode='off', enforce_czech=False, max_concurrency=8, max_admitted_tokens=3557376)
    app = create_app(settings, upstream_transport=httpx.MockTransport(upstream), vision_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        yield client, upstream


@pytest.mark.parametrize('body', [
    {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Hlavní agent'}]},
    image_body(), image_body(reasoning_effort='max'), image_body(reasoning_effort='xhigh'),
    image_body(tools=[{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}]),
    {'model': MODEL, 'messages': [{'role': 'system', 'content': 'Hlavní agent'},
        {'role': 'user', 'content': [deepcopy(IMAGE)]}, {'role': 'assistant', 'content': 'Historie'},
        {'role': 'user', 'content': 'Pokračuj v agentní práci.'}]},
])
def test_main_agent_remains_max_even_with_images_tools_or_history(api, body):
    client, upstream = api
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200, response.text
    actual = upstream.generations[-1]
    assert actual['reasoning_effort'] == 'max'
    assert actual['chat_template_kwargs'] == {'thinking': True, 'reasoning_effort': 'max'}
    assert actual['model'] == MODEL
    assert actual['max_tokens'] == 32768


@pytest.mark.parametrize('selectors', [{}, {'reasoning_effort': 'max'}, {'reasoning': {'effort': 'max'}},
    {'chat_template_kwargs': {'thinking': False, 'reasoning_effort': 'none'}}])
def test_explicit_vision_alias_is_medium_on_actual_wire_and_keeps_originals(api, selectors):
    client, upstream = api
    body = image_body(MODEL + ':vision', max_tokens=4096, **selectors)
    original = deepcopy(body)
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200, response.text
    actual = upstream.generations[-1]
    assert actual['reasoning_effort'] == 'high'  # public medium -> native high
    assert actual['chat_template_kwargs'] == {'thinking': True, 'reasoning_effort': 'high'}
    assert actual['max_tokens'] == 3072
    assert actual['model'] == MODEL
    assert actual['messages'][0]['content'][-1] == IMAGE
    assert body == original


def test_normal_main_request_is_not_rewritten():
    main = image_body(reasoning_effort='max', max_tokens=131072)
    assert apply_auxiliary_vision_policy(main, model=MODEL, enabled=True) is main


def test_smaller_explicit_vision_cap_is_respected(api):
    client, upstream = api
    r = client.post('/v1/chat/completions', json=image_body(MODEL+':vision', max_completion_tokens=512))
    assert r.status_code == 200
    assert upstream.generations[-1]['max_tokens'] == 512


@pytest.mark.parametrize('body', [
    {'model': MODEL+':vision', 'messages': [{'role': 'user', 'content': 'No image'}]},
    image_body(MODEL+':vision', tools=[{'type': 'function', 'function': {'name': 'agent'}}]),
    image_body(MODEL+':vision', max_tokens=0), image_body(MODEL+':vision', max_tokens='bad'),
])
def test_invalid_auxiliary_request_rejected_before_generation(api, body):
    client, upstream = api
    r = client.post('/v1/chat/completions', json=body)
    assert r.status_code == 400
    assert not upstream.generations


def test_responses_alias_uses_medium_without_changing_main_responses(api):
    client, _ = api
    engine = client.app.state.engine
    body = {'model': MODEL+':vision', 'input': [{'role': 'user', 'content': [
        {'type': 'input_image', 'image_url': IMAGE['image_url']['url']}]}], 'reasoning': {'effort': 'max'}, 'max_output_tokens': 8192}
    auxiliary = apply_auxiliary_vision_policy(body, model=MODEL, enabled=True)
    normalized, cap, *_ = engine.normalise_responses_request(auxiliary)
    assert normalized['reasoning']['effort'] == 'medium'
    assert cap == 3072
    body['model'] = MODEL
    normalized, cap, *_ = engine.normalise_responses_request(body)
    assert normalized['reasoning']['effort'] == 'max'
    assert cap == 8192


def test_model_catalog_keeps_main_max_and_advertises_vision_medium(api):
    client, _ = api
    response = client.get('/v1/models')
    assert response.status_code == 200
    catalog = response.json()
    assert [x['id'] for x in catalog['data']] == [MODEL, MODEL+':vision']
    assert catalog['models'][0]['default_reasoning_level'] == 'max'
    assert catalog['models'][1]['default_reasoning_level'] == 'medium'
