from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ds41f_gateway import create_app, production_settings
from test_vision_jobs import Fake, body

ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_requires_an_operator_secret(monkeypatch):
    monkeypatch.delenv('LAGUNA_GATEWAY_API_KEY', raising=False)
    with pytest.raises(ValueError, match='required'):
        production_settings()


def test_every_production_policy_setting_is_preserved(monkeypatch):
    monkeypatch.setenv('LAGUNA_GATEWAY_API_KEY', 'offline-fixture-only')
    monkeypatch.delenv('DS41F_GATEWAY_PROFILE', raising=False)
    monkeypatch.delenv('LAGUNA_GATEWAY_ENFORCE_CZECH', raising=False)
    actual = asdict(production_settings())
    for name in ('api_key', 'upstream_base_url', 'vision_upstream_base_url', 'local_tokenizer_path'):
        actual.pop(name)
    actual['served_model_aliases'] = list(actual['served_model_aliases'])
    assert actual == json.loads((ROOT/'laguna_gateway/production-settings.json').read_text())


def test_portable_defaults_and_language_override(monkeypatch):
    monkeypatch.setenv('LAGUNA_GATEWAY_API_KEY', 'offline-fixture-only')
    monkeypatch.delenv('LAGUNA_GATEWAY_UPSTREAM', raising=False)
    monkeypatch.delenv('LAGUNA_GATEWAY_VISION_UPSTREAM', raising=False)
    monkeypatch.setenv('LAGUNA_GATEWAY_ENFORCE_CZECH', 'false')
    settings = production_settings()
    assert settings.upstream_base_url == settings.vision_upstream_base_url == 'http://127.0.0.1:8003'
    assert settings.enforce_czech is False


def test_original_source_and_explicit_address_only_adaptations():
    manifest = json.loads((ROOT/'source-manifest.json').read_text())
    names = {p.name for p in (ROOT/'laguna_gateway').glob('*.py')}
    assert names == set(manifest['modules'])
    adaptations = []
    for name, record in manifest['modules'].items():
        raw = (ROOT/'laguna_gateway'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['public_sha256']
        if record['adaptation']:
            adaptations.append(name)
        else:
            assert record['public_sha256'] == record['production_sha256']
    assert sorted(adaptations) == ['lan_bridge.py', 'tailscale_bridge.py']


def test_installed_profile_runs_authenticated_durable_vision(monkeypatch, tmp_path):
    monkeypatch.delenv('DGX_MEDIA_MAX_CONCURRENCY', raising=False)
    monkeypatch.delenv('DGX_VISION_JOBS_REVISION', raising=False)
    monkeypatch.delenv('DS41F_RECIPE_VERSION', raising=False)
    monkeypatch.setenv('LAGUNA_GATEWAY_API_KEY', 'offline-fixture-only')
    monkeypatch.setenv('DGX_VISION_JOBS_DB', str(tmp_path/'private/jobs.sqlite3'))
    monkeypatch.setenv('LAGUNA_GATEWAY_ENFORCE_CZECH', 'false')
    fake = Fake()
    app = create_app(upstream_transport=httpx.MockTransport(fake), vision_transport=httpx.MockTransport(fake))
    with TestClient(app) as client:
        assert client.post('/v1/vision/jobs', json=body()).status_code == 401
        headers = {'Authorization':'Bearer offline-fixture-only'}
        accepted = client.post('/v1/vision/jobs', json=body(), headers=headers)
        assert accepted.status_code == 202
        job = accepted.json()['job_id']
        result = client.post('/v1/chat/completions', json=body(), headers=headers)
        assert result.status_code == 200
        assert result.headers['X-DGX-Vision-Job-ID'] == job
        assert len(fake.calls) == 1
        assert fake.calls[0]['reasoning_effort'] == 'high'
        assert fake.calls[0]['max_tokens'] == 6144
        assert client.get('/health').json()['vision_jobs']['states'] == {'succeeded':1}
        normalized, cap, *_ = app.state.engine.normalise_request({'model':'deepseek-v4.1-flash',
            'messages':[{'role':'user','content':'A text task'}]})
        assert normalized['reasoning_effort'] == 'max' and cap == 32768
        assert app.state.settings.max_concurrency == 12
        assert app.state.vision_jobs.ledger.max_running == 2
        import os
        assert os.environ['DS41F_RECIPE_VERSION'] == '1.2.3'
        # This release changes scheduling without expiring existing job identities.
        assert os.environ['DGX_VISION_JOBS_REVISION'] == 'd021774d38c0e7c684e6da44f39f22e735deb14b'
