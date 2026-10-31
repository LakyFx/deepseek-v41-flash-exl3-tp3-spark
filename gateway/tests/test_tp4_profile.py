"""Exercise the actual public entrypoint with the separate TP4 capacity profile."""
from pathlib import Path

import ds41f_gateway


def test_tp4_profile_is_effective_and_tp3_default_is_preserved(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv('LAGUNA_GATEWAY_API_KEY', 'local-test-key')
    monkeypatch.delenv('DS41F_GATEWAY_PROFILE', raising=False)
    before = ds41f_gateway.production_settings()
    assert before.max_concurrency == 12
    assert before.max_admitted_tokens == 5336064
    monkeypatch.setenv('DS41F_GATEWAY_PROFILE', str(root/'tensorfold/tp4/gateway-profile.json'))
    selected = ds41f_gateway.production_settings()
    assert selected.max_concurrency == 16
    assert selected.max_admitted_tokens == 8000000
    assert selected.max_model_len == before.max_model_len == 401408
    monkeypatch.delenv('DS41F_GATEWAY_PROFILE')
    assert ds41f_gateway.production_settings() == before
