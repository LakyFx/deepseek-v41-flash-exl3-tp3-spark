"""Do not let portable-entrypoint defaults leak into legacy transport fixtures."""
import pytest


@pytest.fixture(autouse=True)
def isolate_portable_defaults(monkeypatch):
    for name in ('DGX_VISION_CONTROL_ENABLED', 'DGX_VISION_FINAL_ANSWER_TOKENS',
                 'DGX_VISION_JOB_TIMEOUT_SECONDS', 'DGX_VISION_QUEUE_TIMEOUT_SECONDS',
                 'DGX_VISION_CANCEL_GRACE_SECONDS'):
        monkeypatch.delenv(name, raising=False)
