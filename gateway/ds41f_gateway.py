"""Portable entry point for the production gateway, with operator-owned state."""
from dataclasses import replace
from importlib import resources
import json
import os
from pathlib import Path

from laguna_gateway.app import create_app as _create_app
from laguna_gateway.config import Settings

SOURCE_REVISION = '8b4e06481aa7e29911026670e2718a7bd91d3c38'
RECIPE_VERSION = '1.2.3'
# Preserve implicit deduplication across this compatible scheduler-only upgrade.
VISION_JOBS_REVISION = 'd021774d38c0e7c684e6da44f39f22e735deb14b'


def production_settings():
    override = os.environ.get('DS41F_GATEWAY_PROFILE')
    raw = (Path(override).read_text(encoding='utf-8') if override else
           resources.files('laguna_gateway').joinpath('production-settings.json').read_text(encoding='utf-8'))
    profile = json.loads(raw)
    if {'api_key', 'upstream_base_url', 'vision_upstream_base_url', 'local_tokenizer_path'} & profile.keys():
        raise ValueError('Keep credentials and operator paths in environment variables')
    profile['served_model_aliases'] = tuple(profile['served_model_aliases'])
    settings = Settings(**profile)
    key = os.environ.get('LAGUNA_GATEWAY_API_KEY', '')
    if not key.strip():
        raise ValueError('LAGUNA_GATEWAY_API_KEY is required; generate your own secret')
    upstream = os.environ.get('LAGUNA_GATEWAY_UPSTREAM', 'http://127.0.0.1:8003').rstrip('/')
    vision = os.environ.get('LAGUNA_GATEWAY_VISION_UPSTREAM', upstream).rstrip('/')
    language = os.environ.get('LAGUNA_GATEWAY_ENFORCE_CZECH')
    settings = replace(settings, api_key=key, upstream_base_url=upstream, vision_upstream_base_url=vision,
        local_tokenizer_path=os.environ.get('LAGUNA_GATEWAY_LOCAL_TOKENIZER_PATH', ''),
        enforce_czech=settings.enforce_czech if language is None else language.lower() in ('1','true','yes','on'))
    settings.validate()
    return settings


def create_app(*, upstream_transport=None, vision_transport=None):
    settings = production_settings()
    os.umask(0o077)
    state = Path(os.environ.get('DGX_VISION_JOBS_DB') or (
        Path(os.environ.get('DS41F_GATEWAY_STATE_DIR', str(Path.home()/'.local/state/ds41f-gateway')))/'jobs.sqlite3'))
    state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == 'posix' and state.parent.stat().st_mode & 0o077:
        raise ValueError('Vision state directory must have mode0700')
    os.environ['DGX_VISION_JOBS_DB'] = str(state)
    os.environ.setdefault('DGX_VISION_JOBS_REVISION', VISION_JOBS_REVISION)
    os.environ.setdefault('DGX_MEDIA_SERIAL_REQUESTS', '1')
    os.environ.setdefault('DGX_MEDIA_MAX_CONCURRENCY', '2')
    os.environ.setdefault('DS41F_RECIPE_VERSION', RECIPE_VERSION)
    return _create_app(settings, upstream_transport=upstream_transport, vision_transport=vision_transport)


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get('DS41F_GATEWAY_BIND', '127.0.0.1'),
        port=int(os.environ.get('DS41F_GATEWAY_PORT', '18002')), workers=1,
        access_log=False, timeout_graceful_shutdown=15)


if __name__ == '__main__':
    main()
