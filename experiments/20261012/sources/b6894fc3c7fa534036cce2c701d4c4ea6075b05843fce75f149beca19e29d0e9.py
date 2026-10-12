"""CPU-only ownership and all-rank evidence contract for future test runs."""
import hashlib
import json
import re

from .registry import CANDIDATES

CAMPAIGN = 'x11c-next-20261011'
PROTECTED = {'tp': 3, 'ep': False, 'max_context': 401408, 'prefill_chunk': 4096,
             'max_seqs': 8, 'kv_dtype': 'fp8_ds_mla', 'kv_bytes': 8074035200,
             'kv_block': 128, 'draft_depth': 5, 'adaptive': True}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def arm_for_version(version):
    prefix = '0.8.9X11c-'
    if not isinstance(version, str) or not version.startswith(prefix):
        raise ValueError('unknown next version')
    arm = version[len(prefix):]
    if arm != 'B0' and arm not in CANDIDATES:
        raise ValueError('unknown next arm')
    return arm


def authorize(payload, owner, env):
    data = json.loads(payload)
    if set(data) != {'campaign_id', 'plan_sha256', 'version', 'stage', 'nonce'}:
        raise ValueError('exact next qualification request required')
    if (data['campaign_id'] != CAMPAIGN
            or not re.fullmatch('[a-f0-9]{64}', data['plan_sha256'])
            or not re.fullmatch('[a-f0-9]{32}', data['nonce'])):
        raise ValueError('invalid next qualification identity')
    arm = arm_for_version(data['version'])
    if (owner.get('window_closed') is not True or owner.get('head_ready') is not True
            or owner.get('user_test_authorization') is not True
            or any(owner.get(k) != data[k] for k in ('campaign_id', 'plan_sha256'))
            or owner.get('active_version') != data['version']
            or env.get('DGX_V089_VERSION') != data['version']
            or env.get('DGX_NEXT_ARM') != arm or env.get('DGX_NEXT_CAMPAIGN') != CAMPAIGN
            or owner.get('fingerprint') != env.get('DGX_V089_FINGERPRINT')
            or not re.fullmatch('[a-f0-9]{64}', owner.get('container_id', ''))
            or not re.fullmatch('[a-f0-9]{64}', owner.get('fingerprint', ''))
            or not re.fullmatch('[a-f0-9]{64}', owner.get('arm_manifest_sha256', ''))):
        raise ValueError('qualification requires exact authorized closed-window owner')
    return data, arm


def validate_rows(request, response, expected):
    """Reject stale, partial, duplicate, wrong-source or mixed-generation RPCs.

    A response proves only the requested stage. It never makes an arm globally
    qualified or overrides another outstanding numerical/quality check.
    """
    rows = response.get('results') if isinstance(response, dict) else None
    if (not isinstance(rows, list) or len(rows) != 3
            or any(not isinstance(r, dict) or type(r.get('rank')) is not int for r in rows)
            or {r['rank'] for r in rows} != {0, 1, 2}):
        raise ValueError('exactly three unique actual worker receipts required')
    if set(expected) != {'0', '1', '2'}:
        raise ValueError('all-rank expected source/generation manifest required')
    for row in rows:
        pin = expected[str(row['rank'])]
        if (row.get('complete') is not True
                or any(row.get(k) != v for k, v in request.items())
                or any(row.get(k) != pin[k] for k in (
                    'container_id', 'fingerprint', 'arm_manifest_sha256',
                    'qualification_source_sha256', 'source_hashes', 'native_sha256'))
                or row.get('effective') != PROTECTED
                or type(row.get('kv_min_tokens')) is not int or row['kv_min_tokens'] < 3299563):
            raise ValueError('current-generation all-rank qualification mismatch')
    return sorted(rows, key=lambda row: row['rank'])
