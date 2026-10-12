"""Future test-only RPC. No model/GPU work at import and no public CLI."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from .qualification_protocol import authorize, arm_for_version, PROTECTED

LEASE = Path('/opt/dgx_campaign/lease/lease.json')
MANIFEST = Path('/opt/dgx_next/CANDIDATE.json')
NATIVE = Path('/usr/local/lib/python3.12/dist-packages/cuda_exl3/_C.cpython-312-aarch64-linux-gnu.so')


def source_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('worker source escapes candidate')
    for prefix, root in (
        ('vllm/', '/usr/local/lib/python3.12/dist-packages/'),
        ('dgx_campaign_runtime/', '/opt/dgx_campaign/runtime/'),
        ('b12x/', '/opt/dgx_campaign/vendor/b12x-package/'),
        ('dgx_next_runtime/', '/opt/dgx_next/runtime/')):
        if relative.startswith(prefix):
            return Path(root+relative)
    if relative == 'native/gemm.cu':
        return None
    raise ValueError('unreviewed worker source path')


def attest(owner):
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != owner['arm_manifest_sha256']:
        raise ValueError('candidate manifest changed')
    manifest = json.loads(raw)
    if manifest['candidate'] != arm_for_version(owner['active_version']):
        raise ValueError('wrong candidate manifest')
    hashes = {}
    for relative, wanted in manifest['files'].items():
        path = source_path(relative)
        if path is None:
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError('actual worker overlay differs: '+relative)
        hashes[relative] = actual
    native = None
    if 'N4' in manifest['parts']:
        native = hashlib.sha256(NATIVE.read_bytes()).hexdigest()
        if native != owner.get('native_sha256'):
            raise ValueError('worker native extension differs')
    elif owner.get('native_sha256') is not None:
        raise ValueError('unexpected native override')
    return manifest, hashes, native


def run(worker, payload):
    if LEASE.is_symlink() or not LEASE.is_file():
        raise RuntimeError('owned test-window lease missing')
    owner = json.loads(LEASE.read_bytes())
    request, arm = authorize(payload, owner, os.environ)
    manifest, sources, native = attest(owner)
    stage = request['stage']
    if stage not in {'contract', 'adaptive', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7a', 'N7b', 'N8a', 'N8b', 'N9', 'trace_begin', 'trace_end'}:
        # Unsupported stages must never become a silent successful no-op.
        raise ValueError('next worker gate not implemented: '+stage)
    if stage in {'N1','N2','N3','N4','N5','N6','N7a','N7b','N8a','N8b','N9'} and stage not in manifest['parts']:
        raise ValueError(stage+' gate on an arm without '+stage)
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank
    from dgx_campaign_runtime import worker_checks as existing

    with set_current_vllm_config(worker.vllm_config), torch.no_grad():
        contract = existing.contract(worker)
        if contract['effective'] != PROTECTED:
            raise ValueError('effective next model parameters differ')
        if stage == 'contract':
            result = {'complete': True}
        elif stage == 'adaptive':
            result = existing.adaptive_gate(worker)
        elif stage == 'N1':
            from .speculation_gate import adaptive
            result = adaptive(worker)
        elif stage == 'N9':
            from .graph_gate import inventory
            result = inventory(worker.model_runner)
        elif stage in ('trace_begin', 'trace_end'):
            if arm != 'B0':
                raise ValueError('shape selection must use the baseline generation')
            from . import trace
            result = (trace.begin(get_tensor_model_parallel_rank(), request['nonce'])
                      if stage == 'trace_begin' else trace.end())
        else:
            available = next(int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
            if available < 2*1024**3:
                raise RuntimeError('insufficient memory reserve for numerical gate')
            if stage == 'N2':
                from .mhc_gate import run as mhc_gate
                result = mhc_gate(worker)
            elif stage == 'N6':
                from .replay_gate import run as replay_gate
                result = replay_gate(worker)
            elif stage in ('N7a','N7b'):
                from .row_gate import run as row_gate
                result = row_gate(worker,stage)
            elif stage in ('N8a','N8b'):
                from .dense_gate import run as dense_gate
                result = dense_gate(worker)
            elif stage == 'N3':
                from .speculation_gate import draft
                result = draft(worker)
            elif stage == 'N5':
                from .indexer_gate import run as indexer_gate
                result = indexer_gate(worker)
            else:
                from .native_gate import run as native_gate
                result = native_gate(worker)
        torch.cuda.synchronize()
    return {**result, **request, 'arm': arm, **contract,
            'rank': get_tensor_model_parallel_rank(),
            'container_id': owner['container_id'], 'fingerprint': owner['fingerprint'],
            'arm_manifest_sha256': owner['arm_manifest_sha256'],
            'qualification_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_hashes': sources, 'native_sha256': native}
