"""Load attested ARM64 CUDA extensions without inference-time compilation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_extension(key, source):
    root = Path(__file__).resolve().parent
    receipt = root / 'prebuilt_extensions.json'
    if not receipt.exists():
        return None  # source-only development candidates still support builds
    record = json.loads(receipt.read_text())
    import torch
    if record.get('torch') != torch.__version__ or record.get('python') != list(sys.version_info[:2]):
        raise RuntimeError('prebuilt extension Python/Torch ABI differs')
    item = record['extensions'][key]
    binary = (root / item['binary']).resolve()
    if not binary.is_relative_to(root) or hashlib.sha256(binary.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('prebuilt extension artifact hash differs')
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != item['source_sha256']:
        raise RuntimeError('prebuilt extension kernel source differs')
    raw = binary.read_bytes()[:20]
    if raw[:6] != b'\x7fELF\x02\x01' or int.from_bytes(raw[18:20], 'little') != 183:
        raise RuntimeError('prebuilt extension requires ARM64 ELF64')
    spec = importlib.util.spec_from_file_location(item['module_name'], binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('DGX_V089_PREBUILT_LOADED ' + key + ' ' + item['sha256'], flush=True)
    return module
