"""CPU-only binary and manifest check inside a built ARM64 recipe image.

This registers CUDA operators but does not initialize a GPU or execute kernels.
It is not a numerical GPU qualification or model startup test.
"""
from pathlib import Path
import ctypes
import hashlib
import importlib.metadata
import json
import sys


def main():
    expected = json.loads(Path('/opt/recipe/config/dependency-versions.json').read_text())
    actual = {key: importlib.metadata.version(key) for key in expected}
    if actual != expected:
        raise ValueError('dependency versions differ')
    import torch
    # Load the exact new extension directly; no model/tokenizer dependency.
    extension = list(Path('/usr/local/lib/python3.12/dist-packages/cuda_exl3').glob('_C*.so'))
    if len(extension) != 1:
        raise ValueError('one EXL3 extension is required')
    torch.ops.load_library(str(extension[0]))
    torch.ops.load_library('/nds-t01/compiled/nds_t01_fused.so')
    operations = ['exl3_linear', 'exl3_moe_gemm', 'exl3_moe_had_in',
                  'exl3_moe_glu_had_in_t01', 'mla_decode']
    for operation in operations:
        if not torch._C._dispatch_has_kernel_for_dispatch_key('cuda_exl3_C::' + operation, 'CUDA'):
            raise ValueError('CUDA registration absent: ' + operation)
    from v089_runtime.prebuilt import load_extension
    runtime = Path('/usr/local/lib/python3.12/dist-packages/v089_runtime')
    for module, source in [('moe_prep', 'moe_prep_kernel.cu'), ('dense_gemv', 'dense_gemv_kernel.cu')]:
        if load_extension(module, runtime / source) is None:
            raise ValueError('prebuilt extension absent')
    for filename, symbol in [('libnative_engram.so', 'dgx_engram_abi'),
                             ('libengram_dequant.so', 'dgx_engram_dequant_abi')]:
        library = ctypes.CDLL(str(runtime / filename))
        abi = getattr(library, symbol)
        abi.restype = ctypes.c_int
        if abi() != 1:
            raise ValueError('Engram native ABI differs')
    nccl = ctypes.CDLL('/opt/nccl/lib/libnccl.so.2')
    value = ctypes.c_int()
    if nccl.ncclGetVersion(ctypes.byref(value)) != 0 or value.value != 23007:
        raise ValueError('NCCL is not 2.30.7')
    from dgx_campaign_runtime.aot_transport import resolve
    root = Path('/opt/dgx_campaign/native/roce-aot')
    raw = (root / 'AOT-MANIFEST.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if len(manifest['objects']) != 12:
        raise ValueError('twelve TP3 AOT specializations are required')
    for row in manifest['objects']:
        resolve(root, digest, row['name'], row['cache_key'])
    if torch.cuda.is_initialized():
        raise ValueError('CPU qualification unexpectedly initialized CUDA')
    print(json.dumps({'schema': 'dsv41.public-image-cpu-qualification.v1',
                      'dependency_versions': actual, 'cuda_operator_registrations': operations,
                      'prebuilt_extensions': ['moe_prep', 'dense_gemv'], 'engram_abi': 1,
                      'nccl_version': value.value, 'aot_objects': len(manifest['objects']),
                      'aot_manifest_sha256': digest, 'gpu_initialized': False,
                      'gpu_kernels_executed': 0, 'model_loaded': False,
                      'numerical_gpu_qualification': False}))


if __name__ == '__main__':
    main()
