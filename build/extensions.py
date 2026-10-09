"""Compile CUDA extensions beside production without launching GPU kernels.

Run in the exact pinned image, with runtime mounted at the final Python path
and TORCH_EXTENSIONS_DIR pointing to the candidate's independent cache.
No model weights or service management. Keep the build container and outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess

MODULES = [('moe_prep', '_build', 'moe_prep_kernel.cu'),
           ('dense_gemv', '_ext', 'dense_gemv_kernel.cu')]


class CompilePlanReady(BaseException):
    """Exit before compilation, through runtime handlers for ordinary errors."""


def plan(runtime, output):
    import importlib
    from torch.utils import cpp_extension
    def capture(build_directory, *args, **kwargs):
        # cpp_extension has already written its exact version's Ninja file.
        raise CompilePlanReady(build_directory)
    cpp_extension._run_ninja_build = capture
    directories = []
    for module, builder, _source in MODULES:
        implementation = importlib.import_module(runtime.name + '.' + module)
        try:
            getattr(implementation, builder)()
        except CompilePlanReady as ready:
            directories.append(str(ready.args[0]))
        else:
            raise RuntimeError('planner unexpectedly loaded an existing extension')
    (output / 'compile-plan.json').write_text(json.dumps(directories, indent=2))


def load_and_record(runtime, output):
    import importlib
    import torch
    outputs = {}
    for module, builder, source in MODULES:
        implementation = importlib.import_module(runtime.name + '.' + module)
        extension = getattr(implementation, builder)()
        path = Path(extension.__file__)
        copied = output / (module + '_ext.so')
        shutil.copy2(path, copied)
        outputs[module] = {'binary': copied.name, 'module_name': extension.__name__, 'build_path': str(path),
                           'sha256': hashlib.sha256(copied.read_bytes()).hexdigest(),
                           'source_sha256': hashlib.sha256((runtime / source).read_bytes()).hexdigest()}
    (output / 'prebuilt_extensions.json').write_text(json.dumps({'architecture': 'sm_121a', 'extensions': outputs,
                                  'torch': torch.__version__, 'python': list(sys.version_info[:2]),
                                  'gpu_numerically_qualified': False, 'split_build_processes': True,
                                  'cache': os.environ['TORCH_EXTENSIONS_DIR']}, indent=2), encoding='utf-8')
    print(json.dumps({'compiled': list(outputs), 'gpu_numerically_qualified': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['plan', 'load'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.phase:
        raise FileExistsError('retain existing build receipt')
    if not os.environ.get('TORCH_EXTENSIONS_DIR'):
        raise ValueError('an independent extension cache must be explicit')
    os.environ['MAX_JOBS'] = '1'
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.1a'
    sys.path.insert(0, str(args.runtime.parent))
    if args.phase == 'plan':
        plan(args.runtime, output)
        return
    if args.phase == 'load':
        load_and_record(args.runtime, output)
        return
    output.mkdir(parents=True)
    child = [sys.executable, '-u', str(Path(__file__).resolve()), '--runtime', str(args.runtime), '--output', str(output)]
    subprocess.run(child + ['--phase', 'plan'], check=True)
    # Torch has exited. The parent remains stdlib-only while NVCC compiles.
    for directory in json.loads((output / 'compile-plan.json').read_text()):
        subprocess.run(['ninja', '-v', '-j1', '-C', directory], check=True)
    subprocess.run(child + ['--phase', 'load'], check=True)


if __name__ == '__main__':
    main()
