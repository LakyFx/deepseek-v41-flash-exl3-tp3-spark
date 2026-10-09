"""Compile the experimental TP3 C ABI on an authorized SM121 toolchain host.

No GPU kernel is launched and no service/container is changed. Output must be a
new directory. Run only in a preparation environment; source compilation alone
does not qualify the kernel for production.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('retain existing build: '+str(out))
    compiler=shutil.which('nvcc')
    if not compiler:raise RuntimeError('CUDA13+ nvcc with SM121a support required')
    from torch.utils.cpp_extension import include_paths
    source=Path(__file__).resolve().with_name('cooperative_source')
    out.mkdir(parents=True)
    library=out/'cooperative_moe.so'
    command=[compiler,'-std=c++17','-O3','--shared','-Xcompiler=-fPIC',
             '--expt-relaxed-constexpr',
             '-gencode=arch=compute_121a,code=sm_121a','-I'+str(source),
             *['-I'+str(path) for path in include_paths()],str(source/'cooperative_moe.cu'),'-o',str(library)]
    proc=subprocess.run(command,text=True,capture_output=True)
    (out/'build.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError('compile failed; build.log retained')
    sources={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source.rglob('*') if p.is_file()}
    receipt={'abi':2,'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
             'sources':sources,'command':command,'gpu_numerically_qualified':False}
    (out/'cooperative_moe.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'compiled':True,'gpu_numerically_qualified':False}))

if __name__=='__main__':main()
