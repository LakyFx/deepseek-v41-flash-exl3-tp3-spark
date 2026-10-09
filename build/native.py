"""Rebuild native components of the selected X11c profile from source."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys


def checked(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def build_c_libraries():
    runtime=Path('/usr/local/lib/python3.12/dist-packages/v089_runtime')
    for stem in ('native_engram','engram_dequant'):
        checked(['gcc','-O3','-std=c11','-shared','-fPIC','-o',str(runtime/('lib'+stem+'.so')),
                 str(runtime/(stem+'.c'))])
    source=Path('/opt/dgx_campaign/vendor/sparknet/sparknet/oneshot/_roce_proxy.c')
    output=Path('/opt/dgx_campaign/native/roce')
    output.mkdir(parents=True,exist_ok=True)
    # Sparknet itself selects the proxy cache name from the exact source hash.
    # Its normal loader performs ABI checks before starting any transport.
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    checked(['gcc','-O2','-std=gnu11','-shared','-fPIC','-o',str(output/('roce_proxy-'+digest+'.so')),
             str(source),'-libverbs','-lpthread'])


def build_exl3(root):
    source=root/'native/cuda-exl3'
    os.environ['TORCH_CUDA_ARCH_LIST']='12.1'
    output=Path('/opt/recipe-build/exl3')
    checked([sys.executable,'setup.py','build_ext','--build-temp','/opt/recipe-build/exl3-temp',
             '--build-lib',str(output)],cwd=source)
    binaries=list((output/'cuda_exl3').glob('_C*.so'))
    if len(binaries)!=1:raise RuntimeError('one ARM64 EXL3 extension required')
    import shutil
    destination=Path('/usr/local/lib/python3.12/dist-packages/cuda_exl3')/binaries[0].name
    shutil.copy2(binaries[0],destination)


def build_t01(root):
    os.environ['TORCH_CUDA_ARCH_LIST']='12.1a'
    from torch.utils.cpp_extension import load
    source=root/'native/t01'
    output=Path('/nds-t01/compiled');output.mkdir(parents=True,exist_ok=True)
    load(name='nds_t01_fused',sources=[str(source/'bindings.cpp'),str(source/'fused.cu')],
         extra_include_paths=[str(source)],extra_cflags=['-O3','-std=c++17','-g0'],
         extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','--expt-relaxed-constexpr',
                           '--expt-extended-lambda','-U__CUDA_NO_HALF_OPERATORS__',
                           '-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',
                           '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-Xcudafe','--diag_suppress=177'],
         build_directory=str(output),is_python_module=False,verbose=True)


def build_aot():
    """Compile TP3 specializations with fake pointers, never invoke a kernel."""
    if importlib.metadata.version('nvidia-cutlass-dsl')!='4.6.2':
        raise RuntimeError('CuTe DSL 4.6.2 is part of the measured contract')
    from cutlass.cute.runtime import make_fake_stream
    from sparknet.oneshot import _oneshot_cute as reduce, _allgather_cute as gather
    output=Path('/opt/dgx_campaign/native/roce-aot')
    objects=output/'objects';objects.mkdir(parents=True,exist_ok=True)
    reduce.current_cuda_stream=make_fake_stream
    gather.current_cuda_stream=make_fake_stream
    import cutlass.cute as cute
    exported=[]
    def export(launch,*args,name,cache_key):
        # Serving _compile.py intentionally loads an AOT object. The builder
        # uses the pinned upstream compiler directly to create those objects.
        compiled=cute.compile(launch,*args)
        key=json.dumps([name,list(cache_key)],separators=(',',':'))
        prefix='dgx_'+hashlib.sha256(key.encode()).hexdigest()[:24]
        compiled.export_to_c(str(objects),prefix,function_prefix=prefix)
        path=objects/(prefix+'.o')
        exported.append({'name':name,'cache_key':list(cache_key),'prefix':prefix,
                         'file':'objects/'+path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return compiled
    reduce.compile_kernel=export;gather.compile_kernel=export
    for rank in range(3):
        for dtype in ('float16','bfloat16','float32'):
            reduce.get_launcher(dtype,3,rank,512,2,128,2,0)
        gather.get_launcher(3,rank,512,2,128,2,0)
    value={'schema':'dgx.cute-aot-objects.v1','arch':'sm_121a',
           'dependency_versions':{name:importlib.metadata.version(name)
                                  for name in ('nvidia-cutlass-dsl','cuda-bindings','torch')},
           'objects':exported,'launches':0,'runtime_qualified':False}
    (output/'AOT-MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n')


def build_extensions():
    runtime=Path('/usr/local/lib/python3.12/dist-packages/v089_runtime')
    os.environ['TORCH_EXTENSIONS_DIR']='/opt/recipe-build/extensions-cache'
    checked([sys.executable,str(Path(__file__).with_name('extensions.py')),
             '--runtime',str(runtime),'--output','/opt/recipe-build/extensions'])
    import shutil
    output=Path('/opt/recipe-build/extensions')
    for p in output.iterdir():
        if p.name.endswith('.so') or p.name=='prebuilt_extensions.json':
            shutil.copy2(p,runtime/p.name)


def verify_versions(root):
    expected=json.loads((root/'config/dependency-versions.json').read_text())
    observed={name:importlib.metadata.version(name) for name in expected}
    if observed!=expected:
        raise RuntimeError('dependency versions differ: '+json.dumps(observed))
    print(json.dumps(observed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/opt/recipe'))
    parser.add_argument('--part',required=True,choices=['c','exl3','t01','aot','extensions','versions'])
    args=parser.parse_args()
    os.environ['MAX_JOBS']='1'
    os.environ['CUTE_DSL_ARCH']='sm_121a'
    if args.part=='versions':verify_versions(args.root)
    elif args.part=='c':build_c_libraries()
    elif args.part=='exl3':build_exl3(args.root)
    elif args.part=='t01':build_t01(args.root)
    elif args.part=='aot':build_aot()
    else:build_extensions()


if __name__=='__main__':main()
