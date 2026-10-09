import pathlib,subprocess,sys,re,glob,shutil,torch
src=pathlib.Path('/opt/dsv41-src')
p=src/'CMakeLists.txt';p.write_text(re.sub(r'(?m)^(\s*)include\(cmake/external_projects/',r'\1# Stable-only: include(cmake/external_projects/',p.read_text()))
nvrtc=glob.glob('/usr/local/cuda/lib64/libnvrtc.so*')+glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so*')
assert nvrtc
args=['cmake','-S',str(src),'-B',str(src/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DVLLM_TARGET_DEVICE=cuda','-DVLLM_PYTHON_EXECUTABLE='+sys.executable,'-DVLLM_PYTHON_PATH='+':'.join(x for x in sys.path if x),'-DFETCHCONTENT_SOURCE_DIR_CUTLASS=/opt/cutlass-stable','-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,'-DNVCC_THREADS=1','-DCUDA_nvrtc_LIBRARY='+nvrtc[0]]
subprocess.run(args,check=True)
subprocess.run(['cmake','--build',str(src/'build'),'--target','_C_stable_libtorch','-j','4'],check=True)
target=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
shutil.copytree(src/'vllm',target,dirs_exist_ok=True)
so=list((src/'build').glob('**/_C_stable_libtorch*.so'));assert len(so)==1,so
shutil.copy2(so[0],target/so[0].name)
