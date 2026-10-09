import os
import subprocess
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(THIS_DIR, "src", "cuda_exl3", "csrc")


def _default_arch_list() -> str:
    """Pick the arch list to build for.

    Honour TORCH_CUDA_ARCH_LIST if the caller set it. Otherwise build only for
    the GPUs actually present, which keeps the (already long) nvcc time down
    during development. Falls back to the set of archs EXL3 kernels support at
    all if no device is visible at build time (e.g. a CI wheel builder).
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        import torch

        if torch.cuda.is_available():
            caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
            return ";".join(f"{maj}.{min}" for maj, min in sorted(caps))
    except Exception:
        pass
    # mma.m16n8k16 + cp.async are the floor -> sm_80 and up. 12.1 is GB10
    # (DGX Spark) and only exists from CUDA 12.8, so ask nvcc before naming it.
    archs = ["8.0", "8.6", "8.9", "9.0", "12.0"]
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        listed = subprocess.run(
            [os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc"),
             "--list-gpu-arch"], capture_output=True, text=True, timeout=30
        ).stdout
        if "compute_121" in listed:
            archs.append("12.1")
    except Exception:
        pass
    return ";".join(archs)


def _fallback_include_dir():
    """The pip-installed CUDA headers, if and only if the toolkit lacks some.

    Base images that ship torch against nvidia-* wheels can leave those include
    directories off the search path, so a build fails on a header the toolkit
    does not have. The trap is that such a wheel directory is not a few extra
    headers -- it is a *complete* CUDA header tree at a different patch level,
    so putting it on the include path ahead of the toolkit makes nvcc generate
    its device stub against one crt/host_runtime.h and compile it against the
    other ("__cudaLaunch passed 2 arguments, but takes just 1").

    So it goes on CPATH, which is searched after every -I, applies to both the
    device and host passes, and is exactly what a hand-set CPATH was doing. The
    toolkit wins every header it has; the wheel fills only genuine gaps.
    (nvcc rejects -idirafter outright, and passing it only via -Xcompiler would
    leave nvcc's own front-end search unfixed.)
    """
    from torch.utils.cpp_extension import CUDA_HOME

    # torch needs both of these; a toolkit can have one and not the other.
    wanted = ("cusparse.h", "cusolverDn.h")
    toolkit = os.path.join(CUDA_HOME or "/usr/local/cuda", "include")
    missing = [h for h in wanted if not os.path.exists(os.path.join(toolkit, h))]
    if not missing:
        return None
    import site
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        nv = os.path.join(root, "nvidia")
        if not os.path.isdir(nv):
            continue
        for entry in sorted(os.listdir(nv)):
            inc = os.path.join(nv, entry, "include")
            if all(os.path.exists(os.path.join(inc, h)) for h in missing):
                print(f"[cuda-exl3] toolkit lacks {', '.join(missing)}; "
                      f"adding {inc} to CPATH")
                return inc
    print(f"[cuda-exl3] warning: {', '.join(missing)} not found anywhere; "
          f"set CPATH if the build fails")
    return None


os.environ["TORCH_CUDA_ARCH_LIST"] = _default_arch_list()
print(f"[cuda-exl3] building for TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

_fallback_inc = _fallback_include_dir()
if _fallback_inc:
    _cpath = os.environ.get("CPATH", "")
    os.environ["CPATH"] = f"{_cpath}:{_fallback_inc}" if _cpath else _fallback_inc

sources = [
    os.path.join("src", "cuda_exl3", "csrc", f)
    for f in [
        "bindings.cpp",
        "hadamard.cu",
        "gemm.cu",
        "mla_decode.cu",
    ]
]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # 177: declared but never referenced (template instantiation fan-out)
    "-Xcudafe", "--diag_suppress=177",
]
if os.environ.get("CUDA_EXL3_DEBUG"):
    nvcc_flags += ["-g", "-G", "-DCUDA_EXL3_DEBUG"]

cxx_flags = ["-O3", "-std=c++17", "-g0"]

missing = [s for s in sources if not os.path.exists(s)]
if missing:
    # Lets the pure-Python plugin be installed before the kernels are written;
    # at runtime it falls back to the exllamav3 backend (see cuda_exl3/ops.py).
    print(f"[cuda-exl3] WARNING: skipping CUDA extension, missing sources: {missing}")
    ext_modules = []
    cmdclass = {}
else:
    ext_modules = [
        CUDAExtension(
            name="cuda_exl3._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
