ARG BASE_IMAGE=dsv41-x11c-base:local
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV MAX_JOBS=1 CUTE_DSL_ARCH=sm_121a TORCH_CUDA_ARCH_LIST=12.1a
RUN apt-get update && apt-get install -y --no-install-recommends libibverbs-dev gcc g++ make
COPY runtime/rootfs/ /
COPY native/ /opt/recipe/native/
COPY build/native.py /opt/recipe/build/native.py
COPY build/extensions.py /opt/recipe/build/extensions.py
COPY config/dependency-versions.json /opt/recipe/config/dependency-versions.json
ENV PYTHONPATH=/opt/dgx_campaign/vendor/b12x-package:/opt/dgx_campaign/vendor/sparknet:/opt/dgx_campaign/runtime
RUN ln -s /usr/local/lib/python3.12/dist-packages/v089_runtime /opt/dgx-v089 && python3 /opt/recipe/build/native.py --part versions
RUN python3 /opt/recipe/build/native.py --part c
RUN python3 /opt/recipe/build/native.py --part exl3
RUN python3 /opt/recipe/build/native.py --part t01
RUN python3 /opt/recipe/build/native.py --part extensions
RUN python3 /opt/recipe/build/native.py --part aot
RUN mkdir -p /opt/nccl-src /opt/nccl/lib && curl -fL --retry 5 https://codeload.github.com/NVIDIA/nccl/tar.gz/refs/tags/v2.30.7-1 | tar xz -C /opt/nccl-src --strip-components=1
RUN make -C /opt/nccl-src -j2 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && cp -a /opt/nccl-src/build/lib/libnccl.so* /opt/nccl/lib/
RUN cp -L /opt/nccl/lib/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
ENV VLLM_NCCL_SO_PATH=/opt/nccl/lib/libnccl.so.2 LD_PRELOAD=/opt/nccl/lib/libnccl.so.2
LABEL org.opencontainers.image.source="https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark" dgx.recipe.version="0.8.9X11c"
