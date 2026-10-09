FROM vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_HAS_FLASHINFER_CUBIN=1 FLASHINFER_NVCC_THREADS=1 MAX_JOBS=4
RUN mkdir -p /opt/dsv41-src /opt/cutlass-stable && curl -fL --retry 5 https://codeload.github.com/vllm-project/vllm/tar.gz/e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba | tar xz -C /opt/dsv41-src --strip-components=1 && curl -fL --retry 5 https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.7.1 | tar xz -C /opt/cutlass-stable --strip-components=1
COPY build_stable.py /opt/build_stable.py
RUN python3 /opt/build_stable.py
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin flashinfer-python
RUN mkdir -p /opt/fi-src && curl -fL --retry 5 https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/07869c61ba581e6d6b8ad8d142f4a6c89b707cc1 | tar xz -C /opt/fi-src --strip-components=1
RUN mkdir -p /opt/fi-src/3rdparty/{cutlass,cccl,spdlog} && curl -fL --retry 5 https://codeload.github.com/NVIDIA/cutlass/tar.gz/b46b16d003484063bca4ed365e44095c4c6ed633 | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1 && curl -fL --retry 5 https://codeload.github.com/NVIDIA/cccl/tar.gz/16bd510c9b712e82b0ab6cbb630d8e29ba1f7116 | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1 && curl -fL --retry 5 https://codeload.github.com/gabime/spdlog/tar.gz/c3aed4b68373955e1cc94307683d44dca1515d2b | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1
RUN cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation .
COPY upstream-build/prewarm5.py /opt/prewarm5.py
RUN python3 /opt/prewarm5.py
RUN mkdir -p /opt/cuda-exl3 && curl -fL --retry 5 https://codeload.github.com/Zeuss5/cuda-exl3/tar.gz/6a1ffc34866e23f484574ce1922a8bca93eb33b2 | tar xz -C /opt/cuda-exl3 --strip-components=1 && cd /opt/cuda-exl3 && MAX_JOBS=2 pip install --no-deps --no-build-isolation .
COPY patch /opt/dsv41-patch
COPY apply_patches.py /opt/apply_patches.py
RUN python3 /opt/apply_patches.py && python3 -c "import vllm,flashinfer,cuda_exl3,cuda_exl3._C; print(vllm.__version__,flashinfer.__version__)"
LABEL dgx.recipe.revision="fc725ecf10869c184f4347dd73336536d395753c" dgx.vllm.revision="e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba"
