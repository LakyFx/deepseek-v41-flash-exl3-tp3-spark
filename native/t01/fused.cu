// EXL3 input transform: A_had[g] = had128(x * suh[g]).
//
// ExLlamaV3 folds this into the front of its gemm and pays for it with a
// cooperative launch (the grid-wide sync between transform and matmul).
// Splitting it out costs one extra pass over the activations -- a few
// microseconds, since x is small and L2-resident -- and in exchange the gemm
// becomes an ordinary kernel: no cooperative launch, no grid size tied to the SM
// count, and nothing special to do under CUDA graph capture.
//
// A fused layer (qkv_proj, gate_up_proj) needs one transform per shard, because
// each shard was quantized with its own input scales. All of them are produced
// in a single launch here, loading x once and writing G outputs, rather than one
// launch and one re-read of x per shard.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"
#include "exl3_had.cuh"

namespace cuda_exl3 {
template <typename IN_T>
__global__ void exl3_moe_glu_had_in_t01_kernel(const IN_T* __restrict__ x,
                                           half* __restrict__ a_had,
                                           const half* __restrict__ suh,
                                           const int* __restrict__ expert_ids,
                                           const int* __restrict__ n_rows,
                                           int rows, int k, int block_m, float limit)
{
    int blocks_per_row = k / 128;
    long long total = (long long) rows * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    if (n_rows && row >= *n_rows) return;
    int blk = (int) (w % blocks_per_row);
    int lane = threadIdx.x & 31;

    int e = expert_ids[row / block_m];
    if (e < 0) return;                  // block belongs to no expert

    // x is (rows, 2k): gate in the first half of the row, up in the second.
    const IN_T* gate = x + (long long) row * 2 * k + blk * 128;
    had128_warp_glu_in_t01<IN_T>(gate, gate + k,
                             a_had + (long long) row * k + blk * 128,
                             suh + (long long) e * k + blk * 128, lane, limit);
}

void exl3_moe_glu_had_in_t01(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         int64_t block_m, double limit)
{
    const c10::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_moe_glu_had_in_t01: out must be float16");
    TORCH_CHECK(suh.dim() == 3 && suh.size(1) == 1,
                "exl3_moe_glu_had_in_t01: suh must be (experts, 1, k)");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "exl3_moe_glu_had_in_t01: expert_ids must be int32");
    TORCH_CHECK(x.dim() == 2, "exl3_moe_glu_had_in_t01: x must be 2-D (rows, 2k)");

    int rows = (int) x.size(0);
    TORCH_CHECK(x.size(1) % 2 == 0, "exl3_moe_glu_had_in_t01: x must have an even width");
    int k = (int) (x.size(1) / 2);
    TORCH_CHECK(k % 128 == 0, "exl3_moe_glu_had_in_t01: k must be a multiple of 128");
    TORCH_CHECK((int) suh.size(2) == k, "exl3_moe_glu_had_in_t01: suh k mismatch");
    TORCH_CHECK(out.numel() >= (long long) rows * k, "exl3_moe_glu_had_in_t01: out too small");
    TORCH_CHECK((long long) expert_ids.numel() * block_m >= rows,
                "exl3_moe_glu_had_in_t01: expert_ids covers ", expert_ids.numel() * block_m,
                " rows but x has ", rows);

    long long total_warps = (long long) rows * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int* nr = n_rows.numel() ? n_rows.data_ptr<int>() : nullptr;

    if (x.scalar_type() == at::kHalf)
        exl3_moe_glu_had_in_t01_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), expert_ids.data_ptr<int>(), nr,
            rows, k, (int) block_m, (float) limit);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_moe_glu_had_in_t01_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), expert_ids.data_ptr<int>(), nr,
            rows, k, (int) block_m, (float) limit);
    else
        TORCH_CHECK(false, "exl3_moe_glu_had_in_t01: x must be float16 or bfloat16");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_exl3
