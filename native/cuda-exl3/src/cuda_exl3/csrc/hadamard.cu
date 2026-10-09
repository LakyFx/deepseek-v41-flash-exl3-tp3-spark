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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"
#include "exl3_had.cuh"

namespace cuda_exl3 {

template <typename IN_T>
__global__ void exl3_had_in_kernel(const IN_T* __restrict__ x, half* __restrict__ a_had,
                                   const half* __restrict__ suh, int m, int k, int groups)
{
    int blocks_per_row = k / 128;
    long long total = (long long) m * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    int blk = (int) (w % blocks_per_row);
    long long off = (long long) row * k + blk * 128;
    int lane = threadIdx.x & 31;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(x + off, a_had + (long long) g * m * k + off,
                             suh + (long long) g * k + blk * 128, lane);
}

void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_had_in: out must be float16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf, "exl3_had_in: suh must be float16");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "exl3_had_in: needs contiguous tensors");

    int k = x.size(-1);
    int m = x.numel() / k;
    int groups = suh.dim() == 2 ? (int) suh.size(0) : 1;
    TORCH_CHECK(k % 128 == 0, "exl3_had_in: k must be a multiple of 128, got ", k);
    TORCH_CHECK(suh.numel() == (long long) groups * k, "exl3_had_in: suh size mismatch");
    TORCH_CHECK(out.numel() >= (long long) groups * m * k, "exl3_had_in: out too small");

    long long total_warps = (long long) m * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    half* o = (half*) out.data_ptr();
    const half* s = (const half*) suh.data_ptr();

    if (x.scalar_type() == at::kHalf)
        exl3_had_in_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), o, s, m, k, groups);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), o, s, m, k, groups);
    else
        TORCH_CHECK(false, "exl3_had_in: x must be float16 or bfloat16, got ", x.scalar_type());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// MoE variant: gather the routed rows and apply that row's expert's scales.
//
// Rows arrive already sorted by expert and padded so each expert's run is a
// whole multiple of block_m (vLLM's moe_align_block_size), which is what lets
// the GEMM treat the expert as uniform over a row block. Padding rows are
// zeroed here so they contribute nothing downstream.
// ---------------------------------------------------------------------------

template <typename IN_T>
__global__ void exl3_moe_had_in_kernel(const IN_T* __restrict__ x,
                                       half* __restrict__ a_had,
                                       const half* __restrict__ suh,
                                       const int* __restrict__ sorted_ids,
                                       const int* __restrict__ expert_ids,
                                       const int* __restrict__ n_rows,
                                       int rows, int k, int groups, int block_m,
                                       int top_k, int m_valid, bool skip_padding)
{
    // Row and 128-block come straight from the grid. They used to be recovered
    // from a flat warp index with a 64-bit divide and modulo -- and there is no
    // hardware 64-bit integer divide, so that was a software sequence per warp,
    // once for every (row, 128-block) pair: 65k of them at M=256 and 1.1M at a
    // 2048-token prefill chunk. The kernel was reading and writing at 57% of
    // what the device delivers with nothing else to blame.
    const int blocks_per_row = k / 128;
    const int warps_per_block = blockDim.x / 32;
    const int row = (int) blockIdx.x;
    const int blk = (int) blockIdx.y * warps_per_block + (int) (threadIdx.x >> 5);
    if (row >= rows || blk >= blocks_per_row) return;
    // Surplus rows from the worst-case padding are never read by the gemm
    // (it retires those blocks on the same count), so skip them entirely.
    if (n_rows && row >= *n_rows) return;
    int lane = threadIdx.x & 31;
    long long dst = (long long) row * k + blk * 128;

    // An empty sorted_ids means the rows are already in place (the second
    // projection consumes the first one's output), so the gather is the
    // identity and no index array needs building.
    // Retire a block that belongs to no expert before doing anything else. This
    // has to come above the padding branch, not below it: the gemm skips the
    // whole block, so nobody reads these rows and even writing zeros to them is
    // wasted bandwidth. Under expert parallel most blocks are another rank's,
    // and at small batch most rows are padding, so that was the bulk of this
    // kernel's traffic. Reported by @NNNtrance in #1.
    int e = expert_ids[row / block_m];
    if (e < 0) return;

    int idx = sorted_ids ? sorted_ids[row] : row;
    if (idx >= m_valid)
    {
        // Padding row. When the gemm skips fetching these -- cp.async zero-fills
        // a row it does not read -- writing zeros here is traffic nobody
        // consumes. Both sides take the same decision from the caller and use
        // the same predicate, sorted_ids[row] against m_valid, so they cannot
        // disagree; if the gemm is going to read the row, it needs the zeros.
        if (skip_padding) return;
        for (int g = 0; g < groups; ++g)
            ((half4*) (a_had + (long long) g * rows * k + dst))[lane] =
                half4{__float2half2_rn(0.f), __float2half2_rn(0.f)};
        return;
    }

    int token = idx / top_k;
    const IN_T* src = x + (long long) token * k + blk * 128;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(src, a_had + (long long) g * rows * k + dst,
                             suh + ((long long) e * groups + g) * k + blk * 128, lane);
}

void exl3_moe_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                     const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                     const at::Tensor& n_rows, int64_t block_m, int64_t top_k,
                     int64_t m_valid, bool skip_padding)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_moe_had_in: out must be float16");
    TORCH_CHECK(suh.dim() == 3, "exl3_moe_had_in: suh must be (experts, groups, k)");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "exl3_moe_had_in: expert_ids must be int32");
    TORCH_CHECK(!sorted_ids.numel() || sorted_ids.scalar_type() == at::kInt,
                "exl3_moe_had_in: sorted_ids must be int32");

    int k = (int) x.size(-1);
    int groups = (int) suh.size(1);
    int rows = (int) (sorted_ids.numel() ? sorted_ids.numel()
                                          : expert_ids.numel() * block_m);
    TORCH_CHECK(k % 128 == 0, "exl3_moe_had_in: k must be a multiple of 128");
    TORCH_CHECK((int) suh.size(2) == k, "exl3_moe_had_in: suh k mismatch");
    TORCH_CHECK(out.numel() >= (long long) groups * rows * k, "exl3_moe_had_in: out too small");
    TORCH_CHECK((long long) expert_ids.numel() * block_m >= rows,
                "exl3_moe_had_in: expert_ids covers ", expert_ids.numel() * block_m,
                " rows but sorted_ids has ", rows);

    // One block per (row, chunk of 128-blocks): the kernel then reads row and
    // block straight from the grid instead of recovering them from a flat warp
    // index with a 64-bit divide.
    const int threads = 256;
    const int bpr = k / 128;
    dim3 blocks((unsigned) rows,
                (unsigned) ((bpr + threads / 32 - 1) / (threads / 32)));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int* nr = n_rows.numel() ? n_rows.data_ptr<int>() : nullptr;
    const int* sids = sorted_ids.numel() ? sorted_ids.data_ptr<int>() : nullptr;

    if (x.scalar_type() == at::kHalf)
        exl3_moe_had_in_kernel<half><<<blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), (half*) out.data_ptr(), (const half*) suh.data_ptr(),
            sids, expert_ids.data_ptr<int>(), nr, rows, k, groups,
            (int) block_m, (int) top_k, (int) m_valid, skip_padding);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_moe_had_in_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), sids, expert_ids.data_ptr<int>(), nr, rows, k,
            groups, (int) block_m, (int) top_k, (int) m_valid, skip_padding);
    else
        TORCH_CHECK(false, "exl3_moe_had_in: x must be float16 or bfloat16");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combine routed rows back into per-token outputs.
//
// Replaces a python index_copy + broadcast-multiply + sum, which together cost
// ~9% of MoE decode GPU time and needed an (M*top_k+1, H) scratch buffer. Each
// (token, k) pair appears exactly once among the live rows, so inverting
// sorted_ids gives a direct gather -- no atomics, no scratch, deterministic.
// ---------------------------------------------------------------------------

__global__ void exl3_moe_build_inv_kernel(const int* __restrict__ sorted_ids,
                                          int* __restrict__ inv, int rows, int m_valid)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int idx = sorted_ids[r];
    if (idx < m_valid) inv[idx] = r;
}

template <typename T_>
__global__ void exl3_moe_combine_kernel(const T_* __restrict__ rows_out,
                                        const int* __restrict__ inv,
                                        const float* __restrict__ w,
                                        T_* __restrict__ out, int top_k, int H,
                                        const int* __restrict__ expert_ids,
                                        int block_m)
{
    int token = blockIdx.x;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
    {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            int r = inv[token * top_k + k];
            // Under expert parallel this pair may be routed to an expert another
            // rank owns. Every producer already skips those rows -- the gemm and
            // the glu transform both return on e < 0 -- so the row was never
            // written and must not be read: this rank simply contributes nothing
            // and the all-reduce takes the owner's value. Reading it and scaling
            // by the routing weight would not do, because NaN * 0 is NaN.
            if (expert_ids && expert_ids[r / block_m] < 0) continue;
            acc += w[token * top_k + k] * (float) rows_out[(size_t) r * H + h];
        }
        out[(size_t) token * H + h] = (T_) acc;
    }
}

at::Tensor exl3_moe_combine(const at::Tensor& rows_out, const at::Tensor& sorted_ids,
                            const at::Tensor& topk_weights, int64_t num_tokens,
                            const std::optional<at::Tensor>& expert_ids,
                            int64_t block_m)
{
    const at::cuda::OptionalCUDAGuard guard(rows_out.device());
    TORCH_CHECK(sorted_ids.scalar_type() == at::kInt, "exl3_moe_combine: sorted_ids int32");
    int H = (int) rows_out.size(1);
    int rows = (int) sorted_ids.numel();
    int M = (int) num_tokens;
    int top_k = (int) topk_weights.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto w = topk_weights.to(at::kFloat).contiguous();
    auto inv = at::empty({(long long) M * top_k},
                         at::TensorOptions().dtype(at::kInt).device(rows_out.device()));
    exl3_moe_build_inv_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sorted_ids.data_ptr<int>(), inv.data_ptr<int>(), rows, M * top_k);

    const int* eids = nullptr;
    if (expert_ids.has_value() && expert_ids->numel())
    {
        TORCH_CHECK(expert_ids->scalar_type() == at::kInt,
                    "exl3_moe_combine: expert_ids int32");
        TORCH_CHECK(block_m > 0, "exl3_moe_combine: block_m must be positive");
        eids = expert_ids->data_ptr<int>();
    }

    at::Tensor out = at::empty({M, H}, rows_out.options());
    int threads = H < 1024 ? ((H + 31) / 32) * 32 : 1024;
    if (rows_out.scalar_type() == at::kHalf)
        exl3_moe_combine_kernel<half><<<M, threads, 0, stream>>>(
            (const half*) rows_out.data_ptr(), inv.data_ptr<int>(), w.data_ptr<float>(),
            (half*) out.data_ptr(), top_k, H, eids, (int) block_m);
    else
        exl3_moe_combine_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
            (const __nv_bfloat16*) rows_out.data_ptr(), inv.data_ptr<int>(),
            w.data_ptr<float>(), (__nv_bfloat16*) out.data_ptr(), top_k, H,
            eids, (int) block_m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ---------------------------------------------------------------------------
// Fused SwiGLU + input transform for the MoE down-projection.
//
// The rows here are already in routed order (they are the first GEMM's output),
// so the gather is the identity and there is no sorted_ids, no top_k and only
// one group. Padding rows carry zeros through from the first transform, so
// silu(0)*0 = 0 needs no special case.
// ---------------------------------------------------------------------------
template <typename IN_T>
__global__ void exl3_moe_glu_had_in_kernel(const IN_T* __restrict__ x,
                                           half* __restrict__ a_had,
                                           const half* __restrict__ suh,
                                           const int* __restrict__ expert_ids,
                                           const int* __restrict__ n_rows,
                                           int rows, int k, int block_m)
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
    had128_warp_glu_in<IN_T>(gate, gate + k,
                             a_had + (long long) row * k + blk * 128,
                             suh + (long long) e * k + blk * 128, lane);
}

void exl3_moe_glu_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         int64_t block_m)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_moe_glu_had_in: out must be float16");
    TORCH_CHECK(suh.dim() == 3 && suh.size(1) == 1,
                "exl3_moe_glu_had_in: suh must be (experts, 1, k)");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "exl3_moe_glu_had_in: expert_ids must be int32");
    TORCH_CHECK(x.dim() == 2, "exl3_moe_glu_had_in: x must be 2-D (rows, 2k)");

    int rows = (int) x.size(0);
    TORCH_CHECK(x.size(1) % 2 == 0, "exl3_moe_glu_had_in: x must have an even width");
    int k = (int) (x.size(1) / 2);
    TORCH_CHECK(k % 128 == 0, "exl3_moe_glu_had_in: k must be a multiple of 128");
    TORCH_CHECK((int) suh.size(2) == k, "exl3_moe_glu_had_in: suh k mismatch");
    TORCH_CHECK(out.numel() >= (long long) rows * k, "exl3_moe_glu_had_in: out too small");
    TORCH_CHECK((long long) expert_ids.numel() * block_m >= rows,
                "exl3_moe_glu_had_in: expert_ids covers ", expert_ids.numel() * block_m,
                " rows but x has ", rows);

    long long total_warps = (long long) rows * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int* nr = n_rows.numel() ? n_rows.data_ptr<int>() : nullptr;

    if (x.scalar_type() == at::kHalf)
        exl3_moe_glu_had_in_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), expert_ids.data_ptr<int>(), nr,
            rows, k, (int) block_m);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_moe_glu_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), expert_ids.data_ptr<int>(), nr,
            rows, k, (int) block_m);
    else
        TORCH_CHECK(false, "exl3_moe_glu_had_in: x must be float16 or bfloat16");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_exl3
