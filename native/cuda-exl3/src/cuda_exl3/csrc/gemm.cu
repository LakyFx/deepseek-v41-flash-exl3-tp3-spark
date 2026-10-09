// EXL3 GEMM with M-tiling.
//
// ExLlamaV3's gemm fixes TILESIZE_M at 16 and loops over the batch in 16-row
// chunks, re-reading the whole trellis for every chunk. That is optimal for
// single-stream decode but makes cost linear in the batch size: on an RTX PRO
// 6000 the kernel sits at ~1.65 TB/s (i.e. HBM-bound) from m=24 upward, so a
// 512-row batch reads a 47 MB tensor 32 times.
//
// vLLM's continuous batching lives exactly in that range, so this kernel tiles
// M instead: one block owns BM rows x BN columns and the full k extent, reads
// its slice of the trellis *once*, and amortizes the dequant over all BM rows.
// Owning the full k extent also means the block holds a complete output row
// segment, so the output Hadamard and svh scaling fold into the epilogue with
// no second pass and no k x n scratch buffer.
//
// The trellis decode itself (exl3_dq.cuh) and the fragment layout come from
// ExLlamaV3 and are part of the on-disk format.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

#include "exl3_common.cuh"
#include "exl3_dq.cuh"
#include "exl3_had.cuh"

namespace cuda_exl3 {

// Output Hadamard block size; BN must equal this so a block owns whole blocks.
constexpr int HAD_N = 128;

// Which shard of a fused layer each BN-wide column block belongs to. Shards are
// contiguous column ranges, so a block just walks the (at most 8) boundaries.
// Only the activation slice differs per shard -- trellis, svh and the output are
// all addressed by absolute column -- so one launch can cover a whole qkv_proj.
struct ShardMap
{
    int nblk_end[8];
    int n_groups;
};

// Accumulator policy.
//
// MEASURED AND REJECTED (kept opt-in so the result is not re-derived): fp16
// accumulation halves the accumulator registers, which affords BM=256 and so
// twice the MMA work per dequantized weight -- the obvious lever against this
// kernel's dequant-vs-MMA issue pressure. It does not pay off. BM=256 needs 145
// registers and 69 KB of shared, dropping to 1 block/SM, and that occupancy loss
// cancels the gain exactly: 8192-row q_proj went 3518 -> 3599 us, and 512-row
// down_proj regressed 403 -> 694 us. Relative error meanwhile rose 8-16x
// (3.5e-4 -> 2.6e-3, and 4.9e-3 on down_proj where k=17408 gives the most
// accumulation steps). fp32 accumulation stays the default.
//
// (There is no bf16 accumulator to try instead: bf16 mma inputs always
// accumulate to fp32.)
template <bool H_ACC>
struct Acc;

template <>
struct Acc<false>
{
    using T = FragC;
    __device__ __forceinline__ static void zero(T& c)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    __device__ __forceinline__ static void mma(const FragA& a, const FragB& b, T& c)
    {
        mma_m16n8k16(a, b, c);
    }
    __device__ __forceinline__ static half geth(const T& c, int i)
    {
        return __float2half(c[i]);
    }
    __device__ __forceinline__ static float getf(const T& c, int i) { return c[i]; }
};

template <>
struct Acc<true>
{
    using T = FragC_h;
    __device__ __forceinline__ static void zero(T& c)
    {
        c[0] = __float2half2_rn(0.0f);
        c[1] = __float2half2_rn(0.0f);
    }
    __device__ __forceinline__ static void mma(const FragA& a, const FragB& b, T& c)
    {
        mma_m16n8k16_h(a, b, c);
    }
    // Element i of the mma D fragment: regs pack (0,1) and (2,3).
    __device__ __forceinline__ static half geth(const T& c, int i)
    {
        half2 h = c[i >> 1];
        return (i & 1) ? __high2half(h) : __low2half(h);
    }
    __device__ __forceinline__ static float getf(const T& c, int i)
    {
        return __half2float(geth(c, i));
    }
};

// WARP_N is the width of a warp's tile. Each 16x16 trellis tile is decoded by
// exactly one warp, so a narrower warp tile means fewer warps decode the same
// tile: with WARP_N=16 and one warp row, every tile is dequantized once per
// block instead of WARPS_M times. That matters at small batch sizes, where the
// kernel is dequant-bound rather than memory-bound.
template <int BITS, int CB, int BM, int BN, int BK, int NWARPS, int STAGES, int WARP_N_>
struct GemmCfg
{
    static constexpr int NTHREADS = NWARPS * 32;
    static constexpr int WARPS_N = BN / WARP_N_;
    static constexpr int WARPS_M = NWARPS / WARPS_N;
    static constexpr int WARP_M = BM / WARPS_M;
    static constexpr int WARP_N = WARP_N_;
    static constexpr int MBLK = WARP_M / 16;             // m16n8k16 blocks
    static constexpr int NBLK = WARP_N / 8;
    static constexpr int KSTEPS = BK / 16;
    static constexpr int A_COLS = BK / 8;                // int4 per A row
    // ldmatrix reads 8 rows at one column offset, so those 8 rows must land on
    // 8 distinct bank groups.
    //
    // With BK=64 an A row is exactly 128 B = 32 banks, and an 8-way XOR swizzle
    // achieves that with no padding at all (this is what Marlin does). With
    // BK=32 a row is only 64 B, giving 4 columns to permute -- not enough for 8
    // rows, and no XOR can fix it -- so there we pad the stride by one 16-byte
    // element instead (80 B -> banks 0,20,8,28,16,4,24,12).
    static constexpr bool A_SWIZZLE = (A_COLS >= 8);
    static constexpr int A_STRIDE = A_SWIZZLE ? A_COLS : A_COLS + 1;
    static constexpr int NTILES = BN / 16;               // B tiles per k step
    static constexpr int TILE_U32 = 8 * BITS;            // uint32 per 16x16 tile
    static constexpr int TILE_I4 = 2 * BITS;             // int4 per 16x16 tile

    static constexpr int SH_A_I4 = BM * A_STRIDE;        // int4
    static constexpr int SH_B_I4 = KSTEPS * NTILES * TILE_I4;
    static constexpr int SH_STAGE_I4 = SH_A_I4 + SH_B_I4;

    static constexpr int C_STRIDE = BN + 8;              // pad: bank conflicts
    static constexpr int SH_C_BYTES = BM * C_STRIDE * 2;
    static constexpr int SH_PIPE_BYTES = STAGES * SH_STAGE_I4 * 16;

    // Split-k stages fp32 partials in shared so the atomics into the global
    // accumulator come out coalesced. Only worth the shared memory for the small
    // BM values split-k actually uses; BM=128 falls back to direct atomics.
    static constexpr bool SPLIT_STAGED = BM <= 64;
    static constexpr int F_STRIDE = BN + 4;
    static constexpr int SH_F_BYTES = SPLIT_STAGED ? BM * F_STRIDE * 4 : 0;

    static constexpr int MAX2(int a, int b) { return a > b ? a : b; }
    static constexpr int SMEM = MAX2(MAX2(SH_PIPE_BYTES, SH_C_BYTES), SH_F_BYTES);
};

// SPLIT: this block covers only part of k, so it accumulates fp32 partials into
// `acc` and a later pass does the Hadamard. Splitting k is how narrow-n layers
// (Qwen3.5's down_proj is only 40 blocks wide at BN=128) and small batches keep
// all 188 SMs busy. Unlike shrinking BN it adds blocks without multiplying the
// number of times A is re-read.
template <int BITS, int CB, int BM, int BN, int BK, int NWARPS, int STAGES, bool SPLIT,
          typename OUT_T, int WARP_N_, bool H_ACC>
__global__ __launch_bounds__(NWARPS * 32) void exl3_gemm_m_kernel(
    const half* __restrict__ A,        // (groups, m, k), Hadamard-transformed
    const uint16_t* __restrict__ Bq,   // (k/16, n/16, 16*BITS) trellis
    OUT_T* __restrict__ C,             // (m, ldc)
    const half* __restrict__ svh,      // (n_full,)
    int m, int k, int n, int ldc,
    int n_off,                         // first column of this shard, in features
    int n_tiles_full,                  // trellis dim-1 extent, i.e. the row stride
    float* __restrict__ acc,           // (m, ldc) fp32 partials, SPLIT only
    int kt_per_split,
    ShardMap smap,
    const int* __restrict__ expert_ids,  // MoE: one expert per BM-row block
    int64_t b_expert_stride,             // uint16 elements between experts
    int64_t svh_expert_stride,
    const int* __restrict__ n_rows,      // MoE: live row count, device-side
    // Fused combine. When sorted_ids is non-null this gemm finishes the MoE by
    // itself: each routed row is scaled by its routing weight and accumulated
    // into its token's row of C, so C is (tokens, n) rather than (rows, n) and
    // the separate combine kernel -- plus the (rows, n) tensor it read -- go
    // away entirely.
    const int* __restrict__ moe_sorted_ids,
    const float* __restrict__ moe_weights,
    int moe_top_k, int moe_m_valid)
{
    using Cfg = GemmCfg<BITS, CB, BM, BN, BK, NWARPS, STAGES, WARP_N_>;

    extern __shared__ __align__(16) int4 smem[];

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int warp_m = warp / Cfg::WARPS_N;
    const int warp_n = warp % Cfg::WARPS_N;

    // Column ranges are expressed relative to the shard, then offset by n_off.
    // Fused layers (qkv_proj, gate_up_proj) keep ONE trellis tensor covering
    // every shard, so slicing it in python would produce a non-contiguous view
    // whose real row stride the kernel cannot see. Pass the offset and the full
    // dim-1 extent instead, and write straight into the merged output.
    const int n0 = blockIdx.x * BN;          // first output column, shard-relative
    const int m0 = blockIdx.y * BM;          // first output row
    const int kt_total = k / BK;
    const int kt_begin = SPLIT ? (int) blockIdx.z * kt_per_split : 0;
    const int kt_end = SPLIT ? min(kt_begin + kt_per_split, kt_total) : kt_total;

    // Pick this block's shard, and with it the activation slice to read.
    int grp = 0;
    while (grp + 1 < smap.n_groups && (int) blockIdx.x >= smap.nblk_end[grp]) ++grp;
    A += (size_t) grp * m * k;

    // MoE: every row of a BM block belongs to the same expert (the caller pads
    // each expert's token run out to a multiple of BM), so the expert -- and
    // with it the weight tensor -- is uniform across the block.
    if (expert_ids)
    {
        // The alignment pass sizes its output for the worst case (every expert
        // padded out to a full block), but only reports the live row count on
        // the device. Retire the surplus blocks immediately rather than sync to
        // find out how many there are.
        if (n_rows && m0 >= *n_rows) return;
        int e = expert_ids[blockIdx.y];
        // moe_align_block_size marks blocks that belong to no expert with -1,
        // which under expert parallel also covers every block routed to an
        // expert another rank owns. Nothing downstream reads these rows: the
        // glu transform and exl3_moe_combine both skip them on the same
        // predicate, so the rank contributes nothing and the all-reduce takes
        // the owner's value. Leaving them unwritten is what makes that cheap --
        // zeroing them here cost a full-width store per dead row, and under EP
        // three quarters of the rows are dead.
        if (e < 0) return;
        Bq += (size_t) e * b_expert_stride;
        svh += (size_t) e * svh_expert_stride;
    }

    // Which of this thread's rows are real, worked out once. The staging loop
    // runs k/BK times over the same BM*A_COLS elements and each thread always
    // draws the same ones, so the test is loop-invariant per thread. Doing it
    // inline cost a dependent global load per element ahead of every cp.async
    // (-4% at one token); doing it once in shared needed a barrier before the
    // pipeline prologue, which cost about as much again at that size. Registers
    // need neither.
    constexpr int A_ELEMS = (BM * Cfg::A_COLS + Cfg::NTHREADS - 1) / Cfg::NTHREADS;
    bool real_row[A_ELEMS];
#pragma unroll
    for (int j = 0; j < A_ELEMS; ++j)
    {
        const int i = t + j * Cfg::NTHREADS;
        const int grow = m0 + (i / Cfg::A_COLS);
        real_row[j] = (i < BM * Cfg::A_COLS) && grow < m
                      && (!moe_sorted_ids || moe_sorted_ids[grow] < moe_m_valid);
    }

    // ---- global -> shared staging -----------------------------------------
    auto load_stage = [&](int stage, int k0) {
        int4* sh = smem + stage * Cfg::SH_STAGE_I4;
        int4* sh_a = sh;
        int4* sh_b = sh + Cfg::SH_A_I4;

        // A tile: BM rows x BK halfs, XOR-swizzled so ldmatrix is conflict-free
#pragma unroll
        for (int i = t; i < BM * Cfg::A_COLS; i += Cfg::NTHREADS)
        {
            int row = i / Cfg::A_COLS;
            int c = i % Cfg::A_COLS;
            int grow = m0 + row;
            int cw = Cfg::A_SWIZZLE ? (c ^ (row & (Cfg::A_COLS - 1))) : c;
            const bool real = real_row[(i - t) / Cfg::NTHREADS];
            const int4* src = ((const int4*) A) + (size_t) grow * (k / 8) + k0 / 8 + c;
            // A routed block is block_m rows, but decode routes about one row to
            // each expert, so most rows are padding carrying zeros. cp.async
            // with a zero source size zero-fills, which is exactly what those
            // rows must hold, so skip the fetch. At 16 rows per block with one
            // live row that is 15/16 of this tile's traffic, and the transform
            // kernel no longer has to write them either.
            cp_async16_pred(sh_a + row * Cfg::A_STRIDE + cw, src, real);
        }

        // B tiles: KSTEPS x NTILES packed 16x16 trellis tiles
#pragma unroll
        for (int i = t; i < Cfg::SH_B_I4; i += Cfg::NTHREADS)
        {
            int chunk = i % Cfg::TILE_I4;
            int tile = i / Cfg::TILE_I4;
            int ks = tile / Cfg::NTILES;
            int nt = tile % Cfg::NTILES;
            const int4* src = ((const int4*) Bq) +
                              ((size_t) (k0 / 16 + ks) * n_tiles_full +
                               (n_off + n0) / 16 + nt) * Cfg::TILE_I4 + chunk;
            cp_async16(sh_b + tile * Cfg::TILE_I4 + chunk, src);
        }
    };

    using A_ = Acc<H_ACC>;
    typename A_::T frag_c[Cfg::MBLK][Cfg::NBLK];
#pragma unroll
    for (int i = 0; i < Cfg::MBLK; ++i)
#pragma unroll
        for (int j = 0; j < Cfg::NBLK; ++j) A_::zero(frag_c[i][j]);

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (kt_begin + s < kt_end) load_stage(s, (kt_begin + s) * BK);
        cp_async_fence();
    }

    // ---- main loop ---------------------------------------------------------
    for (int kt = kt_begin; kt < kt_end; ++kt)
    {
        int slot = (kt - kt_begin) % STAGES;
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        // Prefetch into the buffer just retired by iteration kt-1.
        int nxt_kt = kt + STAGES - 1;
        if (nxt_kt < kt_end) load_stage((nxt_kt - kt_begin) % STAGES, nxt_kt * BK);
        cp_async_fence();

        const int4* sh = smem + slot * Cfg::SH_STAGE_I4;
        const int4* sh_a = sh;
        const uint32_t* sh_b = (const uint32_t*) (sh + Cfg::SH_A_I4);

#pragma unroll
        for (int ks = 0; ks < Cfg::KSTEPS; ++ks)
        {
            FragA frag_a[Cfg::MBLK];
#pragma unroll
            for (int mb = 0; mb < Cfg::MBLK; ++mb)
            {
                // ldmatrix.x4 addressing: lane -> (row, 8-col group)
                int r = (lane & 7) + 8 * ((lane >> 3) & 1);
                int R = warp_m * Cfg::WARP_M + mb * 16 + r;
                int c = ks * 2 + (lane >> 4);
                int cw = Cfg::A_SWIZZLE ? (c ^ (R & (Cfg::A_COLS - 1))) : c;
                ldsm4(frag_a[mb], sh_a + R * Cfg::A_STRIDE + cw);
            }

            FragB frag_b[Cfg::NBLK];
#pragma unroll
            for (int nb = 0; nb < Cfg::NBLK; nb += 2)
            {
                // One 16x16 trellis tile decodes to two n8 B fragments.
                int nt = (warp_n * Cfg::WARP_N + nb * 8) / 16;
                const uint32_t* tile =
                    sh_b + (ks * Cfg::NTILES + nt) * Cfg::TILE_U32;
                dq_dispatch<BITS, CB>(tile, lane << 3, frag_b[nb], frag_b[nb + 1]);
            }

#pragma unroll
            for (int mb = 0; mb < Cfg::MBLK; ++mb)
#pragma unroll
                for (int nb = 0; nb < Cfg::NBLK; ++nb)
                    A_::mma(frag_a[mb], frag_b[nb], frag_c[mb][nb]);
        }
    }

    // ---- epilogue ----------------------------------------------------------
    if constexpr (SPLIT)
    {
        // Partial sums only: accumulate and let exl3_epilogue finish the row
        // once every split has landed.
        if constexpr (Cfg::SPLIT_STAGED)
        {
            __syncthreads();
            float* sh_f = (float*) smem;
#pragma unroll
            for (int mb = 0; mb < Cfg::MBLK; ++mb)
            {
                int r0 = warp_m * Cfg::WARP_M + mb * 16 + (lane >> 2);
#pragma unroll
                for (int nb = 0; nb < Cfg::NBLK; ++nb)
                {
                    int col = warp_n * Cfg::WARP_N + nb * 8 + 2 * (lane & 3);
                    float* p0 = sh_f + r0 * Cfg::F_STRIDE + col;
                    float* p1 = p0 + 8 * Cfg::F_STRIDE;
                    p0[0] = A_::getf(frag_c[mb][nb], 0);
                    p0[1] = A_::getf(frag_c[mb][nb], 1);
                    p1[0] = A_::getf(frag_c[mb][nb], 2);
                    p1[1] = A_::getf(frag_c[mb][nb], 3);
                }
            }
            __syncthreads();

            // Consecutive threads hit consecutive addresses, so each warp's
            // atomics coalesce into whole cache lines.
            for (int i = t; i < BM * BN; i += Cfg::NTHREADS)
            {
                int r = i / BN;
                int c = i - r * BN;
                int gr = m0 + r;
                if (gr >= m) continue;
                atomicAdd(&acc[(size_t) gr * ldc + n_off + n0 + c],
                          sh_f[r * Cfg::F_STRIDE + c]);
            }
        }
        else
        {
#pragma unroll
            for (int mb = 0; mb < Cfg::MBLK; ++mb)
            {
                int r0 = m0 + warp_m * Cfg::WARP_M + mb * 16 + (lane >> 2);
#pragma unroll
                for (int nb = 0; nb < Cfg::NBLK; ++nb)
                {
                    int col = n_off + n0 + warp_n * Cfg::WARP_N + nb * 8 + 2 * (lane & 3);
                    if (r0 < m)
                    {
                        atomicAdd(&acc[(size_t) r0 * ldc + col], A_::getf(frag_c[mb][nb], 0));
                        atomicAdd(&acc[(size_t) r0 * ldc + col + 1], A_::getf(frag_c[mb][nb], 1));
                    }
                    if (r0 + 8 < m)
                    {
                        atomicAdd(&acc[(size_t) (r0 + 8) * ldc + col], A_::getf(frag_c[mb][nb], 2));
                        atomicAdd(&acc[(size_t) (r0 + 8) * ldc + col + 1], A_::getf(frag_c[mb][nb], 3));
                    }
                }
            }
        }
        return;
    }

    // Non-split: this block owns the whole k extent for a full 128-wide output
    // group, so the Hadamard and svh fold in here -- no second pass, no scratch.
    __syncthreads();
    half* sh_c = (half*) smem;

#pragma unroll
    for (int mb = 0; mb < Cfg::MBLK; ++mb)
    {
        int base_row = warp_m * Cfg::WARP_M + mb * 16 + (lane >> 2);
#pragma unroll
        for (int nb = 0; nb < Cfg::NBLK; ++nb)
        {
            int col = warp_n * Cfg::WARP_N + nb * 8 + 2 * (lane & 3);
            half* p0 = sh_c + base_row * Cfg::C_STRIDE + col;
            half* p1 = p0 + 8 * Cfg::C_STRIDE;
            p0[0] = A_::geth(frag_c[mb][nb], 0);
            p0[1] = A_::geth(frag_c[mb][nb], 1);
            p1[0] = A_::geth(frag_c[mb][nb], 2);
            p1[1] = A_::geth(frag_c[mb][nb], 3);
        }
    }
    __syncthreads();

    for (int r = warp; r < BM; r += NWARPS)
    {
        int grow = m0 + r;
        if (grow >= m) continue;
        if (moe_weights)
        {
            const int idx = moe_sorted_ids[grow];
            // Padding rows carry zeros and belong to no token; nothing to add.
            if (idx >= moe_m_valid) continue;
            had128_warp_out_acc<OUT_T>(sh_c + r * Cfg::C_STRIDE,
                                       C + (size_t) (idx / moe_top_k) * ldc + n_off + n0,
                                       svh + n_off + n0, lane, moe_weights[idx]);
        }
        else
            had128_warp_out<OUT_T>(sh_c + r * Cfg::C_STRIDE,
                                   C + (size_t) grow * ldc + n_off + n0,
                                   svh + n_off + n0, lane);
    }
}

// Finishes a split-k result: Hadamard + svh over each 128-column group, fp32 ->
// fp16, and re-zeroes the accumulator so no memset is needed next time.
template <typename OUT_T>
__global__ void exl3_epilogue_kernel(float* __restrict__ acc, OUT_T* __restrict__ C,
                                     const half* __restrict__ svh, int m, int ldc,
                                     int n_off, int n_size,
                                     const int* __restrict__ expert_ids,
                                     const int* __restrict__ n_rows, int block_m,
                                     int64_t svh_expert_stride)
{
    int blocks_per_row = n_size / HAD_N;
    long long total = (long long) m * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    int blk = (int) (w % blocks_per_row);

    // MoE: svh is per expert, and the accumulator invariant requires this to
    // retire exactly the blocks the gemm retired -- had128_warp_acc re-zeroes
    // what it reads, so skipping a block the gemm wrote (or touching one it did
    // not) leaves stale partials behind for the next call. Hence the same
    // block-granular predicate the gemm uses, not a per-row one.
    if (expert_ids)
    {
        int blk_m = row / block_m;
        if (n_rows && blk_m * block_m >= *n_rows) return;
        int e = expert_ids[blk_m];
        if (e < 0) return;
        svh += (size_t) e * svh_expert_stride;
    }

    size_t off = (size_t) row * ldc + n_off + blk * HAD_N;
    had128_warp_acc<OUT_T>(acc + off, C + off, svh + n_off + blk * HAD_N, threadIdx.x & 31);
}

}  // namespace cuda_exl3

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

namespace {

using cuda_exl3::HAD_N;
using cuda_exl3::ShardMap;

constexpr int BN_ = 128;   // must equal HAD_N: a block owns whole Hadamard blocks
constexpr int BK_ = 32;
constexpr int NW_ = 8;

// Cap on the fused activation workspace. Fusing every shard of a layer into one
// launch needs all of their transformed activations live at once; past this the
// shards are run in sequence instead, reusing a single buffer. Launch overhead
// is irrelevant at those batch sizes anyway.
constexpr int64_t FUSE_MAX_ELEMS = 32ll << 20;   // 64 MiB of fp16

// Raising the dynamic-smem cap is a per-kernel, per-device property; do it once.
bool raise_smem(const void* fn, int smem)
{
    static std::set<const void*> done[8];
    int dev = 0;
    cudaGetDevice(&dev);
    auto& s = done[dev & 7];
    if (s.find(fn) == s.end())
    {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        s.insert(fn);
    }
    return true;
}

// CUDA_EXL3_FP16_ACC=1 enables fp16 accumulation in the GEMM (see Acc<>).
bool h_acc_enabled()
{
    static const bool v = [] {
        const char* e = exl3_env("CUDA_EXL3_FP16_ACC");
        return e && *e && *e != '0';
    }();
    return v;
}

int pick_bm(int m)
{
    // Override for tuning sweeps; 0 = use the heuristic.
    static const int forced = [] {
        const char* e = exl3_env("CUDA_EXL3_FORCE_BM");
        return e && *e ? atoi(e) : 0;
    }();
    if (forced) return forced;

    // Smallest BM that still covers the batch: BM >= m means the trellis is read
    // exactly once. Past 128 the accumulator register file binds, so larger
    // batches re-read in BM-sized passes.
    if (m <= 16) return 16;
    if (m <= 32) return 32;
    if (m <= 64) return 64;
    return 128;
}

// How many ways to split k. Enough blocks to fill the machine, but split-k costs
// an extra ~8*(S-1)*m*n bytes of accumulator traffic, so it is capped at a
// fraction of the weight bytes it is trying to stream faster.
int pick_split(int m, int k, int n, int bits, int bm, bool allowed, int weight_mult = 1)
{
    if (!allowed) return 1;
    const int sms = exl3_dev_sms();

    long long blocks = (long long) (n / BN_) * ((m + bm - 1) / bm);
    if (blocks <= 0) return 1;

    // MoE only: never split a grid that already fills the machine. Past one
    // full wave the extra accumulator traffic and the epilogue launch cost more
    // than the occupancy they buy, and the two projections of one GLM layer
    // land on opposite sides of that line -- at TP=4, w13 is 64 blocks on 188
    // SMs (0.34 waves) and splitting takes it 27.8 -> 18.7 us, while w2 is 256
    // blocks (1.36 waves) and splitting costs it 10.5 -> 12.7. No single split
    // target serves both; wave occupancy is what separates them.
    if (weight_mult > 1 && blocks >= sms) return 1;
    static const double target_mult = [] {
        const char* e = exl3_env("CUDA_EXL3_SPLIT_TARGET");
        return e && *e ? atof(e) : 3.0;
    }();
    long long target = (long long) (target_mult * sms);
    int want = (int) ((target + blocks - 1) / blocks);
    if (want <= 1) return 1;

    // Split-k's cost is the extra read-modify-write of the fp32 accumulator.
    // Charging that against HBM weight bytes was far too conservative: on a GPU
    // with a large L2 (128 MiB here) the accumulator is usually resident, so it
    // is L2 traffic, not memory traffic. Discounting it by L2_GAIN when it fits
    // was worth up to 1.5x in the m=16..256 range -- the region where the kernel
    // is block-starved and split-k is exactly what it needs. Not free, though:
    // treating it as free over-splits and regresses (down_proj m=128 went
    // 101 -> 121 us), so the discount is a factor, not a bypass.
    const double l2_bytes = (double) exl3_dev_l2();
    double acc_bytes = 4.0 * (double) m * n;
    int by_traffic;
    static const double budget = [] {
        const char* e = exl3_env("CUDA_EXL3_SPLIT_BUDGET");
        return e && *e ? atof(e) : 0.30;
    }();
    static const double l2_gain = [] {
        const char* e = exl3_env("CUDA_EXL3_L2_GAIN");
        return e && *e ? atof(e) : 2.0;
    }();
    // Split-k is charged against the weight bytes it is trying to stream
    // faster. A dense gemm reads one k*n matrix however many row-blocks it has,
    // but an MoE grid reads a whole expert slice per row-block, so its weight
    // traffic is weight_mult times larger. Without that the budget caps the
    // split at 1 and the MoE path never splits at all.
    double weight_bytes = (double) k * n * bits / 8.0 * (double) weight_mult;
    double per_extra = 8.0 * (double) m * n;            // one extra RMW of acc
    double b_eff = budget * (acc_bytes < 0.25 * l2_bytes ? l2_gain : 1.0);
    by_traffic = 1 + (int) (b_eff * weight_bytes / per_extra);

    int kt_total = k / BK_;
    int s = want;
    if (s > by_traffic) s = by_traffic;
    if (s > kt_total) s = kt_total;
    if (s > 16) s = 16;
    return s < 1 ? 1 : s;
}

template <int BITS, int CB, int BM, bool SPLIT, typename OUT_T, int WARP_N_, int ST_,
          bool H_ACC = false, int BK = BK_, int NWARPS = NW_>
void launch(const half* A, const uint16_t* Bq, OUT_T* C, const half* svh, int m, int k,
            int n, int ldc, int n_off, int n_tiles_full, float* acc, int split,
            ShardMap smap, cudaStream_t stream, const int* expert_ids = nullptr,
            int64_t b_expert_stride = 0, int64_t svh_expert_stride = 0,
            const int* n_rows = nullptr, const int* moe_sorted_ids = nullptr,
            const float* moe_weights = nullptr, int moe_top_k = 0,
            int moe_m_valid = 0)
{
    using Cfg = cuda_exl3::GemmCfg<BITS, CB, BM, BN_, BK, NWARPS, ST_, WARP_N_>;
    auto fn = cuda_exl3::exl3_gemm_m_kernel<BITS, CB, BM, BN_, BK, NWARPS, ST_, SPLIT, OUT_T,
                                            WARP_N_, H_ACC>;
    raise_smem((const void*) fn, Cfg::SMEM);
    int kt_total = k / BK;
    int kt_per_split = (kt_total + split - 1) / split;
    dim3 grid(n / BN_, (m + BM - 1) / BM, SPLIT ? split : 1);
    fn<<<grid, Cfg::NTHREADS, Cfg::SMEM, stream>>>(A, Bq, C, svh, m, k, n, ldc, n_off,
                                                  n_tiles_full, acc, kt_per_split, smap,
                                                  expert_ids, b_expert_stride,
                                                  svh_expert_stride, n_rows,
                                                  moe_sorted_ids, moe_weights,
                                                  moe_top_k, moe_m_valid);
}

// ---------------------------------------------------------------------------
// Autotuner for the block-M tier.
//
// Which BM wins is shape-dependent, not just batch-dependent: at m=128 up_proj
// (n=17408) is 16% faster with BM=64 while q_proj (n=12288) prefers BM=128. A
// static rule cannot capture that, so time the candidates once per distinct
// shape and remember the winner. All candidates compute the same result, so
// timing them on the real operands is safe -- the winner is simply run last.
// ---------------------------------------------------------------------------

uint64_t tune_key(int bits, int64_t cb, int m, int k, int n, bool bf16, bool can_split)
{
    uint64_t mb = 1;                       // bucket m by power of two
    while (mb < (uint64_t) m && mb < 4096) mb <<= 1;
    uint64_t h = 1469598103934665603ull;
    // can_split belongs in the key: the cached choice carries a split factor,
    // and replaying a split entry when no accumulator was allocated (the
    // deterministic path) sends the kernel through a null pointer.
    for (uint64_t v : {(uint64_t) bits, (uint64_t) cb, mb, (uint64_t) k, (uint64_t) n,
                       (uint64_t) bf16, (uint64_t) can_split})
    {
        h ^= v;
        h *= 1099511628211ull;
    }
    return h;
}

std::map<uint64_t, int>& tune_cache()
{
    static std::map<uint64_t, int> c;
    return c;
}

bool tuning_enabled()
{
    static const bool v = [] {
        const char* e = exl3_env("CUDA_EXL3_AUTOTUNE");
        return !(e && *e == '0');
    }();
    return v;
}

// Runs `run(bm)` for each candidate, returns the fastest. `run` must leave the
// output correct for whichever bm it was last called with.
template <typename F>
int autotune_cfg(uint64_t key, int m, int k, int n, int bits, bool split_k,
                 bool can_split, int split_fixed, const F& run, cudaStream_t stream,
                 int heuristic_bm)
{
    auto pack = [](int bm, int sp) { return bm | (sp << 16); };
    auto& cache = tune_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto split_for = [&](int bm) {
        if (split_fixed) return split_fixed;
        return can_split ? pick_split(m, k, n, bits, bm, split_k) : 1;
    };

    // Never time inside graph capture: it needs syncs, and the capture would
    // record whichever candidate ran last.
    if (!tuning_enabled() ||
        at::cuda::currentStreamCaptureStatusMayInitCtx() != at::cuda::CaptureStatus::None)
        return pack(heuristic_bm, split_for(heuristic_bm));

    // Search the split alongside the block size. pick_split is a cost model, and
    // in the m=64..256 band -- where the grid is block-starved but the extra
    // accumulator traffic is not yet free -- the model is guessing. Measuring
    // one step either side of its answer costs a few hundred microseconds once
    // per shape and lets the tuner correct it.
    const int bms[4] = {16, 32, 64, 128};
    // Doubling the heuristic's split can push past what pick_split would ever
    // return, so bound it by the *actual* k-tile count for the BK this shape
    // runs with -- BK=64 when k allows it, otherwise the BK=32 fallback.
    // Overshooting hands some splits an empty k range.
    int cap = k / (k % 64 == 0 ? 64 : BK_);
    if (cap > 16) cap = 16;
    if (cap < 1) cap = 1;

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    int best = pack(heuristic_bm, split_for(heuristic_bm));
    float best_ms = 1e30f;
    for (int bm : bms)
    {
        int base = split_for(bm);
        int sps[3] = {base, 0, 0};
        int nsp = 1;
        if (!split_fixed && can_split)
        {
            if (base > 1) sps[nsp++] = base / 2;
            if (base * 2 <= cap) sps[nsp++] = base * 2;
        }
        for (int i = 0; i < nsp; ++i)
        {
            int sp = sps[i];
            if (sp < 1 || sp > cap) continue;
            run(bm, sp);                                  // warm
            cudaEventRecord(beg, stream);
            for (int r = 0; r < 3; ++r) run(bm, sp);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, beg, end);
            if (ms < best_ms)
            {
                best_ms = ms;
                best = pack(bm, sp);
            }
        }
    }
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cache[key] = best;
    return best;
}

template <int BITS, int CB, typename OUT_T>
void launch_bm(const half* A, const uint16_t* Bq, OUT_T* C, const half* svh, int m, int k,
               int n, int ldc, int n_off, int n_tiles_full, float* acc, int split,
               ShardMap smap, cudaStream_t stream, int bm_override = 0,
               const int* expert_ids = nullptr, int64_t b_expert_stride = 0,
               int64_t svh_expert_stride = 0, const int* n_rows = nullptr,
               const int* moe_sorted_ids = nullptr, const float* moe_weights = nullptr,
               int moe_top_k = 0, int moe_m_valid = 0)
{
    // fp16 accumulation is opt-in and only used where it can pay for itself: no
    // split-k (which reduces in fp32 anyway) and a batch large enough for the
    // 256-row tile it unlocks. Partial sums are then kept in fp16, so it trades
    // accuracy for MMA work per dequantized weight.
    if (h_acc_enabled() && split == 1 && m > 128)
    {
        launch<BITS, CB, 256, false, OUT_T, 16, 3, true>(A, Bq, C, svh, m, k, n, ldc,
                                                         n_off, n_tiles_full, acc, 1,
                                                         smap, stream);
        return;
    }

    int bm = bm_override ? bm_override : pick_bm(m);
    // Campaign K: only the measured-checkpoint family and small routed
    // query shapes. 4 warps x32 output columns = same128-wide tile;
    // each warp reuses its A fragments for twice as many N fragments.
    // The reduction remains FP32 MMA in the original K order.
    static const bool campaign_k = [] {
        const char* value = exl3_env("CUDA_EXL3_CAMPAIGN_K");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if constexpr (CB == 2 && (BITS == 3 || BITS == 4)) {
        if (campaign_k && expert_ids && bm == 16 && split == 1
            && moe_sorted_ids && moe_m_valid > 0
            && moe_m_valid <= 48 * moe_top_k
            && ((k == 5120 && n == 1536) || (k == 768 && n == 5120))) {
            // BK64 is valid for both exact TP3 input widths.
            launch<BITS, CB, 16, false, OUT_T, 32, 3, false, 64, 4>(
                A, Bq, C, svh, m, k, n, ldc, n_off, n_tiles_full, acc,
                1, smap, stream, expert_ids, b_expert_stride,
                svh_expert_stride, n_rows, moe_sorted_ids, moe_weights,
                moe_top_k, moe_m_valid);
            return;
        }
    }
    // One warp row (WARPS_M == 1) keeps each trellis tile decoded exactly once.
    // Pipeline depth trades against occupancy: the padded A tile is larger, so
    // the biggest block tile uses one stage fewer to stay at 2 blocks/SM.
#define VE3_ONE(BM_, WN_, ST, BKT)                                                     \
    if (split > 1)                                                                     \
        launch<BITS, CB, BM_, true, OUT_T, WN_, ST, false, BKT>(A, Bq, C, svh, m,      \
                      k, n, ldc, n_off, n_tiles_full, acc, split, smap, stream,        \
                      expert_ids, b_expert_stride, svh_expert_stride, n_rows);         \
    else                                                                               \
        /* n_rows is not optional here. Without it the unsplit kernel has no live-row \
           bound, and the surplus tail of expert_ids is not reliably -1: the alignment \
           marks it -1, then expert_map[expert_ids] indexes with -1, which is negative \
           indexing -- it returns the local id of the LAST global expert. On the rank  \
           owning the top of the range that is a real expert, so that rank runs a full \
           gemm over every surplus block: 206 of 540 at M=2048, 38% of the grid, and   \
           the step waits for it. Reported and diagnosed by @NNNtrance in #1. */       \
        launch<BITS, CB, BM_, false, OUT_T, WN_, ST, false, BKT>(A, Bq, C, svh, m,     \
                      k, n, ldc, n_off, n_tiles_full, acc, 1, smap, stream,            \
                      expert_ids, b_expert_stride, svh_expert_stride, n_rows,          \
                      moe_sorted_ids, moe_weights, moe_top_k, moe_m_valid);

// BK=64 borrows Marlin's shape: an A row is then exactly 128 B = 32 banks, so the
// XOR swizzle is conflict-free with no padding at all. It needs k % 64 == 0
// though (EXL3 only guarantees multiples of 16), so BK=32 stays as a fallback --
// there a row is 64 B with just 4 columns to permute, and the stride is padded
// instead.
#define VE3_BM(BM_, WN_, ST64, ST32)                                                   \
    if (bm == BM_)                                                                     \
    {                                                                                  \
        if (k % 64 == 0) { VE3_ONE(BM_, WN_, ST64, 64) }                               \
        else             { VE3_ONE(BM_, WN_, ST32, 32) }                               \
        return;                                                                        \
    }
    VE3_BM(16, 16, 2, 4) VE3_BM(32, 16, 3, 4) VE3_BM(64, 16, 3, 4) VE3_BM(128, 16, 2, 3)
#undef VE3_BM
#undef VE3_ONE
}

template <typename OUT_T>
void launch_epilogue(float* acc, OUT_T* C, const half* svh, int m, int ldc, int n_off,
                     int n_size, cudaStream_t stream,
                     const int* expert_ids = nullptr, const int* n_rows = nullptr,
                     int block_m = 0, int64_t svh_expert_stride = 0)
{
    long long warps = (long long) m * (n_size / HAD_N);
    const int threads = 256;
    long long blocks = (warps + threads / 32 - 1) / (threads / 32);
    cuda_exl3::exl3_epilogue_kernel<OUT_T><<<(unsigned) blocks, threads, 0, stream>>>(
        acc, C, svh, m, ldc, n_off, n_size, expert_ids, n_rows, block_m,
        svh_expert_stride);
}

template <typename OUT_T>
void dispatch_gemm(int bits, int64_t cb, const half* A, const uint16_t* B, OUT_T* C,
                   const half* S, int m, int k, int n, int ldc, int n_off,
                   int n_tiles_full, float* acc, int split_fixed, ShardMap smap,
                   cudaStream_t stream, const int* expert_ids = nullptr,
                   int64_t b_expert_stride = 0, int64_t svh_expert_stride = 0,
                   int force_bm = 0, const int* n_rows = nullptr,
                   bool split_k = false, int bits_for_split = 0,
                   const int* moe_sorted_ids = nullptr,
                   const float* moe_weights = nullptr,
                   int moe_top_k = 0, int moe_m_valid = 0)
{
#define VE3_CASE(B_, C_)                                                        \
    if (bits == B_ && cb == C_)                                                 \
    {                                                                           \
        {                                                                       \
            /* Both the block size and the split are tuned, and they interact: \
               a bigger tile means fewer blocks, which means more splitting.  */\
            auto run = [&](int bm_, int sp_) {                                  \
                launch_bm<B_, C_, OUT_T>(A, B, C, S, m, k, n, ldc, n_off,       \
                                 n_tiles_full, acc, sp_, smap, stream, bm_,     \
                                 expert_ids, b_expert_stride, svh_expert_stride, \
                                 n_rows, moe_sorted_ids, moe_weights, moe_top_k, \
                                 moe_m_valid);                                   \
                if (sp_ > 1)                                                    \
                    launch_epilogue<OUT_T>(acc, C, S, m, ldc, n_off, n, stream,  \
                                           expert_ids, n_rows, bm_,              \
                                           svh_expert_stride);                   \
            };                                                                  \
            /* MoE pins BM: the caller padded each expert's rows to that block, \
               so the grid's row tiling has to match it exactly. */             \
            int bm, sp;                                                         \
            if (force_bm)                                                       \
            {                                                                   \
                bm = force_bm;                                                  \
                sp = split_fixed ? split_fixed : 1;                             \
            }                                                                   \
            else                                                                \
            {                                                                   \
                int c = autotune_cfg(tune_key(B_, C_, m, k, n,                  \
                                          sizeof(OUT_T) == 2 && !std::is_same<OUT_T, half>::value, \
                                          acc != nullptr),                       \
                                     m, k, n, bits_for_split, split_k,          \
                                     acc != nullptr, split_fixed, run, stream,  \
                                     pick_bm(m));                               \
                bm = c & 0xffff;                                                \
                sp = c >> 16;                                                   \
            }                                                                   \
            run(bm, sp);                                                        \
        }                                                                       \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
        return;                                                                 \
    }
    // Every bitrate (1-8) and every EXL3 codebook: 3inst (0), mcg (1), mul1 (2).
    // The codebook multiplier is a constant of the codebook id, not per tensor,
    // so an id is all the kernel needs to be format-complete.
#define VE3_ALL_BITS(C_)                                                        \
    VE3_CASE(1, C_) VE3_CASE(2, C_) VE3_CASE(3, C_) VE3_CASE(4, C_)             \
    VE3_CASE(5, C_) VE3_CASE(6, C_) VE3_CASE(7, C_) VE3_CASE(8, C_)
    VE3_ALL_BITS(2) VE3_ALL_BITS(1) VE3_ALL_BITS(0)
#undef VE3_ALL_BITS
#undef VE3_CASE
    TORCH_CHECK(false, "exl3: unsupported (bits=", bits, ", cb=", cb,
                "). Rebuild with this combination instantiated.");
}

// Process-wide workspaces. Kept here rather than per layer so they do not count
// against the weight budget: one activation-transform buffer and one split-k
// accumulator serve every EXL3 layer in the model.
at::Tensor g_ahad[16];
at::Tensor g_acc[16];

// Growing a workspace hands back a *new* allocation, which silently invalidates
// any pointer already baked into a captured CUDA graph -- the replay then writes
// wherever the caching allocator reassigned that block, corrupting whatever now
// lives there. So retired buffers are kept alive (never returned to the
// allocator), and exl3_reserve() sizes both workspaces to the model's maximum at
// load time, before anything is captured.
std::vector<at::Tensor> g_retired;

at::Tensor& get_ws(at::Tensor* pool, int64_t numel, at::ScalarType dt,
                   const at::Device& device, bool zeroed)
{
    at::Tensor& t = pool[device.index() & 15];
    if (!t.defined() || t.numel() < numel)
    {
        TORCH_CHECK(
            at::cuda::currentStreamCaptureStatusMayInitCtx() == at::cuda::CaptureStatus::None,
            "cuda-exl3: workspace must grow (", (t.defined() ? t.numel() : 0), " -> ", numel,
            ") during CUDA graph capture. This should not happen: call "
            "exl3_reserve() at load time.");
        if (t.defined()) g_retired.push_back(t);
        auto opts = at::TensorOptions().dtype(dt).device(device);
        t = zeroed ? at::zeros({numel}, opts) : at::empty({numel}, opts);
    }
    return t;
}

}  // namespace

namespace cuda_exl3 {

// Defined in hadamard.cu.
void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh);

// Size both workspaces for the largest forward this model can issue, so that
// nothing reallocates once CUDA graphs start being captured.
void exl3_reserve(const at::Tensor& like, int64_t max_tokens, int64_t k, int64_t n,
                  int64_t groups)
{
    const at::cuda::OptionalCUDAGuard guard(like.device());
    int64_t fused = groups * max_tokens * k;
    if (fused > FUSE_MAX_ELEMS) fused = FUSE_MAX_ELEMS;
    int64_t ahad = std::max<int64_t>(max_tokens * k, fused);
    get_ws(g_ahad, ahad, at::kHalf, like.device(), false);

    // The split-k accumulator is only touched when the shape actually splits,
    // which needs the layer to be block-starved. A wide output (lm_head, with
    // n=248k, is 1940 blocks) never splits, so reserving max_tokens*n there
    // would burn gigabytes on a buffer nothing ever writes. Reserve for the
    // largest batch that genuinely splits, and nothing if none does.
    const int bits_ub = 8;  // more bits -> more weight bytes -> larger split cap
    int64_t acc_elems = 0;
    for (int64_t m = 1; m <= max_tokens; m = (m < 8 ? m + 1 : m * 2))
        if (pick_split((int) m, (int) k, (int) n, bits_ub, 128, true) > 1)
            acc_elems = std::max<int64_t>(acc_elems, m * n);
    if (acc_elems) get_ws(g_acc, acc_elems, at::kFloat, like.device(), true);
}

// Cap on the MoE split-k accumulator, in fp32 elements. Zero disables MoE
// split-k entirely, which is the default -- see exl3_moe_gemm for the numbers.
// Mutable so tests can exercise the split path in-process; the environment only
// supplies the initial value.
int64_t& moe_acc_cap_ref()
{
    static int64_t v = [] {
        const char* e = exl3_env("CUDA_EXL3_MOE_ACC_MAX_ELEMS");
        return e && *e ? (int64_t) atoll(e) : (8ll << 20);   // 32 MiB of fp32
    }();
    return v;
}

int64_t moe_acc_cap_elems() { return moe_acc_cap_ref(); }

void exl3_set_moe_acc_cap(int64_t elems) { moe_acc_cap_ref() = elems; }

int64_t exl3_get_moe_acc_cap() { return moe_acc_cap_ref(); }

// The MoE gemm sizes its accumulator from the routed row count, which is not
// known until the forward runs. Reserve the ceiling now: get_ws hard-errors if
// it has to grow once graphs are being captured, and the MoE path caps its own
// split at this many elements, so reserving the cap makes growth impossible.
void exl3_reserve_acc(const at::Tensor& like, int64_t elems)
{
    const at::cuda::OptionalCUDAGuard guard(like.device());
    if (elems > 0) get_ws(g_acc, elems, at::kFloat, like.device(), true);
}

at::Tensor exl3_linear(const at::Tensor& x, const at::Tensor& trellis,
                       const at::Tensor& suh, const at::Tensor& svh,
                       at::IntArrayRef group_n, int64_t cb, bool split_k)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(trellis.is_contiguous(), "exl3_linear: trellis must be contiguous");
    TORCH_CHECK(trellis.dim() == 3, "exl3_linear: trellis must be 3-D");
    TORCH_CHECK(suh.dim() == 2, "exl3_linear: suh must be (groups, k)");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "exl3_linear: activations must be float16 or bfloat16");

    const int k = (int) x.size(-1);
    const int m = (int) (x.numel() / k);
    const int n_tiles_full = (int) trellis.size(1);
    const int n_total = n_tiles_full * 16;
    const int bits = (int) trellis.size(2) / 16;
    const int groups = (int) group_n.size();

    TORCH_CHECK((int) suh.size(0) == groups, "exl3_linear: suh has ", suh.size(0),
                " rows for ", groups, " shards");
    TORCH_CHECK((int) suh.size(1) == k, "exl3_linear: suh k mismatch");
    TORCH_CHECK((int) trellis.size(0) * 16 == k, "exl3_linear: trellis k mismatch");
    TORCH_CHECK(svh.numel() == n_total, "exl3_linear: svh size mismatch");
    TORCH_CHECK(k % 32 == 0, "exl3_linear: k must be a multiple of 32, got ", k);
    TORCH_CHECK(groups >= 1 && groups <= 8, "exl3_linear: 1..8 shards supported");

    auto out_sizes = x.sizes().vec();
    out_sizes.back() = n_total;
    at::Tensor out = at::empty(out_sizes, x.options());
    if (m == 0) return out;

    // Every shard boundary must land on a BN-wide block for one launch to cover
    // them all; otherwise fall back to running the shards in sequence.
    bool aligned = true;
    int64_t sum = 0;
    for (int g = 0; g < groups; ++g)
    {
        if (group_n[g] % BN_) aligned = false;
        sum += group_n[g];
    }
    TORCH_CHECK(sum == n_total, "exl3_linear: shard widths sum to ", sum, ", expected ",
                n_total);

    const bool fuse =
        groups == 1 || (aligned && (int64_t) groups * m * k <= FUSE_MAX_ELEMS);
    const int64_t ahad_elems = (fuse ? (int64_t) groups : 1) * m * k;

    at::Tensor& ahad = get_ws(g_ahad, ahad_elems, at::kHalf, x.device(), false);
    auto stream = at::cuda::getCurrentCUDAStream();

    const uint16_t* B = (const uint16_t*) trellis.data_ptr();
    const half* S = (const half*) svh.data_ptr();
    auto x2 = x.view({m, k});

    auto run = [&](int n_off, int n_size, ShardMap smap, const at::Tensor& suh_rows) {
        at::Tensor ahad_view = ahad.narrow(0, 0, (int64_t) smap.n_groups * m * k);
        exl3_had_in(x2, ahad_view, suh_rows);

        float* acc = nullptr;
        if (pick_split(m, k, n_size, bits, 128, split_k) > 1)
        {
            at::Tensor& a = get_ws(g_acc, (int64_t) m * n_total, at::kFloat, x.device(), true);
            acc = (float*) a.data_ptr();
        }
        const half* A = (const half*) ahad.data_ptr();
        if (x.scalar_type() == at::kHalf)
            dispatch_gemm<half>(bits, cb, A, B, (half*) out.data_ptr(), S, m, k, n_size,
                                n_total, n_off, n_tiles_full, acc, 0, smap, stream,
                                nullptr, 0, 0, 0, nullptr, split_k, bits);
        else
            dispatch_gemm<__nv_bfloat16>(bits, cb, A, B, (__nv_bfloat16*) out.data_ptr(), S,
                                         m, k, n_size, n_total, n_off, n_tiles_full, acc,
                                         0, smap, stream, nullptr, 0, 0, 0, nullptr,
                                         split_k, bits);
    };

    if (fuse)
    {
        ShardMap smap{};
        smap.n_groups = groups;
        int acc_blk = 0;
        for (int g = 0; g < groups; ++g)
        {
            acc_blk += (int) (group_n[g] / BN_);
            smap.nblk_end[g] = acc_blk;
        }
        run(0, n_total, smap, suh);
    }
    else
    {
        int64_t off = 0;
        for (int g = 0; g < groups; ++g)
        {
            ShardMap smap{};
            smap.n_groups = 1;
            smap.nblk_end[0] = (int) (group_n[g] / BN_);
            run((int) off, (int) group_n[g], smap, suh.narrow(0, g, 1));
            off += group_n[g];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// MoE grouped GEMM.
//
// Rows are pre-sorted by expert and padded to whole BM blocks, so every row of a
// block shares an expert; the kernel just offsets the trellis and svh by that
// expert. Split-k is off here: the block-count heuristic assumes a dense row
// range, and MoE already has plenty of blocks from the expert dimension.
// ---------------------------------------------------------------------------

at::Tensor exl3_moe_gemm(const at::Tensor& a_had, const at::Tensor& trellis,
                         const at::Tensor& suh_unused, const at::Tensor& svh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         at::IntArrayRef group_n, int64_t cb, int64_t block_m,
                         at::ScalarType out_dtype,
                         const std::optional<at::Tensor>& sorted_ids,
                         const std::optional<at::Tensor>& topk_weights,
                         int64_t num_tokens, int64_t top_k)
{
    const at::cuda::OptionalCUDAGuard guard(a_had.device());
    TORCH_CHECK(trellis.is_contiguous() && trellis.dim() == 4,
                "exl3_moe_gemm: trellis must be contiguous (experts, k/16, n/16, 16*bits)");
    TORCH_CHECK(svh.dim() == 2, "exl3_moe_gemm: svh must be (experts, n)");

    const int k = (int) a_had.size(-1);
    const int rows = (int) expert_ids.size(0) * (int) block_m;
    const int n_tiles_full = (int) trellis.size(2);
    const int n_total = n_tiles_full * 16;
    const int bits = (int) trellis.size(3) / 16;
    const int groups = (int) group_n.size();

    TORCH_CHECK((int) trellis.size(1) * 16 == k, "exl3_moe_gemm: trellis k mismatch");
    TORCH_CHECK((int) svh.size(1) == n_total, "exl3_moe_gemm: svh n mismatch");
    TORCH_CHECK(a_had.numel() >= (long long) groups * rows * k,
                "exl3_moe_gemm: a_had holds ", a_had.numel(), " elements, need ",
                (long long) groups * rows * k, " for ", rows, " rows");

    auto opts = at::TensorOptions().dtype(out_dtype).device(a_had.device());
    // Fused combine: the output is per token, not per routed row, and the rows
    // accumulate into it, so it starts at zero. That is one (tokens, n) memset
    // against the (rows, n) tensor plus a whole extra kernel it replaces.
    const bool have_sids = sorted_ids.has_value() && sorted_ids->numel();
    const bool fuse_combine = have_sids && topk_weights.has_value();
    const int* moe_sid = nullptr;
    const float* moe_w = nullptr;
    at::Tensor w_f32;
    if (have_sids)
    {
        TORCH_CHECK(sorted_ids->scalar_type() == at::kInt,
                    "exl3_moe_gemm: sorted_ids must be int32");
        TORCH_CHECK(num_tokens > 0 && top_k > 0,
                    "exl3_moe_gemm: sorted_ids needs num_tokens and top_k");
        moe_sid = sorted_ids->data_ptr<int>();
    }
    if (fuse_combine)
    {
        w_f32 = topk_weights->to(at::kFloat).contiguous();
        moe_w = w_f32.data_ptr<float>();
    }
    at::Tensor out = fuse_combine ? at::zeros({num_tokens, n_total}, opts)
                                  : at::empty({rows, n_total}, opts);
    const int moe_m_valid = have_sids ? (int) (num_tokens * top_k) : 0;

    ShardMap smap{};
    smap.n_groups = groups;
    int acc_blk = 0;
    int64_t sum = 0;
    for (int g = 0; g < groups; ++g)
    {
        TORCH_CHECK(group_n[g] % BN_ == 0, "exl3_moe_gemm: shard widths must be a "
                    "multiple of ", BN_);
        acc_blk += (int) (group_n[g] / BN_);
        smap.nblk_end[g] = acc_blk;
        sum += group_n[g];
    }
    TORCH_CHECK(sum == n_total, "exl3_moe_gemm: shard widths sum to ", sum, ", expected ",
                n_total);

    // At low concurrency the expert gemm is badly block-starved: c=1 with
    // top_k=8 gives 8 padded rows, so w13 (n=2*inter) is ~96 blocks against 188
    // SMs. Splitting k fills the machine. The accumulator is rows*n floats,
    // which is only affordable while rows is small -- but rows is small exactly
    // when splitting helps, so cap it and let prefill run unsplit.
    // Off by default: measured, and the kernel does get faster (rows=512,
    // block_m=32: 33.0 -> 27.5 us) but it does not reach end to end. Splitting
    // adds an epilogue launch per gemm per layer -- ~96 extra kernels per decode
    // step -- and at decode batch sizes that costs more than the parallelism
    // buys (MoE 8k/1k, tok/s: c=8 742.9 -> 712.8, c=32 1211.6 -> 1196.2, c=1
    // and c=64 a wash). Kept behind the knob so the result is re-testable on
    // hardware with a different SM count / launch cost.
    int64_t moe_acc_cap = moe_acc_cap_elems();
    int split = 1;
    float* acc = nullptr;
    int64_t acc_elems = (int64_t) rows * n_total;
    if (acc_elems <= moe_acc_cap)
        split = pick_split(rows, k, n_total, bits, (int) block_m, true,
                           (int) expert_ids.size(0));
    if (split > 1)
    {
        at::Tensor& a = get_ws(g_acc, acc_elems, at::kFloat, a_had.device(), true);
        acc = (float*) a.data_ptr();
    }

    const int64_t b_stride = (int64_t) trellis.size(1) * n_tiles_full * trellis.size(3);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* A = (const half*) a_had.data_ptr();
    const uint16_t* B = (const uint16_t*) trellis.data_ptr();
    const half* S = (const half*) svh.data_ptr();
    const int* eids = expert_ids.data_ptr<int>();
    const int* nrows = n_rows.numel() ? n_rows.data_ptr<int>() : nullptr;

    if (out_dtype == at::kHalf)
        dispatch_gemm<half>(bits, cb, A, B, (half*) out.data_ptr(), S, rows, k, n_total,
                            n_total, 0, n_tiles_full, acc, split, smap, stream, eids,
                            b_stride, n_total, (int) block_m, nrows, false, 0,
                            moe_sid, moe_w, (int) top_k, moe_m_valid);
    else
        dispatch_gemm<__nv_bfloat16>(bits, cb, A, B, (__nv_bfloat16*) out.data_ptr(), S,
                                     rows, k, n_total, n_total, 0, n_tiles_full, acc,
                                     split, smap, stream, eids, b_stride, n_total,
                                     (int) block_m, nrows, false, 0,
                                     moe_sid, moe_w, (int) top_k, moe_m_valid);
    return out;
}

}  // namespace cuda_exl3
