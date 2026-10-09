// Fused sparse-MLA decode for SM120 (consumer Blackwell).
//
// GLM-5.3-Flash routes each query to `topk` selected latent rows. In MLA the
// key and the value are the *same* cached row -- K uses all `D` dims, V the
// first `DV` -- so a block that owns all query heads for a chunk of keys reads
// each row once and reuses it across every head. That reuse is the whole point:
// parallelising over heads instead would re-read the cache H times and put the
// kernel H x above its own roofline.
//
// The work at decode is small (topk x D bytes, ~1.3 us of HBM at topk=2048), so
// occupancy, not bandwidth, is what a naive kernel loses to. Keys are therefore
// split across blocks flash-decoding style, each block keeping an online softmax
// and a partial output, and a second pass merges the partials by log-sum-exp.
//
// Only mma-free math is used here: the per-key work is a dot product and an
// axpy, both of which map to plain FMA and warp shuffles, so nothing depends on
// wgmma/TMA and the kernel is portable to any SM80+ part.

// Where the time actually goes (ncu, batch 1, topk 2048, 16 heads at TP=4):
//
//     duration 21.95 us   DRAM 6.9%   L1/TEX 59.8%   compute 23.8%
//     warp cycles per issued instruction 10.08 -- warps stall ~90% of the time
//     stalls/issue: mio_throttle 1.93, short_scoreboard 1.66,
//                   long_scoreboard 1.52, wait 1.49, not_selected 1.10
//     achieved occupancy 33.2% against a theoretical 66.7%
//
// So this is bound by the shared-memory/LSU pipe, not by DRAM (6.9%) and not by
// math (23.8%). The cost is 18 separate 2-byte shared loads per key per lane:
// each wastes half of the 128 B/cycle the pipe delivers and occupies a whole
// MIO slot.
//
// Three fixes for that were tried and each made it slower; they are recorded so
// the next pass does not repeat them:
//
//   * Hoisting the k row into registers once per key and reusing it across both
//     heads and both projections (68 shared reads -> 18). 28.7 -> 30.8 us. It
//     removes redundant reads but lengthens the live range enough to cost more
//     than it saves.
//   * 16 warps per block instead of 8 (one head per warp), which drops the
//     kernel from 109 to 64 registers and doubles warps per SM. 28.7 -> 32.8 us:
//     at batch 1 the grid is only ~64 blocks on 188 SMs, so occupancy per SM was
//     never the binding constraint.
//   * Vectorising the shared reads by giving each lane a contiguous 16-dim slice
//     (18 scalar loads -> 3 int4/bf16x2 loads). 28.7 -> 35.6 us, because
//     contiguous ownership makes lane L read shared word L*8, so lanes 0, 4,
//     8 ... collide on bank 0 -- an 8-way conflict. The strided layout this
//     kernel uses is conflict-free. Vectorising here needs an XOR swizzle of the
//     16-byte chunks, the same trick gemm.cu applies to its A tiles.
//
// The structural answer is to stop doing QK as an FMA chain plus a warp-wide
// shuffle reduction per key and make it an mma: a tile of keys is a matrix, so
// S = Q @ K^T maps onto m16n8k16 with m = 16 heads exactly, reading K from
// shared through ldmatrix (vectorised and conflict-free by construction), and
// O += P @ V is a second mma. That also folds the reduce kernel's launch into
// the same pass. It is a rewrite rather than a tweak, which is why it is not in
// this version.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <map>
#include <string>
#include <algorithm>
#include <vector>

#include "exl3_common.cuh"

// bf16 -> fp32 is a 16-bit shift, so a packed pair unpacks with two ALU ops and
// no round trip through memory. Taking the address of a loaded int4 to read it
// as bf16 would have spilled it to local.
__device__ __forceinline__ void bf16x2_f32(uint32_t u, float& a, float& b)
{
    a = __uint_as_float(u << 16);
    b = __uint_as_float(u & 0xffff0000u);
}

// The other direction, round-to-nearest-even, packed two at a time. Done with
// bit math rather than __float2bfloat16 so nothing has its address taken: an
// int4 built through a pointer spills to local memory.
// fp8 e4m3 pair -> bf16 pair. Both conversions are exact: e4m3 carries three
// mantissa bits against bf16's seven, and its 448 maximum is far inside range.
__device__ __forceinline__ uint32_t e4m3x2_bf16(uint32_t packed)
{
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t) packed));
    float lo, hi;
    asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"((uint16_t) (h2 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"((uint16_t) (h2 >> 16)));
    const uint32_t x = __float_as_uint(lo), y = __float_as_uint(hi);
    return ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16)
         | ((y + 0x7fffu + ((y >> 16) & 1u)) & 0xffff0000u);
}

__device__ __forceinline__ uint32_t f32x2_bf16(float a, float b)
{
    const uint32_t x = __float_as_uint(a), y = __float_as_uint(b);
    const uint32_t rx = (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
    const uint32_t ry = (y + 0x7fffu + ((y >> 16) & 1u));
    return rx | (ry & 0xffff0000u);
}

namespace cuda_exl3 {

#ifndef MLA_MAX_HEADS
#define MLA_MAX_HEADS 32
#endif

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// One block owns (batch b, key-split s) and every query head.
//   grid  = (splits, batch)
//   block = NWARPS * 32, each warp taking HEADS_PER_WARP heads
// One block owns (batch b, key-split s) and every query head.
//   grid  = (splits, batch)
//   block = NWARPS * 32, each warp taking HPW heads for the whole chunk
//
// HPW is a template parameter, not a runtime bound: the accumulator has to live
// in registers, and a runtime-indexed local array spills every byte of it to
// local memory (v1 did exactly that and ran 3.4x slower than the kernel it was
// meant to beat). Keys are staged a tile at a time through cp.async so several
// latent rows are in flight instead of one.
// Both matmuls run on tensor cores. A tile of keys is a matrix, so
// S = Q @ K^T is an mma with m = 16 heads exactly, and `row.col` wants B stored
// [n][k] = [key][dim], which is already the cache layout. O += P @ V is a second
// mma with the roles rotated -- m = dim, n = head, k = key -- so V arrives as an
// A fragment through the transposing ldmatrix, and P, produced [head][key], is
// the B fragment as-is.
//
// Softmax runs once per tile rather than once per key. That is what makes the
// mma possible at all (the accumulator can only be rescaled between mmas) and it
// also cuts the rescale traffic by a factor of TILE.
//
// The shared row stride is padded by 8 halves: at the natural 576 the stride is
// 1152 B = 288 words, a multiple of 32, so every group of lanes reading down a
// column would land on the same four banks -- an 8-way conflict. At 584 the rows
// step by 4 banks and every ldmatrix covers all 32 exactly once.
template <int D, int DV, int NWARPS, int TILE, typename KT, int NBUF>
__global__ __launch_bounds__(NWARPS * 32) void mla_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,     // (B, H, D)
    const KT* __restrict__ kv,               // (rows, D) latent cache
    const int* __restrict__ sel,             // (B, topk) row ids, -1 = empty
    const int* __restrict__ seqlens,         // (B) valid entries of sel
    // (B, H, splits, DV): splits contiguous, so the merge walks them 1 KB apart
    // instead of H * DV * 2 = 16 KB apart, which is a different page every time.
    __nv_bfloat16* __restrict__ part_o,
    float* __restrict__ part_m,              // (B, H, splits)
    float* __restrict__ part_l,              // (B, H, splits)
    // Set when splits == 1: there is nothing to merge, so normalise here and
    // write the answer straight out. At prefill that removes a round trip of
    // hundreds of megabytes through the partial buffer, and a kernel launch.
    __nv_bfloat16* __restrict__ out_direct, float out_scale,
    int H, int topk, int splits, int chunk, float scale, int hpb)
{
    constexpr int KS = D + 8;                // padded shared row stride
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int MMA_M = 16;                // heads per QK tile
    constexpr int NBLK = TILE / 8;           // QK n-blocks of 8 keys
    constexpr int NBLK_E = NBLK < NWARPS ? NBLK : NWARPS;
    constexpr int KSL = NWARPS / NBLK_E;     // leftover warps split D instead
    constexpr int KSTEPS = (D / 16) / KSL;
    constexpr int MTILES = DV / 16;          // PV m-tiles (dims)
    constexpr int MTPW = MTILES / NWARPS;    // m-tiles per warp
    constexpr int NT = MMA_M / 8;            // PV n-tiles (heads)
    constexpr int PS = TILE + 8;             // padded P row stride
    // An fp8 cache is read into registers, widened, and written to shared as
    // bf16, so there is nothing for cp.async to do and only one buffer is
    // needed; the load latency is hidden by prefetching the next tile into
    // registers instead. The dequant scale never appears here: it folds into
    // the softmax scale for QK and into the merged output for PV.
    constexpr bool KV8 = !std::is_same<KT, __nv_bfloat16>::value;
    static_assert(!KV8 || NBUF == 1, "the fp8 path widens through registers");
    constexpr int EPL = KV8 ? 16 : 8;        // cache elements per 16-byte load
    constexpr int NPF = (TILE * (D / EPL) + NWARPS * 32 - 1) / (NWARPS * 32);
    // The softmax pass is grid-strided over 16 x TILE scores and reduces across
    // the threads that share a head, so a head must not straddle a warp.
    static_assert(TILE == 16 || TILE == 32, "softmax reduction assumes 16 or 32 keys");
    static_assert((MMA_M * TILE) % NTHREADS == 0, "softmax pass must tile evenly");
    static_assert(TILE % 16 == 0, "PV mma consumes 16 keys at a time");

    // Q needs 16 x KS halves and a K tile needs TILE x KS. At TILE = 16 those
    // are the same size, so Q is staged into the second K buffer, read out into
    // register fragments, and the buffer is handed back to the pipeline. Double
    // buffering therefore costs no shared memory at all.
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);      // NBUF x TILE x KS
    __nv_bfloat16* q_s = k_s + (NBUF - 1) * TILE * KS;                    // aliases a buffer
    __nv_bfloat16* p_s = k_s + NBUF * TILE * KS;                          // 16 x PS
    float* s_s = reinterpret_cast<float*>(p_s + MMA_M * PS);              // KSL x 16 x TILE
    float* m_s = s_s + KSL * MMA_M * TILE;                                // 16
    float* l_s = m_s + MMA_M;                                             // 16
    float* c_s = l_s + MMA_M;                                             // 16
    // The selection list is held a slab at a time, not a whole chunk. A whole
    // chunk is 8 KB at chunk 2048, and that is exactly what pushed the block
    // over 50 KB and down to one resident block per SM. A slab is 1 KB and costs
    // one extra barrier every SLAB_T tiles. One tile of lookahead, because
    // staging runs a tile ahead of compute.
    constexpr int SLAB_T = 16;
    constexpr int SLAB = (SLAB_T + 1) * TILE;
    int* sel_s = reinterpret_cast<int*>(c_s + MMA_M);                     // SLAB

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = blockIdx.x;
    const int hg = blockIdx.y;
    const int b = blockIdx.z;
    const int h0 = hg * hpb;

    const int lo = s * chunk;
    const int valid = seqlens ? seqlens[b] : topk;
    const int hi = min(lo + chunk, valid);

    // Q for this block's heads, staged once, 16 bytes at a time -- one scalar
    // half per iteration meant 36 dependent global loads before the first tile.
    constexpr int QV = D / 8;
    for (int i = tid; i < MMA_M * QV; i += NTHREADS) {
        const int r = i / QV, c = (i % QV) * 8;
        const int h = h0 + r;
        int4 v = int4{0, 0, 0, 0};
        if (r < hpb && h < H) v = *reinterpret_cast<const int4*>(q + ((size_t) b * H + h) * D + c);
        *reinterpret_cast<int4*>(q_s + r * KS + c) = v;
    }
    if (tid < MMA_M) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    int slab0 = 0;                                    // first tile held by the slab
    auto load_slab = [&](int t0) {
        slab0 = t0;
        for (int i = tid; i < SLAB; i += NTHREADS) {
            const int g = lo + t0 * TILE + i;
            sel_s[i] = g < hi ? sel[(size_t) b * topk + g] : -1;
        }
    };
    load_slab(0);

    // acc[t][n] holds C[dim][head] for m-tile (warp + t*NWARPS), n-tile n.
    FragC acc[MTPW][NT];
#pragma unroll
    for (int t = 0; t < MTPW; ++t)
#pragma unroll
        for (int nn = 0; nn < NT; ++nn)
#pragma unroll
            for (int f = 0; f < 4; ++f) acc[t][nn][f] = 0.f;

    __syncthreads();

    // Q's A fragments never change, so read them once and free the buffer.
    const int nb_w = warp % NBLK_E, ksl = warp / NBLK_E;
    FragA q_f[KSTEPS];
#pragma unroll
    for (int ki = 0; ki < KSTEPS; ++ki)
        ldsm4(q_f[ki], q_s + ((lane & 7) + 8 * ((lane >> 3) & 1)) * KS
                            + (ksl * KSTEPS + ki) * 16 + (lane >> 4) * 8);
    __syncthreads();

    const int ntiles = (hi - lo + TILE - 1) / TILE;

    // The selection list is a gather: reading it inside the staging loop would put
    // a dependent global load in front of every cp.async, which pins the kernel on
    // long_scoreboard. It was hoisted to shared above. Always commit a group, even
    // past the last tile, so wait_group(1) names the tile about to be read.
    auto stage = [&](int t) {
        if (t < ntiles) {
            __nv_bfloat16* dst_b = k_s + (NBUF > 1 ? (t & 1) : 0) * (TILE * KS);
            const int n = min(TILE, hi - lo - t * TILE);
            for (int i = tid; i < TILE * (D / 8); i += NTHREADS) {
                const int r = i / (D / 8), c = (i % (D / 8)) * 8;
                const int row = (r < n) ? sel_s[(t - slab0) * TILE + r] : -1;
                void* dst = dst_b + r * KS + c;
                if (row >= 0) {
                    const __nv_bfloat16* src =
                        reinterpret_cast<const __nv_bfloat16*>(kv) + (size_t) row * D + c;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src));
                } else {
                    *reinterpret_cast<int4*>(dst) = int4{0, 0, 0, 0};
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // fp8 path: 16 cache bytes per thread per slot, widened on the way to shared.
    uint4 pf[NPF];
    auto fetch = [&](int t) {
#pragma unroll
        for (int u = 0; u < NPF; ++u) {
            const int i = tid + u * NTHREADS;
            if (i >= TILE * (D / EPL)) break;
            const int r = i / (D / EPL), c = (i % (D / EPL)) * EPL;
            const int row = (t < ntiles && r < min(TILE, hi - lo - t * TILE))
                          ? sel_s[(t - slab0) * TILE + r] : -1;
            pf[u] = (row >= 0) ? *reinterpret_cast<const uint4*>(kv + (size_t) row * D + c)
                               : uint4{0, 0, 0, 0};
        }
    };
    auto widen = [&]() {
#pragma unroll
        for (int u = 0; u < NPF; ++u) {
            const int i = tid + u * NTHREADS;
            if (i >= TILE * (D / EPL)) break;
            const int r = i / (D / EPL), c = (i % (D / EPL)) * EPL;
            const uint32_t* w = &pf[u].x;
            uint32_t out[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                out[2 * e] = e4m3x2_bf16(w[e]);
                out[2 * e + 1] = e4m3x2_bf16(w[e] >> 16);
            }
            __nv_bfloat16* dst = k_s + r * KS + c;
            *reinterpret_cast<int4*>(dst) =
                make_int4(out[0], out[1], out[2], out[3]);
            *reinterpret_cast<int4*>(dst + 8) =
                make_int4(out[4], out[5], out[6], out[7]);
        }
    };

    if constexpr (KV8) fetch(0);
    else if constexpr (NBUF > 1) stage(0);
    for (int t = 0; t < ntiles; ++t) {
        if (t > 0 && (t - slab0) == SLAB_T) {
            __syncthreads();                          // the old slab is still being read
            load_slab(t);
            __syncthreads();
        }
        const int base = lo + t * TILE;
        const int n = min(TILE, hi - base);
        const __nv_bfloat16* k_cur = k_s + (NBUF > 1 ? (t & 1) * (TILE * KS) : 0);
        if constexpr (KV8) {
            widen();
            __syncthreads();
            fetch(t + 1);
        } else if constexpr (NBUF > 1) {
            stage(t + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
        } else {
            stage(t);
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }

        // S[16 heads][TILE keys] = Q @ K^T
        {
            for (int nb = nb_w; nb < NBLK; nb += NBLK_E) {
                // Two accumulators, not one. Every mma in this loop reads the
                // accumulator the previous one wrote, so a single fragment makes
                // the whole k-loop one dependency chain and `wait` becomes the
                // top stall. Two independent chains cost four registers.
                FragC c{}, c2{};
#pragma unroll
                for (int f = 0; f < 4; ++f) { c[f] = 0.f; c2[f] = 0.f; }
                // One ldmatrix feeds two mmas. A B fragment for m16n8k16 is
                // eight keys by sixteen dims, which is two of ldmatrix's 8x8
                // tiles in exactly the register order it produces, so an x4 load
                // covers 32 dims: four scalar shared loads become one, and the
                // eight rows it reads are KS apart, which the D+8 pad makes
                // conflict-free.
                constexpr int PAIRS = KSTEPS / 2;
#pragma unroll
                for (int kp = 0; kp < PAIRS; ++kp) {
                    const int k0 = (ksl * KSTEPS + kp * 2) * 16;
                    FragA kb4;
                    ldsm4(kb4, k_cur + (nb * 8 + (lane & 7)) * KS
                                     + k0 + 8 * (lane >> 3));
                    const FragB* fb2 = reinterpret_cast<const FragB*>(&kb4);
                    mma_m16n8k16_bf16(q_f[kp * 2], fb2[0], c);
                    mma_m16n8k16_bf16(q_f[kp * 2 + 1], fb2[1], c2);
                }
                if (KSTEPS & 1) {                 // odd tail, D = 576 only
                    const int ki = KSTEPS - 1;
                    const int k0 = (ksl * KSTEPS + ki) * 16;
                    FragB fb;
                    const __nv_bfloat16* krow = k_cur + (nb * 8 + (lane >> 2)) * KS
                                              + k0 + (lane & 3) * 2;
                    reinterpret_cast<uint32_t*>(&fb)[0] =
                        *reinterpret_cast<const uint32_t*>(krow);
                    reinterpret_cast<uint32_t*>(&fb)[1] =
                        *reinterpret_cast<const uint32_t*>(krow + 8);
                    mma_m16n8k16_bf16(q_f[ki], fb, c);
                }
#pragma unroll
                for (int f = 0; f < 4; ++f) c[f] += c2[f];
                const int cr = lane >> 2, cc = (lane & 3) * 2;
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    float* sp = s_s + ksl * (MMA_M * TILE) + nb * 8 + cc + f;
                    sp[(cr + 0) * TILE] = c[f];
                    sp[(cr + 8) * TILE] = c[2 + f];
                }
            }
        }
        __syncthreads();

        // Tile softmax: one thread per (head, key), reduced across the TILE
        // lanes that share a head.
        for (int i = tid; i < MMA_M * TILE; i += NTHREADS) {
            const int h = i / TILE, j = i % TILE;
            float raw = s_s[h * TILE + j];
#pragma unroll
            for (int u = 1; u < KSL; ++u) raw += s_s[u * (MMA_M * TILE) + h * TILE + j];
            float sc = (j < n && sel_s[(t - slab0) * TILE + j] >= 0) ? raw * scale : -INFINITY;

            float mt = sc;
#pragma unroll
            for (int off = TILE >> 1; off; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, mt);
            const bool empty = !isfinite(m_new);
            const float corr = empty ? 1.f : __expf(m_old - m_new);
            const float p = empty ? 0.f : __expf(sc - m_new);
            p_s[h * PS + j] = __float2bfloat16(p);

            float ps = p;
#pragma unroll
            for (int off = TILE >> 1; off; off >>= 1)
                ps += __shfl_xor_sync(0xffffffff, ps, off);
            __syncwarp();
            if (j == 0) { m_s[h] = m_new; l_s[h] = l_s[h] * corr + ps; c_s[h] = corr; }
        }
        __syncthreads();

        // O = O * corr + P @ V, with V transposed on the way out of shared.
#pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int hb = nn * 8 + (lane & 3) * 2;
            const float c0 = c_s[hb], c1 = c_s[hb + 1];
#pragma unroll
            for (int mt = 0; mt < MTPW; ++mt) {
                acc[mt][nn][0] *= c0; acc[mt][nn][1] *= c1;
                acc[mt][nn][2] *= c0; acc[mt][nn][3] *= c1;
            }
        }
        for (int kb = 0; kb < TILE; kb += 16) {
            FragB fb[NT];
#pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const __nv_bfloat16* pr = p_s + (nn * 8 + (lane >> 2)) * PS
                                        + kb + (lane & 3) * 2;
                reinterpret_cast<uint32_t*>(&fb[nn])[0] =
                    *reinterpret_cast<const uint32_t*>(pr);
                reinterpret_cast<uint32_t*>(&fb[nn])[1] =
                    *reinterpret_cast<const uint32_t*>(pr + 8);
            }
#pragma unroll
            for (int mt = 0; mt < MTPW; ++mt) {
                const int mbase = (warp + mt * NWARPS) * 16;
                FragA fa;
                ldsm4_trans(fa, k_cur + (kb + 8 * (lane >> 4) + (lane & 7)) * KS
                                    + mbase + 8 * ((lane >> 3) & 1));
#pragma unroll
                for (int nn = 0; nn < NT; ++nn) mma_m16n8k16_bf16(fa, fb[nn], acc[mt][nn]);
            }
        }
        __syncthreads();
    }

    if (out_direct == nullptr && tid < MMA_M && tid < hpb && h0 + tid < H) {
        const size_t o = ((size_t) b * H + h0 + tid) * splits + s;
        part_m[o] = m_s[tid]; part_l[o] = l_s[tid];
    }
    if (out_direct != nullptr && tid < MMA_M)
        c_s[tid] = l_s[tid] > 0.f ? out_scale / l_s[tid] : 0.f;
    __syncthreads();

    // The accumulator layout puts consecutive lanes on different heads at the
    // same dim, so writing it straight out is 32 scattered dwords per lane. Bounce
    // it through the (now dead) K tile instead and write coalesced rows. The 260
    // stride keeps the staging conflict-free: at 256 every head lands on one bank.
    constexpr int OSH = 256, OSS = OSH + 4;
    float* osh = reinterpret_cast<float*>(k_s);
    // The merge reads these back, but this kernel pushes many times their volume
    // of cache through L2 first and evicts them. Ask L2 to keep them.
    uint64_t keep;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(keep));
#pragma unroll
    for (int pass = 0; pass < DV / OSH; ++pass) {
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MTPW; ++mt) {
            const int mi = warp + mt * NWARPS;
            if (mi / (OSH / 16) != pass) continue;
            const int mbase = mi * 16 - pass * OSH;
#pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const int hb = nn * 8 + (lane & 3) * 2, d0 = mbase + (lane >> 2);
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    osh[(hb + (f & 1)) * OSS + d0 + 8 * (f >> 1)] = acc[mt][nn][f];
            }
        }
        __syncthreads();
        // Eight dims per thread in one 16-byte store, instead of one bf16 each.
        for (int i = tid; i < MMA_M * (OSH / 8); i += NTHREADS) {
            const int hl = i / (OSH / 8), d = (i % (OSH / 8)) * 8;
            if (hl >= hpb || h0 + hl >= H) continue;
            const float* src = osh + hl * OSS + d;
            const float w = out_direct ? c_s[hl] : 1.f;
            __nv_bfloat16* dst =
                out_direct ? out_direct + ((size_t) b * H + h0 + hl) * DV
                                        + pass * OSH + d
                           : part_o + (((size_t) b * H + h0 + hl) * splits + s) * DV
                                    + pass * OSH + d;
            asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1,%2,%3,%4}, %5;\n"
                         :: "l"(dst), "r"(f32x2_bf16(src[0] * w, src[1] * w)),
                            "r"(f32x2_bf16(src[2] * w, src[3] * w)),
                            "r"(f32x2_bf16(src[4] * w, src[5] * w)),
                            "r"(f32x2_bf16(src[6] * w, src[7] * w)), "l"(keep)
                         : "memory");
        }
    }
}

// Eviction for the autotuner: streams a buffer through L2 by reading it.
//
// Doing this with a memset is worse than useless. Dirty lines have to be
// written back, and those write-backs contend with the kernel's own reads, so
// the inflation is not uniform -- measured 1.57x at chunk 96 against 1.33x at
// chunk 256. That is a gradient, and a gradient re-ranks: it favours large
// chunks, and it compresses the landscape from a 20% spread to 3%, back inside
// the timer's resolution. Reads leave the cache just as cold with no write-back
// traffic, and are what a real model streams between two decode steps anyway.
__global__ void exl3_evict_kernel(const int4* __restrict__ p, size_t n, int* sink)
{
    int4 acc = {0, 0, 0, 0};
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n;
         i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // Never true; exists so the loads cannot be optimised away.
    if (acc.x == 0x7fffffff && acc.y == 0x7ffffffe) sink[0] = acc.z ^ acc.w;
}

// Merge the per-split partials. Standard log-sum-exp combine: rescale each
// split's output by exp(m_s - m*) and divide by the summed weight.
//
// The merge is small -- a megabyte at batch 1 -- but it was costing as much as
// the attention itself, for three reasons: every thread rescanned the whole
// per-split max/sum before doing any output work, exp(m_s - m*) was recomputed
// once per output dim instead of once per split, and 64-thread blocks left the
// machine mostly idle. Now the weights are computed once into shared, and the
// split axis is spread over SG thread groups so there are SG independent loads
// in flight per output element.
// DIMS output dims per block, VEC of them per thread, and SG groups splitting
// the split axis so there are SG independent loads in flight per dim.
template <int DV, int DIMS, int VEC, int SG>
__global__ __launch_bounds__(DIMS / VEC* SG) void mla_decode_reduce_kernel(
    const __nv_bfloat16* __restrict__ part_o,   // (B, H, splits, DV)
    const float* __restrict__ part_m,           // (B, H, splits)
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
    int H, int splits, float out_scale)
{
    constexpr int TPG = DIMS / VEC;          // threads per split group
    static_assert(TPG <= 32, "a split group must fit in one warp");
    constexpr int NT = TPG * SG;
    constexpr int NW = NT / 32;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d0 = blockIdx.z * DIMS;
    const int tid = threadIdx.x;
    const int g = tid / TPG, t = tid % TPG;

    extern __shared__ __align__(16) char red_raw[];
    float* w = reinterpret_cast<float*>(red_raw);       // splits
    float* red = w + splits;                            // NW x DIMS
    __shared__ float xchg[NW], xchl[NW];
    __shared__ float m_all, inv;

    // (max, weight) is an associative pair under the log-sum-exp combine, so one
    // reduction produces both. The old code ran two -- and read the per-split max
    // from global twice, then finished each reduction on a single thread.
    constexpr int RMAX = 8;
    const size_t mbase = ((size_t) b * H + h) * splits;
    float m_l = -INFINITY, l_l = 0.f;
#pragma unroll 1
    for (int r = 0; r < RMAX; ++r) {
        const int sp = tid + r * NT;
        if (sp >= splits) break;
        const float mm = part_m[mbase + sp], ll = part_l[mbase + sp];
        const float M = fmaxf(m_l, mm);
        l_l = isfinite(M) ? l_l * __expf(m_l - M) + ll * __expf(mm - M) : 0.f;
        m_l = M;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffff, m_l, off);
        const float l2 = __shfl_xor_sync(0xffffffff, l_l, off);
        const float M = fmaxf(m_l, m2);
        l_l = isfinite(M) ? l_l * __expf(m_l - M) + l2 * __expf(m2 - M) : 0.f;
        m_l = M;
    }
    if ((tid & 31) == 0) { xchg[tid >> 5] = m_l; xchl[tid >> 5] = l_l; }
    __syncthreads();
    if (tid < 32) {
        float mv = (tid < NW) ? xchg[tid] : -INFINITY;
        float lv = (tid < NW) ? xchl[tid] : 0.f;
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            const float m2 = __shfl_xor_sync(0xffffffff, mv, off);
            const float l2 = __shfl_xor_sync(0xffffffff, lv, off);
            const float M = fmaxf(mv, m2);
            lv = isfinite(M) ? lv * __expf(mv - M) + l2 * __expf(m2 - M) : 0.f;
            mv = M;
        }
        if (tid == 0) { m_all = mv; inv = lv > 0.f ? 1.f / lv : 0.f; }
    }
    __syncthreads();

    if (!isfinite(m_all)) {                       // no keys landed here
        for (int d = tid; d < DIMS; d += NT)
            out[((size_t) b * H + h) * DV + d0 + d] = __float2bfloat16(0.f);
        return;
    }
    for (int sp = tid; sp < splits; sp += NT)
        w[sp] = __expf(part_m[mbase + sp] - m_all);
    __syncthreads();

    float acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.f;
    for (int sp = g; sp < splits; sp += SG) {
        const uint4 r = *reinterpret_cast<const uint4*>(
            part_o + (((size_t) b * H + h) * splits + sp) * DV + d0 + t * VEC);
        const float ws = w[sp];
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&r);
#pragma unroll
        for (int e = 0; e < VEC; e += 2) {
            float lo, hi;
            bf16x2_f32(u[e >> 1], lo, hi);
            acc[e] = fmaf(lo, ws, acc[e]);
            acc[e + 1] = fmaf(hi, ws, acc[e + 1]);
        }
    }
    // Lanes of one warp span 32/TPG groups; fold those with shuffles, then the
    // NW warps through shared.
#pragma unroll
    for (int off = TPG; off < 32; off <<= 1)
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] += __shfl_xor_sync(0xffffffff, acc[e], off);
    if ((tid & 31) < TPG) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) red[(tid >> 5) * DIMS + t * VEC + e] = acc[e];
    }
    __syncthreads();
    for (int d = tid; d < DIMS; d += NT) {
        float v = red[d];
#pragma unroll
        for (int i = 1; i < NW; ++i) v += red[i * DIMS + d];
        out[((size_t) b * H + h) * DV + d0 + d] =
            __float2bfloat16(v * inv * out_scale);
    }
}

}  // namespace cuda_exl3

namespace cuda_exl3 {

// Which (chunk, block shape) wins is shape-dependent and not guessable:
// splitting keys buys blocks but grows the partial buffer the reduce must read,
// while splitting heads buys blocks for free in the reduce but re-reads K per
// group. At batch 1 the answer is a 4-head block; by batch 4 it is a 16-head
// one. So time the candidates once per shape and remember the winner.
namespace {

// Where a persisted tuner cache lives, or empty if CUDA_EXL3_TUNE_CACHE is unset.
//
// The device name and a format tag go in the *filename*, so a different card or
// a changed candidate grid cannot read another's entries: no header to parse and
// no way to mix them. Bump TUNE_TAG whenever the search space changes -- a stale
// entry is still a *valid* config (chunk and shape are only parameters, and the
// kernel is correct for any of them), so the cost of missing a bump is a slower
// pick, not a wrong answer.
#define TUNE_TAG "v1"
std::string mla_tune_cache_file() {
    const char* dir = exl3_env("CUDA_EXL3_TUNE_CACHE");
    if (!dir || !*dir) return {};
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess) return {};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return {};
    std::string name(prop.name);
    for (char& c : name)
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
            c = '_';
    return std::string(dir) + "/mla-tune-" TUNE_TAG "-" + name + ".txt";
}

// The map is process-local, so without this every boot re-tunes and every new
// batch shape costs a tune at runtime -- measured on a GB10 as 11 events before
// the server is up and more as B moves, ~15 ms each, which is what pollutes the
// first rounds of a serving A/B (@NNNtrance, #2). Persisting makes it a one-off
// per (device, build).
std::map<uint64_t, std::pair<int, int>>& mla_tune_cache() {
    static std::map<uint64_t, std::pair<int, int>> c = [] {
        std::map<uint64_t, std::pair<int, int>> m;
        const std::string path = mla_tune_cache_file();
        if (path.empty()) return m;
        FILE* f = fopen(path.c_str(), "r");
        if (!f) return m;
        unsigned long long k; int chunk, wide;
        while (fscanf(f, "%llu %d %d", &k, &chunk, &wide) == 3)
            if (chunk > 0 && wide > 0) m[(uint64_t) k] = {chunk, wide};
        fclose(f);
        if (exl3_env("CUDA_EXL3_MLA_TUNE_VERBOSE"))
            printf("[cuda-exl3] mla tune cache: %zu entries from %s\n",
                   m.size(), path.c_str());
        return m;
    }();
    return c;
}

// Appended one short line at a time. O_APPEND is atomic on POSIX for writes this
// small, so the tensor-parallel ranks share one file without a lock; a duplicate
// key from two ranks is harmless, since either is a config the search could have
// picked and the last one read simply wins.
void mla_tune_cache_store(uint64_t key, std::pair<int, int> v) {
    const std::string path = mla_tune_cache_file();
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "a");
    if (!f) return;
    fprintf(f, "%llu %d %d\n", (unsigned long long) key, v.first, v.second);
    fclose(f);
}

// Decode batches are small and the optimum moves between them, so they are
// keyed exactly. Prefill arrives as whatever the scheduler chunked to, the
// optimum is flat across it, and tuning a 4k-row shape costs hundreds of
// milliseconds -- so above 32 rows the key is bucketed to a power of two.
int mla_tune_bucket(int B) {
    if (B <= 32) return B;
    int b = 32;
    while (b < B) b <<= 1;
    return b;
}

// topk was keyed exactly, which was wrong for the same reason B is bucketed.
// The indexer returns min(index_topk, context), so during a chunked prefill the
// count grows chunk by chunk and every chunk minted a fresh key and re-tuned --
// measured on a GB10 as 1728 eviction-kernel launches inside a 6.5 s prefill,
// 2.6% of the window, by @NNNtrance in #1. Round up to a multiple of 256: the
// chunk is only a split size, so any value stays correct and a coarser key
// costs at most a slightly worse pick, while the number of distinct keys a
// prefill can mint is bounded by index_topk / 256.
int mla_tune_topk_bucket(int topk) {
    constexpr int Q = 256;
    return ((topk + Q - 1) / Q) * Q;
}

uint64_t mla_tune_key(int B, int H, int topk, int D, bool kv8) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t v : {(uint64_t) mla_tune_bucket(B), (uint64_t) H,
                       (uint64_t) mla_tune_topk_bucket(topk),
                       (uint64_t) D, (uint64_t) kv8}) {
        h ^= v; h *= 1099511628211ull;
    }
    return h;
}

// Ask the device rather than assuming sm_120's 99 KB: GB10 is the same major
// version and this plugin is meant to build for it from the same source.
int mla_max_dyn_smem() {
    static const int cap = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return v;
    }();
    return cap;
}

bool mla_tuning_enabled() {
    static const bool on = [] {
        const char* e = exl3_env("CUDA_EXL3_MLA_TUNE");
        return !(e && *e == '0');
    }();
    return on;
}

}  // namespace

at::Tensor mla_decode(const at::Tensor& q, const at::Tensor& kv,
                      const at::Tensor& sel, const at::Tensor& seqlens,
                      double scale, int64_t v_head_dim, int64_t split_chunk,
                      int64_t heads_per_block, double kv_scale)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dim() == 3, "mla_decode: q must be (batch, heads, head_dim)");
    TORCH_CHECK(kv.dim() == 2, "mla_decode: kv must be (rows, head_dim)");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "mla_decode: q must be bfloat16");
    const bool kv8 = kv.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16 || kv8,
                "mla_decode: kv must be bfloat16 or float8_e4m3fn");
    // An fp8 cache stores k / kv_scale. Both matmuls are linear in it, so the
    // scale never has to touch an element: it folds into the softmax scale for
    // Q@K^T and into the merged output for P@V.
    const double sscale = kv8 ? scale * kv_scale : scale;
    const double oscale = kv8 ? kv_scale : 1.0;
    TORCH_CHECK(sel.scalar_type() == at::kInt, "mla_decode: sel must be int32");

    const int B = (int) q.size(0), H = (int) q.size(1), D = (int) q.size(2);
    const int topk = (int) sel.size(1);
    const int DV = (int) v_head_dim;
    // 576 is the DeepSeek latent (512 nope + 64 rope); 512 is GLM-5.3-Flash,
    // which has qk_rope_head_dim = 0.
    TORCH_CHECK((D == 576 || D == 512) && DV == 512,
                "mla_decode: built for head_dim 576 or 512 with v_head_dim 512, got ",
                D, "/", DV);

    int chunk = (int) split_chunk;
    int wide = (int) heads_per_block;  // 1: 8 warps x 16 keys, 2: 16 warps x 32
    if (chunk <= 0 || wide <= 0) {
        const uint64_t key = mla_tune_key(B, H, topk, D, kv8);
        auto& cache = mla_tune_cache();
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::pair<int, int> best{B >= 64 ? 512 : 32, 1};
            float best_ms_report = 0.f;
            int ring_report = 0;
            if (mla_tuning_enabled() &&
                at::cuda::currentStreamCaptureStatusMayInitCtx() ==
                    at::cuda::CaptureStatus::None) {
                // Two regimes, and searching the wrong one is expensive. Few
                // rows means the machine is starved and the split count is what
                // fills it, so search small chunks and the half-head shapes.
                // Many rows -- prefill -- already has all the parallelism it
                // needs from the rows themselves, and every extra split is pure
                // partial-buffer traffic, so search large chunks only. One timed
                // rep there instead of three: a 4k-row candidate is milliseconds.
                const bool many = B >= 64;
                const std::vector<int> chunks =
                    many ? std::vector<int>{256, 384, 512, 768, 1024, 2048}
                         : std::vector<int>{16, 32, 48, 64, 96, 128, 192, 256, 384};
                const std::vector<int> shapes =
                    many ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 3, 4};
                // A batch-1 decode is a few microseconds, so a fixed rep count
                // times something the same order as the timer and the launch
                // overhead around it -- on a GB10 that left the winner 4-6% off
                // the grid optimum. Size the window to the kernel instead:
                // probe, run enough reps to fill ~400 us, and take the best of
                // three so one scheduling hiccup cannot decide it.
                // Which cache regime should the tuner measure? It used to time
                // one selection repeatedly, so from the second rep the rows were
                // L2-resident. That is not what decode does, and the reason is
                // not the one I first argued. A sequence's top-k list really does
                // change by only about a row in 2048 per step, so consecutive
                // steps re-touch almost the same rows -- but between layer L at
                // step t and layer L at step t+1 the whole rest of the model
                // streams through, gigabytes against tens of megabytes of L2.
                // Nothing survives, so the overlap buys nothing.
                //
                // Measured: with that traffic modelled, warm, drifting and
                // independent selections all collapse onto the same curve, and
                // on this card the evicted optimum at batch 16 is two chunk
                // tiers away from the warm one (c96 -> c192) at 1.85x the time.
                //
                // So rotate enough distinct selections that the rows they touch
                // overrun the L2 twice over, which costs only the index tensors
                // -- the rows themselves are already in the cache tensor.
                // CUDA_EXL3_MLA_TUNE_RING overrides the count; 0 restores the
                // old warm behaviour for A/B.
                // Rotate always, sized so the rows overrun L2 twice over.
                //
                // This briefly had a gate -- rotate only when one selection was
                // at least an eighth of L2 -- on the theory that below that the
                // rows stay resident whatever else runs. That theory is exactly
                // what the eviction experiment disproved: at batch 1 a selection
                // is 8% of a 24 MB L2 and still loses 1.15-1.27x once traffic
                // streams past it, and in a real step gigabytes do. The gate was
                // generalised from this card measuring 1.00x either way at batch
                // 1, which is a different thing entirely -- there the kernel is
                // latency-bound on split-k merge traffic, not residency-bound.
                //
                // Measured with the gate removed: batch 1 goes 1.034x -> 1.007x
                // of the evicted optimum here, and the gate was costing 15% on a
                // 24 MB L2, where it switched rotation off at exactly the batch
                // size that needed it most.
                std::vector<at::Tensor> sel_ring;
                if (!many) {
                    const int64_t per_sel = (int64_t) B * topk * D * (kv8 ? 1 : 2);
                    int nsel = (int) ((2.0 * (double) exl3_dev_l2()) / (double) per_sel);
                    nsel = std::min(std::max(nsel, 8), 64);
                    const char* e = exl3_env("CUDA_EXL3_MLA_TUNE_RING");
                    if (e && *e) nsel = std::max(0, std::min(atoi(e), 256));
                    sel_ring.reserve(nsel);
                    for (int i = 0; i < nsel; ++i)
                        sel_ring.push_back(
                            at::randint(0, kv.size(0), {B, topk}, sel.options()));
                }
                auto pick_sel = [&](int r) -> const at::Tensor& {
                    return sel_ring.empty() ? sel : sel_ring[r % sel_ring.size()];
                };

                // Rotating the selection makes the cache *rows* cold, and that
                // is not the same as a cold cache: everything else the kernel
                // touches stays resident, and back-to-back reps pipeline. At
                // batch 1 that leaves the candidates within a microsecond of one
                // another -- flat inside the timer's resolution -- so the winner
                // was chosen by noise. Six of twelve picks flipped between runs
                // with identical reported times, and chunk 16 was being picked
                // where it is the worst candidate under eviction by 16%.
                //
                // So clear the cache between candidate reps, and bracket events
                // around the kernel alone so the clearing is not what is timed.
                // Traffic, not footprint, is what evicts, so a modest buffer is
                // memset repeatedly rather than allocating twice L2 -- at server
                // startup the KV cache is already placed and a 256 MB transient
                // is a poor thing to demand.
                int evict_passes = 0;
                at::Tensor evict_buf;
                if (!many) {
                    const int64_t l2 = exl3_dev_l2();
                    const int64_t span = std::min<int64_t>(l2, 32 << 20);
                    // 1.5x L2 of distinct lines is enough to clear it; 2x only
                    // costs tuning time, which on a 48-SM part is already the
                    // dominant term.
                    evict_passes = (int) ((3 * l2 / 2 + span - 1) / span);
                    evict_buf = at::zeros({span}, at::TensorOptions()
                                                      .dtype(at::kByte).device(q.device()));
                }
                auto stream = at::cuda::getCurrentCUDAStream();
                auto evict = [&]() {
                    const size_t n16 = (size_t) evict_buf.numel() / 16;
                    const int threads = 256;
                    const int blocks = (int) std::min<size_t>((n16 + threads - 1) / threads,
                                                              (size_t) exl3_dev_sms() * 8);
                    for (int i = 0; i < evict_passes; ++i)
                        exl3_evict_kernel<<<blocks, threads, 0, stream>>>(
                            (const int4*) evict_buf.data_ptr(), n16,
                            (int*) evict_buf.data_ptr());
                };

                const int reps = many ? 1 : 8;
                const int trials = many ? 1 : 3;
                std::vector<cudaEvent_t> ev_b(reps), ev_e(reps);
                for (int i = 0; i < reps; ++i) {
                    cudaEventCreate(&ev_b[i]); cudaEventCreate(&ev_e[i]);
                }
                float best_ms = 1e30f;
                for (int c : chunks) {
                    for (int wd : shapes) {
                        if (c > topk && c != chunks.front()) continue;
                        mla_decode(q, kv, pick_sel(0), seqlens, scale, v_head_dim,
                                   c, wd, kv_scale);
                        float samples[8];
                        for (int tr = 0; tr < trials; ++tr) {
                            for (int r = 0; r < reps; ++r) {
                                if (evict_passes) evict();
                                cudaEventRecord(ev_b[r], stream);
                                mla_decode(q, kv, pick_sel(r + tr * reps), seqlens,
                                           scale, v_head_dim, c, wd, kv_scale);
                                cudaEventRecord(ev_e[r], stream);
                            }
                            cudaEventSynchronize(ev_e[reps - 1]);
                            float sum = 0.f;
                            for (int r = 0; r < reps; ++r) {
                                float one = 0.f;
                                cudaEventElapsedTime(&one, ev_b[r], ev_e[r]);
                                sum += one;
                            }
                            samples[tr] = sum;
                        }
                        std::sort(samples, samples + trials);
                        const float ms = samples[trials / 2];
                        if (ms < best_ms) { best_ms = ms; best = {c, wd}; }
                    }
                }
                for (int i = 0; i < reps; ++i) {
                    cudaEventDestroy(ev_b[i]); cudaEventDestroy(ev_e[i]);
                }
                ring_report = (int) sel_ring.size();
                best_ms_report = best_ms * 1000.f / (float) reps;
            }
            if (exl3_env("CUDA_EXL3_MLA_TUNE_VERBOSE")) {
                printf("[cuda-exl3] mla tune B=%d H=%d topk=%d D=%d kv=%s "
                       "-> chunk=%d shape=%d ring=%d (%.1f us/call, evicted)\n",
                       B, H, topk, D, kv8 ? "fp8" : "bf16", best.first, best.second,
                       ring_report, best_ms_report);
                fflush(stdout);
            }
            it = cache.emplace(key, best).first;
            mla_tune_cache_store(key, best);
        }
        if (chunk <= 0) chunk = it->second.first;
        if (wide <= 0) wide = it->second.second;
    }
    // Splitting keys needs an LSE merge, so more splits means a bigger partial
    // buffer and a costlier reduce. Splitting *heads* costs nothing -- their
    // outputs are disjoint -- at the price of re-reading K once per head group.
    // The mma tile is 16 heads wide, so that is the only useful group size.
    // Codes 3 and 4 halve the head group. That wastes half of every mma, but at
    // batch 1 the kernel is nowhere near bandwidth-bound and the extra blocks
    // cost nothing but a second read of the cache -- the partial write, which is
    // the expensive part, is per head and so unchanged.
    const int hpb = std::min(H, (wide >= 3 && H >= 16) ? 8 : 16);
    const int hgroups = (H + hpb - 1) / hpb;

    const int splits = (topk + chunk - 1) / chunk;

    auto dev = q.device();
    // Allocated per call rather than kept in a global workspace. A workspace
    // that grows reallocates, and a reallocation between two CUDA graph captures
    // leaves the earlier graph pointing at freed memory, which faults during
    // capture of the later one. Under capture these land in the graph's private
    // pool and stay valid; outside it the caching allocator makes them cheap.
    const auto fopt = at::TensorOptions().dtype(at::kFloat).device(dev);
    auto po = at::empty({(int64_t) B * splits * H * DV},
                        at::TensorOptions().dtype(at::kBFloat16).device(dev));
    auto pm = at::empty({(int64_t) B * splits * H}, fopt);
    auto pl = at::empty({(int64_t) B * splits * H}, fopt);
    auto out = at::empty({B, H, DV}, q.options());
    // Only when a single block owns every head of a row: with head groups the
    // heads are disjoint, so each still writes its own slice, which is fine.
    const bool direct = (splits == 1);

    auto stream = at::cuda::getCurrentCUDAStream();
    // Two block shapes. The wide one halves the block count for the same thread
    // count, which halves both the Q re-read and the partial write: those are
    // per block, not per key, and at low batch they outweigh the cache traffic.
#define MLA_LAUNCH(D_, NW_, TILE_, KT_, NB_)                                             \
    do {                                                                       \
        constexpr int KSL_ = (NW_) / ((TILE_) / 8);                            \
        constexpr int NBUF_ = std::is_same<KT_, __nv_bfloat16>::value ? (NB_) : 1;\
        const size_t smem = NBUF_ * (TILE_) * ((D_) + 8)                       \
                              * sizeof(__nv_bfloat16)                          \
                          + 16 * ((TILE_) + 8) * sizeof(__nv_bfloat16)         \
                          + (size_t) KSL_ * 16 * (TILE_) * sizeof(float)       \
                          + 3 * 16 * sizeof(float)                             \
                          + 17 * (TILE_) * sizeof(int);                        \
        auto kern = mla_decode_partial_kernel<D_, 512, NW_, TILE_, KT_, NBUF_>;\
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,\
                             mla_max_dyn_smem());                              \
        kern<<<grid, (NW_) * 32, smem, stream>>>(                              \
            (const __nv_bfloat16*) q.data_ptr(),                               \
            (const KT_*) kv.data_ptr(), sel.data_ptr<int>(),                   \
            seqlens.numel() ? seqlens.data_ptr<int>() : nullptr,               \
            (__nv_bfloat16*) po.data_ptr(), pm.data_ptr<float>(),              \
            pl.data_ptr<float>(),                                              \
            direct ? (__nv_bfloat16*) out.data_ptr() : nullptr,                \
            (float) oscale, H, topk, splits, chunk, (float) sscale, hpb);      \
    } while (0)

    dim3 grid(splits, hgroups, B);
// This shape is a local optimum, and the search around it is worth recording
// because every neighbour is worse for a different reason. Prefill sits at
// ~150 TFLOPS against this card's 417, with the tensor pipe at 30%, so the
// question was always what the other 70% is waiting for.
//
//   8 warps, 32-key tile, one buffer   1.8x slower. Twice the mma between
//     barriers, but losing the cp.async prefetch costs more than the barriers
//     save.
//   16 warps, 16-key tile              2.2x slower, and the most informative.
//     It halves per-thread accumulators and held Q fragments, so the compiler
//     lands at 61 registers instead of 122 and occupancy doubles, 33% -> 66% of
//     peak warps. The tensor pipe does not move off 30%. Occupancy is simply
//     not what binds here; halving the work each warp does between the same
//     four barriers costs more than the extra warps return.
//   8 warps, 32-key tile, two buffers  2.5x slower. Keeps the prefetch and
//     doubles the mma per warp per barrier, but a 32-key tile forces the
//     k-slices down to two, so each warp holds 16 Q fragments instead of 8 --
//     64 registers of them -- and the register file gives out.
//   16 warps, 32-key tile              already in the search as shape 2, and
//     the autotuner declines it at every size.
//
// So the tile geometry is boxed in: the two knobs that would raise mma per
// barrier both raise register pressure through the held Q fragments, and the
// one that lowers register pressure lowers work per barrier instead. Moving
// past 30% needs the ratio of mma to softmax-and-staging work inside a tile to
// change, which is a different kernel, not a different tiling of this one.
#define MLA_PICK(D_, KT_)                                                      \
    do {                                                                       \
        if (wide == 2 || wide == 4) { MLA_LAUNCH(D_, 16, 32, KT_, 2); }        \
        else { MLA_LAUNCH(D_, 8, 16, KT_, 2); }                                \
    } while (0)

    if (D == 576) {
        if (kv8) { MLA_PICK(576, __nv_fp8_e4m3); } else { MLA_PICK(576, __nv_bfloat16); }
    } else {
        if (kv8) { MLA_PICK(512, __nv_fp8_e4m3); } else { MLA_PICK(512, __nv_bfloat16); }
    }
#undef MLA_PICK
#undef MLA_LAUNCH

    // The merge splits its 256 threads between output dims and the split axis.
    // Spreading over more splits than exist just idles the extra groups, so the
    // shape follows the split count: at 11 splits, 32 groups left two thirds of
    // the block doing nothing and still paid for the cross-group reduce.
#define MLA_REDUCE(SG_)                                                        \
    do {                                                                       \
        constexpr int VEC_ = 8;                                                \
        constexpr int DIMS_ = VEC_ * 256 / (SG_) < 256                         \
                            ? VEC_ * 256 / (SG_) : 256;                        \
        constexpr int NT_ = DIMS_ / VEC_ * (SG_);                              \
        const size_t rsm = ((size_t) splits + NT_ / 32 * DIMS_) * sizeof(float);\
        mla_decode_reduce_kernel<512, DIMS_, VEC_, SG_>                        \
            <<<dim3(H, B, 512 / DIMS_), NT_, rsm, stream>>>(                   \
                (const __nv_bfloat16*) po.data_ptr(), pm.data_ptr<float>(),    \
                pl.data_ptr<float>(), (__nv_bfloat16*) out.data_ptr(), H,      \
                splits, (float) oscale);                                       \
    } while (0)

    if (direct) { /* the partial kernel already wrote the answer */ }
    else if (splits >= 24) { MLA_REDUCE(32); }
    else if (splits >= 12) { MLA_REDUCE(16); }
    else if (splits >= 6) { MLA_REDUCE(8); }
    else { MLA_REDUCE(4); }
#undef MLA_REDUCE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace cuda_exl3
