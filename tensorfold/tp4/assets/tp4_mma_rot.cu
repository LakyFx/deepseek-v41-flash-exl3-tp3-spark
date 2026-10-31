#ifndef TF_TP4_GU_CTAS
#define TF_TP4_GU_CTAS 1
#endif
#ifndef TF_TP4_DOWN_CTAS
#define TF_TP4_DOWN_CTAS 2
#endif
// Grouped EXL3 expert GEMV (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): rows stay independent, K ranges fixed by shape, warps summed in order.
#include <stdint.h>
#include <stddef.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>




namespace tf_exl3z {

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2)
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    // right shift of window j of a run (run starts at an even position) relative to the run's last window
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight.
template <int CB, int K2, int NT, int PF>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[4] = {load_pair(x0 + k, ok0), load_pair(x1 + k, ok1), load_pair(x0 + k + 8, ok0),
                                 load_pair(x1 + k + 8, ok1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// warp_tiles for G member tiles at once: each k tile's weights are decoded once and fed to every live tile's rows (off:
// a row's element offset in X, -1 for none); a row's mma sequence is warp_tiles' (same k tiles, same order, from zero).
template <int CB, int K2, int NT, int PF, int G>
__device__ __forceinline__ void warp_tiles_rows(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                                const half* __restrict__ X, const int (&off)[G][2], int live, int lane,
                                                float (&acc)[G][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    // the rows' A fragments a k tile ahead (the loads' latency behind this tile's decode and mma)
    const half* xr[G][2];
    bool ok[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            ok[g][j] = off[g][j] >= 0;
            xr[g][j] = X + (ok[g][j] ? off[g][j] : 0) + kt0 * 16;
        }
    uint32_t an[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
        if (g < live) {
            an[g][0] = load_pair(xr[g][0], ok[g][0]);
            an[g][1] = load_pair(xr[g][1], ok[g][1]);
            an[g][2] = load_pair(xr[g][0] + 8, ok[g][0]);
            an[g][3] = load_pair(xr[g][1] + 8, ok[g][1]);
        }

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[G][4];
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int c = 0; c < 4; ++c) a[g][c] = an[g][c];
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        if (g < live) {
                            an[g][0] = load_pair(xr[g][0] + k, ok[g][0]);
                            an[g][1] = load_pair(xr[g][1] + k, ok[g][1]);
                            an[g][2] = load_pair(xr[g][0] + k + 8, ok[g][0]);
                            an[g][3] = load_pair(xr[g][1] + k + 8, ok[g][1]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        if (g < live) {
                            mma16816(acc[g][i][0], a[g], b0);
                            mma16816(acc[g][i][1], a[g], b1);
                        }
                    }
                }
            }
        }
    }
}

// The K2 values an instance covering [LO, HI] compiles (half-bits 2..16).
__host__ __device__ constexpr bool k2_supported(int k2) {
    return k2 >= 2 && k2 <= 16;
}

// Program (expert u, n block, split and member tile): up to 16 members times W_q over the split's K range; warps added in order.
template <int CB, int NT, int W, int PF, int LO, int HI>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);    \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

    // warps' partial sums through shared memory, added in warp order
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

// Prompt chunks: a program per (group of G member tiles, n block, expert u, split), the group's weights decoded once.
// Every row gets grouped_kernel's bits: the same K range a warp, the same mma sequence, warps added in order.
// Launch order runs an expert's member groups back to back (its weights from L2), then its column blocks (its rows).
// FOLD: one program runs every split in order and writes their sum from 0 (the gate/up epilogue's order: the epilogue
// then reads one split instead of SK), so the fp32 partials written and read drop SK times, the bits unchanged.
template <int CB, int NT, int W, int PF, int LO, int HI, int G, bool FOLD>
__global__ void __launch_bounds__(W * 32) grouped_rows_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int nexp,
    const int* __restrict__ work) {
    constexpr int GR = 16 * G;
    constexpr int TPT = 16 * NT * 16 / (W * 32);          // a thread's outputs of a member tile in the warp sum
    // the work list (prompt chunks): grid.z walks (expert place, member group) pairs where it walked the expert places
    // (nexp is then the list's length and grid.x 1), so no program is launched for a group an expert does not have, and
    // an expert's programs still run back to back (its rows stay in L2 across its column blocks)
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mg = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int zi = (int)(blockIdx.z / nexp);              // (matrix, split)
    const int splits = FOLD ? 1 : SK;                     // grid splits (FOLD runs them all in one program)
    const int split0 = FOLD ? 0 : zi % SK;
    const int mat = zi / splits;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[GR];
    for (int i = threadIdx.x; i < GR; i += W * 32) {
        const int m = mg * GR + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this group is empty
    int live = 0;                                         // its occupied tiles are a prefix
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;
    int off[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int r0 = rows_sh[g * 16 + g8], r1 = rows_sh[g * 16 + g8 + 8];
        off[g][0] = r0 < 0 ? -1 : r0 * K + 2 * t;
        off[g][1] = r1 < 0 ? -1 : r1 * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = blockIdx.y * NT;
    float tot[G][TPT];                                    // FOLD: the splits' sums so far, from 0
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < TPT; ++j) tot[g][j] = 0.f;
    __shared__ float red[W][16][NT * 16];

    for (int split = split0; split < (FOLD ? SK : split0 + 1); ++split) {
        const int kt0 = split * per_split + warp * per_warp;
        float acc[G][NT][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;

        switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles_rows<CB, K2_, NT, PF, G>(T, NTILES, kt0, per_warp, nt0, X, off, live, lane, acc);        \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
            TF_EXL3X_CASE(2)
            TF_EXL3X_CASE(3)
            TF_EXL3X_CASE(4)
            TF_EXL3X_CASE(5)
            TF_EXL3X_CASE(6)
            TF_EXL3X_CASE(7)
            TF_EXL3X_CASE(8)
            TF_EXL3X_CASE(9)
            TF_EXL3X_CASE(10)
            TF_EXL3X_CASE(11)
            TF_EXL3X_CASE(12)
            TF_EXL3X_CASE(13)
            TF_EXL3X_CASE(14)
            TF_EXL3X_CASE(15)
            TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
            default:
                __trap();
        }

        // warps' partial sums through shared memory, added in warp order, one member tile after another
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        red[warp][g8][col] = acc[g][i][h][0];
                        red[warp][g8][col + 1] = acc[g][i][h][1];
                        red[warp][g8 + 8][col] = acc[g][i][h][2];
                        red[warp][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    if constexpr (FOLD) {
                        tot[g][j] += s;
                    } else {
                        const int r = rows_sh[g * 16 + row];
                        if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
                    }
                }
                __syncthreads();
            }
        }
    }
    if constexpr (FOLD) {                                 // one split's worth: Z [mats, 1, P, N]
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[g][j];
                }
            }
        }
    }
}

// Prompt chunks with the weights shared through shared memory: a program per (64 member rows, 64 columns, expert u,
// matrix), eight warps: warp w runs member rows 16 (w % 4) .. + 15 against columns 32 (w / 4) .. + 31. The k tiles go
// in steps of MMA_KB (4): during a step the warps decode the next step's sixteen 16 x 16 weight tiles (two each) into
// shared memory as mma B fragments, while the step's mma read the fragments decoded during the step before (two
// buffers, one barrier a step), so a weight is decoded once for up to 64 rows (grouped_rows: 32). The rows come through
// a ring of shared-memory stages (cp.async, MMA_STAGES - 1 steps ahead, ldmatrix), the trellis words two steps ahead.
// (Measured on GB10, 2,048 rows: 2 k tiles a step or 128 rows a program were slower; ncu showed the kernel latency- and
// issue-bound, now close to its weight and row traffic.)
// Every row keeps grouped_kernel's bits: the K range splits into SK x WK chains, each an mma chain from zero over the
// same k tiles in the same order; the WK chains of a split are added in order; with FOLD the splits are added in order
// from 0 (the gate/up epilogue's order: it then reads one split), without (SK 1) the split's sum is written as is.
// Transposed MMA/member-group adaptation after TensorFold x3gm2.cuh/exl3_prompt.cuh
// 612332f60668744b32a82eca8a91afa2b2929d69, Apache-2.0.
// Their common prompt helpers credit BobClawblaw #976 and MiaAI-Lab patches0004/0009.
// Trellis helpers above retain ExLlamaV3 MIT Copyright2025 Turboderp provenance.
constexpr int R_ROWS=64, R_KB=4, R_STAGES=3, R_LD=R_KB*16+8;
__device__ __forceinline__ void r_copy(void* dst,const void* src,bool valid) {
    const unsigned s=(unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void r_commit(){asm volatile("cp.async.commit_group;\n" ::);}
template<int N> __device__ __forceinline__ void r_wait(){asm volatile("cp.async.wait_group %0;\n" ::"n"(N));}
__device__ __forceinline__ void r_ld4(uint32_t (&a)[4],const void* p) {
    const unsigned s=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(s));
}
template<int K2,bool FOLD,int NB>
__device__ __forceinline__ void r_slice(const uint32_t* T,int NTILES,int nt0,
    const half* X,int K,const int* rows,int warp,int lane,
    half (&As)[R_STAGES][R_ROWS][R_LD],uint32_t (&Ws)[R_STAGES][R_KB][8][Fmt<K2>::TW],float (&tot)[8][4]) {
    constexpr int TW=Fmt<K2>::TW;
    constexpr int KT=FOLD?320:72,per_range=FOLD?20:18,WK=4;
    if(KT%R_KB!=0 || KT%(per_range*WK)!=0){__trap();}
    const LaneMap<K2> map(lane);const int steps=KT/R_KB;
    auto load=[&](int st) {
        const int slot=st%R_STAGES;
        for(int z=threadIdx.x;z<NB*16*(R_KB*2);z+=256) {
            const int row=z/(R_KB*2),chunk=z%(R_KB*2),cr=rows[row];
            r_copy(&As[slot][row][chunk*8],X+(size_t)(cr<0?0:cr)*K+st*R_KB*16+chunk*8,cr>=0);
        }
        for(int z=threadIdx.x;z<R_KB*8*K2;z+=256) {
            const int part=z%K2,nt=(z/K2)%8,kk=z/(K2*8);
            r_copy(&Ws[slot][kk][nt][part*4],T+(((size_t)st*R_KB+kk)*NTILES+nt0+nt)*TW+part*4,true);
        }
    };
#pragma unroll
    for(int st=0;st<R_STAGES-1;++st){if(st<steps)load(st);r_commit();}
    float acc[2*NB][4],sacc[2*NB][4];
#pragma unroll
    for(int j=0;j<2*NB;++j)
#pragma unroll
      for(int q=0;q<4;++q)acc[j][q]=sacc[j][q]=0.f;
    int left=per_range,wk=0;
    auto fold_chain=[&]() {
#pragma unroll
            for(int j=0;j<2*NB;++j)
#pragma unroll
              for(int q=0;q<4;++q){sacc[j][q]=wk==0?acc[j][q]:sacc[j][q]+acc[j][q];acc[j][q]=0.f;}
            left=per_range;
            if(++wk==WK){wk=0;
#pragma unroll
                for(int j=0;j<2*NB;++j)
#pragma unroll
                  for(int q=0;q<4;++q)tot[j][q]=FOLD?tot[j][q]+sacc[j][q]:sacc[j][q];
            }
    };
    for(int st=0;st<steps;++st) {
        r_wait<R_STAGES-2>();__syncthreads();
        if(st+R_STAGES-1<steps)load(st+R_STAGES-1);r_commit();
#pragma unroll
        for(int kk=0;kk<R_KB;++kk) {
            uint32_t words[Fmt<K2>::LW];
#pragma unroll
            for(int j=0;j<Fmt<K2>::LW;++j){const int i=j*32+lane;words[j]=i<TW?Ws[st%R_STAGES][kk][warp][i]:0u;}
            uint32_t b0[2],b1[2];decode_tile<2,K2>(words,map,lane,b0,b1);
            const uint32_t a[4]={b0[0],b1[0],b0[1],b1[1]};
#pragma unroll
            for(int jb=0;jb<NB;++jb) {
                const int row=jb*16+(lane&7)+((lane>>4)&1)*8,col=((lane>>3)&1)*8;
                uint32_t r[4];r_ld4(r,&As[st%R_STAGES][row][kk*16+col]);
                const uint32_t bl[2]={r[0],r[1]},bh[2]={r[2],r[3]};
                mma16816(acc[2*jb],a,bl);mma16816(acc[2*jb+1],a,bh);
            }
            if constexpr(!FOLD) {if(--left==0) fold_chain();}
        }
        if constexpr(FOLD) {left-=R_KB;if(left==0) fold_chain();}
    }
    r_wait<0>();__syncthreads();
}
template<int CB,int LO,int HI,bool FOLD,int KB>
__global__ void __launch_bounds__(256,(FOLD?TF_TP4_GU_CTAS:TF_TP4_DOWN_CTAS)) grouped_mma_kernel(
 const half* X0,const half* X1,const int64_t* TP0,const int64_t* TP1,const int* K2_0,const int* K2_1,
 const int* uids,const int* ucount,const int* members,float* Z,int K,int N,int P,int SK,int WK,int maxm,int slots,int nexp,const int* work) {
    static_assert(CB==2&&KB==4,"checkpoint scope");
    // Payload is confined to the native TP4 geometries qualified below.
    if(K!=(FOLD?5120:1152) || SK!=(FOLD?4:1) || WK!=4 ||
        (FOLD?(N!=640 && N!=512):N!=2560)){__trap();return;}
    const int zz=blockIdx.z%nexp,u=work?work[2*zz]:zz;if(u>=ucount[0])return;
    const int mb=work?work[2*zz+1]:(int)blockIdx.x,mat=blockIdx.z/nexp,e=uids[u];
    const half* X=mat?X1:X0;const uint32_t* T=reinterpret_cast<const uint32_t*>(mat?TP1[e]:TP0[e]);
    const int k2=mat?K2_1[e]:K2_0[e],warp=threadIdx.x>>5,lane=threadIdx.x&31;
    __shared__ int rows[R_ROWS];
    if(threadIdx.x<R_ROWS){const int m=mb*R_ROWS+threadIdx.x,code=m<maxm?members[u*maxm+m]:-1;rows[threadIdx.x]=code>=0?(code>>5)*slots+(code&31):-1;}
    __syncthreads();if(rows[0]<0)return;
    const int nb=(rows[48]>=0?4:rows[32]>=0?3:rows[16]>=0?2:1),nt0=blockIdx.y*8;
    __shared__ half As[R_STAGES][R_ROWS][R_LD];
    // Static maximum accommodates every width; r_slice uses its compact typed view.
    __shared__ uint32_t Ws[R_STAGES][R_KB][8][4*HI];
    float tot[8][4];
#pragma unroll
    for(int j=0;j<8;++j)
#pragma unroll
      for(int q=0;q<4;++q)tot[j][q]=0.f;
#define R_RUN(Q,B) r_slice<Q,FOLD,B>(T,N/16,nt0,X,(FOLD?5120:1152),rows,warp,lane,As,*reinterpret_cast<uint32_t (*)[R_STAGES][R_KB][8][Fmt<Q>::TW]>(Ws),tot)
#define R_NB(Q) do {switch(nb){case 1:R_RUN(Q,1);break;case 2:R_RUN(Q,2);break;case 3:R_RUN(Q,3);break;default:R_RUN(Q,4);break;}}while(0)
#define R_CASE(Q) case Q:if constexpr(Q>=LO&&Q<=HI){R_NB(Q);}else{__trap();}break;
    switch(k2){R_CASE(2)R_CASE(3)R_CASE(4)R_CASE(5)R_CASE(6)R_CASE(7)R_CASE(8)R_CASE(9)R_CASE(10)default:__trap();}
#undef R_CASE
#undef R_NB
#undef R_RUN
    const int g=lane>>2,t=lane&3,col=nt0*16+warp*16+g;
#pragma unroll
    for(int j=0;j<8;++j){const int r0=rows[j*8+2*t],r1=rows[j*8+2*t+1];
        if(r0>=0){Z[((size_t)mat*P+r0)*N+col]=tot[j][0];Z[((size_t)mat*P+r0)*N+col+8]=tot[j][2];}
        if(r1>=0){Z[((size_t)mat*P+r1)*N+col]=tot[j][1];Z[((size_t)mat*P+r1)*N+col+8]=tot[j][3];}
    }
}

template __global__ void grouped_mma_kernel<2,8,8,true,4>(const half*,const half*,const int64_t*,const int64_t*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int,int,int,int,int,int,const int*);
template __global__ void grouped_mma_kernel<2,8,8,false,4>(const half*,const half*,const int64_t*,const int64_t*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int,int,int,int,int,int,const int*);
template __global__ void grouped_mma_kernel<2,2,10,true,4>(const half*,const half*,const int64_t*,const int64_t*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int,int,int,int,int,int,const int*);
template __global__ void grouped_mma_kernel<2,2,10,false,4>(const half*,const half*,const int64_t*,const int64_t*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int,int,int,int,int,int,const int*);
} // namespace tf_exl3z

// Prompt rotation body from TensorFold EXL3 (Apache-2.0), after Turboderp.
// Only independent CTA packing changed; original butterfly and FP16 conversion retained.
namespace tf_tp4_rot {
constexpr float HAD_SCALE=0.08838834764831845f;
// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
extern "C" __global__ void __launch_bounds__(256) tf_tp4_rot_prompt(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    if(K!=5120 || blockDim.x!=256){__trap();return;}
    const int p=blockIdx.x,blk=blockIdx.y*8+(threadIdx.x>>5),mat=blockIdx.z;
    if(blk>=K/128)return;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x&31;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<__nv_bfloat16>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}


}
