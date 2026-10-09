// Warp-level 128-point Hadamard transform.
//
// Structure follows ExLlamaV3's hadamard_inner.cuh (MIT, (c) turboderp): a
// 128-point transform factored as (4-point) x (32-point), the 32-point half
// done with butterfly shuffles across the warp's lanes. Rewritten here to take
// explicit scale pointers rather than deriving them from the launch geometry,
// so the same routine serves both the standalone input transform and the fused
// gemm epilogue.
#pragma once

#include <type_traits>
#include <cuda_bf16.h>

#include "exl3_common.cuh"

struct half4
{
    half2 x, y;
};

// Activation dtype adapters. EXL3's codebook decodes natively to fp16 and every
// kernel accumulates from fp16 fragments, but vLLM runs most models in bf16.
// Converting in the Hadamard load/store (rather than with separate .to() calls)
// keeps the conversion off the critical path entirely -- it was ~10% of decode
// GPU time as two extra elementwise passes per linear.
template <typename T>
struct ActVec;

template <>
struct ActVec<half>
{
    __device__ __forceinline__ static half4 load(const half* p, int lane)
    {
        return ((const half4*) p)[lane];
    }
    __device__ __forceinline__ static void store(half* p, int lane, half4 v)
    {
        ((half4*) p)[lane] = v;
    }
};

template <>
struct ActVec<__nv_bfloat16>
{
    __device__ __forceinline__ static half4 load(const __nv_bfloat16* p, int lane)
    {
        const float4 raw = ((const float4*) p)[lane / 2];
        // Each float4 holds 8 bf16; pick this lane's half of it.
        const __nv_bfloat162* b = (const __nv_bfloat162*) &raw;
        int o = (lane & 1) * 2;
        float2 f0 = __bfloat1622float2(b[o]);
        float2 f1 = __bfloat1622float2(b[o + 1]);
        half4 v;
        v.x = __floats2half2_rn(f0.x, f0.y);
        v.y = __floats2half2_rn(f1.x, f1.y);
        return v;
    }
    __device__ __forceinline__ static void store(__nv_bfloat16* p, int lane, half4 v)
    {
        float2 f0 = __half22float2(v.x);
        float2 f1 = __half22float2(v.y);
        __nv_bfloat162 o0 = __floats2bfloat162_rn(f0.x, f0.y);
        __nv_bfloat162 o1 = __floats2bfloat162_rn(f1.x, f1.y);
        __nv_bfloat162* dst = (__nv_bfloat162*) p;
        dst[lane * 2] = o0;
        dst[lane * 2 + 1] = o1;
    }
};

// 1/sqrt(128): makes the transform orthonormal, matching how the quantizer
// applied it. Both the input and output transforms carry this factor.
#define EXL3_HAD128_RSCALE 0.088388347648318447f

__device__ __forceinline__ void shuffle_had_f4x32(float& h0, float& h1, float& h2,
                                                  float& h3, const int lane_id)
{
#pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        uint64_t h01 = (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 = (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        // Lanes whose bit `i` is set negate their own value before adding.
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

// One warp transforms one 128-element row.
//   pre_scale : out = had(in * scale_pre)
//   post_scale: out = had(in) * scale_post
// `lane` is the caller's lane id; scale pointers address the same 128 elements.
// Input-transform variant: reads activations in their native dtype, writes fp16.
template <typename IN_T>
__device__ __forceinline__ void had128_warp_in(const IN_T* __restrict__ in,
                                               half* __restrict__ out,
                                               const half* __restrict__ scale_pre,
                                               int lane)
{
    half4 v = ActVec<IN_T>::load(in, lane);
    half4 s = ((const half4*) scale_pre)[lane];
    v.x = __hmul2(v.x, s.x);
    v.y = __hmul2(v.y, s.y);

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v.x = __floats2half2_rn(h0 * EXL3_HAD128_RSCALE, h1 * EXL3_HAD128_RSCALE);
    v.y = __floats2half2_rn(h2 * EXL3_HAD128_RSCALE, h3 * EXL3_HAD128_RSCALE);
    ((half4*) out)[lane] = v;
}

// SwiGLU folded into the input transform. The MoE down-projection's activation
// is silu(gate) * up over the first GEMM's two output halves; computing it here
// saves materialising that (rows x inter) tensor only to read it straight back.
// The product is formed in fp32, where the separate elementwise kernel used the
// GEMM's output dtype, so this is if anything slightly more accurate.
template <typename IN_T>
__device__ __forceinline__ void had128_warp_glu_in(const IN_T* __restrict__ gate,
                                                   const IN_T* __restrict__ up,
                                                   half* __restrict__ out,
                                                   const half* __restrict__ scale_pre,
                                                   int lane)
{
    half4 g = ActVec<IN_T>::load(gate, lane);
    half4 u = ActVec<IN_T>::load(up, lane);
    half4 s = ((const half4*) scale_pre)[lane];

    // silu(x) = x / (1 + exp(-x)). __expf is ~2^-21 accurate, far below the
    // quantization error it feeds into.
    auto glu = [] (half hg, half hu, half hs) {
        float x = __half2float(hg);
        return x * __frcp_rn(1.0f + __expf(-x)) * __half2float(hu) * __half2float(hs);
    };
    float v0 = glu(__low2half(g.x),  __low2half(u.x),  __low2half(s.x));
    float v1 = glu(__high2half(g.x), __high2half(u.x), __high2half(s.x));
    float v2 = glu(__low2half(g.y),  __low2half(u.y),  __low2half(s.y));
    float v3 = glu(__high2half(g.y), __high2half(u.y), __high2half(s.y));

    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    half4 v;
    v.x = __floats2half2_rn(h0 * EXL3_HAD128_RSCALE, h1 * EXL3_HAD128_RSCALE);
    v.y = __floats2half2_rn(h2 * EXL3_HAD128_RSCALE, h3 * EXL3_HAD128_RSCALE);
    ((half4*) out)[lane] = v;
}

template <bool pre_scale, bool post_scale>
__device__ __forceinline__ void had128_warp(const half* __restrict__ in,
                                            half* __restrict__ out,
                                            const half* __restrict__ scale_pre,
                                            const half* __restrict__ scale_post,
                                            int lane)
{
    half4 v = ((const half4*) in)[lane];

    if constexpr (pre_scale)
    {
        half4 s = ((const half4*) scale_pre)[lane];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    // 4-point Hadamard held entirely in one lane
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

    // 32-point Hadamard across the warp
    shuffle_had_f4x32(h0, h1, h2, h3, lane);

    v.x = __floats2half2_rn(h0 * EXL3_HAD128_RSCALE, h1 * EXL3_HAD128_RSCALE);
    v.y = __floats2half2_rn(h2 * EXL3_HAD128_RSCALE, h3 * EXL3_HAD128_RSCALE);

    if constexpr (post_scale)
    {
        half4 s = ((const half4*) scale_post)[lane];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    ((half4*) out)[lane] = v;
}

// Split-k epilogue: read an fp32 partial-sum row, Hadamard it, apply svh, emit
// fp16 -- and leave the accumulator zeroed again so the next call can atomically
// accumulate into it without a separate memset.
template <typename OUT_T>
__device__ __forceinline__ void had128_warp_acc(float* __restrict__ acc,
                                                OUT_T* __restrict__ out,
                                                const half* __restrict__ svh,
                                                int lane)
{
    float4 a = ((float4*) acc)[lane];
    ((float4*) acc)[lane] = make_float4(0.f, 0.f, 0.f, 0.f);

    float v0 = a.x, v1 = a.y, v2 = a.z, v3 = a.w;
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, lane);

    half4 v;
    v.x = __floats2half2_rn(h0 * EXL3_HAD128_RSCALE, h1 * EXL3_HAD128_RSCALE);
    v.y = __floats2half2_rn(h2 * EXL3_HAD128_RSCALE, h3 * EXL3_HAD128_RSCALE);
    half4 s = ((const half4*) svh)[lane];
    v.x = __hmul2(v.x, s.x);
    v.y = __hmul2(v.y, s.y);
    ActVec<OUT_T>::store(out, lane, v);
}


// Fused epilogue: Hadamard + svh straight out of the shared C tile, emitting the
// caller's activation dtype.
template <typename OUT_T>
__device__ __forceinline__ void had128_warp_out(const half* __restrict__ in,
                                                OUT_T* __restrict__ out,
                                                const half* __restrict__ svh,
                                                int lane)
{
    half4 v = ((const half4*) in)[lane];
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v.x = __floats2half2_rn(h0 * EXL3_HAD128_RSCALE, h1 * EXL3_HAD128_RSCALE);
    v.y = __floats2half2_rn(h2 * EXL3_HAD128_RSCALE, h3 * EXL3_HAD128_RSCALE);
    half4 s = ((const half4*) svh)[lane];
    v.x = __hmul2(v.x, s.x);
    v.y = __hmul2(v.y, s.y);
    ActVec<OUT_T>::store(out, lane, v);
}

// Same transform, but scaled by a routing weight and accumulated into a token's
// row instead of written to a routed row. This is what lets the down projection
// finish the MoE by itself: the separate combine kernel read back a (rows,
// hidden) tensor that nothing else wanted, and at decode it ran on one block per
// token -- 8 blocks on a 188-SM part, pure latency. Accumulating here removes
// both the kernel and the tensor.
//
// top_k rows land on the same token, so this has to be atomic. bf16/half pair
// atomics are one instruction per two elements and the contention is only top_k
// deep, which is why this is cheaper than the round trip it replaces.
template <typename OUT_T>
__device__ __forceinline__ void had128_warp_out_acc(const half* __restrict__ in,
                                                    OUT_T* __restrict__ out,
                                                    const half* __restrict__ svh,
                                                    int lane, float w)
{
    half4 v = ((const half4*) in)[lane];
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    half4 sc = ((const half4*) svh)[lane];
    const float sw = EXL3_HAD128_RSCALE * w;
    float o0 = h0 * sw * __half2float(__low2half(sc.x));
    float o1 = h1 * sw * __half2float(__high2half(sc.x));
    float o2 = h2 * sw * __half2float(__low2half(sc.y));
    float o3 = h3 * sw * __half2float(__high2half(sc.y));
    if constexpr (std::is_same<OUT_T, __nv_bfloat16>::value)
    {
        __nv_bfloat162* dst = (__nv_bfloat162*) out;
        atomicAdd(dst + lane * 2,     __floats2bfloat162_rn(o0, o1));
        atomicAdd(dst + lane * 2 + 1, __floats2bfloat162_rn(o2, o3));
    }
    else
    {
        __half2* dst = (__half2*) out;
        atomicAdd(dst + lane * 2,     __floats2half2_rn(o0, o1));
        atomicAdd(dst + lane * 2 + 1, __floats2half2_rn(o2, o3));
    }
}
