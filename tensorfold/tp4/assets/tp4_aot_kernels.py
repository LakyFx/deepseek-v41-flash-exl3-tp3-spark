"""Unmodified upstream kernel bodies; origins and SHA256 in AOT-SOURCE.json."""
import math
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@triton.jit
def _attn_step(q_lo, q_hi, k_lo, k_hi, ok, m_i, l_i, acc_lo, acc_hi, scale):
    s = (tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))) * scale
    s = tl.where(ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)
    pr = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(pr, axis=1)
    p16 = pr.to(tl.bfloat16)
    acc_lo = acc_lo * alpha[:, None] + tl.dot(p16, k_lo)
    acc_hi = acc_hi * alpha[:, None] + tl.dot(p16, k_hi)
    return m_new, l_i, acc_lo, acc_hi

@triton.jit
def _bfly(v, e, BIT: tl.constexpr):
    o = tl.gather(v, e ^ BIT, axis=1)
    return tl.where((e & BIT) != 0, o - v, v + o)

@triton.jit
def _comp_keys(COMP, CSC, row, ok, hc, HD: tl.constexpr, BN: tl.constexpr, PACKED: tl.constexpr):
    HALF: tl.constexpr = HD // 2
    if PACKED:
        cb = tl.load(COMP + row[:, None] * HALF + hc[None, :], mask=ok[:, None], other=0).to(tl.int32)
        g = tl.arange(0, HALF // 16)
        sl = tl.load(CSC + row[:, None] * (HD // 16) + g[None, :], mask=ok[:, None], other=0)
        sh = tl.load(CSC + row[:, None] * (HD // 16) + HALF // 16 + g[None, :], mask=ok[:, None], other=0)
        sl = sl.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sh = sh.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        lo = tl.reshape(tl.reshape(_e2m1(cb & 15), (BN, HALF // 16, 16)) * sl[:, :, None], (BN, HALF))
        hi = tl.reshape(tl.reshape(_e2m1(cb >> 4), (BN, HALF // 16, 16)) * sh[:, :, None], (BN, HALF))
        return lo.to(tl.bfloat16), hi.to(tl.bfloat16)
    else:
        kb = COMP + row[:, None] * HD
        return (tl.load(kb + hc[None, :], mask=ok[:, None], other=0.0),
                tl.load(kb + HALF + hc[None, :], mask=ok[:, None], other=0.0))

@triton.jit
def _discard_l2(ptrs):
    """discard.global.L2 of the 128-byte line at every address in ptrs: the line leaves L2 without a write-back (its
    data is dead: nothing reads it again before it is rewritten)."""

    return tl.inline_asm_elementwise("discard.global.L2 [$1], 128; mov.b32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                                     is_pure=False, pack=1)

@triton.jit
def _e2m1(code):
    """E2M1 nibble (sign bit 3) -> fp32 by building the float's bits: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6."""

    m = code & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 0x3F000000, 0))
    bits = bits | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)

@triton.jit
def _hc_mix_part(X, XO, G, RS, POST, COMB, FN, PART, rows, WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
                 SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr, POSTED: tl.constexpr, RB: tl.constexpr = 1,
                 PDL: tl.constexpr = False, XPF: tl.constexpr = False):
    r0 = tl.program_id(0) * RB
    b = tl.program_id(1)
    mg = tl.program_id(2)
    KB: tl.constexpr = WIDE // NB
    m = mg * MB + tl.arange(0, MB)
    k = tl.arange(0, SUB)
    ws = ()                                  # the mixing weights first: they may load while the kernel before runs
    for t in tl.static_range(KB // SUB):
        ws = ws + (tl.load(FN + m[:, None] * WIDE + b * KB + t * SUB + k[None, :]),)
    if XPF:
        # (switch "hc_xpf", posted mixes) the old streams' lines this program reads, and its rows' post / comb, into
        # L2 beside the weights' loads: the kernel before (the gather) writes neither, so their (often DRAM) latency
        # runs with the weights' instead of after the wait (a prefetch moves no data the kernel reads)
        xs_ = (b * KB) // D
        xc0 = b * KB - xs_ * D
        li = tl.arange(0, 4 * (KB // 64))                    # 4 streams x the block's 128-byte lines
        for i in tl.static_range(RB):
            rx = tl.minimum(r0 + i, rows - 1)
            _prefetch_l2(X + rx * WIDE + (li // (KB // 64)) * D + xc0 + (li % (KB // 64)) * 64)
            _prefetch_l2(COMB + rx * 16 + tl.arange(0, 1))
            _prefetch_l2(POST + rx * 4 + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    if RB == 1:
        _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r0, b, mg, m, k, WIDE, D, NB, KB, SUB, MB, WORLD, POSTED)
    else:
        for i in tl.static_range(RB):
            if r0 + i < rows:
                _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r0 + i, b, mg, m, k, WIDE, D, NB, KB, SUB, MB,
                             WORLD, POSTED)

@triton.jit
def _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r, b, mg, m, k, WIDE: tl.constexpr, D: tl.constexpr,
                 NB: tl.constexpr, KB: tl.constexpr, SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr,
                 POSTED: tl.constexpr):
    """One row of _hc_mix_part (the weight tiles ws already loaded)."""

    s = (b * KB) // D
    col0 = b * KB - s * D
    acc = tl.zeros((MB,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    if POSTED:
        c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
        c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
        c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
        c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
        ps = tl.load(POST + r * 4 + s)
    for t in tl.static_range(KB // SUB):
        d = col0 + t * SUB + k
        if POSTED:
            ya = tl.load(G + r * D + d)
            for q in tl.static_range(1, WORLD):
                ya = ya + tl.load(G + q * RS + r * D + d)
            y = ya.to(tl.bfloat16).to(tl.float32)
            x0 = tl.load(X + r * WIDE + d).to(tl.float32)
            x1 = tl.load(X + r * WIDE + D + d).to(tl.float32)
            x2 = tl.load(X + r * WIDE + 2 * D + d).to(tl.float32)
            x3 = tl.load(X + r * WIDE + 3 * D + d).to(tl.float32)
            # _hc_post's contraction, spelled out: fma(ps, y, fma(c3, x3, fma(c2, x2, fma(c0, x0, c1 x1))))
            v = tl.fma(ps, y, tl.fma(c3, x3, tl.fma(c2, x2, tl.fma(c0, x0, c1 * x1))))
            xb = v.to(tl.bfloat16)
            if mg == 0:
                tl.store(XO + r * WIDE + s * D + d, xb)
            x = xb.to(tl.float32)
        else:
            x = tl.load(X + r * WIDE + s * D + d).to(tl.float32)
        acc += tl.sum(ws[t] * x[None, :], axis=1)
        ss = tl.fma(x, x, ss)                                   # _hc_partial's fma(x, x, ss)
    tl.store(PART + (r * NB + b) * 32 + m, acc)
    if mg == 0:
        tl.store(PART + (r * NB + b) * 32 + 24, tl.sum(ss, axis=0))

@triton.jit
def _prefetch_l2(ptrs):
    """prefetch.global.L2 of every address in ptrs (no data moved into registers, nothing computed)."""

    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.b32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                                     is_pure=False, pack=1)

@triton.jit
def _rope_heads(X, COS, SIN, POS, H: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, INV: tl.constexpr,
                HB: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * HB + tl.arange(0, HB)
    j = tl.arange(0, RD // 2)
    p = tl.load(POS + r)
    cs = tl.load(COS + p * (RD // 2) + j)
    sn = tl.load(SIN + p * (RD // 2) + j)
    if INV:
        sn = -sn
    base = X + r * (H * HD) + h[:, None] * HD + (HD - RD) + 2 * j[None, :]
    xe = tl.load(base).to(tl.float32)
    xo = tl.load(base + 1).to(tl.float32)
    tl.store(base, (xe * cs[None, :] - xo * sn[None, :]).to(tl.bfloat16))
    tl.store(base + 1, (xe * sn[None, :] + xo * cs[None, :]).to(tl.bfloat16))

@triton.jit
def _rot128(v, s, M: tl.constexpr):
    """EXL3's input rotation of M 128-wide blocks (v, s [M, 128] fp32: values and the layer's suh): rot128_in's bits
    (linear_common.cuh): the first butterfly as fma(v0, s0, +-(v1 s1)), then six butterflies lo + hi / lo - hi in bit
    order, then the scale."""

    e = tl.broadcast_to(tl.arange(0, 128)[None, :], (M, 128))
    p = v * s
    vo = tl.gather(v, e ^ 1, axis=1)
    so = tl.gather(s, e ^ 1, axis=1)
    po = tl.gather(p, e ^ 1, axis=1)
    v = tl.where((e & 1) != 0, tl.fma(vo, so, -p), tl.fma(v, s, po))
    v = _bfly(v, e, 2)
    v = _bfly(v, e, 4)
    v = _bfly(v, e, 8)
    v = _bfly(v, e, 16)
    v = _bfly(v, e, 32)
    v = _bfly(v, e, 64)
    return v * 0.08838834764831845

@triton.jit
def _sparse_attn_merge_rot2(PM, PL, PO, SINK, OUT, COS, SIN, POS, SUH, XH, rows, H: tl.constexpr, HD: tl.constexpr,
                            HB: tl.constexpr, SPLITS: tl.constexpr, RD: tl.constexpr, GH: tl.constexpr,
                            PDL: tl.constexpr = False, DISCARD: tl.constexpr = False):
    h = tl.program_id(1)
    _prefetch_l2(SUH + h * HD + tl.arange(0, HD // 64) * 64)    # wo_a's suh of this head
    _prefetch_l2(SINK + h + tl.arange(0, 1))
    pp = tl.load(POS + tl.program_id(0))                       # (positions: a round input, no kernel writes them)
    _prefetch_l2(COS + pp * (RD // 2) + tl.arange(0, 1))
    _prefetch_l2(SIN + pp * (RD // 2) + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.arange(0, HD)
    base0 = (r * (H // HB) + h // HB) * SPLITS
    hi = h % HB
    m = -1e30
    for sp in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + (base0 + sp) * HB + hi))
    l = 0.0
    acc = tl.zeros((HD,), dtype=tl.float32)
    for sp in range(SPLITS):
        ms = tl.load(PM + (base0 + sp) * HB + hi)
        a = tl.exp(ms - m)
        l += tl.load(PL + (base0 + sp) * HB + hi) * a
        acc += tl.load(PO + (base0 + sp) * HB * HD + hi * HD + d) * a
    l += tl.exp(tl.load(SINK + h) - m)
    y = (acc / l).to(tl.bfloat16).to(tl.float32)               # the merge's bf16 row
    if DISCARD:
        tl.debug_barrier()                                     # every thread's split loads are consumed
        q = tl.arange(0, SPLITS * (HD // 32))
        _discard_l2(PO + (base0 + q // (HD // 32)) * HB * HD + hi * HD + (q % (HD // 32)) * 32)
    # inverse RoPE on the last RD values (pairs j = HD / 2 - RD / 2 ..): rope_heads' contraction as in
    # _sparse_attn_merge_rot, on the same bf16 values
    ye, yo = tl.split(tl.reshape(y, (HD // 2, 2)))
    jj = tl.arange(0, HD // 2)
    J0: tl.constexpr = (HD - RD) // 2
    rot = jj >= J0
    jr = tl.maximum(jj - J0, 0)
    p = tl.load(POS + r)
    cs = tl.load(COS + p * (RD // 2) + jr)
    sn = tl.load(SIN + p * (RD // 2) + jr)
    sn = -sn
    re = tl.where(rot, tl.fma(ye, cs, -(yo * sn)).to(tl.bfloat16).to(tl.float32), ye)
    im = tl.where(rot, tl.fma(ye, sn, yo * cs).to(tl.bfloat16).to(tl.float32), yo)
    o = tl.reshape(tl.join(re, im), (HD,))                     # the roped bf16 row (as fp32)
    tl.store(OUT + r * (H * HD) + h * HD + d, o.to(tl.bfloat16))
    # wo_a's input rotation of the head: its HD / 128 blocks (slice g = h // GH, columns (h % GH) * HD ..)
    NBK: tl.constexpr = HD // 128
    bi = tl.arange(0, NBK)
    c = tl.arange(0, 128)
    v = tl.reshape(o, (NBK, 128))
    sv = tl.load(SUH + h * HD + bi[:, None] * 128 + c[None, :]).to(tl.float32)
    yr = _rot128(v, sv, NBK)
    g = h // GH
    col = (h % GH) * HD + bi * 128
    tl.store(XH + g * (rows * GH * HD) + r * (GH * HD) + col[:, None] + c[None, :], yr.to(tl.float16))

@triton.jit
def _sparse_attn_part(Q, WSRC, WLO, COMP, CSC, IDX, POS, PM, PL, PO, SINK, OUT, scale, ring_size, n_idx, RBASE, CBASE,
                      H: tl.constexpr, HD: tl.constexpr, HB: tl.constexpr, WIN: tl.constexpr, BN: tl.constexpr,
                      RING: tl.constexpr, HAS_COMP: tl.constexpr, PACKED: tl.constexpr, SPLITS: tl.constexpr,
                      NBLK: tl.constexpr, FINAL: tl.constexpr, HAS_BASE: tl.constexpr, PDL: tl.constexpr = False,
                      WPF: tl.constexpr = False):
    if WPF:
        # (switch "attn_pf") the split's window block into L2 before the PDL wait: ring rows written kernels ago (the
        # window KV norm), at slots of round inputs (positions, ring bases); a prefetch moves no data a kernel reads
        spw = tl.program_id(2)
        if spw < WIN // BN:
            nw = tl.arange(0, BN)
            pw = tl.load(POS + tl.program_id(0))
            wpw = pw - (WIN - 1) + spw * BN + nw
            okw = wpw >= 0
            sw = tl.where(okw, wpw % ring_size, 0)
            if HAS_BASE:
                sw = sw + tl.load(RBASE + tl.program_id(0))
            _prefetch_l2(WSRC + sw[:, None] * HD + (tl.arange(0, HD // 64) * 64)[None, :])
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    HALF: tl.constexpr = HD // 2
    h = hb * HB + tl.arange(0, HB)
    hc = tl.arange(0, HALF)
    qb = Q + r * (H * HD) + h[:, None] * HD
    q_lo = tl.load(qb + hc[None, :])
    q_hi = tl.load(qb + HALF + hc[None, :])
    p = tl.load(POS + r)
    wlo = tl.load(WLO)
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    acc_lo = tl.zeros((HB, HALF), dtype=tl.float32)
    acc_hi = tl.zeros((HB, HALF), dtype=tl.float32)
    n = tl.arange(0, BN)
    WB: tl.constexpr = WIN // BN
    # key blocks 0..WB-1 are the window, WB.. the picks; this program takes blocks sp, sp + SPLITS, ...
    for b in range(sp, NBLK, SPLITS):
        if b < WB:
            wp = p - (WIN - 1) + b * BN + n
            ok = wp >= 0
            if RING:
                slot = wp % ring_size
            else:
                slot = wp - wlo
                ok = ok & (slot >= 0)
            slot = tl.where(ok, slot, 0)
            if HAS_BASE:                   # a concurrent round: the row's stream's ring starts at pool row RBASE[r]
                slot = slot + tl.load(RBASE + r)
            kb = WSRC + slot[:, None] * HD
            k_lo = tl.load(kb + hc[None, :], mask=ok[:, None], other=0.0)
            k_hi = tl.load(kb + HALF + hc[None, :], mask=ok[:, None], other=0.0)
        else:
            t = (b - WB) * BN
            ii = tl.load(IDX + r * n_idx + t + n, mask=(t + n) < n_idx, other=-1)
            ok = ii >= 0
            row = tl.where(ok, ii, 0)
            if HAS_BASE:                   # ... and its compressed rows at pool row CBASE[r]
                row = row + tl.load(CBASE + r)
            k_lo, k_hi = _comp_keys(COMP, CSC, row, ok, hc, HD, BN, PACKED)
        m_i, l_i, acc_lo, acc_hi = _attn_step(q_lo, q_hi, k_lo, k_hi, ok, m_i, l_i, acc_lo, acc_hi, scale)
    if FINAL:                         # one split (prompt chunks): the sink and the division here, no merge pass
        l_i = l_i + tl.exp(tl.load(SINK + h) - m_i)
        ob = OUT + r * (H * HD) + h[:, None] * HD
        tl.store(ob + hc[None, :], (acc_lo / l_i[:, None]).to(tl.bfloat16))
        tl.store(ob + HALF + hc[None, :], (acc_hi / l_i[:, None]).to(tl.bfloat16))
        return
    base = (r * (H // HB) + hb) * SPLITS + sp
    tl.store(PM + base * HB + tl.arange(0, HB), m_i)
    tl.store(PL + base * HB + tl.arange(0, HB), l_i)
    ob = PO + base * HB * HD + tl.arange(0, HB)[:, None] * HD
    tl.store(ob + hc[None, :], acc_lo)
    tl.store(ob + HALF + hc[None, :], acc_hi)

@triton.jit(do_not_specialize=["M"])
def _gemm(X, W, H, SVH, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
          BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, SCALE: tl.constexpr):
    """OUT[m, block] = ((xh[m] @ W_q[:, block]) @ H) * SCALE * svh + bias; K in BK steps in order, a row alone."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // 128)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, 128), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(W + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, w, acc)
    hi = tl.arange(0, 128)
    h = tl.load(H + hi[:, None] * 128 + hi[None, :])
    top = acc.to(tl.bfloat16)
    rest = (acc - top.to(tl.float32)).to(tl.bfloat16)
    y = tl.dot(rest, h, tl.dot(top, h))
    y = y * SCALE * tl.load(SVH + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], y.to(OUT.dtype.element_ty), mask=ok[:, None])


@triton.jit(do_not_specialize=["n_part", "o_stride", "send"])
def _finish(PV, PI, n_part, OUT, o_stride, SEND, send, BP: tl.constexpr, PDL: tl.constexpr = False):
    """Row r's best of its n_part program bests: the token into OUT[r * o_stride] (int64), or (``send``) (value,
    index, 0, 0) fp32 into SEND[r] for the ranks' gather."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    p = tl.arange(0, BP)
    pm = p < n_part
    v = tl.load(PV + r * n_part + p, mask=pm, other=float("-inf"))
    i = tl.load(PI + r * n_part + p, mask=pm, other=BIG)
    bv, bi = tl.reduce((v, i), 0, _best)
    if send != 0:
        q = tl.arange(0, 4)
        val = tl.where(q == 0, bv, tl.where(q == 1, bi.to(tl.float32), 0.0))
        tl.store(SEND + r * 4 + q, val)
    else:
        tl.store(OUT + r * o_stride, bi.to(tl.int64))


@triton.jit(do_not_specialize=["n_rows", "o_stride"])
def _pick(G, n_rows, OUT, o_stride, WORLD: tl.constexpr, PDL: tl.constexpr = False):
    """Row r's token from the ranks' (value, index) bests [WORLD, rows, 4], in the same order."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    bv = tl.load(G + r * 4)
    bi = tl.load(G + r * 4 + 1).to(tl.int32)
    for w in tl.static_range(1, WORLD):
        v = tl.load(G + (w * n_rows + r) * 4)
        i = tl.load(G + (w * n_rows + r) * 4 + 1).to(tl.int32)
        bv, bi = _best(v, i, bv, bi)
    tl.store(OUT + r * o_stride, bi.to(tl.int64))


BIG = tl.constexpr(1 << 30)


@triton.jit
def _best(v1, i1, v2, i2):
    t = _better(v1, i1, v2, i2)
    return tl.where(t, v1, v2), tl.where(t, i1, i2)


@triton.jit
def _better(v1, i1, v2, i2):
    """(v1, i1) before (v2, i2) in torch.argmax's order: NaN above everything, then the larger value, ties (and NaN
    against NaN) to the lower index."""

    n1 = v1 != v1
    n2 = v2 != v2
    gt = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (n1 & (~n2 | (i1 < i2))) | (~n1 & ~n2 & gt)
