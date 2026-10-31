# Upstream attention with three head-tail masks for TP3 HB16; see SOURCE.json.
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents,gdc_wait

@triton.jit
def _prefetch_l2(ptrs):
    """prefetch.global.L2 of every address in ptrs (no data moved into registers, nothing computed)."""

    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.b32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                                     is_pure=False, pack=1)

@triton.jit
def _e2m1(code):
    """E2M1 nibble (sign bit 3) -> fp32 by building the float's bits: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6."""

    m = code & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 0x3F000000, 0))
    bits = bits | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)

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
    q_lo = tl.load(qb + hc[None, :], mask=h[:, None] < H, other=0.0)
    q_hi = tl.load(qb + HALF + hc[None, :], mask=h[:, None] < H, other=0.0)
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
        l_i = l_i + tl.exp(tl.load(SINK + h, mask=h < H, other=0.0) - m_i)
        ob = OUT + r * (H * HD) + h[:, None] * HD
        tl.store(ob + hc[None, :], (acc_lo / l_i[:, None]).to(tl.bfloat16), mask=h[:, None] < H)
        tl.store(ob + HALF + hc[None, :], (acc_hi / l_i[:, None]).to(tl.bfloat16), mask=h[:, None] < H)
        return
    base = (r * (H // HB) + hb) * SPLITS + sp
    tl.store(PM + base * HB + tl.arange(0, HB), m_i)
    tl.store(PL + base * HB + tl.arange(0, HB), l_i)
    ob = PO + base * HB * HD + tl.arange(0, HB)[:, None] * HD
    tl.store(ob + hc[None, :], acc_lo)
    tl.store(ob + HALF + hc[None, :], acc_hi)
