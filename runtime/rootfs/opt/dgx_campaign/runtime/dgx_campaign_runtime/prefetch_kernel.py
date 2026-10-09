# SPDX-License-Identifier: Apache-2.0
"""Two-CTA bounded L2-hint stream. No values/scales are modified or decoded."""
from vllm.triton_utils import tl, triton


@triton.jit
def _hint(SEGS, COUNT: tl.constexpr, CTAS: tl.constexpr):
    lanes = tl.arange(0, 32)
    offset = (tl.program_id(0) * 32 + lanes) * 128
    for s in range(COUNT):
        ptr = tl.load(SEGS + s * 2)
        size = tl.load(SEGS + s * 2 + 1)
        for start in range(0, size, CTAS * 32 * 128):
            address = ptr + offset + start
            valid = offset + start < size
            # Non-pure inline assembly survives DCE; mask is checked in PTX.
            tl.inline_asm_elementwise(
                "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.u32 $0, 0; }",
                constraints="=r,l,r", args=[address, valid.to(tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1)


def hint(segments, *, ctas=2):
    if isinstance(ctas, bool) or not isinstance(ctas, int) or ctas not in (1, 2):
        raise ValueError("P followup permits only integer one/two-CTA pacing")
    _hint[(ctas,)](segments, segments.shape[0], ctas, num_warps=1)
