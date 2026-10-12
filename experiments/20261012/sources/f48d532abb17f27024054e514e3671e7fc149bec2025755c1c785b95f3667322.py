"""Frozen per-arm launch policy; imported before B12X graph warmup."""
import os

SPLIT=int(os.environ.get('DGX_NEXT_DENSE_SMALL_SPLIT','2'))
if SPLIT not in (1,2):
    raise ValueError('N8 small split must be 1 or 2')


def small_split(recipe, rows):
    return SPLIT if recipe=='mxfp8' and 1<=rows<=6 else 4


def row_capacity(recipe, rows):
    # The upstream heuristic normally maps real5/6 to capacity8, then erases
    # expected_m. Preserve the small actual capacity so its default config
    # really selects N8. Other recipes and all larger batches keep baseline.
    return rows if recipe == 'mxfp8' and 1 <= rows <= 6 else 1 << (rows-1).bit_length()
