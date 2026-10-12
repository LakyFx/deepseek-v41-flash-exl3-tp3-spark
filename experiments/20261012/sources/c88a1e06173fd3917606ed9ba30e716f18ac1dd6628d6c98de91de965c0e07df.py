# Exact selectors from the pinned baseline B12X source.
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x._lib.quant.block_codec import BLOCK_CODECS

def _automatic_a16(query, device):
    return device is not None and device.compute_capability in ((12, 0), (12, 1)) and (query.in_features == query.padded_in_features) and (query.in_features % 32 == 0) and (query.out_features % 8 == 0) and query.source_contiguous and query.source_aligned and (query.num_tokens <= 8)

def _default_config(query, device):
    if query.activation_mode == 'quantized':
        return BlockscaledConfig(mode='quantized')
    if _automatic_a16(query, device):
        return BlockscaledConfig(mode='a16', tile_n=128, tile_k=64, split_k=4)
    if query.activation_mode == 'a16' or query.recipe in BLOCK_CODECS:
        return BlockscaledConfig(mode='a16', tile_n=64, tile_k=64, split_k=1)
    return BlockscaledConfig(mode='quantized')
