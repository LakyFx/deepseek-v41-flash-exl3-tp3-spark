"""Column-sharded main_proj, padded at complete checkpoint FP8 block boundaries.

Retains checkpoint weights and a full gathered result before the original norm.
Does not use B12X's non-stock allow_tp_padding loader contract.
"""
import math
import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear

def pad_checkpoint_rows(weight, expected, logical, output_dim, *, scale=False):
    rows = weight.shape[output_dim]
    if rows != logical or expected < logical:
        raise ValueError('unexpected draft checkpoint output geometry')
    if rows == expected:
        return weight
    shape = list(weight.shape)
    shape[output_dim] = expected
    out = weight.new_full(shape, 1 if scale else 0)
    out.narrow(output_dim, 0, rows).copy_(weight)
    return out

class PaddedDraftColumn(ColumnParallelLinear):
    def __init__(self, input_size, output_size, **kwargs):
        tp = get_tensor_model_parallel_world_size()
        if tp != 3 or kwargs.get('bias', False) or kwargs.get('return_bias', True):
            raise ValueError('draft projection port requires TP3, no bias and tensor output')
        quant = kwargs.get('quant_config')
        block = getattr(quant, 'weight_block_size', None)
        # Known baseline MXFP8 uses 32 rows. Block-FP8 can also use 128.
        # Other quantizers must be separately qualified, never inferred.
        if quant is not None and (block is None or int(block[0]) not in (32,128)):
            raise ValueError('draft main_proj quantizer requires explicit FP8 block rows')
        block_rows = int(block[0]) if block is not None else 1
        self.logical_output = output_size
        self.padded_output = math.ceil(output_size/(tp*block_rows))*tp*block_rows
        super().__init__(input_size, self.padded_output, gather_output=True, **kwargs)
        for name, parameter in self.named_parameters(recurse=False):
            original = getattr(parameter, 'weight_loader', None)
            axis = getattr(parameter, 'output_dim', None)
            if original is None or axis is None:
                raise ValueError('unsupported draft parameter loader '+name)
            scale = name in ('weight_scale_inv','weight_scale')
            logical = math.ceil(output_size/block_rows) if scale else output_size
            expected = parameter.shape[axis]*tp
            def loader(param, loaded, *, original=original, axis=axis,
                       logical=logical, expected=expected, scale=scale):
                return original(param, pad_checkpoint_rows(
                    loaded, expected, logical, axis, scale=scale))
            parameter.weight_loader = loader

    def forward(self, inputs):
        result = super().forward(inputs)
        return result[..., :self.logical_output].contiguous()
