"""Cache only existing FP32 weight scale packing; never requantize weights."""
class ScaleCache:
    def __init__(self):
        self.entries={};self.rejected=set();self.packs=0

    @staticmethod
    def key(scale,recipe,groups,rank):
        return (scale.data_ptr(),getattr(scale,'_version',0),tuple(scale.shape),tuple(scale.stride()),
                str(scale.dtype),str(scale.device),tuple(recipe),groups,rank)

    def resolve(self,scale,recipe,groups,rank,*,pack,verify,capturing=False):
        key=self.key(scale,recipe,groups,rank)
        if key in self.entries:
            return self.entries[key]
        if capturing or key in self.rejected:
            return scale
        try:
            candidate=pack(scale)
            if not verify(scale,candidate):
                raise ValueError('packed scale does not reproduce original output')
            # Preserve the exact returned strides/layout. Never .contiguous().
            self.entries.clear()  # one current weight scale per layer, bounded
            self.entries[key]=candidate;self.packs+=1
            return candidate
        except Exception as exc:
            self.rejected.add(key)
            print(f'DGX_V089_WOA_DISARMED {type(exc).__name__}: {exc}',flush=True)
            return scale

def resolve_scale(layer,scale,n_groups,o_lora_rank,recipe):
    import torch
    if scale.dtype!=torch.float32 or layer.weight.dtype!=torch.float8_e4m3fn:
        return scale  # BF16 emulation stays exactly as loaded; no stage-1 conversion
    from vllm.utils.deep_gemm import fp8_einsum,transform_sf_into_required_layout
    cache=getattr(layer,'_dgx_woa_scale_cache',None)
    if cache is None:
        cache=layer._dgx_woa_scale_cache=ScaleCache()
    k=layer.weight.shape[-1]
    def pack(value):
        return transform_sf_into_required_layout(value,o_lora_rank,k,recipe,n_groups,False)
    def verify(original,packed):
        gen=torch.Generator(device=layer.weight.device);gen.manual_seed(8924)
        # Qualify the actual K5/C8 physical row buckets, not only foreign M=4.
        for m in (1,5,6,10,12,15,18,20,24,25,30,35,36,40,42,48):
            x=torch.randn((m,n_groups,k),generator=gen,device=layer.weight.device).to(torch.float8_e4m3fn)
            xs=torch.ones((m,n_groups,k//int(recipe[2])),device=x.device,dtype=torch.float32)
            old=torch.empty((m,n_groups,o_lora_rank),device=x.device,dtype=torch.bfloat16)
            new=torch.empty_like(old)
            fp8_einsum('bhr,hdr->bhd',(x,xs),(layer.weight,original),old,recipe=recipe)
            fp8_einsum('bhr,hdr->bhd',(x,xs),(layer.weight,packed),new,recipe=recipe)
            if not torch.equal(old.view(torch.uint16),new.view(torch.uint16)):
                return False
        return True
    return cache.resolve(scale,recipe,n_groups,o_lora_rank,pack=pack,verify=verify,
                         capturing=torch.cuda.is_current_stream_capturing())
