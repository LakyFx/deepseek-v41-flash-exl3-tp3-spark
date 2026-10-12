"""Experimental full TP3 MoE adapter. Strict raw-BF16 gate; stock on mismatch.

No engine is imported or patched at module import. The native source port keeps
the checkpoint's mixed K3/K4 mul1 weights and combined gate/up packing. Different
MMA/reduction schedules can still fail the strict gate; no quality relaxation is
silently selected. GPU launch failures after arming propagate for rollback.
"""
import ctypes as C
import hashlib
import math
import os
from pathlib import Path

_pool={}
_qualifying=False
_rejected=set()

class NumericalMismatch(ValueError):
    """A synchronized comparison failed; no CUDA failure was swallowed."""

def pointer_offsets(expert,w13_bits,w2_bits,*,hidden=5120,inter=768):
    if not 0<=expert<384 or w13_bits not in (3,4) or w2_bits not in (3,4):
        raise ValueError('unsupported expert metadata')
    # Byte offsets in the ORIGINAL combined [E,H/16,2I/16,16*K] storage.
    gate=expert*(hidden//16)*(2*inter//16)*(16*w13_bits)*2
    up=gate+(inter//16)*(16*w13_bits)*2
    down=expert*(inter//16)*(hidden//16)*(16*w2_bits)*2
    return gate,up,down

def _metadata(layer):
    names=('w13_trellis','w13_suh','w13_svh','w2_trellis','w2_suh','w2_svh')
    return tuple((name,getattr(layer,name).data_ptr(),getattr(getattr(layer,name),'_version',0),
                  tuple(getattr(layer,name).shape),str(getattr(layer,name).dtype)) for name in names)

class Native:
    def __init__(self,device):
        import torch
        if torch.cuda.get_device_capability(device)!=(12,1):raise ValueError('requires SM121')
        if torch.cuda.is_current_stream_capturing():raise ValueError('initialize outside graph capture')
        self.device=device;self.stream=torch.cuda.current_stream(device).cuda_stream
        path=Path(__file__).with_name('cooperative_moe.so')
        manifest=path.with_suffix('.json')
        import json
        receipt=json.loads(manifest.read_text())
        if receipt.get('abi')!=2 or receipt.get('sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('native ABI/hash receipt missing or mismatched')
        self.lib=C.CDLL(str(path));self.lib.dgx_tp3_coop_abi.restype=C.c_int
        if self.lib.dgx_tp3_coop_abi()!=2:raise ValueError('cooperative ABI differs')
        info_fn=self.lib.dgx_tp3_coop_info
        info_fn.argtypes=[C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int)];info_fn.restype=C.c_int
        for gate in (3,4):
            for down in (3,4):
                info=(C.c_int*18)()
                if info_fn(gate,down,2,info):raise RuntimeError('cooperative occupancy/info failed')
                if info[16]!=4227:raise ValueError('cooperative counter geometry differs')
        self.launch=self.lib.dgx_tp3_coop_launch
        self.launch.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_int,C.c_int,
                              C.c_float,C.c_int,C.c_int,C.c_void_p]
        self.launch.restype=C.c_int
        self.scratch=[torch.empty(shape,dtype=dtype,device=device) for shape,dtype in (
            ((288,5120),torch.float16),((288,5120),torch.float16),
            ((288,768),torch.float32),((288,768),torch.float32),
            ((288,768),torch.float16),((288,5120),torch.float32))]
        self.counters=torch.zeros(4227,dtype=torch.int32,device=device)
        self.tables={}

    def tables_for(self,layer):
        import torch
        metadata=_metadata(layer);key=id(layer)
        if key in self.tables:
            old,tables=self.tables[key]
            if old!=metadata:raise ValueError('weights were replaced after qualification')
            return tables
        if torch.cuda.is_current_stream_capturing():raise ValueError('pointer tables were not prepared')
        gate=[];up=[];down=[]
        for e in range(384):
            g,u,d=pointer_offsets(e,layer.exl3_w13_bits,layer.exl3_w2_bits)
            gate.append(layer.w13_trellis.data_ptr()+g);up.append(layer.w13_trellis.data_ptr()+u)
            down.append(layer.w2_trellis.data_ptr()+d)
        def rows(tensor,stride,offset=0):return [tensor.data_ptr()+e*stride+offset for e in range(384)]
        values=[gate,rows(layer.w13_suh,2*5120*2),rows(layer.w13_svh,2*768*2),
            up,rows(layer.w13_suh,2*5120*2,5120*2),rows(layer.w13_svh,2*768*2,768*2),
            down,rows(layer.w2_suh,768*2),rows(layer.w2_svh,5120*2)]
        tables=[torch.tensor(v,dtype=torch.int64,device=self.device) for v in values]
        self.tables[key]=(metadata,tables)
        return tables

    def run(self,method,layer,x,weights,ids):
        import torch
        tables=self.tables_for(layer)
        xh=x.contiguous().to(torch.float16)
        rw=weights.contiguous().to(torch.float32)  # keep original routing precision
        local=ids.contiguous().to(torch.int64)
        out=torch.empty((x.shape[0],5120),dtype=torch.float32,device=x.device)
        self.counters.zero_()  # graph replay never inherits completion state
        tensors=[xh,local,rw,*tables,*self.scratch,self.counters,out]
        pointers=(C.c_void_p*20)(*[t.data_ptr() for t in tensors])
        status=self.launch(pointers,layer.exl3_w13_bits,layer.exl3_w2_bits,x.shape[0],384,
            float(method.swiglu_limit or 0),2,1,C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
        if status:raise RuntimeError('cooperative CUDA launch status '+str(status))
        return out.to(x.dtype)

def _qualify(native,method,layer):
    global _qualifying
    import torch
    gen=torch.Generator(device=native.device);gen.manual_seed(8955)
    _qualifying=True
    try:
        for m in range(1,49):
            x=torch.randn((m,5120),device=native.device,generator=gen).to(torch.bfloat16)
            ids=torch.randint(0,384,(m,6),device=native.device,generator=gen,dtype=torch.int64)
            weights=torch.rand((m,6),device=native.device,generator=gen,dtype=torch.float32)
            weights/=weights.sum(1,keepdim=True)
            reference=method._dgx_apply_stock(layer,x,weights,ids)
            result=native.run(method,layer,x,weights,ids)
            if not torch.equal(reference.view(torch.uint16),result.view(torch.uint16)):
                raise NumericalMismatch('cooperative/0.8.8 raw BF16 differs at M='+str(m))
        # Verify graph replay and changed routing, including all slots to one
        # expert. No sampler or model-generation performance test is sent.
        x=torch.randn((48,5120),device=native.device,generator=gen).to(torch.bfloat16)
        ids=torch.zeros((48,6),device=native.device,dtype=torch.int64)
        weights=torch.full((48,6),1/6,device=native.device,dtype=torch.float32)
        graph=torch.cuda.CUDAGraph();side=torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream(native.device))
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph):result=native.run(method,layer,x,weights,ids)
        ids.fill_(383);graph.replay();torch.cuda.synchronize()
        reference=method._dgx_apply_stock(layer,x,weights,ids)
        if not torch.equal(reference.view(torch.uint16),result.view(torch.uint16)):
            raise NumericalMismatch('cooperative changed-routing graph differs from 0.8.8')
    finally:_qualifying=False

def maybe_apply(method,layer,x,weights,ids):
    import torch
    if _qualifying or os.environ.get('DGX_V089_COOPERATIVE')!='1' or id(layer) in _rejected:return None
    if (not x.is_cuda or x.dtype!=torch.bfloat16 or x.ndim!=2 or not 1<=x.shape[0]<=48
        or x.shape[1]!=5120 or tuple(ids.shape)!=(x.shape[0],6) or weights.shape!=ids.shape
        or getattr(layer,'expert_map',None) is not None or layer.exl3_num_experts!=384
        or layer.exl3_inter!=768 or layer.exl3_cb!=2 or layer.exl3_w13_bits not in (3,4)
        or layer.exl3_w2_bits not in (3,4) or not math.isfinite(float(method.swiglu_limit or 0))):return None
    native=_pool.get(str(x.device))
    if native is None:
        if torch.cuda.is_current_stream_capturing():return None
        try:native=_pool[str(x.device)]=Native(x.device)
        except (ValueError, FileNotFoundError, OSError) as exc:
            _rejected.add(id(layer));print('DGX_V089_COOP_DISARMED '+repr(exc)[:400],flush=True);return None
    # Shared scratch belongs to one serialized stream. Other streams use stock.
    if torch.cuda.current_stream(x.device).cuda_stream!=native.stream:return None
    if not getattr(layer,'_dgx_coop_qualified',False):
        if torch.cuda.is_current_stream_capturing():return None
        try:
            _qualify(native,method,layer);layer._dgx_coop_qualified=True
            print('DGX_V089_COOP_QUALIFIED K'+str(layer.exl3_w13_bits)+str(layer.exl3_w2_bits),flush=True)
        except NumericalMismatch as exc:
            _rejected.add(id(layer));print('DGX_V089_COOP_DISARMED '+repr(exc)[:400],flush=True);return None
    return native.run(method,layer,x,weights,ids)
