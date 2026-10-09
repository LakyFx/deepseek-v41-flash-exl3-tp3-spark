"""TP3 route alignment plus an explicit eager correctness gate."""
import hashlib
import os
from pathlib import Path

_extension=None
_rejected=False
_qualified=set()

class NumericalMismatch(ValueError):
    """Only exact comparison failures may select the stock implementation."""

def oracle(ids,block_m,experts=384):
    flat=[int(e) for row in ids for e in row]
    if not flat or len(flat)>288 or experts!=384 or block_m not in (16,32,64):
        raise ValueError('unsupported route geometry')
    if any(e<0 or e>=experts for e in flat):raise ValueError('invalid expert')
    sorted_ids=[];expert_ids=[]
    for e in range(experts):
        slots=[i for i,value in enumerate(flat) if value==e]
        pad=(-len(slots))%block_m
        sorted_ids.extend(slots+[len(flat)]*pad)
        expert_ids.extend([e]*((len(slots)+pad)//block_m))
    return sorted_ids,expert_ids,len(sorted_ids)

def _build():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        source=Path(__file__).with_name('moe_prep_kernel.cu')
        from .prebuilt import load_extension
        _extension=load_extension('moe_prep',source)
        if _extension is not None:
            return _extension
        tag=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        root=Path(os.environ.get('TORCH_EXTENSIONS_DIR','/tmp'))/('dgx-tp3-moe-prep-'+tag)
        root.mkdir(parents=True,exist_ok=True)
        _extension=load(name='dgx_tp3_moe_prep_'+tag,sources=[str(source)],build_directory=str(root),
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-gencode=arch=compute_121a,code=sm_121a'],verbose=False)
    return _extension

def _qualify(ext,stock,ids,block_m):
    import torch
    gen=torch.Generator(device=ids.device);gen.manual_seed(8951)
    # Physical K5/C8 rows including the graph buckets and partial batches.
    for m in range(1,49):
        for pattern in ('random','same','cyclic'):
            if pattern=='random':test=torch.randint(0,384,(m,6),generator=gen,device=ids.device,dtype=ids.dtype)
            elif pattern=='same':test=torch.zeros((m,6),device=ids.device,dtype=ids.dtype)
            else:test=(torch.arange(m*6,device=ids.device,dtype=ids.dtype)%384).reshape(m,6)
            a,b,n=ext.align(test,block_m);count=int(n.item())
            expected=oracle(test.cpu().tolist(),block_m)
            if count!=expected[2] or a[:count].cpu().tolist()!=expected[0] or b[:count//block_m].cpu().tolist()!=expected[1]:
                raise NumericalMismatch('routing kernel differs from CPU oracle')
            sa,sb,sn=stock(test,block_m,384,pad_sorted_ids=True)
            if int(sn.item())!=count or not torch.equal(sb[:count//block_m],b[:count//block_m]):
                raise NumericalMismatch('routing kernel differs from original expert runs')
            # Stock may choose a different valid order inside an expert.
            for offset in range(0,count,block_m):
                if not torch.equal(sa[offset:offset+block_m].sort()[0],a[offset:offset+block_m].sort()[0]):
                    raise NumericalMismatch('routing row multiset differs')
    # A captured launch must overwrite sentinels/padding on every replay.
    x=torch.zeros((48,6),device=ids.device,dtype=ids.dtype)
    graph=torch.cuda.CUDAGraph()
    side=torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream(ids.device))
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph):captured=ext.align(x,block_m)
    x.fill_(383);graph.replay();torch.cuda.synchronize()
    blocks=(48*6+block_m-1)//block_m
    if int(captured[2].item())!=blocks*block_m or captured[1][:blocks].cpu().tolist()!=[383]*blocks:
        raise NumericalMismatch('routing graph retained stale expert runs')

def align_or_stock(ids,block_m,experts,expert_map,stock):
    global _rejected
    import torch
    fallback=lambda:stock(ids,block_m,experts,expert_map=expert_map,pad_sorted_ids=True)
    if (_rejected or os.environ.get('DGX_V089_MOE_PREP')!='1' or expert_map is not None
        or experts!=384 or ids.shape[0]>48 or ids.shape[1]!=6 or not ids.is_cuda
        or not ids.is_contiguous() or block_m not in (16,32,64)):
        return fallback()
    key=(str(ids.device),str(ids.dtype),block_m)
    if key not in _qualified:
        if torch.cuda.is_current_stream_capturing():return fallback()
        try:
            ext=_build();_qualify(ext,stock,ids,block_m);_qualified.add(key)
            print('DGX_V089_MOE_PREP_QUALIFIED '+str(key),flush=True)
        except NumericalMismatch as exc:
            _rejected=True
            print('DGX_V089_MOE_PREP_DISARMED '+repr(exc)[:400],flush=True)
            return fallback()
    # A launch error after qualification propagates, never retry partially
    # executed device work. The service-level rollback handles this case.
    return tuple(_extension.align(ids,block_m))
