"""Direct CPU rank-cache export from pinned tensor manifests.

Uses native plan_main.zig keys, not a second implementation of sharding policy.
Virtual padding is streamed into the rank's slices; no full padded checkpoint
copy is needed. Dry run is the default. GPU correctness remains a separate gate.
"""
import argparse
import ast
from bisect import bisect_right
import hashlib
import json
from math import prod
import os
from pathlib import Path
import struct
from materialize_checkpoint import SIZES, read_header
from tp3_layout import transformation, new_shape, spans
from source_verify import check_receipt
from io_budget import Budget, Reader

DTYPE={'BF16':'bfloat16','F16':'float16','F32':'float32','I16':'int16',
       'I32':'int32','I64':'int64','U8':'uint8','F8_E4M3':'float8_e4m3fn'}
CHUNK=1<<20


class DigestWriter:
    """Hash exactly the complete artifact, including its index and footer."""
    def __init__(self,file):
        self.file=file; self.hash=hashlib.sha256()

    def write(self,data):
        n=self.file.write(data)
        if n!=len(data): raise OSError('Short rank-cache write')
        self.hash.update(data)
        return n

    def tell(self): return self.file.tell()
    def flush(self): self.file.flush()
    def fileno(self): return self.file.fileno()


def parse_range(text):
    if text=='None': return None
    r=ast.literal_eval(text)
    if not isinstance(r,tuple) or len(r)!=2 or any(type(x)!=int for x in r) or not 0<=r[0]<=r[1]:
        raise ValueError('Invalid range')
    return r


def layout(key,tensors, *, padded=True):
    fields=key.split('|')
    if len(fields)==2:
        name,cast=fields; rows=cols=None; part=None
        if cast not in ('None','torch.float16','torch.float32'): raise ValueError('Invalid cast')
    elif len(fields)==4:
        name,c,r,part=fields; cols=parse_range(c); rows=parse_range(r); cast='None'
        name+='.'+{'tr':'trellis','suh':'suh','svh':'svh'}[part]
    else: raise ValueError('Invalid native key')
    source_name=name if name in tensors else name.replace('.attn.wo_a.slice.8.','.attn.wo_a.slice.0.')
    t=tensors[source_name]; op=transformation(name) if padded else None
    shape=new_shape(t['shape'],op) if op else list(t['shape'])
    bounds=[(0,n) for n in shape]
    if part=='tr':
        for axis,ran in [(0,rows),(1,cols)]:
            if ran is not None:
                if any(x%128 for x in ran): raise ValueError('EXL3 slice crosses Hadamard block')
                bounds[axis]=(ran[0]//16,ran[1]//16)
    elif part in ('suh','svh'):
        ran=rows if part=='suh' else cols
        if ran is not None: bounds[0]=ran
    if any(lo<0 or hi>n or lo>hi for (lo,hi),n in zip(bounds,shape)):
        raise ValueError('Slice outside tensor: '+key)
    outshape=[hi-lo for lo,hi in bounds]
    dtype=DTYPE[t['dtype']] if cast=='None' else cast.removeprefix('torch.')
    if dtype!=DTYPE[t['dtype']] and not (t['dtype']=='F16' and dtype=='float32'):
        raise ValueError('Unimplemented exact conversion: '+key+' '+t['dtype'])
    es=4 if dtype=='float32' else SIZES[t['dtype']]
    return dict(name=name,source_name=source_name,spec=t,op=op,shape=shape,bounds=bounds,
                outshape=outshape,dtype=dtype,bytes=prod(outshape)*es)


def selected_intervals(shape,bounds,itemsize):
    """Contiguous last-axis intervals in row-major order, relative to tensor."""
    def walk(axis,offset):
        lo,hi=bounds[axis]
        stride=prod(shape[axis+1:])
        if all(r==(0,n) for r,n in zip(bounds[axis+1:],shape[axis+1:])):
            # Merge complete trailing axes. Otherwise packed trellis shards
            # cause one seek/read per codeword instead of one per tile row.
            if hi>lo: yield (offset+lo*stride)*itemsize,(hi-lo)*stride*itemsize
        else:
            for i in range(lo,hi): yield from walk(axis+1,offset+i*stride)
    if not shape: yield 0,itemsize
    elif all(r==(0,n) for r,n in zip(bounds,shape)):
        yield 0,prod(shape)*itemsize
    else: yield from walk(0,0)


class VirtualTensor:
    def __init__(self,file,base,spec,op):
        self.file=file; self.base=base+spec['data_offsets'][0]
        parts=list(spans(spec['shape'],SIZES[spec['dtype']],op)) if op else [(0,spec['bytes'])]
        self.starts=[]; self.parts=parts; at=0
        for _,n in parts: self.starts.append(at); at+=n
        self.size=at

    def chunks(self,offset,size):
        if not 0<=offset<=offset+size<=self.size: raise ValueError('Virtual read outside tensor')
        while size:
            i=bisect_right(self.starts,offset)-1
            src,n=self.parts[i]; within=offset-self.starts[i]
            take=min(size,n-within,CHUNK)
            if src is None: data=bytes(take)
            else:
                self.file.seek(self.base+src+within); data=self.file.read(take)
                if len(data)!=take: raise ValueError('Truncated source payload')
            yield data
            offset+=take; size-=take


def write_tensor(file,base,l,out):
    v=VirtualTensor(file,base,l['spec'],l['op'])
    for offset,n in selected_intervals(l['shape'],l['bounds'],SIZES[l['spec']['dtype']]):
        for data in v.chunks(offset,n):
            if l['spec']['dtype']=='F16' and l['dtype']=='float32':
                data=b''.join(struct.pack('<f',x[0]) for x in struct.iter_unpack('<e',data))
            out.write(data)


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','keys','pollard','mia','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--source-receipt',action='append',default=[],metavar='SOURCE=PATH')
    p.add_argument('--read-mib-s',type=float,default=0,help='Shared source read limit, 0=unlimited')
    p.add_argument('--native-layout',action='store_true',help='TP4: native shapes, no TP3 virtual padding')
    p.add_argument('--write',action='store_true'); a=p.parse_args()
    plan_bytes=Path(a.plan).read_bytes(); plan=json.loads(plan_bytes)
    key_bytes=Path(a.keys).read_bytes(); keys=key_bytes.decode().splitlines()
    if len(set(keys))!=len(keys): raise ValueError('Duplicate native keys')
    entries=[layout(k,plan['tensors'],padded=not a.native_layout) for k in keys]
    receipt={'arm':plan['arm'],'keys_sha256':hashlib.sha256(key_bytes).hexdigest(),
             'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),'entries':len(keys),
             'payload_bytes':sum(x['bytes'] for x in entries),'status':'DRY_RUN'}
    receipt['native_layout']=a.native_layout
    if not a.write:
        print(json.dumps(receipt,indent=2)); return
    target=Path(a.out); pending=target.with_suffix('.partial')
    receipt_path=target.with_suffix('.receipt.json')
    if target.exists() or pending.exists() or receipt_path.exists(): raise FileExistsError(target)
    roots={'pollard':Path(a.pollard),'mia':Path(a.mia)}
    receipts={}; needed={}
    for item in a.source_receipt:
        source,path=item.split('=',1)
        if source in receipts: raise ValueError('Duplicate source receipt')
        receipts[source]=json.loads(Path(path).read_bytes())
    for l in entries:
        spec=l['spec']; needed.setdefault(spec['source'],set()).add(spec['shard'])
    for source,shards in needed.items():
        if source not in receipts: raise ValueError('Missing --source-receipt for '+source)
        check_receipt(receipts[source],roots[source],plan['sources'][source],shards)
    files={}; index=[]; at=0; budget=Budget(a.read_mib_s)
    try:
        with pending.open('xb') as raw:
            out=DigestWriter(raw)
            for number,(key,l) in enumerate(zip(keys,entries)):
                spec=l['spec']; loc=(spec['source'],spec['shard'])
                if loc not in files:
                    f=Reader((roots[loc[0]]/loc[1]).open('rb'),budget); h,base=read_header(f)
                    files[loc]=(f,h,base)
                f,h,base=files[loc]
                if h[l['source_name']]!={k:spec[k] for k in ('dtype','shape','data_offsets')}:
                    raise ValueError('Source changed: '+l['source_name'])
                write_tensor(f,base,l,out)
                index.append([key,l['dtype'],l['outshape'],at,l['bytes']]); at+=l['bytes']
                if out.tell()!=at: raise ValueError('Rank-cache byte accounting mismatch')
                if number%10000==0: print(f'{number}/{len(keys)} {at} bytes',flush=True)
            out.write(json.dumps(index,separators=(',',':')).encode())
            out.write(struct.pack('<Q',at)+b'TFDSRK01');out.flush();os.fsync(out.fileno())
            receipt['artifact_sha256']=out.hash.hexdigest()
            receipt['artifact_bytes']=out.tell()
        for source,shards in needed.items():
            check_receipt(receipts[source],roots[source],plan['sources'][source],shards)
        os.link(pending,target)
        receipt['status']='WRITTEN_NOT_QUALIFIED'
        with receipt_path.open('x') as f:
            json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    finally:
        for f,_,_ in files.values(): f.close()


if __name__=='__main__': main()
