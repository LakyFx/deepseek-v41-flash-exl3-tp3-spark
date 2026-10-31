"""Byte-exact TP4 export from the three fully verified 1017 rank caches.

No dequantization, arithmetic or new precision. Native planner keys select
rectangles in the union of the old slices; only TP3 virtual padding is removed.
This is a data-preparation tool, never an engine/runtime patch.
"""
import argparse
import ast
import gzip
import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import time

SIZES={'uint8':1,'int16':2,'int32':4,'int64':8,'float16':2,'bfloat16':2,'float32':4,'float8_e4m3fn':1}
PLAN='dce5e37b0e61b13d6cf517a00271bd2ded407f0b0f2c25beed50ebe008d35007'


def publish_rank(pending, path, digest):
    """Keep interchange and native-discovery names on the same verified inode."""
    os.link(pending, path)
    os.link(path, path.with_name(f'{path.stem}-{digest[:20]}.bin'))


def identity(key):
    parts=key.split('|')
    if len(parts)==2:return parts[0],parts[1]
    if len(parts)!=4:raise ValueError('Bad key')
    return parts[0]+'.'+{'tr':'trellis','suh':'suh','svh':'svh'}[parts[3]],'None'


def bounds(key,shape):
    ranges=[(0,n) for n in shape];parts=key.split('|')
    if len(parts)==4:
        cols=ast.literal_eval(parts[1]);rows=ast.literal_eval(parts[2])
        pairs=[(0,rows,16),(1,cols,16)] if parts[3]=='tr' else [(0,rows if parts[3]=='suh' else cols,1)]
        for axis,ran,div in pairs:
            if ran is None:continue
            if any(n%div for n in ran):raise ValueError('Unaligned slice')
            ranges[axis]=tuple(n//div for n in ran)
    return ranges


def index(path):
    with path.open('rb') as f:
        f.seek(-16,2);tail=f.read(16)
        if tail[8:]!=b'TFDSRK01':raise ValueError('Bad footer')
        at=struct.unpack('<Q',tail[:8])[0]
        if not 0<at<path.stat().st_size-16:raise ValueError('Bad index offset')
        f.seek(at);entries=json.loads(f.read(path.stat().st_size-at-16))
    pos=0;seen=set()
    for key,dt,shape,offset,size in entries:
        if key in seen or offset!=pos or size!=math.prod(shape)*SIZES[dt]:raise ValueError('Bad index entry')
        seen.add(key);pos+=size
    if pos!=at:raise ValueError('Bad payload end')
    return entries


def groups(indexes):
    result={}
    for rank,entries in enumerate(indexes):
        for key,dt,shape,offset,size in entries:
            b=bounds(key,shape)
            if [hi-lo for lo,hi in b]!=shape:raise ValueError('Source shape/key mismatch: '+key)
            name=identity(key)
            g=result.setdefault(name,dict(dtype=dt,shape=[0]*len(shape),sources=[]))
            if g['dtype']!=dt or len(g['shape'])!=len(shape):raise ValueError('Inconsistent replicated dtype')
            g['shape']=[max(n,hi) for n,(_,hi) in zip(g['shape'],b)]
            g['sources'].append(dict(rank=rank,bounds=b,shape=shape,offset=offset,key=key))
    return result


def target(key,g,transform):
    name,_=identity(key);shape=list(g['shape']);op=transform(name)
    if op:shape[op.axis]-=op.count
    if '.attn.wo_a.slice.8.' in name:raise ValueError('Native target retained padded group')
    b=bounds(key,shape)
    if any(lo<0 or hi>n or lo>=hi for (lo,hi),n in zip(b,shape)):raise ValueError('Target outside native geometry: '+key)
    return b


def cells(wanted,sources):
    # Cut only at source rectangle boundaries. Each cell has exactly one
    # chosen source even when non-sharded parameters are replicated.
    axes=[]
    for axis,(lo,hi) in enumerate(wanted):
        cuts=sorted({lo,hi}|{n for s in sources for n in s['bounds'][axis] if lo<n<hi})
        axes.append(list(zip(cuts,cuts[1:])))
    for cell in itertools.product(*axes):
        choices=[s for s in sources if all(sl<=lo and hi<=sh for (lo,hi),(sl,sh) in zip(cell,s['bounds']))]
        if not choices:raise ValueError('Native cell not covered by TP3 sources: '+str(cell))
        yield cell,choices[0]


def flat(coords,shape):
    out=0
    for c,n in zip(coords,shape):out=out*n+c
    return out


def extract(wanted,g,maps,vectorized_bytes=False):
    shape=[hi-lo for lo,hi in wanted];item=SIZES[g['dtype']]
    output=bytearray(math.prod(shape)*item)
    if not shape:
        s=g['sources'][0];return maps[s['rank']][s['offset']:s['offset']+item]
    volume=0
    if vectorized_bytes:
        import numpy as np
        dst_view=np.ndarray((*shape,item),dtype=np.uint8,buffer=output)
    for cell,s in cells(wanted,g['sources']):
        volume+=math.prod(hi-lo for lo,hi in cell)
        if vectorized_bytes:
            src_view=np.ndarray((*s['shape'],item),dtype=np.uint8,buffer=maps[s['rank']],offset=s['offset'])
            src_slice=tuple(slice(lo-bound[0],hi-bound[0]) for (lo,hi),bound in zip(cell,s['bounds']))
            dst_slice=tuple(slice(lo-bound[0],hi-bound[0]) for (lo,hi),bound in zip(cell,wanted))
            dst_view[dst_slice]=src_view[src_slice]
            continue
        for outer in itertools.product(*(range(lo,hi) for lo,hi in cell[:-1])):
            coords=(*outer,cell[-1][0]);n=(cell[-1][1]-cell[-1][0])*item
            src=s['offset']+flat([c-lo for c,(lo,_) in zip(coords,s['bounds'])],s['shape'])*item
            dst=flat([c-lo for c,(lo,_) in zip(coords,wanted)],shape)*item
            output[dst:dst+n]=maps[s['rank']][src:src+n]
    if volume!=math.prod(shape):raise ValueError('Incomplete or overlapping output')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--source-code',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--write',action='store_true');p.add_argument('--vectorized-bytes',action='store_true');p.add_argument('--rank',type=int,choices=range(4));a=p.parse_args()
    sys.path.insert(0,str(a.source_code/'tools/dsv41-tp3'));from tp3_layout import transformation
    paths=[a.sources/f'rank{r}of3.bin' for r in range(3)]
    receipts=[json.loads(p.with_suffix('.receipt.json').read_bytes()) for p in paths]
    verified=[json.loads(p.with_suffix('.verified.json').read_bytes()) for p in paths]
    for r in range(3):
        if receipts[r]['plan_sha256']!=PLAN or verified[r]['status']!='ALL_PAYLOAD_BYTES_SOURCE_CHECKED_NOT_GPU_QUALIFIED' or not verified[r]['full_sha256_checked'] or verified[r]['producer_sha256']!=receipts[r]['artifact_sha256']:raise ValueError('Wrong or unverified 1017 source')
    indexes=[index(p) for p in paths];g=groups(indexes)
    keys=[gzip.decompress((a.recipe/f'rank{r}.keys.gz').read_bytes()).decode().splitlines() for r in range(4)]
    key_manifest=json.loads((a.recipe/'KEYS.json').read_bytes())
    sizes=[]
    for r,kk in enumerate(keys):
        raw=gzip.decompress((a.recipe/f'rank{r}.keys.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=key_manifest[str(r)]['sha256'] or len(set(kk))!=143530:raise ValueError('Native planner keys changed')
        total=0
        for k in kk:
            gg=g[identity(k)];wanted=target(k,gg,transformation)
            if wanted:list(cells(wanted,gg['sources']))
            total+=math.prod(hi-lo for lo,hi in wanted)*SIZES[gg['dtype']]
        sizes.append(total)
    print(json.dumps(dict(event='PLAN_PASS',payload_bytes=sizes,source_plan=PLAN)),flush=True)
    if not a.write:return
    a.output.mkdir(exist_ok=False)
    fingerprints=[(x.stat().st_size,x.stat().st_mtime_ns,x.stat().st_ino) for x in paths]
    for r,path in enumerate(paths):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(8<<20):h.update(block)
        if h.hexdigest()!=receipts[r]['artifact_sha256']:raise ValueError('Source full hash mismatch')
        print(json.dumps(dict(event='SOURCE_HASH_PASS',rank=r)),flush=True)
    files=[p.open('rb') for p in paths];maps=[mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) for f in files]
    try:
        for r in range(4) if a.rank is None else [a.rank]:
            kk=keys[r]
            path=a.output/f'rank{r}of4.bin';pending=path.with_suffix('.partial');h=hashlib.sha256();entries=[];pos=0;start=time.monotonic();digests=[]
            with pending.open('xb') as f:
                for i,k in enumerate(kk):
                    gg=g[identity(k)];b=target(k,gg,transformation);data=extract(b,gg,maps,a.vectorized_bytes)
                    if f.write(data)!=len(data):raise OSError('Short write')
                    h.update(data);digests.append(hashlib.sha256(data).hexdigest());entries.append([k,gg['dtype'],[hi-lo for lo,hi in b],pos,len(data)]);pos+=len(data)
                    if i%20000==0:print(json.dumps(dict(event='EXPORT',rank=r,entries=i,bytes=pos,seconds=time.monotonic()-start)),flush=True)
                tail=json.dumps(entries,separators=(',',':')).encode()+struct.pack('<Q',pos)+b'TFDSRK01';f.write(tail);h.update(tail);f.flush();os.fsync(f.fileno())
            if pos!=sizes[r]:raise ValueError('Output accounting differs')
            # Independent reread of every payload byte and complete artifact.
            with pending.open('rb') as f:
                reread=hashlib.sha256()
                for e,digest in zip(entries,digests):
                    data=f.read(e[4]);reread.update(data)
                    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Output readback differs')
                reread.update(f.read())
            if reread.hexdigest()!=h.hexdigest():raise ValueError('Full output hash differs')
            # Native rank_cache.find discovers rankNof4-*.bin, not the plain
            # interchange filename. Keep both as hardlinks to the same bytes.
            publish_rank(pending, path, h.hexdigest())
            result=dict(status='TP3_TO_TP4_BYTES_REPARTITIONED_FULL_READBACK_NOT_GPU_QUALIFIED',plan_sha256=PLAN,rank=r,entries=len(entries),payload_bytes=pos,artifact_bytes=path.stat().st_size,artifact_sha256=h.hexdigest(),keys_sha256=key_manifest[str(r)]['sha256'],source_hashes=[x['artifact_sha256'] for x in receipts],requantized=False,seconds=time.monotonic()-start)
            path.with_suffix('.receipt.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(event='RANK_COMPLETE',**result)),flush=True)
    finally:
        for m in maps:m.close()
        for f in files:f.close()
    if fingerprints!=[(x.stat().st_size,x.stat().st_mtime_ns,x.stat().st_ino) for x in paths]:raise ValueError('Source changed during export')


if __name__=='__main__':main()
