"""CPU verification of a direct mixed-source rank export.

Native loader keys define ordering; frozen source headers define layout. Probe
values are independently mapped by coordinates (not VirtualTensor/spans).
Three scalar probes per nonempty entry do NOT prove full numerical correctness.
Optional full SHA256 detects transport/storage corruption against the producer.
"""
import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import struct
from io_budget import Budget, Reader
from materialize_checkpoint import SIZES, read_header
from rank_cache import layout
from source_verify import check_receipt


def source_value(file, base, info, flat):
    """Independent scalar coordinate mapping through slices and TP3 padding."""
    spec=info['spec']; coords=[0]*len(info['outshape'])
    for axis in range(len(coords)-1,-1,-1):
        flat,coords[axis]=divmod(flat,info['outshape'][axis])
        coords[axis]+=info['bounds'][axis][0]
    if flat: raise ValueError('Probe exceeds entry')
    op=info['op']; zero=False
    if op and coords[op.axis]>=spec['shape'][op.axis]:
        zero=op.zero
        coords[op.axis]-=spec['shape'][op.axis]
    size=SIZES[spec['dtype']]
    if zero: value=bytes(size)
    else:
        at=0
        for n,c in zip(spec['shape'],coords):
            if not 0<=c<n: raise ValueError('Probe outside original tensor')
            at=at*n+c
        file.seek(base+spec['data_offsets'][0]+at*size)
        value=file.read(size)
        if len(value)!=size: raise ValueError('Short source probe')
    if spec['dtype']=='F16' and info['dtype']=='float32':
        value=struct.pack('<f',struct.unpack('<e',value)[0])
    return value


def verify(plan_path,keys_path,artifact,roots,receipts,*,full_hash=False,read_mib_s=0,native_layout=False):
    plan_raw=Path(plan_path).read_bytes();plan=json.loads(plan_raw)
    key_raw=Path(keys_path).read_bytes();keys=key_raw.decode().splitlines()
    receipt=json.loads(artifact.with_suffix('.receipt.json').read_bytes())
    if (receipt.get('status')!='WRITTEN_NOT_QUALIFIED' or receipt.get('arm')!=plan['arm']
            or receipt.get('plan_sha256')!=hashlib.sha256(plan_raw).hexdigest()
            or receipt.get('keys_sha256')!=hashlib.sha256(key_raw).hexdigest()
            or receipt.get('entries')!=len(keys)):
        raise ValueError('Producer receipt does not bind this plan/keys')
    if receipt.get('native_layout',False)!=native_layout: raise ValueError('Producer layout mode differs')
    if len(set(keys))!=len(keys): raise ValueError('Duplicate native key')
    size=artifact.stat().st_size
    if size!=receipt.get('artifact_bytes') or size<16: raise ValueError('Artifact size differs')
    budget=Budget(read_mib_s);files={};needed={};probes=0
    for name,t in plan['tensors'].items():needed.setdefault(t['source'],set()).add(t['shard'])
    for source,shards in needed.items():
        if source not in receipts: raise ValueError('Missing verified source '+source)
        check_receipt(receipts[source],roots[source],plan['sources'][source],shards)
    before=artifact.stat()
    try:
        with Reader(artifact.open('rb'),budget) as f:
            f.seek(size-16);tail=f.read(16)
            if tail[8:]!=b'TFDSRK01': raise ValueError('Invalid rank footer')
            end=struct.unpack('<Q',tail[:8])[0]
            if not 0<=end<=size-16 or size-16-end>128<<20: raise ValueError('Invalid index extent')
            f.seek(end);index=json.loads(f.read(size-16-end))
            if len(index)!=len(keys): raise ValueError('Native entry count differs')
            at=0
            for key,entry in zip(keys,index):
                info=layout(key,plan['tensors'],padded=not native_layout)
                expected=[key,info['dtype'],info['outshape'],at,info['bytes']]
                if entry!=expected: raise ValueError('Index/layout differs: '+key)
                at+=info['bytes'];spec=info['spec'];loc=(spec['source'],spec['shard'])
                if loc not in files:
                    src=Reader((roots[loc[0]]/loc[1]).open('rb'),budget)
                    try:h,base=read_header(src)
                    except BaseException:src.close();raise
                    files[loc]=(src,h,base)
                src,h,base=files[loc]
                if h[info['source_name']]!={k:spec[k] for k in ('dtype','shape','data_offsets')}:
                    raise ValueError('Source header differs: '+key)
                count=prod(info['outshape'])
                if count:
                    itemsize=info['bytes']//count
                    for flat in sorted({0,count//2,count-1}):
                        expected_value=source_value(src,base,info,flat)
                        f.seek(entry[3]+flat*itemsize)
                        if f.read(itemsize)!=expected_value:raise ValueError('Source probe differs: '+key)
                        probes+=1
            if at!=end or at!=receipt['payload_bytes']:raise ValueError('Payload extent differs')
            if full_hash:
                f.seek(0);sha=hashlib.sha256()
                while data:=f.read(8<<20):sha.update(data)
                if sha.hexdigest()!=receipt['artifact_sha256']:raise ValueError('Artifact SHA256 differs')
        after=artifact.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Artifact changed during verification')
        for source,shards in needed.items():
            check_receipt(receipts[source],roots[source],plan['sources'][source],shards)
        return dict(status='CPU_CHECKED_NOT_GPU_QUALIFIED',arm=plan['arm'],entries=len(keys),
                    payload_bytes=at,source_scalar_probes=probes,full_sha256_checked=full_hash,
                    producer_sha256=receipt['artifact_sha256'],gpu_used=False)
    finally:
        for file,_,_ in files.values():file.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','keys','artifact','pollard','mia','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--source-receipt',action='append',default=[])
    p.add_argument('--full-hash',action='store_true')
    p.add_argument('--native-layout',action='store_true')
    p.add_argument('--read-mib-s',type=float,default=0)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    receipts={}
    for text in a.source_receipt:
        source,path=text.split('=',1)
        if source in receipts:raise ValueError('Duplicate source receipt')
        receipts[source]=json.loads(Path(path).read_bytes())
    result=verify(a.plan,a.keys,Path(a.artifact),{'pollard':Path(a.pollard),'mia':Path(a.mia)},
                  receipts,full_hash=a.full_hash,read_mib_s=a.read_mib_s,native_layout=a.native_layout)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
