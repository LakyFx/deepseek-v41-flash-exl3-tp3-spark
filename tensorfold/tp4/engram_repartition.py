"""Copy exact TP2-owned hash columns out of the three verified sparse tables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--kit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.kit/'engram.json').read_bytes())
    meta=[json.loads((a.inputs/f'engram-r{r}/SOURCE.json').read_bytes()) for r in range(3)]
    # Verify the input-owned bytes, not merely the copy/readback of the output.
    for rank, entry in enumerate(meta):
        if (entry['rank'] != rank or entry['world'] != 3
                or entry['original_revision'] != '1bae0e1088993b9b82de08f59de5be5a34e42170'):
            raise ValueError('Wrong 1017 Engram source')
        for name, record in entry['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe Engram source name')
            source = a.inputs/f'engram-r{rank}'/name
            if source.stat().st_size != record['size']:
                raise ValueError('Incomplete Engram source')
            with source.open('rb') as stream:
                if hashlib.sha256(stream.read(record['header_bytes'])).hexdigest() != record['header_sha256']:
                    raise ValueError('Wrong Engram header')
                for owned in record['ranges']:
                    stream.seek(owned['offset']); left = owned['bytes']; check = hashlib.sha256()
                    while left:
                        block = stream.read(min(8 << 20, left))
                        if not block:
                            raise EOFError('Incomplete Engram source range')
                        check.update(block); left -= len(block)
                    if check.hexdigest() != owned['sha256']:
                        raise ValueError('Wrong Engram source range')
    a.output.mkdir(exist_ok=False)
    for rank in range(2):
        out=a.output/f'engram-tp2-r{rank}';out.mkdir();records={};started=time.monotonic()
        for j,layer in enumerate(config['layers']):
            offsets=config['offsets'][j]+[sum(config['primes'][j])]
            lo,hi=offsets[rank*12],offsets[(rank+1)*12]
            name=f'model-000{47+j}-of-00048.safetensors'
            source=a.inputs/'engram-r0'/name
            with source.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];header=f.read(n)
            specs=json.loads(header);base=8+n
            assert meta[0]['files'][name]['header_bytes']==base
            target=out/name;record=dict(header_bytes=base,header_sha256=hashlib.sha256(struct.pack('<Q',n)+header).hexdigest(),size=meta[0]['files'][name]['size'],ranges=[])
            with target.open('xb+') as dst:
                dst.truncate(record['size']);dst.write(struct.pack('<Q',n)+header)
                for suffix,width in [('weight',256),('scale',8)]:
                    tensor=f'layers.{layer}.engram.embed.{suffix}';spec=specs[tensor]
                    assert spec['shape']==[offsets[-1],width]
                    begin=base+spec['data_offsets'][0]+lo*width;end=begin+(hi-lo)*width
                    cursor=begin;h=hashlib.sha256()
                    for oldrank in range(3):
                        entry=meta[oldrank]['files'][name]
                        assert entry['header_sha256']==meta[0]['files'][name]['header_sha256']
                        src_range=next(x for x in entry['ranges'] if x['tensor']==tensor)
                        low=max(begin,src_range['offset']);high=min(end,src_range['offset']+src_range['bytes'])
                        if high<=low:continue
                        if low!=cursor:raise ValueError('Source gap/overlap')
                        with (a.inputs/f'engram-r{oldrank}'/name).open('rb') as src:
                            src.seek(low);dst.seek(low);left=high-low
                            while left:
                                b=src.read(min(8<<20,left))
                                if not b:raise EOFError('Source changed')
                                if dst.write(b)!=len(b):raise OSError('Short write')
                                h.update(b);left-=len(b)
                        cursor=high
                    if cursor!=end:raise ValueError('Missing native rows')
                    dst.flush();dst.seek(begin);check=hashlib.sha256();left=end-begin
                    while left:
                        b=dst.read(min(8<<20,left))
                        if not b:raise EOFError('Incomplete Engram readback')
                        check.update(b);left-=len(b)
                    if check.hexdigest()!=h.hexdigest():raise ValueError('Engram output differs')
                    record['ranges'].append(dict(tensor=tensor,row_range=[lo,hi],offset=begin,bytes=end-begin,sha256=h.hexdigest()))
                dst.flush();os.fsync(dst.fileno())
            records[name]=record
            print(json.dumps(dict(event='ENGRAM_LAYER_COMPLETE',rank=rank,layer=layer,rows=[lo,hi],seconds=time.monotonic()-started)),flush=True)
        result=dict(status='TP2_OWNED_RANGES_FULL_READBACK_VERIFIED',world=2,rank=rank,source_world=3,files=records,requantized=False)
        (out/'SOURCE.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
