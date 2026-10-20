"""Build 1.2.3 from verified extracted inputs; no model load or service changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import patch


def build(build_root, out, zig='zig', *, prepare_only=False):
    source = build_root/'source'
    patch.apply(source)
    manifest = patch.verify(source)
    if prepare_only:
        return dict(status='SOURCE_ONLY_VERIFIED',version='1.2.3',model_requests=0)
    out.mkdir(parents=True,exist_ok=False)
    commands = [
        [zig,'test','zig/src/core/lanes/adaptive.zig','-O','ReleaseSafe'],
        [zig,'build','test','-j2','--summary','all'],
        [zig,'test','zig/src/core/lanes/overhead_check.zig','-O','ReleaseFast'],
        [zig,'build','native','dsv41-tools','-j2','-Dtarget=aarch64-linux-gnu',
         '-Dfatbins='+str((build_root/'kit/build-fatbins').resolve()),'--prefix',str(out.resolve()),'--summary','all'],
    ]
    logs = []
    for i,command in enumerate(commands):
        path = out/f'build-{i+1}.log'
        with path.open('wb') as log:
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        logs.append(dict(step=i+1,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    bins = {}
    for name in ['tensorfold-native','tf-dsv41-lanes']:
        found = [p for p in out.rglob(name) if p.is_file()]
        if len(found)!=1:
            raise ValueError('Expected one freshly built executable: '+name)
        bins[name]=dict(path=found[0].relative_to(out).as_posix(),
            sha256=hashlib.sha256(found[0].read_bytes()).hexdigest())
    receipt = dict(version='1.2.3',adaptive_learning=True,
        source_archive_sha256=manifest['source_archive_sha256'],source_hashes=manifest['patched_sha256'],
        overlay_sha256=manifest['overlay_sha256']['adaptive.zig'],binaries=bins,
        cpu_steps=logs,model_requests=0,gpu_qualified_on_this_build_host=False)
    (out/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root',type=Path,required=True,help='New tensorfold/extract.py output')
    p.add_argument('--out',type=Path,required=True,help='New build output, never an existing runtime')
    p.add_argument('--zig',default='zig')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(build(a.build_root.resolve(),a.out.resolve(),a.zig,prepare_only=a.prepare_only),indent=2))


if __name__=='__main__':main()
