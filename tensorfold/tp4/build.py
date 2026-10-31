"""Stage exact production TP4 binaries/kernels, or rebuild the pinned source.

Preparation only: no inference, services, networking, or existing runtime edits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from tensorfold.tp4 import assets
from tensorfold.variants import patch

HERE = Path(__file__).resolve().parent


def prepare(kit, out, source_build=None, zig='zig'):
    profile = json.loads((HERE/'profile.json').read_bytes())
    spec, identity = patch.recipe('v400')
    # Verify all kernel inputs before creating the new output.
    assets.inputs(kit, profile['prefill_assets_sha256'])
    archive = HERE/'assets/native-linux-arm64.tar.gz'
    if patch.digest(archive.read_bytes()) != profile['native_archive_sha256']:
        raise ValueError('Wrong TP4 native archive')
    if out.exists():
        raise FileExistsError('Use a new build output; previous versions are preserved')
    if source_build:
        from tensorfold.variants.build import build
        receipt = build(source_build, out, 'v400', zig)
        (out/'native').mkdir()
        for name, record in receipt['binaries'].items():
            shutil.copyfile(out.joinpath(*patch.relative(record['path']).parts), out/'native'/name)
        provenance = 'rebuilt_from_exact_source; destination GPU qualification required'
    else:
        out.mkdir(parents=True)
        expected = {'native/'+k: v for k, v in profile['historical_native_binaries'].items()}
        seen = set()
        with tarfile.open(archive) as stream:
            for member in stream:
                patch.relative(member.name)
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected native archive member')
                raw = stream.extractfile(member).read()
                if patch.digest(raw) != expected[member.name]:
                    raise ValueError('Native bytes differ from measured production')
                target = out/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError('Incomplete native archive')
        provenance = 'exact measured P2/P3 production binaries'
    for name in profile['historical_native_binaries']:
        (out/'native'/name).chmod(0o755)
    assets.prepare(kit, out/'prefill', profile['prefill_assets_sha256'])
    receipt = dict(schema='ds41f.tp4-build.v1', version='4.0.0', recipe='v400',
        recipe_manifest_sha256=identity, source_tree_sha256=spec['source_tree_sha256'],
        source_archive_sha256=spec['base_archive_sha256'], adaptive_learning=True,
        prefill_assets_sha256=profile['prefill_assets_sha256'], provenance=provenance,
        binaries={name: dict(path='native/'+name,
                  sha256=patch.digest((out/'native'/name).read_bytes()))
                  for name in profile['historical_native_binaries']},
        gpu_qualified_on_this_host=False, services_touched=False)
    (out/'BUILD.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-build', type=Path, help='New extract.py root, for optional recompilation')
    p.add_argument('--zig', default='zig')
    a = p.parse_args()
    print(json.dumps(prepare(a.kit.resolve(), a.out.resolve(), a.source_build, a.zig), indent=2))
