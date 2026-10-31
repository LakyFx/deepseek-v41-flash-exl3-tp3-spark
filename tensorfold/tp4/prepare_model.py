"""Create native TP4 metadata plus unchanged tokenizer and BF16 vision bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def prepare(source, out, release_manifest):
    if out.exists():
        raise FileExistsError('Use a new model directory')
    manifest = json.loads(release_manifest.read_bytes())
    if (manifest['status'] != 'READY_TP3_GPU_SMOKE_VERIFIED'
            or manifest['source_revision'] != '1bae0e1088993b9b82de08f59de5be5a34e42170'):
        raise ValueError('Wrong pinned 1017 source manifest')
    records = {row['path']: row for row in manifest['files']}
    original = json.loads((source/'config.json').read_bytes())
    if original['tensorfold_tp3'] != json.loads((HERE/'model/config.json').read_bytes())['tensorfold_tp4']['parent_tp3_provenance']:
        raise ValueError('Source metadata differs from the qualified 1017 TP3 arm')
    copies = []
    for name in ['tokenizer.json', 'model.safetensors.index.json', 'vision.safetensors']:
        path = source/name
        expected = records['models/TF351-1017/'+name]
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError('Pinned source bytes differ: '+name)
        copies.append(path)
    out.mkdir(parents=True)
    for source_file in copies:
        shutil.copyfile(source_file, out/source_file.name)
    for name in ['config.json', 'tokenizer_config.json', 'chat_template.jinja']:
        shutil.copyfile(HERE/'model'/name, out/name)
    return dict(version='4.0.0', world=4, weights_requantized=False, services_touched=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--release-manifest', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.out, a.release_manifest), indent=2))
